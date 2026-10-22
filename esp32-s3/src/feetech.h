#pragma once
#include <Arduino.h>

// Feetech STS/SCS half-duplex serial protocol (STS3215 and friends).
//
//   request : FF FF <id> <len> <inst> [params...] <chk>
//   reply   : FF FF <id> <len> <err>  [data...]   <chk>
//   len     = param count + 2
//   chk     = ~(id + len + inst + params)  truncated to 8 bits
//
// The bus is single-wire half-duplex, so a request usually comes straight back
// as an echo before the real reply. Every read here skips that echo explicitly
// rather than hoping it is absent -- on this board it is always present.

namespace feetech {

static const uint8_t INST_PING  = 0x01;
static const uint8_t INST_READ  = 0x02;
static const uint8_t INST_WRITE = 0x03;
static const uint8_t INST_SYNC_WRITE = 0x83;
static const uint8_t INST_SYNC_READ  = 0x82;

// STS3215 register map (the subset used here).
static const uint8_t REG_TORQUE_EN  = 0x28;  // 1 byte
static const uint8_t REG_ACC        = 0x29;  // 1 byte, units of 100 steps/s^2
static const uint8_t REG_GOAL_POS   = 0x2A;  // 2 bytes, 0..4095 over 360 deg
static const uint8_t REG_GOAL_SPEED = 0x2E;  // 2 bytes, steps/s
static const uint8_t REG_PRES_POS   = 0x38;  // 2 bytes
static const uint8_t REG_PRES_VOLT  = 0x3E;  // 1 byte, 0.1 V
static const uint8_t REG_PRES_TEMP  = 0x3F;  // 1 byte, deg C

static const uint16_t POS_MAX = 4095;

struct Echo {
  uint8_t bytes[16];
  uint8_t len;
};

inline void sendPacket(HardwareSerial &bus, uint8_t id, uint8_t inst,
                       const uint8_t *params, uint8_t n, Echo &echo) {
  uint8_t pkt[16];
  uint8_t len = n + 2;
  pkt[0] = 0xFF; pkt[1] = 0xFF; pkt[2] = id; pkt[3] = len; pkt[4] = inst;
  memcpy(&pkt[5], params, n);

  uint16_t sum = id + len + inst;
  for (uint8_t i = 0; i < n; i++) sum += params[i];
  pkt[5 + n] = (uint8_t)~sum;

  echo.len = 6 + n;
  memcpy(echo.bytes, pkt, echo.len);

  while (bus.available()) bus.read();
  bus.write(pkt, echo.len);
  bus.flush();
}

// Waits for a well-formed reply from `id` carrying `nData` payload bytes.
// `rxCount`, when given, reports how many bytes arrived at all -- non-zero with
// no valid reply means the wire is live even though the framing did not match.
inline bool recvReply(HardwareSerial &bus, uint8_t id, uint8_t nData, uint8_t *out,
                      const Echo &echo, size_t *rxCount = nullptr,
                      uint32_t timeoutMs = 20) {
  uint8_t buf[64];
  size_t cnt = 0;
  uint32_t t0 = millis();

  // Return the moment a complete frame has arrived. Waiting out the full
  // timeout on every transaction costs ~20 ms per servo, which is the entire
  // 50 Hz control budget -- fine for a one-shot probe, fatal in a control loop.
  while (millis() - t0 < timeoutMs && cnt < sizeof(buf)) {
    if (!bus.available()) continue;
    buf[cnt++] = (uint8_t)bus.read();
    if (cnt < 6u + nData) continue;

    size_t start = (cnt >= echo.len && memcmp(buf, echo.bytes, echo.len) == 0) ? echo.len : 0;
    for (size_t i = start; i + 6 + nData <= cnt; i++) {
      if (buf[i] != 0xFF || buf[i + 1] != 0xFF) continue;
      if (buf[i + 2] != id || buf[i + 3] != nData + 2) continue;
      uint16_t sum = id + buf[i + 3] + buf[i + 4];
      for (uint8_t k = 0; k < nData; k++) sum += buf[i + 5 + k];
      if ((uint8_t)~sum != buf[i + 5 + nData]) continue;
      if (buf[i + 4] != 0) return false;  // servo reports a hardware/protocol error
      if (nData && out) memcpy(out, &buf[i + 5], nData);
      if (rxCount) *rxCount = cnt;
      return true;
    }
  }
  // Timed out: report everything seen, so a caller can tell a silent bus from
  // a live one that never framed correctly.
  if (rxCount) *rxCount = cnt;
  return false;
}

inline bool ping(HardwareSerial &bus, uint8_t id, size_t *rxCount = nullptr) {
  Echo e;
  sendPacket(bus, id, INST_PING, nullptr, 0, e);
  return recvReply(bus, id, 0, nullptr, e, rxCount);
}

inline bool readRegs(HardwareSerial &bus, uint8_t id, uint8_t addr, uint8_t n,
                     uint8_t *out, uint32_t timeoutMs = 20) {
  uint8_t params[2] = {addr, n};
  Echo e;
  sendPacket(bus, id, INST_READ, params, sizeof(params), e);
  return recvReply(bus, id, n, out, e, nullptr, timeoutMs);
}

inline bool writeRegs(HardwareSerial &bus, uint8_t id, uint8_t addr,
                      const uint8_t *data, uint8_t n) {
  uint8_t params[8];
  params[0] = addr;
  memcpy(&params[1], data, n);
  Echo e;
  sendPacket(bus, id, INST_WRITE, params, n + 1, e);
  return recvReply(bus, id, 0, nullptr, e);
}

inline bool writeU8(HardwareSerial &bus, uint8_t id, uint8_t addr, uint8_t v) {
  return writeRegs(bus, id, addr, &v, 1);
}

inline bool writeU16(HardwareSerial &bus, uint8_t id, uint8_t addr, uint16_t v) {
  uint8_t d[2] = {(uint8_t)(v & 0xFF), (uint8_t)(v >> 8)};
  return writeRegs(bus, id, addr, d, 2);
}

inline bool readPosition(HardwareSerial &bus, uint8_t id, uint16_t *pos) {
  uint8_t d[2];
  if (!readRegs(bus, id, REG_PRES_POS, 2, d)) return false;
  *pos = d[0] | (d[1] << 8);
  return true;
}

// Feetech reports speed and load as sign-magnitude, not two's complement.
inline int signMag(uint16_t v) {
  return (v & 0x8000) ? -(int)(v & 0x7FFF) : (int)(v & 0x7FFF);
}

// SYNC WRITE (0x83): one broadcast packet sets every servo's goal at once.
// Writing 12 goals individually staggers them across ~4 ms of a 20 ms tick,
// so the legs act on commands from slightly different instants; this makes the
// whole pose land together and costs one transaction instead of twelve.
inline void syncWriteGoals(HardwareSerial &bus, const uint8_t *ids,
                           const uint16_t *goals, uint8_t n) {
  uint8_t pkt[8 + 3 * 16];
  uint8_t len = (2 + 1) * n + 4;
  pkt[0] = 0xFF; pkt[1] = 0xFF; pkt[2] = 0xFE; pkt[3] = len;
  pkt[4] = INST_SYNC_WRITE; pkt[5] = REG_GOAL_POS; pkt[6] = 2;
  uint16_t sum = 0xFE + len + INST_SYNC_WRITE + REG_GOAL_POS + 2;
  uint8_t k = 7;
  for (uint8_t i = 0; i < n; i++) {
    uint8_t lo = goals[i] & 0xFF, hi = goals[i] >> 8;
    pkt[k++] = ids[i]; pkt[k++] = lo; pkt[k++] = hi;
    sum += ids[i] + lo + hi;
  }
  pkt[k++] = (uint8_t)~sum;
  while (bus.available()) bus.read();
  bus.write(pkt, k);
  bus.flush();
}

// SYNC READ (0x82): sample every servo in ONE transaction.
//
// Reading 12 servos with 12 separate round trips takes ~5.3 ms, and because the
// id list is ordered right-leg-then-left-leg, it samples one leg up to ~2.6 ms
// staler than the other -- a systematic inter-leg skew in joint_pos and
// joint_vel on every tick. The Pi never had this (scservo_sdk GroupSyncRead is
// one transaction), so it shows up as a left/right asymmetry the moment the
// controller is swapped.
//
// Request: FF FF FE <len> 82 <addr> <dataLen> <id...> <chk>,  len = n + 4
// Reply per servo: FF FF <id> <dataLen+2> <err> <data...> <chk>
inline bool syncRead(HardwareSerial &bus, const uint8_t *ids, uint8_t n,
                     uint8_t addr, uint8_t dataLen, uint8_t *out,
                     uint32_t timeoutMs = 10, uint16_t *validMask = nullptr,
                     uint16_t *errorMask = nullptr, uint8_t *status = nullptr,
                     uint32_t idleBreakUs = 0) {
  if (validMask) *validMask = 0;
  if (errorMask) *errorMask = 0;
  if (!n || n > 16 || !dataLen || dataLen > 16) return false;
  uint8_t pkt[8 + 16];
  uint8_t len = n + 4;
  pkt[0] = 0xFF; pkt[1] = 0xFF; pkt[2] = 0xFE; pkt[3] = len;
  pkt[4] = INST_SYNC_READ; pkt[5] = addr; pkt[6] = dataLen;
  uint16_t sum = 0xFE + len + INST_SYNC_READ + addr + dataLen;
  uint8_t k = 7;
  for (uint8_t i = 0; i < n; i++) { pkt[k++] = ids[i]; sum += ids[i]; }
  pkt[k++] = (uint8_t)~sum;

  while (bus.available()) bus.read();
  bus.write(pkt, k);
  bus.flush();

  // Parse as the bytes arrive and return the instant all n replies are in.
  // Waiting for a fixed byte count instead burns the whole timeout whenever the
  // bus does not echo the request -- measured 9.5 ms, worse than the sequential
  // reads this is meant to replace.
  uint8_t buf[400];
  size_t cnt=0, scan=0;
  uint32_t t0=millis();
  uint16_t seen=0;
  uint8_t remaining=n;
  bool errors=false;
  // Replies arrive back to back (~0.7 ms apart for a 15-byte payload). Once the
  // stream goes quiet for much longer than that, the servos that were going to
  // answer have answered; waiting out the rest of the timeout for a silent one
  // just spends control budget. 0 keeps the old wait-the-whole-timeout policy.
  uint32_t lastRx=micros();
  while(millis()-t0<timeoutMs && cnt<sizeof(buf)) {
    if(!bus.available()) {
      if(idleBreakUs && cnt && uint32_t(micros()-lastRx)>idleBreakUs) break;
      continue;
    }
    buf[cnt++]=uint8_t(bus.read());
    lastRx=micros();
    // Inspect each candidate offset once, only after its entire frame arrives.
    // This keeps partial-group parsing linear instead of rescanning the whole
    // receive buffer for every byte and consuming the control deadline.
    while(scan+6u+dataLen<=cnt) {
      size_t j=scan++;
      if(buf[j]!=0xff||buf[j+1]!=0xff||buf[j+3]!=dataLen+2)continue;
      int index=-1;
      for(uint8_t i=0;i<n;i++)if(ids[i]==buf[j+2]){index=i;break;}
      if(index<0 || (seen&(1u<<index)))continue;
      uint16_t cs=buf[j+2]+buf[j+3]+buf[j+4];
      for(uint8_t d=0;d<dataLen;d++)cs+=buf[j+5+d];
      if(uint8_t(~cs)!=buf[j+5+dataLen])continue;
      seen|=1u<<index;--remaining;
      if(status)status[index]=buf[j+4];
      if(buf[j+4]) {
        errors=true;if(errorMask)*errorMask|=1u<<index;
      } else {
        if(validMask)*validMask|=1u<<index;
        memcpy(out+size_t(index)*dataLen,buf+j+5,dataLen);
      }
      if(!remaining)return !errors;
    }
  }
  return false;
}

}  // namespace feetech
