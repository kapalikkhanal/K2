// Bring-up diagnostic for the XIAO ESP32S3 Sense:
//   1. scan I2C and read the LSM6DS3 (accel + gyro + temp)
//   2. probe the Waveshare serial servo bus for STS/SCS servos
//
// The LSM6DS3 is driven by raw register access over Wire -- no library, so
// there is nothing to install and every transaction is visible here.
//
// The servo probe sends Feetech PING packets only. PING is read-only: it asks
// a servo to identify itself and never commands motion or torque.

#include <Arduino.h>
#include <Wire.h>
#include "feetech.h"

// ---------------------------------------------------------------- pin map
// Seeed XIAO ESP32S3 silkscreen -> GPIO. Change these if wired differently.
static const int PIN_SDA      = 5;    // D4
static const int PIN_SCL      = 6;    // D5
static const int PIN_SERVO_TX = 43;   // D6
static const int PIN_SERVO_RX = 44;   // D7

// ---------------------------------------------------------------- LSM6DS3
// The part answers on 0x6A or 0x6B depending on how SDO/SA0 is strapped.
static const uint8_t LSM_ADDR_CANDIDATES[] = {0x6A, 0x6B};

static const uint8_t REG_WHO_AM_I   = 0x0F;
static const uint8_t REG_CTRL1_XL   = 0x10;
static const uint8_t REG_CTRL2_G    = 0x11;
static const uint8_t REG_CTRL3_C    = 0x12;
static const uint8_t REG_OUT_TEMP_L = 0x20;
static const uint8_t REG_OUTX_L_G   = 0x22;

// WHO_AM_I differs across the family; both are "an LSM6DS3" for our purposes.
static const uint8_t WHOAMI_DS3    = 0x69;  // LSM6DS3
static const uint8_t WHOAMI_DS3TRC = 0x6A;  // LSM6DS3TR-C

static uint8_t lsmAddr = 0;  // 0 = not found

// Full-scale settings chosen below: +/-2 g and 245 dps.
static const float ACC_LSB_TO_G   = 0.061f / 1000.0f;  // mg/LSB  -> g
static const float GYR_LSB_TO_DPS = 8.75f  / 1000.0f;  // mdps/LSB -> dps

static bool regWrite(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

// Reads n consecutive registers. The LSM6DS3 auto-increments its register
// pointer (CTRL3_C.IF_INC, on by default), so one burst read is coherent.
static bool regReadN(uint8_t addr, uint8_t reg, uint8_t *out, uint8_t n) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;   // repeated START
  if (Wire.requestFrom((int)addr, (int)n) != n) return false;
  for (uint8_t i = 0; i < n; i++) out[i] = Wire.read();
  return true;
}

static bool regRead(uint8_t addr, uint8_t reg, uint8_t *out) {
  return regReadN(addr, reg, out, 1);
}

static uint8_t i2cFound[16];
static uint8_t i2cFoundN = 0;

static bool i2cPresent(uint8_t a) {
  for (uint8_t i = 0; i < i2cFoundN; i++)
    if (i2cFound[i] == a) return true;
  return false;
}

static void i2cScan() {
  Serial.println(F("--- I2C scan (SDA=GPIO5/D4, SCL=GPIO6/D5) ---"));
  i2cFoundN = 0;
  for (uint8_t a = 0x08; a < 0x78; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) {
      Serial.printf("  device at 0x%02X\n", a);
      if (i2cFoundN < sizeof(i2cFound)) i2cFound[i2cFoundN++] = a;
    }
  }
  if (i2cFoundN == 0) Serial.println(F("  (nothing responded -- check wiring/pull-ups)"));
}

static bool lsmInit() {
  Serial.println(F("--- LSM6DS3 ---"));
  for (uint8_t i = 0; i < sizeof(LSM_ADDR_CANDIDATES); i++) {
    uint8_t addr = LSM_ADDR_CANDIDATES[i];
    if (!i2cPresent(addr)) continue;   // avoids a noisy Wire error on a vacant addr
    uint8_t who = 0;
    if (!regRead(addr, REG_WHO_AM_I, &who)) continue;
    if (who == WHOAMI_DS3 || who == WHOAMI_DS3TRC) {
      lsmAddr = addr;
      Serial.printf("  found at 0x%02X, WHO_AM_I=0x%02X (%s)\n", addr, who,
                    who == WHOAMI_DS3 ? "LSM6DS3" : "LSM6DS3TR-C");
      break;
    }
    Serial.printf("  0x%02X answered but WHO_AM_I=0x%02X (unexpected)\n", addr, who);
  }
  if (!lsmAddr) {
    Serial.println(F("  NOT FOUND"));
    return false;
  }

  // BDU=1 so the low/high halves of a sample can't be split across updates.
  regWrite(lsmAddr, REG_CTRL3_C,  0x44);   // BDU | IF_INC
  regWrite(lsmAddr, REG_CTRL1_XL, 0x60);   // accel 416 Hz, +/-2 g
  regWrite(lsmAddr, REG_CTRL2_G,  0x60);   // gyro  416 Hz, 245 dps
  delay(20);
  Serial.println(F("  configured: 416 Hz, +/-2 g, 245 dps"));
  return true;
}

static void lsmPrintSample() {
  if (!lsmAddr) return;
  uint8_t b[12];
  // 0x22..0x2D = gyro XYZ then accel XYZ, little-endian int16 each.
  if (!regReadN(lsmAddr, REG_OUTX_L_G, b, 12)) {
    Serial.println(F("imu: read failed"));
    return;
  }
  int16_t gx = (int16_t)(b[0]  | (b[1]  << 8));
  int16_t gy = (int16_t)(b[2]  | (b[3]  << 8));
  int16_t gz = (int16_t)(b[4]  | (b[5]  << 8));
  int16_t ax = (int16_t)(b[6]  | (b[7]  << 8));
  int16_t ay = (int16_t)(b[8]  | (b[9]  << 8));
  int16_t az = (int16_t)(b[10] | (b[11] << 8));

  uint8_t t[2];
  float tempC = NAN;
  if (regReadN(lsmAddr, REG_OUT_TEMP_L, t, 2)) {
    tempC = 25.0f + (int16_t)(t[0] | (t[1] << 8)) / 16.0f;
  }

  float axg = ax * ACC_LSB_TO_G, ayg = ay * ACC_LSB_TO_G, azg = az * ACC_LSB_TO_G;
  Serial.printf(
      "imu  acc[g] % .3f % .3f % .3f  |a|=%.3f   gyr[dps] % 8.2f % 8.2f % 8.2f   %.1fC\n",
      axg, ayg, azg, sqrtf(axg * axg + ayg * ayg + azg * azg),
      gx * GYR_LSB_TO_DPS, gy * GYR_LSB_TO_DPS, gz * GYR_LSB_TO_DPS, tempC);
}

// ------------------------------------------------------------- servo bus
// K2 uses 0..11; the wider sweep exists because a factory-fresh servo ships as
// ID 1 and a mis-set one can land anywhere in 0..253.
static uint8_t scanIdMax = 30;
static bool foundIds[254];

static const uint32_t BAUDS[] = {1000000, 500000, 250000, 128000, 115200,
                                 76800,   57600,  38400,  19200,  9600};

static bool servoPing(uint8_t id, size_t *rxBytes) {
  return feetech::ping(Serial1, id, rxBytes);
}

// Reads the live telemetry block (0x38..0x3F) -- position through temperature.
static void servoReport(uint8_t id) {
  uint8_t d[8];
  if (!feetech::readRegs(Serial1, id, feetech::REG_PRES_POS, sizeof(d), d)) {
    Serial.printf("      id %u: ping OK but register read failed\n", id);
    return;
  }
  uint8_t torque = 0xFF;  // 0xFF = could not read
  feetech::readRegs(Serial1, id, feetech::REG_TORQUE_EN, 1, &torque);

  uint16_t pos = d[0] | (d[1] << 8);
  Serial.printf("      id %u: torque=%s  pos=%u (%.1f deg)  speed=%d  load=%d  %.1fV  %uC\n",
                id, torque == 0 ? "OFF" : (torque == 1 ? "ON" : "?"),
                pos, pos * 360.0f / 4096.0f,
                feetech::signMag(d[2] | (d[3] << 8)),
                feetech::signMag(d[4] | (d[5] << 8)),
                d[6] / 10.0f, d[7]);
}

// Dumps whatever is sitting on the bus as hex + printable ASCII. A board that
// runs its own firmware (rather than passing Feetech frames straight through)
// usually announces itself on boot, and this is what would catch that.
static size_t passiveListen(uint32_t ms) {
  uint8_t buf[48];
  size_t n = 0;
  uint32_t t0 = millis();
  while (millis() - t0 < ms && n < sizeof(buf)) {
    if (Serial1.available()) buf[n++] = (uint8_t)Serial1.read();
  }
  if (n) {
    Serial.print(F("      unsolicited:"));
    for (size_t i = 0; i < n; i++) Serial.printf(" %02X", buf[i]);
    Serial.print(F("  |"));
    for (size_t i = 0; i < n; i++)
      Serial.print((char)(buf[i] >= 32 && buf[i] < 127 ? buf[i] : '.'));
    Serial.println(F("|"));
  }
  return n;
}

// Tells connected-and-powered apart from nothing-on-the-wire.
//
// Apply an internal pull-down, then a pull-up, and see whether the pin obeys.
// A pin that stays HIGH against a pull-down is being driven by something
// outside the XIAO -- which for a UART line idling high means the other board
// is wired here AND powered. A pin that simply follows both pulls is floating.
static void pinDriveTest(int pin, const char *label) {
  pinMode(pin, INPUT_PULLDOWN); delay(5); int lo = digitalRead(pin);
  pinMode(pin, INPUT_PULLUP);   delay(5); int hi = digitalRead(pin);

  const char *verdict;
  if (lo == HIGH && hi == HIGH)      verdict = "driven HIGH externally (wired + powered, UART idle)";
  else if (lo == LOW && hi == LOW)   verdict = "driven LOW externally (wired, but held low)";
  else                               verdict = "FLOATING -- nothing connected, or the other end is unpowered";

  Serial.printf("    GPIO%-2d %-8s pulldown=%d pullup=%d -> %s\n",
                pin, label, lo, hi, verdict);
  pinMode(pin, INPUT);
}

// Proves the MCU end works before we go blaming anyone's wiring.
//
// GPIO43/44 are the ESP32-S3's native UART0 pins, so there is a real chance the
// core is still holding them and Serial1's remap never reaches the pad. The
// chip's GPIO matrix lets one pad be an output and an input at once, so putting
// Serial1's RX and TX on the SAME pin makes it loop back through the physical
// pad. Bytes returning = that pad is genuinely toggling.
static bool uartPinSelfTest(int pin) {
  static const uint8_t probe[4] = {0x55, 0xAA, 0x0F, 0xF0};
  Serial1.begin(115200, SERIAL_8N1, pin, pin);   // rx == tx
  delay(20);
  while (Serial1.available()) Serial1.read();

  Serial1.write(probe, sizeof(probe));
  Serial1.flush();

  uint8_t buf[8];
  size_t n = 0;
  uint32_t t0 = millis();
  while (millis() - t0 < 30 && n < sizeof(buf)) {
    if (Serial1.available()) buf[n++] = (uint8_t)Serial1.read();
  }
  Serial1.end();

  bool ok = (n >= sizeof(probe) && memcmp(buf, probe, sizeof(probe)) == 0);
  Serial.printf("    GPIO%-2d loopback: %s (%u bytes", pin, ok ? "OK" : "FAILED", (unsigned)n);
  for (size_t i = 0; i < n; i++) Serial.printf(" %02X", buf[i]);
  Serial.println(")");
  return ok;
}

// TX/RX swapped is by far the most common miswire, and it costs nothing to
// just try the bus both ways round rather than make someone rewire to find out.
struct PinPair { int tx, rx; const char *label; };
// Orientation is settled: the pull tests showed D6 feeds the board's input and
// D7 is the board's push-pull output. Probing the swapped order would mean
// driving against that live output, so it is deliberately not attempted.
static const PinPair PIN_PAIRS[] = {
    {43, 44, "TX=D6/GPIO43, RX=D7/GPIO44"},
};

static void servoScan() {
  Serial.println(F("--- servo bus probe ---"));
  Serial.println(F("  line state (is anything actually out there?):"));
  pinDriveTest(43, "D6");
  pinDriveTest(44, "D7");
  Serial.println(F("  self-test (pin loops back to itself, nothing external):"));
  // Only D6 is ours to drive; D7 belongs to the board's transmitter.
  if (!uartPinSelfTest(43))
    Serial.println(F("  !! MCU is not driving D6 -- fault is on the XIAO side"));

  bool anyServo = false, anyBytes = false;

  for (uint8_t pi = 0; pi < sizeof(PIN_PAIRS) / sizeof(PIN_PAIRS[0]); pi++) {
    const PinPair &pp = PIN_PAIRS[pi];
    Serial.printf("  %s\n", pp.label);

    for (uint8_t bi = 0; bi < sizeof(BAUDS) / sizeof(BAUDS[0]); bi++) {
      uint32_t baud = BAUDS[bi];
      Serial1.begin(baud, SERIAL_8N1, pp.rx, pp.tx);
      delay(20);

      size_t totalRx = passiveListen(150);   // anything without us asking?

      int hits = 0;
      String ids;
      memset(foundIds, 0, sizeof(foundIds));
      for (uint16_t id = 0; id <= scanIdMax; id++) {
        size_t rx = 0;
        if (servoPing((uint8_t)id, &rx)) {
          hits++;
          foundIds[id] = true;
          ids += " " + String(id);
        }
        totalRx += rx;
      }
      if (hits) {
        Serial.printf("    %7lu baud: %d servo(s) ->%s\n",
                      (unsigned long)baud, hits, ids.c_str());
        // Must stay at this baud with the port still open to read registers.
        for (uint16_t id = 0; id <= scanIdMax; id++)
          if (foundIds[id]) servoReport((uint8_t)id);
        anyServo = true;
      } else {
        Serial.printf("    %7lu baud: none (%u bytes on the wire)\n",
                      (unsigned long)baud, (unsigned)totalRx);
      }
      Serial1.end();
      if (totalRx) anyBytes = true;
    }
  }

  if (anyServo) return;
  if (anyBytes) {
    Serial.println(F("  bytes came back but no valid Feetech reply: TX reaches the"));
    Serial.println(F("  board. Either bus echo, or the board speaks its own protocol."));
  } else {
    Serial.println(F("  totally silent on both pin orders. Not a TX/RX swap --"));
    Serial.println(F("  check: board powered? GND shared with the XIAO? right pins?"));
  }
}

// ------------------------------------------------------------------ main
void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);  // let USB CDC enumerate

  Serial.println();
  Serial.println(F("=== XIAO ESP32S3 Sense bring-up ==="));

  Wire.begin(PIN_SDA, PIN_SCL, 400000);
  i2cScan();
  lsmInit();
  servoScan();

  Serial.println(F("=== streaming IMU  (s = rescan, f = full 0..253 sweep) ==="));
}

void loop() {
  if (Serial.available()) {
    int c = Serial.read() | 0x20;
    if (c == 's') { scanIdMax = 30;  servoScan(); }
    if (c == 'f') { scanIdMax = 253; servoScan(); scanIdMax = 30; }  // full sweep
  }
  lsmPrintSample();
  delay(200);
}
