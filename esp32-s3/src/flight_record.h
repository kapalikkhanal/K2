#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>

// Fixed little-endian IEEE754 records. CRC covers bytes 0..507. A torn write
// occupies its slot but is ignored by the decoder, never mistaken for data.
struct FlightRecord {
  uint32_t magic = 0x324c324b; // K2L2
  // v3 moved per-joint load/current into the control record; q/qd are
  // reconstructed from the observation block, which already carries them.
  uint16_t version = 3, kind = 0; // 1 metadata/event, 2 control, 3 servo telemetry
  uint32_t session = 0, sequence = 0;
  uint64_t timeUs = 0;
  uint32_t flags = 0, intervalUs = 0, readUs = 0, inferUs = 0, workUs = 0;
  uint16_t valid = 0, errors = 0;
  uint32_t dropped = 0, reserved[3] = {};
  union { float values[111]; char text[444]; } payload = {};
  uint32_t crc = 0;
};
static_assert(sizeof(FlightRecord)==512, "flash record format changed");
static_assert(offsetof(FlightRecord, crc)==508, "CRC offset changed");
inline uint32_t recordCrc(const void *data, size_t size) {
  uint32_t crc=0xffffffff;
  auto p=static_cast<const uint8_t *>(data);
  for(size_t i=0;i<size;i++) {
    crc^=p[i];
    for(int bit=0;bit<8;bit++) crc=(crc>>1)^(0xedb88320u & (0u-(crc&1u)));
  }
  return ~crc;
}
