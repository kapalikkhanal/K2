// Sweeps one STS3215 back and forth so you can watch it move.
//
// Deliberately conservative, because this drives real hardware:
//   * the sweep is centred on wherever the horn already is, then clamped well
//     clear of both end stops -- it never commands through a limit
//   * speed and acceleration are set low before torque is ever enabled
//   * torque is released on exit, so the servo is left free rather than holding
//
// Send 'x' over the USB serial monitor to stop and release torque.

#include <Arduino.h>
#include "feetech.h"

static const int PIN_SERVO_TX = 43;   // D6
static const int PIN_SERVO_RX = 44;   // D7
static const uint32_t SERVO_BAUD = 1000000;
static const uint8_t  SERVO_ID   = 1;

// 300 counts either side = about +/-26 deg, plainly visible without being wild.
static const uint16_t AMPLITUDE = 300;
// Keep this much clear of 0 and 4095 so a sweep can never hit an end stop.
static const uint16_t END_MARGIN = 200;

static const uint16_t GOAL_SPEED = 600;  // steps/s, ~52 deg/s
static const uint8_t  ACCEL      = 30;   // x100 steps/s^2

static uint16_t posLo = 0, posHi = 0;
static bool running = false;

static void torqueOff() {
  feetech::writeU8(Serial1, SERVO_ID, feetech::REG_TORQUE_EN, 0);
  running = false;
  Serial.println(F("torque released -- servo is free"));
}

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);

  Serial.println();
  Serial.println(F("=== servo sweep ==="));

  Serial1.begin(SERVO_BAUD, SERIAL_8N1, PIN_SERVO_RX, PIN_SERVO_TX);
  delay(50);

  if (!feetech::ping(Serial1, SERVO_ID)) {
    Serial.printf("no reply from id %u -- not moving anything\n", SERVO_ID);
    return;
  }

  uint16_t pos = 0;
  if (!feetech::readPosition(Serial1, SERVO_ID, &pos)) {
    Serial.println(F("could not read present position -- not moving anything"));
    return;
  }

  // Centre the sweep on the current pose, pulled inside the safe band. Starting
  // from wherever the horn already is keeps the first move small.
  uint16_t lim = AMPLITUDE + END_MARGIN;
  uint16_t center = constrain(pos, lim, (uint16_t)(feetech::POS_MAX - lim));
  posLo = center - AMPLITUDE;
  posHi = center + AMPLITUDE;

  Serial.printf("start pos %u (%.1f deg) -> sweeping %u..%u (%.1f..%.1f deg)\n",
                pos, pos * 360.0f / 4096.0f, posLo, posHi,
                posLo * 360.0f / 4096.0f, posHi * 360.0f / 4096.0f);

  // Order matters: shape the motion profile first, energise second.
  feetech::writeU8 (Serial1, SERVO_ID, feetech::REG_ACC,        ACCEL);
  feetech::writeU16(Serial1, SERVO_ID, feetech::REG_GOAL_SPEED, GOAL_SPEED);
  feetech::writeU8 (Serial1, SERVO_ID, feetech::REG_TORQUE_EN,  1);

  running = true;
  Serial.println(F("sweeping -- send 'x' to stop and release"));
}

void loop() {
  if (Serial.available() && (Serial.read() | 0x20) == 'x') {
    if (running) torqueOff();
    return;
  }
  if (!running) { delay(50); return; }

  static bool toHigh = true;
  uint16_t goal = toHigh ? posHi : posLo;
  toHigh = !toHigh;

  feetech::writeU16(Serial1, SERVO_ID, feetech::REG_GOAL_POS, goal);

  // Watch it get there rather than just assuming it did.
  uint32_t t0 = millis();
  while (millis() - t0 < 1600) {
    uint16_t pos;
    uint8_t vt[2];
    if (feetech::readPosition(Serial1, SERVO_ID, &pos) &&
        feetech::readRegs(Serial1, SERVO_ID, feetech::REG_PRES_VOLT, 2, vt)) {
      Serial.printf("goal %4u  pos %4u (%6.1f deg)  %.1fV %uC\n",
                    goal, pos, pos * 360.0f / 4096.0f, vt[0] / 10.0f, vt[1]);
    }
    delay(200);
    if (Serial.available()) return;   // let 'x' land promptly
  }
}
