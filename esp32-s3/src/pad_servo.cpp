// Drive the servo from the Xbox pad: the whole loop K2 would use, in miniature.
// Pad -> command -> servo, on one ESP32-S3.
//
// Safety, mirroring joystick.py:
//   * DEAD-MAN: a trigger must be held for the servo to move. Release it and the
//     servo holds. This exists to notice a dropped pad, which is why it is a
//     light squeeze rather than a firm pull.
//   * losing the BLE link releases torque immediately
//   * B releases torque
//   * travel is centred on the servo's current pose and clamped clear of both
//     end stops, so a stick command can never drive into a limit

#include <Arduino.h>
#include <XboxSeriesXControllerESP32_asukiaaa.hpp>
#include "feetech.h"

static const char *PAD_ADDRESS = "74:c4:12:e6:84:ef";
XboxSeriesXControllerESP32_asukiaaa::Core xbox(PAD_ADDRESS);

static const int PIN_SERVO_TX = 43;   // D6
static const int PIN_SERVO_RX = 44;   // D7
static const uint32_t SERVO_BAUD = 1000000;
static const uint8_t  SERVO_ID   = 1;

static const float    DEADZONE    = 0.15f;   // matches joystick.py
// Separate arm/disarm thresholds. With one threshold a trigger held near it
// chatters -- observed toggling torque on and off several times a second, which
// on a loaded joint means it repeatedly drops and recatches the load.
static const float    TRIGGER_ARM    = 0.15f;
static const float    TRIGGER_DISARM = 0.08f;
static const uint16_t AMPLITUDE   = 400;     // counts either side (~35 deg)
static const uint16_t END_MARGIN  = 200;
static const uint16_t GOAL_SPEED  = 1500;    // steps/s, responsive but not violent
static const uint8_t  ACCEL       = 50;

static uint16_t center = 0;
static bool servoReady = false;
static bool torqueOn = false;

static float normAxis(uint16_t raw, bool invert) {
  float v = ((float)raw - 32768.0f) / 32768.0f;
  if (invert) v = -v;
  if (fabsf(v) <= DEADZONE) return 0.0f;
  return copysignf((fabsf(v) - DEADZONE) / (1.0f - DEADZONE), v);
}

static void setTorque(bool on) {
  if (on == torqueOn) return;
  feetech::writeU8(Serial1, SERVO_ID, feetech::REG_TORQUE_EN, on ? 1 : 0);
  torqueOn = on;
  Serial.println(on ? F("[torque ON]") : F("[torque released]"));
}

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);

  Serial.println();
  Serial.println(F("=== pad -> servo ==="));

  Serial1.begin(SERVO_BAUD, SERIAL_8N1, PIN_SERVO_RX, PIN_SERVO_TX);
  delay(50);

  uint16_t pos = 0;
  if (feetech::ping(Serial1, SERVO_ID) &&
      feetech::readPosition(Serial1, SERVO_ID, &pos)) {
    uint16_t lim = AMPLITUDE + END_MARGIN;
    center = constrain(pos, lim, (uint16_t)(feetech::POS_MAX - lim));
    feetech::writeU8 (Serial1, SERVO_ID, feetech::REG_ACC, ACCEL);
    feetech::writeU16(Serial1, SERVO_ID, feetech::REG_GOAL_SPEED, GOAL_SPEED);
    servoReady = true;
    Serial.printf("servo id %u at %u -> travel %u..%u\n", SERVO_ID, pos,
                  center - AMPLITUDE, center + AMPLITUDE);
  } else {
    Serial.println(F("no servo on the bus -- pad will still report"));
  }

  xbox.begin();
  Serial.println(F("hold LT or RT (dead-man), then push the LEFT stick up/down"));
  Serial.println(F("B releases torque\n"));
}

void loop() {
  xbox.onLoop();

  static uint32_t lastStatus = 0;
  bool linkUp = xbox.isConnected() && !xbox.isWaitingForFirstNotification();

  if (!linkUp) {
    if (servoReady) setTorque(false);      // a dropped pad must not leave it driven
    if (millis() - lastStatus > 1500) {
      lastStatus = millis();
      Serial.printf("pad: %s\n", xbox.isConnected() ? "connected, awaiting first report"
                                                     : "scanning for pad...");
    }
    delay(20);
    return;
  }

  const auto &n = xbox.xboxNotif;
  float ly = normAxis(n.joyLVert, true);   // up = +
  float lt = (float)n.trigLT / 1023.0f;
  float rt = (float)n.trigRT / 1023.0f;
  static bool wasArmed = false;
  float trig = fmaxf(lt, rt);
  bool armed = wasArmed ? (trig > TRIGGER_DISARM) : (trig > TRIGGER_ARM);
  if (n.btnB) armed = false;
  wasArmed = armed;

  // Never let "nothing printed" be ambiguous between no pad and no dead-man.
  if (!armed && millis() - lastStatus > 1500) {
    lastStatus = millis();
    Serial.printf("pad connected | LT %.2f RT %.2f | hold a trigger to arm%s\n",
                  lt, rt, n.btnB ? "  (B held: torque locked off)" : "");
  }

  if (servoReady) {
    setTorque(armed);
    if (armed) {
      uint16_t goal = (uint16_t)(center + (int)(ly * AMPLITUDE));
      feetech::writeU16(Serial1, SERVO_ID, feetech::REG_GOAL_POS, goal);

      static uint32_t lastPrint = 0;
      if (millis() - lastPrint > 100) {
        lastPrint = millis();
        uint16_t pos = 0;
        feetech::readPosition(Serial1, SERVO_ID, &pos);
        Serial.printf("stick %+.2f  goal %4u  pos %4u (%6.1f deg)  %s\n",
                      ly, goal, pos, pos * 360.0f / 4096.0f,
                      ly > 0.1f ? "stick UP" : (ly < -0.1f ? "stick DOWN" : ""));
      }
    }
  }
  delay(20);   // 50 Hz, the same rate the policy would run at
}
