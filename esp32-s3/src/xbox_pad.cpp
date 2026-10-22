// Connects an Xbox wireless controller to the XIAO ESP32S3 over BLE and shows
// what it would command on K2.
//
// The ESP32-S3 is BLE-only (no Bluetooth Classic). Xbox Series X|S and Xbox
// One S pads speak BLE HID, so they work; an original Xbox One pad (model 1537)
// has no Bluetooth at all and cannot connect here by any means.
//
// Axes are normalised exactly the way joystick.py does it -- deadzone 0.15 with
// the remainder rescaled so the command leaves zero at the deadzone edge rather
// than jumping. The command preview uses the caps from turn_v11's ONNX
// metadata, so what prints here is what the policy would actually be fed.

#include <Arduino.h>
#include <XboxSeriesXControllerESP32_asukiaaa.hpp>

// Pinned to the pad found by the blescan env. Without this the library also
// matches on Microsoft manufacturer data, and there is a second Microsoft
// device in range here (20:1f:df:87:09:cc) that is not a gamepad.
static const char *PAD_ADDRESS = "74:c4:12:e6:84:ef";
XboxSeriesXControllerESP32_asukiaaa::Core xbox(PAD_ADDRESS);

static const float DEADZONE = 0.15f;   // matches joystick.py

// Command caps read from policies/turn_v11_iter3400.onnx metadata.
static const float FORWARD_MAX_MPS = 0.05f;
static const float YAW_MAX_RPS     = 0.25f;
static const float HEIGHT_MIN_M    = -0.045f;
static const float HEIGHT_MAX_M    = 0.006f;

// joystick.py's dead-man: a light squeeze arms, it is there to notice a dropped
// pad rather than to demand a firm pull.
static const float TRIGGER_ARM = 0.15f;

static float normAxis(uint16_t raw, bool invert) {
  float v = ((float)raw - 32768.0f) / 32768.0f;
  if (invert) v = -v;
  if (fabsf(v) <= DEADZONE) return 0.0f;
  return copysignf((fabsf(v) - DEADZONE) / (1.0f - DEADZONE), v);
}

static void printButtons() {
  const auto &n = xbox.xboxNotif;
  struct { bool on; const char *name; } b[] = {
      {n.btnA, "A"}, {n.btnB, "B"}, {n.btnX, "X"}, {n.btnY, "Y"},
      {n.btnLB, "LB"}, {n.btnRB, "RB"}, {n.btnLS, "LS"}, {n.btnRS, "RS"},
      {n.btnSelect, "SELECT"}, {n.btnStart, "START"}, {n.btnShare, "SHARE"},
      {n.btnXbox, "XBOX"}, {n.btnDirUp, "UP"}, {n.btnDirDown, "DOWN"},
      {n.btnDirLeft, "LEFT"}, {n.btnDirRight, "RIGHT"},
  };
  Serial.print(" btn:");
  bool any = false;
  for (auto &e : b) if (e.on) { Serial.print(' '); Serial.print(e.name); any = true; }
  if (!any) Serial.print(" -");
}

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);

  Serial.println();
  Serial.println(F("=== Xbox BLE controller test ==="));
  Serial.printf("target pad: %s\n", PAD_ADDRESS);

  // begin() is what calls NimBLEDevice::init(). Every other NimBLE call must
  // come after it -- querying bonds first asserts on an uninitialised mutex.
  xbox.begin();

  // A bond left over from a failed attempt makes the two ends disagree about
  // keys, and the connection then fails in a way that looks like a missing pad.
  Serial.printf("stored bonds: %d   (send 'c' to erase)\n", NimBLEDevice::getNumBonds());
  Serial.println(F("put the pad in PAIRING mode: hold the small Pair button on the"));
  Serial.println(F("top edge ~3 s until the Xbox light flashes FAST, not slow."));
  Serial.println(F("It leaves pairing mode again after ~20 s.\n"));
}

void loop() {
  if (Serial.available() && (Serial.read() | 0x20) == 'c') {
    Serial.printf("erasing %d bond(s)... ", NimBLEDevice::getNumBonds());
    Serial.println(NimBLEDevice::deleteAllBonds() ? "done" : "failed");
  }
  xbox.onLoop();

  static uint32_t lastPrint = 0;
  static bool wasConnected = false;
  uint32_t now = millis();

  if (!xbox.isConnected()) {
    if (wasConnected) {
      Serial.println(F("!! disconnected -- rescanning"));
      wasConnected = false;
    }
    if (now - lastPrint > 2000) {
      lastPrint = now;
      Serial.printf("scanning...  failed attempts: %u\n",
                    xbox.getCountFailedConnection());
      // The pad drops out of pairing mode on its own after ~20 s.
      if (xbox.getCountFailedConnection() > 0)
        Serial.println(F("   (if this keeps failing, press Pair again -- the "
                         "pad leaves pairing mode after ~20 s)"));
    }
    return;
  }

  if (!wasConnected) {
    wasConnected = true;
    Serial.printf("\nCONNECTED to %s   battery %u%%\n",
                  xbox.buildDeviceAddressStr().c_str(), xbox.battery);
    Serial.println(F("push the sticks and press buttons\n"));
  }

  if (xbox.isWaitingForFirstNotification()) {
    if (now - lastPrint > 500) {
      lastPrint = now;
      Serial.println(F("connected, waiting for the first report..."));
    }
    return;
  }

  if (now - lastPrint < 100) return;   // 10 Hz is plenty to read by eye
  lastPrint = now;

  const auto &n = xbox.xboxNotif;
  // Vertical axes read larger downward on this pad, so invert to make up = +1.
  float lx = normAxis(n.joyLHori, false);
  float ly = normAxis(n.joyLVert, true);
  float rx = normAxis(n.joyRHori, false);
  float ry = normAxis(n.joyRVert, true);
  float lt = (float)n.trigLT / 1023.0f;
  float rt = (float)n.trigRT / 1023.0f;

  // K2 mapping, straight out of joystick.py's docstring.
  float vx = ly * FORWARD_MAX_MPS;
  float wz = -rx * YAW_MAX_RPS;   // stick right = clockwise = negative yaw
  float dh = n.btnLB ? HEIGHT_MIN_M : (n.btnRB ? HEIGHT_MAX_M : 0.0f);
  if (n.btnY) dh = 0.0f;
  const char *mode = n.btnA ? "MARCH" : (n.btnB ? "HOLD" : "-");

  static uint32_t reports = 0;
  static uint16_t prevH = 0;
  uint16_t h = n.joyLHori ^ n.joyLVert ^ n.joyRHori ^ n.joyRVert ^ n.trigLT ^ n.trigRT
               ^ (n.btnA | n.btnB << 1 | n.btnX << 2 | n.btnY << 3
                  | n.btnLB << 4 | n.btnRB << 5 | n.btnXbox << 6);
  if (h != prevH) { prevH = h; reports++; }

  Serial.printf("raw L(%5u,%5u) R(%5u,%5u) T(%4u,%4u) chg%-4lu | ",
                n.joyLHori, n.joyLVert, n.joyRHori, n.joyRVert,
                n.trigLT, n.trigRT, (unsigned long)reports);
  Serial.printf("L(%+.2f,%+.2f) R(%+.2f,%+.2f) LT%.2f RT%.2f %s",
                lx, ly, rx, ry, lt, rt,
                (lt > TRIGGER_ARM || rt > TRIGGER_ARM) ? "ARMED" : "     ");
  Serial.printf("  ->  vx%+.3f m/s  wz%+.3f rad/s  dh%+.3f m  mode:%-5s",
                vx, wz, dh, mode);
  printButtons();
  Serial.println();
}
