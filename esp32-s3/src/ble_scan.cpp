// Raw BLE scan, to see whether the Xbox pad is actually advertising and what it
// looks like on the air. Diagnostic only -- it never tries to connect.
//
// The Xbox pad advertises appearance 964 (HID gamepad) and Microsoft
// manufacturer data starting 06 00. The library distinguishes two states:
//   060000     -- normal (already bonded to some host, not accepting new ones)
//   0600030080 -- pairing/searching (will accept a new host)
// That distinction is the whole game: a pad advertising the NORMAL pattern is
// visible but will refuse to bond, which looks like a failed connection.
//
// Send 'c' to erase stored bonds -- a stale bond from an earlier attempt makes
// the pad and the ESP32 disagree about keys and connection then fails.

#include <Arduino.h>
#include <NimBLEDevice.h>

static String hexOf(const std::string &s) {
  String out;
  for (unsigned char c : s) {
    char buf[4];
    snprintf(buf, sizeof(buf), "%02X", c);
    out += buf;
  }
  return out;
}

class ScanCB : public NimBLEScanCallbacks {
  void onResult(const NimBLEAdvertisedDevice *d) override {
    String mfg = d->haveManufacturerData() ? hexOf(d->getManufacturerData()) : String("-");
    uint16_t appearance = d->getAppearance();

    bool microsoft = mfg.startsWith("0600");
    bool gamepad = (appearance == 964);
    if (!microsoft && !gamepad && d->getName().empty()) return;  // skip noise

    Serial.printf("%s  rssi%4d  appearance %4u  mfg %-12s  name '%s'%s\n",
                  d->getAddress().toString().c_str(), d->getRSSI(), appearance,
                  mfg.c_str(), d->getName().c_str(),
                  (microsoft || gamepad) ? "   <== XBOX PAD" : "");

    if (microsoft) {
      if (mfg.startsWith("060003"))
        Serial.println(F("      state: PAIRING -- will accept a new host"));
      else
        Serial.println(F("      state: NORMAL -- already bound to another host;"
                         " hold Pair until the light flashes FAST"));
    }
  }
  void onScanEnd(const NimBLEScanResults &results, int reason) override {
    Serial.printf("-- scan window ended (%d devices), restarting --\n", results.getCount());
    NimBLEDevice::getScan()->start(0, false, true);
  }
};

static ScanCB scanCb;

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);

  Serial.println();
  Serial.println(F("=== BLE scan (diagnostic, no connect) ==="));
  NimBLEDevice::init("k2-scan");
  Serial.printf("stored bonds: %d   (send 'c' to erase them)\n",
                NimBLEDevice::getNumBonds());
  Serial.println(F("hold the pad's Pair button until the Xbox light flashes FAST\n"));

  NimBLEScan *scan = NimBLEDevice::getScan();
  scan->setScanCallbacks(&scanCb, true);   // duplicates on: state changes show up
  scan->setActiveScan(true);               // needed to receive scan responses
  scan->setInterval(100);
  scan->setWindow(99);
  scan->start(0, false, true);             // 0 = run until stopped
}

void loop() {
  if (Serial.available() && (Serial.read() | 0x20) == 'c') {
    Serial.printf("erasing %d bond(s)... ", NimBLEDevice::getNumBonds());
    Serial.println(NimBLEDevice::deleteAllBonds() ? "done" : "failed");
  }
  delay(50);
}
