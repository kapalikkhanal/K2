// Blink the user LED on a Seeed Studio XIAO ESP32S3 Sense.
//
// Two things about this board that trip people up:
//   1. The user LED is on GPIO21 (LED_BUILTIN in the XIAO variant), and it is
//      wired active LOW -- driving it LOW turns it ON, HIGH turns it OFF.
//   2. Serial goes over the chip's native USB, so the port disappears and
//      re-enumerates on every reset. That is normal, not a failed upload.

#include <Arduino.h>

static const uint8_t  LED_PIN     = LED_BUILTIN;  // GPIO21 on the XIAO ESP32S3
static const uint32_t BLINK_MS    = 500;

// Active-low helpers, so the calling code can stay in on/off terms.
static inline void ledOn()  { digitalWrite(LED_PIN, LOW); }
static inline void ledOff() { digitalWrite(LED_PIN, HIGH); }

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  ledOff();
}

void loop() {
  ledOn();
  Serial.println("LED on");
  delay(BLINK_MS);

  ledOff();
  Serial.println("LED off");
  delay(BLINK_MS);
}
