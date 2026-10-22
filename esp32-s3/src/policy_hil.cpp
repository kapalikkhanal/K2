// Hardware-in-the-loop: the ESP32 runs the real observation assembly and the
// real policy, while the MuJoCo digital twin supplies the sensors and applies
// the actions. Nothing on the robot is touched.
//
// This exists to answer one question before any servo is energised: does the
// C++ observation vector match what hardware/k2_policy_run.py builds? The host
// diffs the returned observation against its own, every tick.
//
// Frame in  (host -> esp): "K2" + 34 floats
//     gyro[3] grav[3] q[12] qd[12] vx wz dh march heading
// Frame out (esp -> host): "K2" + 62 floats
//     action[12] obs[50]

#include <Arduino.h>
#include "k2_obs.h"

static const size_t N_IN = 35, N_OUT = 62;

static bool readExact(uint8_t *dst, size_t n, uint32_t timeoutMs) {
  size_t got = 0;
  uint32_t t0 = millis();
  while (got < n && millis() - t0 < timeoutMs) {
    int c = Serial.read();
    if (c < 0) continue;
    dst[got++] = (uint8_t)c;
    t0 = millis();          // per-byte timeout: a slow host must not abort us
  }
  return got == n;
}

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);
  loadWeights();
  memset(lastAction, 0, sizeof(lastAction));
  Serial.println();
  Serial.printf("HIL ready: weights in %s\n", weightsInRam ? "SRAM" : "flash");
}

void loop() {
  // Resynchronise on the 2-byte magic so a partial frame cannot desync us.
  if (Serial.read() != 'K') return;
  uint32_t t0 = millis();
  int c;
  while ((c = Serial.read()) < 0 && millis() - t0 < 50) {}
  if (c != '2') return;

  float in[N_IN];
  if (!readExact((uint8_t *)in, sizeof(in), 200)) return;

  const float *gyro = in, *grav = in + 3, *q = in + 6, *qd = in + 18;
  float vx = in[30], wz = in[31], dh = in[32];
  bool marching = in[33] > 0.5f;

  // Heading is supplied by the host. Integrating it here would compare the
  // firmware's gyro source against the twin's true yaw, which is a harness
  // artifact rather than a port bug -- and the payload gyro is SITE frame,
  // whose z is the negation of the root-frame yaw rate ImuAttitude integrates.
  // Taking it from the host isolates what actually needs testing: the reference
  // heading this firmware integrates from its own commanded yaw.
  float headingEst = in[34];

  float out[N_OUT];
  float *action = out, *obs = out + 12;
  buildObsFrom(gyro, grav, q, qd, vx, wz, dh, marching, headingEst, obs);
  infer(obs, action);
  for (int i = 0; i < 12; i++) lastAction[i] = action[i];

  Serial.write('K'); Serial.write('2');
  Serial.write((uint8_t *)out, sizeof(out));
}
