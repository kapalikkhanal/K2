// K2 policy on the real robot, joystick-driven -- the deployment build.
//
// This is policy_run.cpp plus the safety layer hardware/k2_policy_run.py has and
// a bare inference loop does not. Every one of these exists because without it a
// single bad tick becomes a broken robot:
//
//   approach ramp  5 s cosine ease from the current pose into the default crouch,
//                  BEFORE the policy takes over. Without it the first tick steps
//                  all 12 joints at once from wherever they happen to be. This
//                  runs from ANY starting orientation -- boot does not check tilt.
//   slew limit     caps per-tick joint motion (1.0 rad/s)
//   joint clamp    q_target clamped to joint_limits.json, counts to [40,4055]
//   tilt cutoff    >24 deg releases torque and latches a fault, ONCE THE POLICY
//                  IS RUNNING. Boot and the approach ramp are exempt.
//   sanity check   a non-finite action releases torque
//   dead-man       release the trigger and the commands go to zero and the robot
//                  HOLDS. Torque stays on -- dropping it on a standing biped just
//                  makes it fall. Same for losing the pad.
//
// The observation assembly in k2_obs.h is byte-for-byte verified against the
// Python on the digital twin (tools/hil_twin.py).

#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include <XboxSeriesXControllerESP32_asukiaaa.hpp>
#include "feetech.h"
#include "k2_obs.h"
#include "robot_control.h"
#include "feedback_guard.h"
#include "web_control.h"
#include "flight_logger.h"
#if K2_RUNTIME_RATE
#include <Preferences.h>
#endif

static const char *PAD_ADDRESS = "74:c4:12:e6:84:ef";
XboxSeriesXControllerESP32_asukiaaa::Core xbox(PAD_ADDRESS);

static const int PIN_SDA = 5, PIN_SCL = 6;
static const int PIN_SERVO_TX = 43, PIN_SERVO_RX = 44;
static const uint32_t SERVO_BAUD = 1000000;

static const float APPROACH_S    = 5.0f;
static const float SLEW_RAD_S    = 1.0f;
static const float FALL_ANGLE    = 24.0f * (float)M_PI / 180.0f;
static const uint16_t COUNT_MIN  = 40, COUNT_MAX = 4055;

static uint8_t lsmAddr = 0;
static float gyroBias[3] = {0, 0, 0}, gChip[3] = {0, 0, -1};
static bool gChipInit = false;
static float headingEst = 0.0f;
static float q[12], qd[12], qCmd[12];
static uint8_t ids[12];
static FeedbackGuard feedback;
static uint16_t freshMask = 0, errorMask = 0;
static float accChip[3], gyroChip[3];
static uint32_t missingTicks = 0, deadlineTicks = 0;

static bool timingPaused = false;
static uint8_t lateTicks = 0;       // consecutive late/missed ticks
static const uint8_t LATE_LATCH_TICKS = 3;
static uint32_t lastTimely = 0;
static float gyrDebiased[3] = {0, 0, 0};   // last chip-frame gyro after bias removal
// Gyro bias refinement while HOLDING. The boot capture leaves a residual
// (measured ~0.06 deg/s on the flight log, far more if the robot is touched
// during capture), and the heading integrator turns any residual into a
// growing heading error the policy then tries to walk out. Only runs when
// the policy is in pure hold and the measured rate is already tiny, with a
// ~40 s time constant, so a real rotation can never be absorbed as bias.
static const float BIAS_REFINE_GATE = 0.03f;      // rad/s (~1.7 deg/s)
static const float BIAS_REFINE_ALPHA = 0.0005f;   // per 20 ms tick
static bool faulted = false;
static const char *faultReason = "";
static bool busReady = false;
static QueueHandle_t padQueue;
static RobotControl controls;
static CommandArbiter commandArbiter;
static RebootChord rebootChord;

static char serialControl() {
  if(!Serial.available())return 0;
  return Serial.read()|0x20;
}

static void padTask(void *) {
  for (;;) {
    // Connection and service discovery can block for seconds. Keep them off
    // Arduino's core-1 control loop, including during reconnects.
    xbox.onLoop();
    PadSample p;
    // Run on the NimBLE host's core and briefly suspend its scheduler while
    // copying the notification, so the callback cannot tear the snapshot.
    vTaskSuspendAll();
    p.connected = xbox.isConnected() && !xbox.isWaitingForFirstNotification();
    p.receivedAt = xbox.getReceiveNotificationAt();
    const auto &n = xbox.xboxNotif;
    p.forward = n.joyLVert; p.turn = n.joyRHori; p.lateral = n.joyLHori;
    p.lt = n.trigLT; p.rt = n.trigRT;
    p.a = n.btnA; p.b = n.btnB; p.lb = n.btnLB; p.rb = n.btnRB;
    p.x = n.btnX;
    p.stop = n.btnSelect;
    xTaskResumeAll();
    xQueueOverwrite(padQueue, &p);
    vTaskDelay(pdMS_TO_TICKS(5));
  }
}

static bool regWrite(uint8_t a, uint8_t r, uint8_t v) {
  Wire.beginTransmission(a); Wire.write(r); Wire.write(v);
  return Wire.endTransmission() == 0;
}
static bool regReadN(uint8_t a, uint8_t r, uint8_t *o, uint8_t n) {
  Wire.beginTransmission(a); Wire.write(r);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)a, (int)n) != n) return false;
  for (uint8_t i = 0; i < n; i++) o[i] = Wire.read();
  return true;
}
static bool imuRead(float *acc, float *gyr) {
  uint8_t b[12];
  if (!regReadN(lsmAddr, 0x22, b, 12)) return false;
  const float GD = 0.00875f * (float)M_PI / 180.0f, AG = 0.000061f;
  for (int i = 0; i < 3; i++) {
    gyr[i] = (int16_t)(b[2*i]   | (b[2*i+1]   << 8)) * GD;
    acc[i] = (int16_t)(b[6+2*i] | (b[6+2*i+1] << 8)) * AG;
  }
  return true;
}
// chip -> root (R_IMU_TO_ROOT) then the measured level correction.
static inline void imuToRoot(const float *v, float *o) {
  float r[3] = {v[1], -v[0], v[2]};
  for (int i = 0; i < 3; i++)
    o[i] = K2_LEVEL[3*i+0]*r[0] + K2_LEVEL[3*i+1]*r[1] + K2_LEVEL[3*i+2]*r[2];
}
static inline void rootToSite(const float *v, float *o) {
  o[0] = -v[1]; o[1] = -v[0]; o[2] = -v[2];
}

static bool attitude(float *gyroSite, float *grav) {
  float acc[3], gyr[3];
  if (!imuRead(acc, gyr)) return false;
  memcpy(accChip,acc,sizeof(accChip));memcpy(gyroChip,gyr,sizeof(gyroChip));
  for (int i = 0; i < 3; i++) { gyr[i] -= gyroBias[i]; gyrDebiased[i] = gyr[i]; }
  float g[3] = {-acc[0], -acc[1], -acc[2]};
  float n = sqrtf(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);
  bool ok = isfinite(n) && n > 0.2f && n < 2.5f;
  if (ok) for (int i = 0; i < 3; i++) g[i] /= n;

  if (!gChipInit) { if (ok) memcpy(gChip, g, sizeof(g)); gChipInit = true; }
  else {
    float c[3] = {gyr[1]*gChip[2]-gyr[2]*gChip[1], gyr[2]*gChip[0]-gyr[0]*gChip[2],
                  gyr[0]*gChip[1]-gyr[1]*gChip[0]};
    for (int i = 0; i < 3; i++) gChip[i] -= DT * c[i];
    float m = sqrtf(gChip[0]*gChip[0]+gChip[1]*gChip[1]+gChip[2]*gChip[2]);
    for (int i = 0; i < 3; i++) gChip[i] /= fmaxf(m, 1e-6f);
    if (ok) {
      float nc = expf(-powf((n-1.0f)/0.12f, 2.0f));
      float d = fmaxf(-1.0f, fminf(1.0f, gChip[0]*g[0]+gChip[1]*g[1]+gChip[2]*g[2]));
      float dc = expf(-powf(acosf(d)/0.25f, 2.0f));
      float a = 0.001f + (0.02f-0.001f)*(nc*dc);
      // Per-tick trust tuned at 50 Hz; keep its time constant at any rate.
      if (fabsf(DT - 0.02f) > 1e-6f) a = 1.0f - powf(1.0f - a, DT / 0.02f);
      for (int i = 0; i < 3; i++) gChip[i] = (1-a)*gChip[i] + a*g[i];
      float m2 = sqrtf(gChip[0]*gChip[0]+gChip[1]*gChip[1]+gChip[2]*gChip[2]);
      for (int i = 0; i < 3; i++) gChip[i] /= fmaxf(m2, 1e-6f);
    }
  }
  imuToRoot(gChip, grav);
  float gr[3]; imuToRoot(gyr, gr);
  headingEst = fmodf(headingEst + gr[2]*DT + (float)M_PI, 2.0f*(float)M_PI) - (float)M_PI;
  rootToSite(gr, gyroSite);
  return isfinite(grav[0]) && isfinite(grav[1]) && isfinite(grav[2]);
}

static bool readJoints(bool retry = true) {
  // Keep the backup's compact position/velocity payload. During policy control,
  // one bounded attempt leaves time for IMU, inference and goal writes.
  uint8_t data[12 * 4] = {};
  freshMask = 0; errorMask = 0;
  for (int attempt = 0; attempt < (retry ? 5 : 1); ++attempt) {
    uint16_t valid = 0, errors = 0;
    feetech::syncRead(Serial1, ids, 12, feetech::REG_PRES_POS, 4,
                      data, 10, &valid, &errors, nullptr, 2000);
    errorMask |= errors;
    for (int j = 0; j < 12; ++j) {
      if (!(valid & (1u << j))) continue;
      const uint8_t *b = data + 4*j;
      uint16_t raw = b[0] | (b[1] << 8);
      if (raw > feetech::POS_MAX) continue;
      q[j] = K2_SIGN[j] * (int(raw) - K2_HOME_RAW[j]) * K2_RAD_PER_COUNT;
      qd[j] = K2_SIGN[j] * feetech::signMag(b[2] | (b[3] << 8)) * K2_RAD_PER_COUNT;
      freshMask |= 1u << j;
    }
    if (freshMask == FeedbackGuard::ALL) return true;
    if (retry) delayMicroseconds(1000);
  }
  return false;
}

// q (rad, SIM_ORDER) -> goal counts, clamped to joint limits then to the
// encoder range the servo will accept.
static void writeGoals(const float *qt) {
  uint16_t goals[12];
  for (int j = 0; j < 12; j++) {
    float v = fmaxf(K2_QMIN[j], fminf(K2_QMAX[j], qt[j]));
    long raw = lroundf(K2_HOME_RAW[j] + K2_SIGN[j] * v / K2_RAD_PER_COUNT);
    goals[j] = (uint16_t)max((long)COUNT_MIN, min((long)COUNT_MAX, raw));
  }
  feetech::syncWriteGoals(Serial1, ids, goals, 12);
}

static bool setTorque(bool on) {
  if (!busReady) return false;
  bool ok = true;
  for (int j = 0; j < 12; j++)
    if (!feetech::writeU8(Serial1, ids[j], feetech::REG_TORQUE_EN, on ? 1 : 0)) ok = false;
  return ok;
}

static void fault(const char *why) {
  if (faulted) return;
  faulted = true; faultReason = why;
  setWebState(3);
  {char m[96];snprintf(m,sizeof m,"fault reason=%s",why);flightlog::event(m);}
  bool released = false;
  for (int attempt = 0; attempt < 3 && !released; attempt++) {
    setTorque(false);
    released = busReady;
    for (int j = 0; busReady && j < 12; j++) {
      uint8_t enabled = 1;
      if (!feetech::readRegs(Serial1, ids[j], feetech::REG_TORQUE_EN, 1, &enabled, 2)
          || enabled != 0) released = false;
    }
  }
  Serial.printf("\n*** SAFETY STOP: %s -- torque release %s ***\n", why,
                released ? "verified" : "NOT verified on all servos");
  flightlog::event(released ? "torque_release verified=1" : "torque_release verified=0");
  if (flightlog::active) {
    if (released) flightlog::active = false;  // downloads need torque verified off
    flightlog::drain();
  } else flightlog::discard();
}

static void rebootRobot() {
  // Never reset away the torque-off failure: verify the actual servo registers.
  setWebState(0);
  bool released = !busReady;
  for (int attempt=0;attempt<3&&!released;attempt++) {
    setTorque(false);released=true;
    for(int j=0;j<12;j++) {
      uint8_t enabled=1;
      if(!feetech::readRegs(Serial1,ids[j],feetech::REG_TORQUE_EN,1,&enabled,2)||enabled)
        released=false;
    }
  }
  if(!released) {faulted=true;faultReason="reboot: torque release unverified";
    setWebState(3);Serial.println(faultReason);return;}
  Serial.println(F("Rebooting with torque OFF. Release the reboot chord."));
  if(flightlog::active){flightlog::event("reboot requested torque_release verified=1");
    flightlog::active=false;flightlog::drain();}
  delay(150);esp_restart();
}

static bool serviceReboot() {
  PadSample p;if(padQueue)xQueuePeek(padQueue,&p,0);
  portENTER_CRITICAL(&webMux);webPadConnected=p.connected;portEXIT_CRITICAL(&webMux);
  bool padRequest=rebootChord.update(p,millis());
  if(takeWebReboot()||padRequest) {rebootRobot();return true;}
  return false;
}

// Boot identity, staged in RAM until arming. The decoder rebuilds q from the
// observation block plus the default_rad values below.
static void stageIdentity() {
  char m[400];   // FlightRecord text holds 444; 200 cut off cpu_mhz
  snprintf(m,sizeof m,"boot firmware_md5=%s reset_reason=%d control_hz=%d action_scale=%.3f "
           "gait_hz=%.3f fall_deg=%.1f slew_rad_s=%.2f kernel=%s weights_sram=%u weights_psram=%u "
           "cpu_mhz=%lu",
           ESP.getSketchMD5().c_str(),(int)esp_reset_reason(),(int)lroundf(1.0f/DT),K2_ACTION_SCALE,
           K2_GAIT_FREQ_HZ,FALL_ANGLE*180.0f/(float)M_PI,SLEW_RAD_S,
           inferDsp?"esp-dsp":"scalar",(unsigned)weightsRamBytes,(unsigned)weightsPsramBytes,
           (unsigned long)getCpuFrequencyMhz());
  flightlog::event(m);
  snprintf(m,sizeof m,"imu gyro_bias=%.6f,%.6f,%.6f",gyroBias[0],gyroBias[1],gyroBias[2]);
  flightlog::event(m);
  for(int j=0;j<12;j++) {
    snprintf(m,sizeof m,"joint index=%d id=%d home=%d sign=%d default_rad=%.6f min_rad=%.6f max_rad=%.6f",
             j,K2_SERVO_ID[j],K2_HOME_RAW[j],K2_SIGN[j],K2_DEFAULT_POS[j],K2_QMIN[j],K2_QMAX[j]);
    flightlog::event(m);
  }
}

#if K2_RUNTIME_RATE
// One binary for the rate comparison: the build no longer differs between
// rates. Saved in NVS so the reset caused by opening the USB port keeps it.
static bool setRate(int hz, bool save) {
  if (hz != 50 && hz != 25 && hz != 10) return false;
  DT = 1.0f / hz;
  if (save) { Preferences p; p.begin("k2", false); p.putUChar("hz", hz); p.end(); }
  return true;
}
// CPU clock and inference mode apply at boot (weights are placed once), so a
// change is saved and followed by the torque-verified reboot.
static void saveBoot(const char *key, uint8_t value) {
  Preferences p; p.begin("k2", false); p.putUChar(key, value); p.end();
}
// Save a boot-time key (4/6/8 clock, F/N inference). True if c was one.
static bool saveBootKey(char c) {
  if (c == 'F' || c == 'N') saveBoot("fast", c == 'F');
  else if (c == '4' || c == '6' || c == '8') saveBoot("mhz4", (c == '4' ? 240 : c == '6' ? 160 : 80) / 4);
  else return false;
  return true;
}
static void loadRuntimeConfig() {
  Preferences p; p.begin("k2", true);
  int hz = p.getUChar("hz", 50), mhz = p.getUChar("mhz4", 60) * 4;  // stored /4: fits a byte
  bool fast = p.getUChar("fast", 0);
  p.end();
  if (!setRate(hz, false)) setRate(50, false);
  if (mhz != 240 && mhz != 160 && mhz != 80) mhz = 240;
  setCpuFrequencyMhz(mhz);
  inferDsp = weightsPsram = fast;
}
#endif

void setup() {
  Serial.begin(115200);
#if K2_RUNTIME_RATE
  loadRuntimeConfig();
#endif
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);
  Serial.println();
  Serial.println(F("=== K2 last_policy: turn + lateral, idle HOLD ==="));
  Serial.setTxTimeoutMs(1);
  flightlog::begin();
  flightlog::status();
#if K2_RUNTIME_RATE
  Serial.printf("RUNTIME build: control %d Hz, CPU %lu MHz, inference %s "
                "(serial before arming: 5/2/1 = 50/25/10 Hz; 4/6/8 = 240/160/80 MHz; "
                "N = deployed scalar, F = PSRAM + ESP-DSP)\n",
                (int)lroundf(1.0f / DT), (unsigned long)getCpuFrequencyMhz(),
                inferDsp ? "psram+esp-dsp" : "deployed-scalar");
#endif
  if (!startWeb()) {fault("web startup failed");return;}
  Serial.println(F("Wi-Fi K2-Robot / k2robot50; browser http://192.168.4.1"));

  for (int j = 0; j < 12; j++) ids[j] = (uint8_t)K2_SERVO_ID[j];
  Serial1.begin(SERVO_BAUD, SERIAL_8N1, PIN_SERVO_RX, PIN_SERVO_TX);
  Serial1.setRxFIFOFull(1);
  Serial1.setRxTimeout(1);
  busReady = true;
  if (!setTorque(false)) {fault("startup torque-off failed");return;}
  Wire.begin(PIN_SDA, PIN_SCL, 400000);
  Wire.setTimeOut(5);
  for (uint8_t a : {0x6A, 0x6B}) {
    uint8_t who;
    Wire.beginTransmission(a);
    if (Wire.endTransmission() != 0) continue;
    if (regReadN(a, 0x0F, &who, 1) && (who == 0x69 || who == 0x6A)) { lsmAddr = a; break; }
  }
  if (!lsmAddr) { Serial.println(F("no IMU -- refusing to run")); fault("no IMU"); return; }
  // Match hardware/lsm6ds3.py exactly. These are not free choices: the Pi
  // software-resets first (the chip survives an MCU restart, so an inherited
  // ODR/full-scale is not safe), runs both sensors at 104 Hz, and waits out the
  // ODR transition before any bias capture. Running at 416 Hz instead widens the
  // accelerometer bandwidth from ~52 Hz to ~208 Hz -- four times the noise into
  // the complementary filter -- and a short settle folds the power-up transient
  // into the gyro bias average.
  if (!regWrite(lsmAddr, 0x12, 0x01)) { fault("IMU reset failed"); return; }  // SW_RESET
  {
    uint32_t deadline = millis() + 200;
    uint8_t c3 = 0x01;
    while (millis() < deadline) {
      if (!regReadN(lsmAddr, 0x12, &c3, 1)) { fault("IMU reset readback failed"); return; }
      if (!(c3 & 0x01)) break;
      delay(2);
    }
    if (c3 & 0x01) { fault("IMU software reset timed out"); return; }
  }
  if (!regWrite(lsmAddr, 0x12, 0x44) || !regWrite(lsmAddr, 0x10, 0x40) ||
      !regWrite(lsmAddr, 0x11, 0x40)) { fault("IMU configuration failed"); return; }
  delay(120);   // discard the ODR transition interval before bias capture

  // 104 Hz ODR: a 2 ms cadence re-reads the same sample ~5 times, so the old
  // 200 x 2 ms capture averaged only ~40 distinct readings (0.4 s). 150 x 10 ms
  // averages 150 distinct readings over 1.5 s. Robot on the floor, torque off.
  Serial.print(F("gyro bias (ROBOT ON THE FLOOR, DO NOT TOUCH, 1.5 s)... "));
  double s[3] = {0, 0, 0};
  const int biasSamples = 150;
  for (int i = 0; i < biasSamples; i++) {
    float a[3], g[3];
    if (!imuRead(a, g)) { fault("IMU bias read failed"); return; }
    for (int k = 0; k < 3; k++) s[k] += g[k];
    delay(10);
  }
  for (int k = 0; k < 3; k++) gyroBias[k] = (float)(s[k]/biasSamples);
  Serial.printf("[%.4f %.4f %.4f] rad/s\n", gyroBias[0], gyroBias[1], gyroBias[2]);
  {
    // Chip-frame gravity, BEFORE any rotation. Compare against
    // imu_level_calibration.json "measured_g_chip" = [+0.027, -0.034, -0.999]:
    // a mismatch means this IMU is not mounted the way the calibration assumes,
    // and every frame transform downstream is then wrong. A yaw-rotated mount is
    // invisible in the level tilt reading but swaps/inverts roll and pitch.
    double ax = 0, ay = 0, az = 0;
    for (int i = 0; i < 100; i++) {
      float a[3], g[3];
      if (imuRead(a, g)) { ax += -a[0]; ay += -a[1]; az += -a[2]; }
      delay(2);
    }
    Serial.printf("g_chip [%+.3f %+.3f %+.3f]  (calibration expects [+0.027 -0.034 -0.999])\n",
                  ax/100.0, ay/100.0, az/100.0);
  }

  delay(30);
  int found = 0;
  for (int j = 0; j < 12; j++) {
    ids[j] = (uint8_t)K2_SERVO_ID[j];
    if (feetech::ping(Serial1, ids[j])) found++;
  }
  Serial.printf("servos: %d of 12\n", found);
  if (found < 12) { fault("not all 12 servos present"); return; }

  xbox.begin();
  padQueue = xQueueCreate(1, sizeof(PadSample));
  if (!padQueue || xTaskCreatePinnedToCore(padTask, "pad", 8192, nullptr, 1,
          nullptr, CONFIG_BT_NIMBLE_PINNED_TO_CORE) != pdPASS) {
    fault("cannot start Bluetooth task"); return;
  }
  // BLE must have memory before the optional weight cache.
  loadWeights(96 * 1024);
  memset(lastAction, 0, sizeof(lastAction));
  float probe[12]; infer(P_PROBE_OBS, probe);
  float probeError = 0;
  for (int i = 0; i < 12; i++) {
    if (!isfinite(probe[i])) { fault("policy self-test non-finite"); return; }
    probeError = fmaxf(probeError, fabsf(probe[i] - P_PROBE_EXPECTED[i]));
  }
  Serial.printf("policy self-test max error %.7f; weights cached %u/%u bytes (+%u PSRAM); "
                "kernel %s; heap %u\n",
                probeError, (unsigned)weightsRamBytes,
                (unsigned)(POLICY_PARAM_COUNT * sizeof(float)), (unsigned)weightsPsramBytes,
                inferDsp ? "esp-dsp" : "scalar", ESP.getFreeHeap());
  if (probeError > 0.0001f) { fault("policy self-test failed"); return; }
  Serial.println(F("pad: LT/RT dead-man; left stick walk/strafe; right stick turn; hold A=march; B=hold"));

  // Prove the sync read on the real chain before anything can move: one
  // transaction must return all twelve joints, or we are back to the
  // leg-skewed sequential path without noticing.
  {
    uint32_t t0s = micros();
    bool ok = readJoints();
    uint32_t us = micros() - t0s;
    if (!ok) { fault("sync read did not return all 12 joints"); return; }
    Serial.printf("sync read: %lu us for 12 joints (sequential was ~5300 us)\n",
                  (unsigned long)us);
    Serial.print(F("  q[deg]:"));
    for (int j = 0; j < 12; j++) Serial.printf(" %+.1f", q[j] * 180.0f / (float)M_PI);
    Serial.println();
  }

#ifdef K2_DIAGNOSTIC_ONLY
  fault("diagnostic mode (no motion)");
  return;
#endif

  setWebState(0);
  float startupGyro[3], startupGravity[3];
  // Earlier bench sketches wrote Goal_Speed/Acceleration to servo id 1 only, so
  // that joint would respond differently from the other eleven -- asymmetric leg
  // dynamics under a gait policy. Set all twelve to the same values (0 = the
  // servo's own maximum, i.e. what the Pi has always run with) so no joint is
  // left carrying a stale profile from a previous session.
  for (int j = 0; j < 12; j++) {
    if (!feetech::writeU16(Serial1, ids[j], feetech::REG_GOAL_SPEED, 0) ||
        !feetech::writeU8(Serial1, ids[j], feetech::REG_ACC, 0)) {
      fault("servo profile write failed"); return;
    }
  }

  // Automatic boot crouch: no policy commands and no flash recording.
  if (!readJoints()) { fault("initial joint read failed"); return; }
  float qStart[12];
  memcpy(qStart, q, sizeof(q));

  // Command the pose it is ALREADY in before energising. The goal register still
  // holds whatever a previous session left there, and enabling torque first
  // would snap every joint to that stale target at full speed.
  writeGoals(qStart);
  delay(20);
  gChipInit=false;
  // Boot no longer gates on orientation: the robot eases to the default pose
  // from whatever attitude it is in, however it happens to be lying. Only an
  // outright IMU failure still faults. The in-policy fall cutoff is unchanged.
  if(!attitude(startupGyro,startupGravity)) {
    fault("IMU read failed before boot crouch");return;
  }
  if (!setTorque(true)) { fault("torque enable failed"); return; }
  int n = (int)(APPROACH_S / DT);
  Serial.printf("approach: easing into the default pose over %.0f s\n", APPROACH_S);
  for (int i = 0; i <= n; i++) {
    if(serviceReboot())return;
    float a = 0.5f * (1.0f - cosf((float)M_PI * i / n));
    for (int j = 0; j < 12; j++) qCmd[j] = qStart[j] + (K2_DEFAULT_POS[j] - qStart[j]) * a;
    if (serialControl() == 'x') { fault("stop requested"); return; }
    PadSample p; if (padQueue) xQueuePeek(padQueue,&p,0);
    if (webStopped() || (p.connected && p.stop)) {fault("stop during approach");return;}
    if (!attitude(startupGyro,startupGravity)) {
      fault("IMU read failed during approach");return;
    }
    writeGoals(qCmd);
    delay((uint32_t)(DT * 1000));
  }
  memcpy(qCmd, K2_DEFAULT_POS, sizeof(qCmd));
  writeGoals(qCmd);
  delay(100); // allow the final goal to settle before releasing
  bool released=false;
  for(int attempt=0;attempt<3&&!released;attempt++) {
    setTorque(false);released=true;
    for(int j=0;j<12;j++) {
      uint8_t enabled=1;
      if(!feetech::readRegs(Serial1,ids[j],feetech::REG_TORQUE_EN,1,&enabled,2)||enabled)
        released=false;
    }
  }
  if(!released) {fault("boot crouch torque release unverified");return;}
  setWebState(1);
  stageIdentity();
  flightlog::status();
  Serial.println(F("READY: boot crouch complete, torque OFF. Enable with browser, trigger+A, or serial g. Trigger+X (1s) reboots."));
  bool armReleased=false;
  while (true) {
    bool request=takeWebArm();
    if(serviceReboot())return;
    if (webStopped()) {fault("web stop");return;}
    PadSample p;if(padQueue)xQueuePeek(padQueue,&p,0);
    bool chord=p.connected&&p.a&&!p.x&&(p.lt>154||p.rt>154);
    if(!chord)armReleased=true;
    if(chord&&armReleased&&RobotControl::axis(p.forward)==0&&
        RobotControl::axis(p.lateral)==0&&RobotControl::axis(p.turn)==0)request=true;
    if (Serial.available()) {
      char c = Serial.read();
      if (flightlog::command(c)) { if (c == 'E') stageIdentity(); }
#if K2_RUNTIME_RATE
      else if (c == '5' || c == '2' || c == '1') {
        setRate(c == '5' ? 50 : c == '2' ? 25 : 10, true);
        flightlog::discard(); stageIdentity();   // identity must carry the new rate
        Serial.printf("RATE set: control %d Hz (saved)\n", (int)lroundf(1.0f / DT));
      }
      else if (saveBootKey(c)) {
        Serial.println(F("BOOT CONFIG saved; rebooting to apply"));
        rebootRobot(); return;
      }
#endif
      else {
        c |= 0x20;
        if (c == 'x') {fault("serial stop");return;}
        if (c == 'g') request=true;
      }
    }
    if(p.connected && p.stop) {fault("controller Back button");return;}
    if(request && (!flightlog::ready || flightlog::stageLost)) {
      Serial.println(F("Cannot start trial: logger unavailable/full. L = status, E = erase (download first)."));
      flightlog::status();
      request=false;
    }
    if(request) break;
    delay(10);
  }
  if (webStopped()) {fault("web stop");return;}
  setWebState(0);
  // The unpowered robot may have sagged: begin from its measured pose. The
  // existing per-tick slew limiter brings it into policy control without a jump.
  gChipInit=false;
  if(!attitude(startupGyro,startupGravity) ||
      acosf(fmaxf(-1,fminf(1,-startupGravity[2])))>FALL_ANGLE) {
    fault("support upright before enabling");return;
  }
  if(!readJoints()) {fault("arming joint read failed");return;}
  memcpy(qCmd,q,sizeof(qCmd));writeGoals(qCmd);delay(20);
  flightlog::active=true;   // torque may be on from here: no download/erase
  flightlog::commit();
  flightlog::event("armed");
  if(!setTorque(true)) {fault("torque enable failed");return;}
  headingEst = headingRef = 0;
  feedback.reset(millis());
  lastTimely = millis();
  setWebState(2);
  Serial.println(F("policy running: sticks activate walking/turning; Back or x releases torque\n"));
  // Telemetry must never hold up balance when USB is connected but unread.
  Serial.setTxTimeoutMs(1);
}

// One kind-2 record per control tick, submitted on EVERY exit path after the
// joint read (paused, skipped and late ticks included). Layout is flight-log v3:
// obs[0..49] action[50..61] goal[62..73] acc[74..76] gyro[77..79]
// target vx,vy,wz[80..82] tilt[83] heading_est[84] heading_ref[85]
// load/current[86..109] (not collected: NaN) phase[110].
struct TickRecord {
  FlightRecord r;
  uint32_t start;
  float *v;
  TickRecord(uint32_t now, uint32_t interval) : start(now), v(r.payload.values) {
    r.kind = 2; r.intervalUs = interval;
    for (int i = 0; i < 111; i++) v[i] = NAN;
  }
  void work() { if (!r.workUs) r.workUs = micros() - start; }
  ~TickRecord() {
    if (faulted || !flightlog::active) return;
    work();
    r.valid = freshMask; r.errors = errorMask;
    r.reserved[0] = deadlineTicks; r.reserved[1] = missingTicks;
    for (int j = 0; j < 12; j++) v[62 + j] = qCmd[j];
    for (int i = 0; i < 3; i++) { v[74 + i] = accChip[i]; v[77 + i] = gyroChip[i]; }
    flightlog::submit(r);
  }
};

void loop() {
  if(serviceReboot())return;
  if (!faulted && webStopped()) {fault("web stop");return;}
  if (faulted) {
    if (Serial.available()) {
      char c = Serial.read();
#if K2_RUNTIME_RATE
      // Torque is already released here; save only, and let a reset apply it.
      if (c == '5' || c == '2' || c == '1') {
        setRate(c == '5' ? 50 : c == '2' ? 25 : 10, true);
        Serial.println(F("RATE saved; applies after reset"));
      } else if (saveBootKey(c)) Serial.println(F("BOOT CONFIG saved; applies after reset"));
      else
#endif
      flightlog::command(c);
    }
    static uint32_t diagnosticAt = 0;
    if (lsmAddr && millis() - diagnosticAt >= 500) {
      diagnosticAt = millis();
      float acc[3], gyr[3];
      PadSample p;
      if (padQueue) xQueuePeek(padQueue, &p, 0);
      if (imuRead(acc, gyr)) {
        float g[3] = {-acc[0], -acc[1], -acc[2]}, root[3];
        imuToRoot(g, root);
        float mag = sqrtf(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);
        float tilt = acosf(fmaxf(-1, fminf(1, -root[2]/fmaxf(mag, 1e-6f))));
        Serial.printf("STOP raw_acc[%+.3f %+.3f %+.3f] |a|%.3f root_g[%+.3f %+.3f %+.3f] "
                      "tilt%.1f gyro[%+.3f %+.3f %+.3f] pad%d age%lu failures%u\n",
                      acc[0], acc[1], acc[2], mag, root[0], root[1], root[2],
                      tilt*180.0f/(float)M_PI, gyr[0]-gyroBias[0], gyr[1]-gyroBias[1],
                      gyr[2]-gyroBias[2], p.connected, (unsigned long)(millis()-p.receivedAt),
                      xbox.getCountFailedConnection());
      }
    }
    delay(10);
    return;
  }
  static uint32_t nextTick = 0, tick = 0, previousTick = 0;
  static uint32_t overruns = 0, maxLoopUs = 0;
  const uint32_t periodUs = (uint32_t)(DT * 1e6f);
  uint32_t now = micros();
  if (nextTick == 0) nextTick = now;
  if ((int32_t)(now - nextTick) < 0) { delay(1); return; }
  uint32_t intervalUs = previousTick ? now - previousTick : periodUs;
  previousTick = now;
  // Do not replay missed ticks in a burst against real servos.
  if (now - nextTick >= periodUs) {
    overruns++;
    nextTick = now;
  }
  nextTick += periodUs;
  if (intervalUs > 2000000u) { fault("control loop stalled >2 s"); return; }
  // A single late tick is not an emergency: goals are held for that tick and
  // the policy resumes on the next one. Only a RUN of late ticks latches the
  // pause (which then needs neutral controls to clear). The old latch-on-first
  // behaviour turned one 7-11 ms inference tail (flash-cache stall while the
  // logger writes) into a forced stop mid-walk.
  bool lateInterval = intervalUs > 2 * periodUs;
  if(intervalUs>periodUs+1000)++deadlineTicks;
  if (lateInterval && ++lateTicks >= LATE_LATCH_TICKS) timingPaused = true;
  if (serialControl() == 'x') { fault("stop requested"); return; }

  TickRecord rec(now, intervalUs);
  readJoints(false);
  if(freshMask!=FeedbackGuard::ALL)++missingTicks;
  feedback.update(freshMask, millis());
  uint32_t readUs = micros() - now;
  rec.r.readUs = readUs;
  float gyro[3], grav[3];
  if (!attitude(gyro, grav)) { fault("IMU read failed"); return; }

  float tilt = acosf(fmaxf(-1.0f, fminf(1.0f, -grav[2])));
  for (int i = 0; i < 3; i++) { rec.v[i] = gyro[i]; rec.v[3 + i] = grav[i]; }
  for (int j = 0; j < 12; j++) if (freshMask & (1u << j)) {
    rec.v[6 + j] = q[j] - K2_DEFAULT_POS[j]; rec.v[18 + j] = qd[j];
  }
  rec.v[83] = tilt; rec.v[84] = headingEst;
  if (tilt > FALL_ANGLE) { fault("tilt exceeded 24 deg"); return; }

  // Commands. Losing the pad or releasing the dead-man zeroes them and holds --
  // it does not cut torque, because a standing robot with no torque falls over.
  PadSample pad;
  xQueuePeek(padQueue, &pad, 0);
  controls.update(pad, millis());
  if (controls.linked && pad.stop) { fault("controller Back button"); return; }
  bool link = controls.linked, armed = controls.armed;
  WebCommand web; xQueuePeek(webQueue,&web,0);
  commandArbiter.update(controls,pad.connected&&pad.b,web,millis());
  bool marching=commandArbiter.moving;
  float vx=commandArbiter.vx,wz=commandArbiter.wz,vy=commandArbiter.vy;
  rec.v[80] = vx; rec.v[81] = vy; rec.v[82] = wz;
  rec.r.flags = 1 | (marching ? 8 : 0) | (link ? 16 : 0) | (armed ? 32 : 0);
  rec.r.reserved[2] = millis() - pad.receivedAt;
  portENTER_CRITICAL(&webMux);
  webOwner=marching?commandArbiter.owner:CommandArbiter::HOLD;
  portEXIT_CRITICAL(&webMux);

  if (feedback.expired(millis())) {
    fault("servo feedback unavailable >2 s"); return;
  }
  bool recovering = feedback.paused || timingPaused;
  feedback.resume(freshMask, !marching);
  if (!marching && !lateInterval) timingPaused = false;
  if (feedback.blocking() || timingPaused) {
    // Isolated misses self-clear. Sustained outages require neutral controls.
    static uint32_t pauseReport = 0;
    if ((feedback.paused || timingPaused) && millis() - pauseReport >= 500) {
      pauseReport = millis();
      const char *message = "PAUSED: torque retained; release controls after feedback recovers\n";
      if (Serial.availableForWrite() >= int(strlen(message))) Serial.print(message);
    }
    writeGoals(qCmd);
    rec.r.flags |= 4; rec.work();
    return;
  }
  if (recovering) {
    commandProfile = CommandProfile();
    gaitAct = phase = 0;
    headingRef = headingEst;
  }
  if (lateInterval) {   // skip this tick only
    writeGoals(qCmd); rec.r.flags |= 4; rec.work(); return;
  }

  // HOLD has no heading target. In training the reference is a frozen number
  // and the policy is rewarded for keeping the robot ON it, so any heading
  // error it sees while standing is a request to shuffle round until the error
  // is gone. On the robot the only thing that moves that error during a hold
  // is gyro-bias drift in headingEst -- and the policy then walks it out in
  // tiny steps ("moves little by little with no command"). Re-anchor the
  // reference to the current heading while holding; the next gait starts from
  // zero error exactly as every training episode does.
  if (!marching && gaitAct <= 0.0f) {
    headingRef = headingEst;
    float w = 0;
    for (int i = 0; i < 3; i++) w += gyrDebiased[i] * gyrDebiased[i];
    if (w < BIAS_REFINE_GATE * BIAS_REFINE_GATE)
      for (int i = 0; i < 3; i++) gyroBias[i] += BIAS_REFINE_ALPHA * gyrDebiased[i];
  }

  float obs[POLICY_OBS_DIM], action[12];
  buildObsFrom(gyro, grav, q, qd, vx, wz, vy, marching, headingEst, obs);
  for (int i=0;i<POLICY_OBS_DIM;i++)
    if (!isfinite(obs[i])) {fault("non-finite observation");return;}
  uint32_t inferStart = micros();
  infer(obs, action);
  uint32_t inferUs = micros() - inferStart;
  for (int i = 0; i < POLICY_OBS_DIM; i++) rec.v[i] = obs[i];
  for (int i = 0; i < 12; i++) rec.v[50 + i] = action[i];
  rec.v[85] = headingRef; rec.v[110] = phase;
  rec.r.inferUs = inferUs;

  for (int i = 0; i < 12; i++)
    if (!isfinite(action[i])) { fault("policy returned a non-finite action"); return; }
  if (micros() - now >= periodUs) {
    ++deadlineTicks;
    if (++lateTicks >= LATE_LATCH_TICKS) timingPaused = true;
    writeGoals(qCmd);
    rec.r.flags |= 4; rec.work();
    if (millis() - lastTimely > 2000) fault("no timely policy update >2 s");
    return;
  }
  lastTimely = millis();
  lateTicks = 0;
  for (int i = 0; i < 12; i++) lastAction[i] = action[i];

  // Slew: cap how far any joint may move in one tick.
  float step = SLEW_RAD_S * DT;
  for (int j = 0; j < 12; j++) {
    qCmd[j] = boundedJointCommand(j, qCmd[j], action[j], step);
  }
  writeGoals(qCmd);
  rec.r.flags |= 64; rec.work();
  maxLoopUs = max(maxLoopUs, (uint32_t)(micros() - now));

  if (++tick % 25 == 0) {
    float mag = 0; for (int i = 0; i < 12; i++) mag += action[i]*action[i];
    char line[240];
    int len = snprintf(line, sizeof(line),
        "t%.1fs tilt%.1f %s %s m%.2f vx%+.3f wz%+.3f vy%+.3f act%.2f "
        "herr%+.1f bz%+.3f dt%luus read%luus infer%luus max%luus over%lu heap%u age%lums\n",
        tick*DT, tilt*180.0f/(float)M_PI, link ? "pad" : "NOPAD",
        armed ? "ARMED" : "web/hold", gaitAct, vx, wz, vy, sqrtf(mag/12.0f),
        (headingEst - headingRef)*180.0f/(float)M_PI, gyroBias[2]*180.0f/(float)M_PI,
        (unsigned long)intervalUs, (unsigned long)readUs, (unsigned long)inferUs,
        (unsigned long)maxLoopUs, (unsigned long)overruns, ESP.getFreeHeap(),
        (unsigned long)(millis() - pad.receivedAt));
    if (len > 0 && len < (int)sizeof(line) && Serial.availableForWrite() >= len)
      Serial.write((const uint8_t *)line, len);
  }
}
