// K2 RL policy running on the ESP32-S3: the same 50 Hz loop hardware/k2_policy_run.py
// runs on the Pi 5, rebuilt in C++.
//
//   LSM6DS3 -> complementary filter -> base_ang_vel + projected_gravity
//   servo bus -> joint_pos_rel, joint_vel_rel
//   + last_action + heading + gait command  = 50-D observation
//   -> MLP (50-256-128-64-12, ELU) -> q_target = default_pose + 0.2 * action
//
// SERVO OUTPUT IS DISABLED. This computes the full loop and reports it, but
// never enables torque and never writes a goal position. Two reasons:
// the K2 robot is not attached to this ESP32 (the bus has one bench servo, not
// twelve), and a 12-DOF gait policy driving one isolated joint is meaningless.
// Set ENABLE_SERVO_OUTPUT only with the real robot wired and supported.
#define ENABLE_SERVO_OUTPUT 0

#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include "feetech.h"
#include "k2_obs.h"

static const int PIN_SDA = 5, PIN_SCL = 6;
static const int PIN_SERVO_TX = 43, PIN_SERVO_RX = 44;
static const uint32_t SERVO_BAUD = 1000000;
static const uint8_t LSM_ADDR_CANDIDATES[2] = {0x6A, 0x6B};

// ---------------------------------------------------------------- IMU
static uint8_t lsmAddr = 0;
static float gyroBias[3] = {0, 0, 0};
static float gChip[3] = {0, 0, -1};
static bool gChipInit = false;
static float headingEst = 0.0f;

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

// accel in g, gyro in rad/s, both chip frame.
static bool imuRead(float *acc, float *gyr) {
  uint8_t b[12];
  if (!regReadN(lsmAddr, 0x22, b, 12)) return false;
  const float GD = 0.00875f * (float)M_PI / 180.0f;   // mdps/LSB -> rad/s
  const float AG = 0.000061f;                          // mg/LSB  -> g
  for (int i = 0; i < 3; i++) {
    gyr[i] = (int16_t)(b[2*i]   | (b[2*i+1]   << 8)) * GD;
    acc[i] = (int16_t)(b[6+2*i] | (b[6+2*i+1] << 8)) * AG;
  }
  return true;
}

// chip -> root, from k2_attitude.R_IMU_TO_ROOT: root_x=chip_y, root_y=-chip_x,
// then the measured level correction. k2_attitude applies BOTH to gravity and
// to the gyro; leaving the level term out gives a standing robot a few degrees
// of false tilt and biases every gravity observation the policy sees.
static inline void imuToRoot(const float *v, float *o) {
  float r[3] = {v[1], -v[0], v[2]};
  for (int i = 0; i < 3; i++)
    o[i] = K2_LEVEL[3*i+0]*r[0] + K2_LEVEL[3*i+1]*r[1] + K2_LEVEL[3*i+2]*r[2];
}
// R_SITE_TO_ROOT.T applied to a root vector: diag chain gives the site frame
// the policy's base_ang_vel was trained in.
static inline void rootToSite(const float *v, float *o) {
  o[0] = -v[1]; o[1] = -v[0]; o[2] = -v[2];
}

// Port of ImuAttitude.attitude(): gyro-propagate gravity, correct with accel
// only in proportion to how trustworthy the accel looks.
static void attitude(float *gyroSite, float *projGravRoot) {
  float acc[3], gyr[3];
  if (!imuRead(acc, gyr)) { gyr[0] = gyr[1] = gyr[2] = 0; acc[0] = acc[1] = 0; acc[2] = -1; }
  for (int i = 0; i < 3; i++) gyr[i] -= gyroBias[i];

  float g[3] = {-acc[0], -acc[1], -acc[2]};
  float n = sqrtf(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);
  bool accelValid = isfinite(n) && n > 0.2f && n < 2.5f;
  if (accelValid) for (int i = 0; i < 3; i++) g[i] /= n;

  if (!gChipInit) {
    if (accelValid) { gChip[0]=g[0]; gChip[1]=g[1]; gChip[2]=g[2]; }
    gChipInit = true;
  } else {
    // A world-fixed vector rotates as -omega x g in the body frame.
    float c[3] = {gyr[1]*gChip[2] - gyr[2]*gChip[1],
                  gyr[2]*gChip[0] - gyr[0]*gChip[2],
                  gyr[0]*gChip[1] - gyr[1]*gChip[0]};
    for (int i = 0; i < 3; i++) gChip[i] -= DT * c[i];
    float m = sqrtf(gChip[0]*gChip[0] + gChip[1]*gChip[1] + gChip[2]*gChip[2]);
    for (int i = 0; i < 3; i++) gChip[i] /= fmaxf(m, 1e-6f);

    if (accelValid) {
      float normConf = expf(-powf((n - 1.0f) / 0.12f, 2.0f));
      float dot = gChip[0]*g[0] + gChip[1]*g[1] + gChip[2]*g[2];
      dot = fmaxf(-1.0f, fminf(1.0f, dot));
      float dirConf = expf(-powf(acosf(dot) / 0.25f, 2.0f));
      float alpha = 0.001f + (0.02f - 0.001f) * (normConf * dirConf);
      for (int i = 0; i < 3; i++) gChip[i] = (1 - alpha) * gChip[i] + alpha * g[i];
      float m2 = sqrtf(gChip[0]*gChip[0] + gChip[1]*gChip[1] + gChip[2]*gChip[2]);
      for (int i = 0; i < 3; i++) gChip[i] /= fmaxf(m2, 1e-6f);
    }
  }

  imuToRoot(gChip, projGravRoot);
  float gyroRoot[3];
  imuToRoot(gyr, gyroRoot);
  headingEst = fmodf(headingEst + gyroRoot[2] * DT + (float)M_PI, 2.0f * (float)M_PI) - (float)M_PI;
  rootToSite(gyroRoot, gyroSite);
}

// ---------------------------------------------------------------- state
static float q[12], qd[12];
static bool servoPresent[12];
static int servoCount = 0;
static float cmdVx = 0, cmdWz = 0, cmdDh = 0;
static bool marching = true;

// Reads whichever servos answer. A joint with no servo holds its default pose,
// so the observation stays well-formed on a partial bench rig.
static void readJoints() {
  for (int j = 0; j < 12; j++) {
    if (!servoPresent[j]) { q[j] = K2_DEFAULT_POS[j]; qd[j] = 0.0f; continue; }
    uint8_t d[4];
    if (!feetech::readRegs(Serial1, K2_SERVO_ID[j], feetech::REG_PRES_POS, 4, d)) continue;
    uint16_t raw = d[0] | (d[1] << 8);
    q[j]  = K2_SIGN[j] * ((int)raw - K2_HOME_RAW[j]) * K2_RAD_PER_COUNT;
    qd[j] = K2_SIGN[j] * feetech::signMag(d[2] | (d[3] << 8)) * K2_RAD_PER_COUNT;
  }
}

static void buildObs(float *obs) {
  float gyro[3], grav[3];
  attitude(gyro, grav);

  int k = 0;
  for (int i = 0; i < 3; i++) obs[k++] = gyro[i];          // base_ang_vel
  for (int i = 0; i < 3; i++) obs[k++] = grav[i];          // projected_gravity
  for (int i = 0; i < 12; i++) obs[k++] = q[i] - K2_DEFAULT_POS[i];  // joint_pos_rel
  for (int i = 0; i < 12; i++) obs[k++] = qd[i];           // joint_vel_rel
  for (int i = 0; i < 12; i++) obs[k++] = lastAction[i];   // previous RAW action

  // gait activation ramps toward the target the way training transitioned it.
  float target = marching ? 1.0f : 0.0f;
  float step = DT / K2_GAIT_TRANSITION_S;
  gaitAct += fmaxf(-step, fminf(step, target - gaitAct));
  if (marching) phase = fmodf(phase + 2.0f * (float)M_PI * K2_GAIT_FREQ_HZ * DT,
                              2.0f * (float)M_PI);
  else phase = 0.0f;

  // Reference heading integrates the yaw we just commanded, then the error is
  // read against it -- the same order as training.
  headingRef = fmodf(headingRef + cmdWz * DT + (float)M_PI, 2.0f*(float)M_PI) - (float)M_PI;
  float err = headingEst - headingRef;
  obs[k++] = sinf(err);
  obs[k++] = cosf(err);

  obs[k++] = gaitAct;                     // march
  obs[k++] = gaitAct * sinf(phase);
  obs[k++] = gaitAct * cosf(phase);
  obs[k++] = cmdVx;
  obs[k++] = cmdWz;
  obs[k++] = cmdDh;
}

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);

  Serial.println();
  Serial.println(F("=== K2 policy runner (turn_v11_iter3400) ==="));
#if ENABLE_SERVO_OUTPUT
  Serial.println(F("!! SERVO OUTPUT ENABLED !!"));
#else
  Serial.println(F("servo output DISABLED -- computes the loop, drives nothing"));
#endif

  Wire.begin(PIN_SDA, PIN_SCL, 400000);
  for (uint8_t i = 0; i < 2; i++) {
    uint8_t who;
    Wire.beginTransmission(LSM_ADDR_CANDIDATES[i]);
    if (Wire.endTransmission() != 0) continue;
    if (regReadN(LSM_ADDR_CANDIDATES[i], 0x0F, &who, 1) && (who == 0x69 || who == 0x6A)) {
      lsmAddr = LSM_ADDR_CANDIDATES[i];
      Serial.printf("IMU at 0x%02X (WHO_AM_I 0x%02X)\n", lsmAddr, who);
      break;
    }
  }
  if (!lsmAddr) { Serial.println(F("no IMU -- cannot build the observation")); return; }
  regWrite(lsmAddr, 0x12, 0x44);
  regWrite(lsmAddr, 0x10, 0x60);
  regWrite(lsmAddr, 0x11, 0x60);
  delay(50);

  // Gyro bias must be captured while still -- left in, it drifts the gravity
  // estimate and the policy slowly leans.
  Serial.print(F("capturing gyro bias (hold still)... "));
  double acc[3] = {0, 0, 0};
  for (int i = 0; i < 200; i++) {
    float a[3], g[3];
    if (imuRead(a, g)) for (int k = 0; k < 3; k++) acc[k] += g[k];
    delay(2);
  }
  for (int k = 0; k < 3; k++) gyroBias[k] = (float)(acc[k] / 200.0);
  Serial.printf("[%.4f %.4f %.4f] rad/s\n", gyroBias[0], gyroBias[1], gyroBias[2]);

  Serial1.begin(SERVO_BAUD, SERIAL_8N1, PIN_SERVO_RX, PIN_SERVO_TX);
  delay(30);
  for (int j = 0; j < 12; j++) {
    servoPresent[j] = feetech::ping(Serial1, K2_SERVO_ID[j]);
    if (servoPresent[j]) servoCount++;
  }
  Serial.printf("servos found: %d of 12", servoCount);
  if (servoCount) { Serial.print(" -> ids"); for (int j=0;j<12;j++) if (servoPresent[j]) Serial.printf(" %d", K2_SERVO_ID[j]); }
  Serial.println();
  if (servoCount < 12)
    Serial.println(F("   (missing joints hold the default pose in the observation)"));

  loadWeights();
  Serial.printf("policy weights in %s\n", weightsInRam ? "internal SRAM" : "FLASH (slow path)");
  Serial.println(F("\nrunning at 50 Hz\n"));
}

static void dumpObs(const float *obs, const float *action) {
  static const char *NAME[12] = {"hipP_R","hipR_R","hipY_R","knee_R","ankP_R","ankR_R",
                                 "hipP_L","hipR_L","hipY_L","knee_L","ankP_L","ankR_L"};
  Serial.println(F("\n--- observation dump ---"));
  Serial.printf("base_ang_vel      [%+.3f %+.3f %+.3f] rad/s\n", obs[0], obs[1], obs[2]);
  Serial.printf("projected_gravity [%+.3f %+.3f %+.3f]  (level stand ~ [0 0 -1])\n",
                obs[3], obs[4], obs[5]);
  Serial.println(F("joint            q_rel(rad)  q_rel(deg)   qd     action"));
  for (int i = 0; i < 12; i++)
    Serial.printf("  %-7s %10.4f %10.2f %8.3f %8.3f\n", NAME[i],
                  obs[6+i], obs[6+i] * 180.0f / (float)M_PI, obs[18+i], action[i]);
  Serial.printf("heading [sin %+.3f cos %+.3f]  gait [%.2f %+.2f %+.2f  vx%+.3f wz%+.3f dh%+.3f]\n",
                obs[42], obs[43], obs[44], obs[45], obs[46], obs[47], obs[48], obs[49]);
  Serial.println(F("joint_pos_rel should be near zero when the robot is in its default pose\n"));
}

void loop() {
  if (!lsmAddr) { delay(1000); return; }

  static uint32_t nextTick = 0, tick = 0;
  static float worstLoopMs = 0, sumLoopMs = 0;
  uint32_t now = micros();
  if (nextTick == 0) nextTick = now;
  if ((int32_t)(now - nextTick) < 0) return;
  nextTick += (uint32_t)(DT * 1e6f);

  uint32_t t0 = micros();
  readJoints();
  uint32_t tRead = micros();

  float obs[POLICY_OBS_DIM];
  buildObs(obs);
  uint32_t tObs = micros();

  float action[12];
  infer(obs, action);
  uint32_t tInfer = micros();

  float qTarget[12];
  for (int i = 0; i < 12; i++) {
    qTarget[i] = K2_DEFAULT_POS[i] + K2_ACTION_SCALE * action[i];
    lastAction[i] = action[i];
  }

#if ENABLE_SERVO_OUTPUT
#error "Enabling servo output requires the full 12-DOF robot wired and supported."
#endif

  float loopMs = (micros() - t0) / 1000.0f;
  worstLoopMs = fmaxf(worstLoopMs, loopMs);
  sumLoopMs += loopMs;
  tick++;

  if (Serial.available() && (Serial.read() | 0x20) == 'o') dumpObs(obs, action);

  if (tick % 25 == 0) {   // ~2 Hz
    float tilt = acosf(fmaxf(-1.0f, fminf(1.0f, -obs[5]))) * 180.0f / (float)M_PI;
    Serial.printf("t%6.1fs tilt%5.1fdeg gyro[%+.2f %+.2f %+.2f] grav[%+.2f %+.2f %+.2f] "
                  "march%.2f |act|%.2f  loop %.2f/%.2f ms (read %lu obs %lu inf %lu us)\n",
                  tick * DT, tilt, obs[0], obs[1], obs[2], obs[3], obs[4], obs[5],
                  gaitAct,
                  sqrtf([&]{ float s=0; for(int i=0;i<12;i++) s+=action[i]*action[i]; return s; }() / 12.0f),
                  loopMs, worstLoopMs,
                  (unsigned long)(tRead - t0), (unsigned long)(tObs - tRead),
                  (unsigned long)(tInfer - tObs));
  }
}
