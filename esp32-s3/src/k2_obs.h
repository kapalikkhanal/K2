#pragma once
// Shared policy core: weight loading, the MLP, and the 50-D observation
// assembly. Both the on-robot runner and the hardware-in-the-loop build include
// this, so the code validated against the digital twin is literally the code
// that runs on the robot -- they cannot drift apart.

#include <Arduino.h>
#include <math.h>
#include "policy_weights.h"
#include "k2_config.h"
#include "command_profile.h"
#ifndef K2_INFER_DSP
#define K2_INFER_DSP 0      // 1: ESP-DSP dot products (env:robot_fast)
#endif
#ifndef K2_WEIGHTS_PSRAM
#define K2_WEIGHTS_PSRAM 0  // 1: weights that miss the SRAM cache go to PSRAM
#endif
#include "esp_dsp.h"
// env:robot_rates chooses these at boot (NVS); every other build fixes them.
#if K2_RUNTIME_RATE
static bool inferDsp = K2_INFER_DSP, weightsPsram = K2_WEIGHTS_PSRAM;
#else
static const bool inferDsp = K2_INFER_DSP, weightsPsram = K2_WEIGHTS_PSRAM;
#endif

// The policy was trained at 50 Hz. Any other rate (env:robot25) is a
// deployment-rate experiment, matching --rate-hz in hardware/k2_policy_run.py.
#ifndef K2_CONTROL_HZ
#define K2_CONTROL_HZ 50
#endif
// env:robot_rates (K2_RUNTIME_RATE) may change this before arming; every other
// build keeps the compiled rate.
static float DT = 1.0f / K2_CONTROL_HZ;

static float *W[4], *B[4], *Wmean, *Wstd;
static const int L_OUT[4] = {L0_OUT, L1_OUT, L2_OUT, L3_OUT};
static const int L_IN[4]  = {L0_IN,  L1_IN,  L2_IN,  L3_IN};
static bool weightsInRam = false;
static size_t weightsPsramBytes = 0;

// Cache whole arrays when SRAM permits, preserving headroom for BLE service
// discovery. The old all-or-nothing 220 KB allocation could starve Bluetooth.
static size_t weightsRamBytes = 0;
static void loadWeights(size_t reserveBytes = 0) {
  // sram: 1 = must try SRAM, 0 = never, -1 = SRAM if the reserve allows.
  auto cache = [&](const float *src, size_t count, int sram = -1) -> float * {
    size_t bytes = count * sizeof(float);
    float *p = nullptr;
    if (sram == 1 || (sram == -1 &&
        heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT) >= bytes + reserveBytes))
      p = (float *)heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (p) weightsRamBytes += bytes;
    // What does not fit beside the BLE reserve goes to octal PSRAM, not flash.
    if (!p && weightsPsram &&
        (p = (float *)heap_caps_aligned_alloc(16, bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)))
      weightsPsramBytes += bytes;
    if (!p) return (float *)src;
    memcpy(p, src, bytes);
    return p;
  };
#if K2_RUNTIME_RATE
  // Fixed placement for clock/rate experiments: exactly the 88,000-byte set the
  // 240 MHz build ends up caching. The reserve test below depends on how far
  // BLE/WiFi start-up has got when it runs, so at 80 MHz it cached only 56,512.
  const int s1 = 1, s0 = 0;
#else
  const int s1 = -1, s0 = -1;
#endif
  Wmean = cache(P_MEAN, POLICY_OBS_DIM, s1); Wstd = cache(P_STD, POLICY_OBS_DIM, s1);
  // Cache the largest layer first: it accounts for most multiply-adds.
  W[1] = cache(P_W1, sizeof(P_W1)/sizeof(float), s0);
  W[0] = cache(P_W0, sizeof(P_W0)/sizeof(float), s1);
  W[2] = cache(P_W2, sizeof(P_W2)/sizeof(float), s1);
  W[3] = cache(P_W3, sizeof(P_W3)/sizeof(float), s1);
  B[0] = cache(P_B0, L0_OUT, s0); B[1] = cache(P_B1, L1_OUT, s1);
  B[2] = cache(P_B2, L2_OUT, s0); B[3] = cache(P_B3, L3_OUT, s1);
  weightsInRam = weightsRamBytes == POLICY_PARAM_COUNT * sizeof(float);
}

static void infer(const float *obs, float *out) {
  alignas(16) float a[256], b[256];
  for (int i = 0; i < POLICY_OBS_DIM; i++) a[i] = (obs[i] - Wmean[i]) / Wstd[i];
  float *src = a, *dst = b;
  for (int l = 0; l < 4; l++) {
    for (int o = 0; o < L_OUT[l]; o++) {
      const float *row = W[l] + (size_t)o * L_IN[l];
      float acc;
      if (inferDsp) {
        dsps_dotprod_f32(row, src, &acc, L_IN[l]);   // ESP-DSP, aes3 on the S3
        acc += B[l][o];
      } else {
        acc = B[l][o];
        for (int i = 0; i < L_IN[l]; i++) acc += row[i] * src[i];
      }
      dst[o] = acc;
    }
    if (l < 3) for (int i = 0; i < L_OUT[l]; i++) if (dst[i] <= 0) dst[i] = expm1f(dst[i]);
    float *t = src; src = dst; dst = t;
  }
  memcpy(out, src, POLICY_ACT_DIM * sizeof(float));
}


// Stateful pieces of the observation: the policy sees its own previous action,
// an integrated gait phase and a heading error against a reference it advances
// itself, so these must persist across ticks exactly as they do in Python.
static float lastAction[12];
static float phase = 0.0f, gaitAct = 0.0f, headingRef = 0.0f;
static CommandProfile commandProfile;

// gyro and grav are supplied by the caller: from the LSM6DS3 on the robot, or
// from the twin over serial in HIL.
static void buildObsFrom(const float *gyro, const float *grav,
                         const float *q, const float *qd,
                         float cmdVx, float cmdWz, float cmdVy, bool marching,
                         float headingEst, float *obs) {
  int k = 0;
  for (int i = 0; i < 3; i++) obs[k++] = gyro[i];
  for (int i = 0; i < 3; i++) obs[k++] = grav[i];
  for (int i = 0; i < 12; i++) obs[k++] = q[i] - K2_DEFAULT_POS[i];
  for (int i = 0; i < 12; i++) obs[k++] = qd[i];
  for (int i = 0; i < 12; i++) obs[k++] = lastAction[i];

  commandProfile.step(cmdVx, cmdVy, cmdWz, marching, DT,
                      K2_GAIT_FREQ_HZ, K2_GAIT_TRANSITION_S, phase, gaitAct);
  float vxG = gaitAct * commandProfile.value[0];
  float wzG = gaitAct * commandProfile.value[2];
  float vyG = gaitAct * commandProfile.value[1];

  headingRef = fmodf(headingRef + wzG * DT + (float)M_PI, 2.0f*(float)M_PI) - (float)M_PI;
  float err = headingEst - headingRef;
  obs[k++] = sinf(err);
  obs[k++] = cosf(err);

  obs[k++] = gaitAct;
  obs[k++] = gaitAct * sinf(phase);
  obs[k++] = gaitAct * cosf(phase);
  obs[k++] = vxG;
  obs[k++] = wzG;
  obs[k++] = vyG;
}
