#pragma once
#include <math.h>

// Exact scalar counterpart of hardware/k2_command_profile.py (smooth_full_v1).
struct CommandProfile {
  float value[3] = {}, rate[3] = {};
  bool settling = false;
  static float clip(float v, float lo, float hi) { return fmaxf(lo, fminf(hi, v)); }
  void step(float vx, float vy, float wz, bool moving, float dt,
            float frequency, float transition, float &phase, float &march) {
    const float target[3] = {moving ? vx : 0, moving ? vy : 0, moving ? wz : 0};
    const float limits[3] = {.04f, .035f, .25f}, jerk[3] = {.12f, .105f, .75f};
    const float stopSpeed[3] = {.001f, .001f, .008f}, stopRate[3] = {.005f, .005f, .04f};
    bool stopped = true;
    for (int i = 0; i < 3; ++i) {
      float acceleration = clip(36*(target[i]-value[i])-12*rate[i], -jerk[i], jerk[i]);
      rate[i] = clip(rate[i]+acceleration*dt, -limits[i], limits[i]);
      value[i] += rate[i]*dt;
      stopped &= fabsf(value[i]) <= stopSpeed[i] && fabsf(rate[i]) <= stopRate[i];
    }
    settling = !moving && (settling || (stopped && fabsf(sinf(phase)) < .30f));
    bool active = moving || (!settling && march > 0);
    if (moving && march <= 0) phase = target[1] < 0 ? (float)M_PI : 0;
    march = clip(march + (active ? dt/transition : -dt/transition), 0, 1);
    if (active && march >= 1-1e-6f)
      phase = fmodf(phase + 2*(float)M_PI*frequency*dt, 2*(float)M_PI);
    if (march <= 0) phase = 0;
  }
};
