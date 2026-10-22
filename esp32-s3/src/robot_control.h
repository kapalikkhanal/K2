#pragma once
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include "k2_config.h"

inline bool parseWebSpeed(const char *text, float maximum, float &value) {
  char *end=nullptr;value=strtof(text,&end);
  return end!=text && *end=='\0' && isfinite(value) && value>0 && value<=maximum;
}

struct WebCommand {
  uint32_t at = 0;
  uint32_t intent = 0; // increments for a new command, never for heartbeat renewal
  float vx = 0, vy = 0, wz = 0;
  bool moving = false;
};

// A held reboot chord must be released once after boot before it can fire.
struct PadSample;
struct RebootChord {
  bool ready = false, holding = false;
  uint32_t since = 0;
  bool update(const PadSample &p, uint32_t now);
};
inline bool freshWebCommand(const WebCommand &c, uint32_t now) {
  return c.moving && (uint32_t)(now-c.at) < 500;
}

// Plain values passed from the BLE task to the 50 Hz control task.
struct PadSample {
  uint32_t receivedAt = 0;
  bool connected = false;
  uint16_t forward = 32768, turn = 32768, lateral = 32768, lt = 0, rt = 0;
  bool a = false, b = false, lb = false, rb = false, x = false, stop = false;
};

inline bool RebootChord::update(const PadSample &p, uint32_t now) {
  bool chord = p.connected && p.x && (p.lt > 154 || p.rt > 154);
  if (!chord) {ready = true;holding = false;return false;}
  if (!ready) return false;
  if (!holding) {holding = true;since = now;}
  if ((uint32_t)(now-since) < 1000) return false;
  ready = false;holding = false;return true;
}

struct RobotControl {
  bool linked = false, armed = false, restingMarch = false, marching = false;
  float vx = 0, wz = 0, vy = 0, dh = 0;

  static float axis(uint16_t raw) {
    float v = ((float)raw - 32768.0f) / (raw >= 32768 ? 32767.0f : 32768.0f);
    return fabsf(v) <= 0.15f ? 0.0f :
        copysignf((fabsf(v) - 0.15f) / 0.85f, v);
  }

  void update(const PadSample &p, uint32_t now) {
    // Liveness comes from the BLE link state, NOT from how recently the pad
    // sent a report. This pad only notifies on CHANGE: holding a steady stick
    // produces no traffic at all (measured >30 s of silence while connected).
    // Any age-based staleness check therefore disarms precisely the operator
    // who is holding a command steady -- which is what made a latched march
    // stop after about a second. Link loss is reported by NimBLE's disconnect
    // callback and is already reflected in p.connected.
    (void)now;
    linked = p.connected;
    vx = wz = vy = dh = 0;
    if (!linked) {
      armed = marching = restingMarch = false;
      return;
    }
    float trigger = fmaxf(p.lt, p.rt) / 1023.0f;
    armed = trigger > (armed ? 0.08f : 0.15f);
    restingMarch = p.a && !p.b; // march only while A is held, never latch motion
    if (armed) {
      vx = -axis(p.forward) * K2_VX_MAX;
      vy = -axis(p.lateral) * K2_VY_MAX;
      wz = -axis(p.turn) * K2_WZ_MAX;
      if (p.b) vx = vy = wz = 0;
    }
    // Match joystick.py: sticks activate the gait, and dead-man release parks
    // even a latched march. Otherwise buildObsFrom gates stick commands to zero.
    marching = armed && !p.b && (restingMarch || vx != 0.0f || vy != 0.0f || wz != 0.0f);
  }
};

// Most recent deliberate input wins. Heartbeats, idle connections and held
// controls do not fight for ownership. Releasing the owner requests HOLD;
// an older command from the other source never resumes by itself.
struct CommandArbiter {
  enum Owner { HOLD, WEB, PAD } owner = HOLD;
  uint32_t webIntent = 0;
  bool padMoving = false, padHold = false;
  float previousVx = 0, previousVy = 0, previousWz = 0;
  float vx = 0, vy = 0, wz = 0;
  bool moving = false;
  void update(const RobotControl &p, bool holdButton, const WebCommand &w, uint32_t now) {
    if (w.intent != webIntent) {webIntent = w.intent;owner = WEB;}
    bool changed = p.marching && (!padMoving || fabsf(p.vx-previousVx) > .002f ||
        fabsf(p.vy-previousVy) > .002f || fabsf(p.wz-previousWz) > .015f);
    if (changed || (holdButton && !padHold)) owner = PAD;
    if (changed || !p.marching) {previousVx=p.vx;previousVy=p.vy;previousWz=p.wz;}
    padMoving=p.marching;padHold=holdButton;
    moving = false;vx=vy=wz=0;
    if (owner == PAD && p.marching) {moving=true;vx=p.vx;vy=p.vy;wz=p.wz;}
    if (owner == WEB && freshWebCommand(w,now)) {moving=true;vx=w.vx;vy=w.vy;wz=w.wz;}
  }
};

inline float boundedJointCommand(int j, float previous, float action, float step) {
  // Clamp BEFORE slew, matching the Pi; don't accumulate a hidden command
  // beyond the joint limit that takes seconds to unwind on direction reversal.
  float target = K2_DEFAULT_POS[j] + K2_ACTION_SCALE * action;
  target = fmaxf(K2_QMIN[j], fminf(K2_QMAX[j], target));
  return previous + fmaxf(-step, fminf(step, target - previous));
}
