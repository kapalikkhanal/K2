#pragma once
#include <stdint.h>

// Time since each joint's last valid sample, not time since the last complete
// group reply. A recovered joint must not mask another joint's outage.
//
// Two distinct states, because a lost UART reply is not evidence of a stopped
// servo and the two failure scales need different responses:
//
//   stale  - this tick's group read was incomplete. Hold the last goals, do not
//            run the policy on stale joint state, and carry on. Self-clearing.
//   paused - the outage persisted (LATCH_TICKS in a row, or LATCH_MS on any one
//            joint). Latches: the operator must release the controls, because at
//            this scale the robot's actual state is genuinely unknown.
//
// Measured on this robot: servo id 7 misses ~8% of group reads, almost always
// for a single tick. Latching on the first miss made walking impossible -- the
// robot moved for one tick per command and then demanded a control release.
// Set LATCH_TICKS to 1 to restore the original latch-on-first-miss behaviour.
struct FeedbackGuard {
  static constexpr uint16_t ALL = 0x0fff;
  static constexpr uint8_t  LATCH_TICKS = 5;    // 100 ms at 50 Hz
  static constexpr uint32_t LATCH_MS = 150;     // any single joint this stale
  uint32_t lastGood[12] = {};
  bool paused = false;   // latched; cleared only by resume()
  bool stale = false;    // this tick only
  uint8_t misses = 0;
  void reset(uint32_t now) {
    for (auto &t : lastGood) t = now;
    paused = false; stale = false; misses = 0;
  }
  uint32_t worstAge(uint32_t now) const {
    uint32_t worst = 0;
    for (int j=0;j<12;j++) {
      uint32_t age = now - lastGood[j];
      if (age > worst) worst = age;
    }
    return worst;
  }
  void update(uint16_t valid, uint32_t now) {
    for (int j=0;j<12;j++) if(valid & (1u<<j)) lastGood[j]=now;
    stale = valid != ALL;
    if (!stale) { misses = 0; return; }
    if (misses < 255) misses++;
    if (misses >= LATCH_TICKS || worstAge(now) > LATCH_MS) paused = true;
  }
  uint16_t expired(uint32_t now) const {
    uint16_t mask=0;
    for(int j=0;j<12;j++) if(uint32_t(now-lastGood[j])>2000) mask|=1u<<j;
    return mask;
  }
  bool resume(uint16_t valid, bool neutral) {
    if(valid==ALL && neutral) {paused=false;misses=0;}
    return !paused;
  }
  // True whenever the policy must not run this tick, latched or not.
  bool blocking() const { return paused || stale; }
};
