#!/usr/bin/env python3
"""Measure the residual yaw-rate bias left in the IMU after ImuAttitude debiasing.

Why this exists: on the first turning hardware run the yaw-tracking error was the
SAME SIGN in both directions -- left +0.013 rad/s, right +0.006 rad/s -- which is
an additive offset of about +0.0094 rad/s (+0.54 deg/s), not a left/right gain
asymmetry. Added to a +/-0.06 rad/s command that makes left turns ~22% fast and
right turns ~10% slow, i.e. left appears ~1.35x faster. That is the whole
reported "left turns fast, right turns slow" symptom in one number.

An offset that size is consistent with residual gyro-z bias: ImuAttitude removes
the bias from 200 samples at construction, and the raw LSM6DS3 z bias measured
during bring-up was about -1.0 dps (0.017 rad/s), so a half-corrected residual
lands exactly here. The policy steers by integrated gyro yaw, so a residual bias
walks its heading reference off course and the policy yaws to chase it.

This isolates the sensor from the robot: by default NO bus, NO torque, nothing
moves. Stand or lay the robot still and run it.

    python -m hardware.test_imu_yaw_bias --duration 60

Pass --bus /dev/ttyAMA0 to hold the default crouch under torque first, which
measures the bias in the orientation the robot actually walks in (gyro bias is
mildly orientation dependent). Torque is always released on exit.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

from . import k2_conventions as C

# Offset inferred from the turning walk log, for direct comparison.
WALK_LOG_OFFSET_RPS = 0.0094
CROUCH_PATH = Path(__file__).with_name("crouch_pose.json")


def _crouch() -> np.ndarray:
  """The validated default crouch, in SIM_ORDER -- the policy's default pose."""
  with open(CROUCH_PATH) as f:
    q = np.asarray(json.load(f)["q"], dtype=np.float32)
  if q.shape != (len(C.SIM_ORDER),):
    raise SystemExit(f"{CROUCH_PATH} does not hold {len(C.SIM_ORDER)} joints")
  return q


def _summarize(name: str, gyro: np.ndarray, dt: float) -> float:
  """Print per-axis statistics; return the yaw-axis mean in rad/s."""
  mean = gyro.mean(axis=0)
  sd = gyro.std(axis=0)
  print(f"  {name}")
  for i, axis in enumerate("xyz"):
    print(f"    {axis}: mean {mean[i]:+.5f} rad/s ({math.degrees(mean[i]):+.3f} "
          f"deg/s)   sd {sd[i]:.5f}")
  return float(mean[2])


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--duration", type=float, default=60.0,
                  help="seconds to measure (default 60)")
  ap.add_argument("--rate", type=float, default=50.0,
                  help="sample rate in Hz, matching the policy loop")
  ap.add_argument("--bus", default=None,
                  help="optional serial device: hold the default crouch under "
                       "torque so the bias is measured standing")
  ap.add_argument("--approach", type=float, default=4.0)
  ap.add_argument("--log", help="write the raw per-sample CSV here")
  args = ap.parse_args(argv)

  dt = 1.0 / args.rate
  bus = None

  def release(*_):
    if bus is not None:
      try:
        bus.torque(False)
      finally:
        bus.close()
    sys.exit(130)

  signal.signal(signal.SIGINT, release)

  try:
    if args.bus:
      from . import k2_bus
      calib = C.load_calibration()
      if calib is None:
        raise SystemExit(f"no calibration at {C.CALIB_PATH}")
      bus = k2_bus.open_bus(args.bus)
      bus.torque(True)
      crouch = _crouch()
      pos, _ = bus.read_pos_speed()
      q_now = C.q_vector(pos, calib)
      steps = max(int(args.approach / dt), 1)
      print(f"easing into the default crouch over {args.approach:g} s")
      for i in range(steps):
        a = 0.5 * (1 - math.cos(math.pi * i / steps))
        bus.write_goals(C.goal_counts(q_now + (crouch - q_now) * a, calib))
        time.sleep(dt)
      print("standing; hold still and do not touch the robot")
      time.sleep(1.0)

    # Construct AFTER any motion so the debias sees a stationary robot -- this
    # is the same 200-sample capture the policy runner performs.
    from .k2_attitude import ImuAttitude
    print("capturing gyro bias (robot must be still) ...")
    att = ImuAttitude()
    print(f"  captured raw bias: "
          f"{np.array2string(att.gyro_bias, precision=5, floatmode='fixed')} rad/s"
          f"  (z = {math.degrees(att.gyro_bias[2]):+.3f} deg/s)")

    n = int(args.duration * args.rate)
    samples = np.zeros((n, 3), np.float32)
    headings = np.zeros(n, np.float32)
    stamps = np.zeros(n, np.float32)
    print(f"measuring {args.duration:g} s at {args.rate:g} Hz ...")
    t0 = time.perf_counter()
    for k in range(n):
      gyro, _ = att.attitude(dt)
      samples[k] = gyro
      headings[k] = att.relative_heading()
      stamps[k] = time.perf_counter() - t0
      remaining = t0 + (k + 1) * dt - time.perf_counter()
      if remaining > 0:
        time.sleep(remaining)
    elapsed = stamps[-1] - stamps[0]

    # `gyro` is the policy observation, in the MJCF site frame. Yaw lives on
    # root z, and R_SITE_TO_ROOT flips it, so root_z = -site_z.
    root = samples @ np.array([[0.0, -1.0, 0.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, 0.0, -1.0]]).T

    print("\nRESIDUAL BIAS AFTER DEBIASING (should be ~0 if the capture worked)")
    _summarize("policy observation frame (MJCF site)", samples, dt)
    yaw_bias = _summarize("root frame (yaw = z)", root, dt)

    drift = float(headings[-1] - headings[0])
    drift_rate = drift / max(elapsed, 1e-6)
    print(f"\nINTEGRATED HEADING (what the policy steers by)")
    print(f"  drift over {elapsed:.1f} s: {math.degrees(drift):+.2f} deg")
    print(f"  drift rate:                {drift_rate:+.5f} rad/s "
          f"({math.degrees(drift_rate):+.3f} deg/s)")

    # Stability across thirds: a steady value is bias, a growing one is thermal.
    third = max(n // 3, 1)
    print("  per-third drift rate (deg/s): ", end="")
    for i in range(3):
      seg = slice(i * third, min((i + 1) * third, n))
      d = float(headings[seg][-1] - headings[seg][0])
      s = float(stamps[seg][-1] - stamps[seg][0])
      print(f"{math.degrees(d / max(s, 1e-6)):+.3f}  ", end="")
    print()

    print(f"\nCOMPARISON WITH THE WALK LOG")
    print(f"  additive yaw offset inferred while turning: "
          f"{WALK_LOG_OFFSET_RPS:+.4f} rad/s "
          f"({math.degrees(WALK_LOG_OFFSET_RPS):+.2f} deg/s)")
    print(f"  static residual measured here:              "
          f"{drift_rate:+.4f} rad/s ({math.degrees(drift_rate):+.2f} deg/s)")
    ratio = drift_rate / WALK_LOG_OFFSET_RPS if WALK_LOG_OFFSET_RPS else float("nan")
    if abs(drift_rate) < 0.2 * abs(WALK_LOG_OFFSET_RPS):
      verdict = ("SENSOR IS CLEAN -- the turning offset is NOT residual gyro "
                 "bias. Look for a physical yaw tendency instead.")
    elif 0.5 <= ratio <= 2.0:
      verdict = ("MATCHES -- residual gyro-z bias explains the turning offset. "
                 "Fix it in ImuAttitude (longer or online bias estimation).")
    else:
      verdict = ("PARTIAL -- the sensor carries real bias but not the whole "
                 "offset; both a sensor and a physical term are present.")
    print(f"  ratio measured/inferred: {ratio:+.2f}\n  VERDICT: {verdict}")

    if args.log:
      rows = np.column_stack([stamps, samples, root, headings])
      np.savetxt(args.log, rows, delimiter=",", comments="",
                 header="t,site_x,site_y,site_z,root_x,root_y,root_z,heading")
      print(f"\nwrote {args.log}")
    att.close()
  finally:
    if bus is not None:
      try:
        bus.torque(False)
        print("servo torque released")
      finally:
        bus.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
