#!/usr/bin/env python3
"""Drive a K2 turning policy through a fixed, scripted command schedule.

Keyboard driving produces ragged phase boundaries that depend on operator
timing, which makes left-vs-right and before-vs-after comparisons noisy. This
runs one continuous session through an exact schedule and writes a single CSV,
so every phase is the same length every time and the segmentation is unambiguous.

It uses `k2_policy_run.run` unchanged -- same observation construction, same
ramps, same safety cutoff, torque always released on exit.

    python -m hardware.run_test_protocol --policy policies/turn_v3_iter3100.onnx

The operator does nothing but spot the robot. Ctrl-C stops and releases torque.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

import onnxruntime as ort

from . import k2_bus
from .k2_policy_run import RATE_HZ, _read_metadata, parse_joint_trim, run
from .k2_stabilizer import UnsafeTiltError


def build_schedule(speed: float, yaw: float, arc_yaw: float,
                   settle: float, phase: float) -> list[tuple[float, str, float, float]]:
  """(seconds, mode, vx, wz). Holds between phases let each start from rest."""
  h = (settle, "hold", 0.0, 0.0)
  return [
    (max(settle, 6.0), "hold", 0.0, 0.0),      # hold quality + IMU settle
    (phase, "march", 0.0, 0.0),                # THE drift measurement
    h,
    (phase, "walk", +speed, 0.0),              # forward
    h,
    (phase, "walk", -speed, 0.0),              # backward
    h,
    (phase, "march", 0.0, +yaw),               # turn left in place
    h,
    (phase, "march", 0.0, -yaw),               # turn right in place
    h,
    (phase * 0.75, "walk", +speed, +arc_yaw),  # forward-left arc
    h,
    (phase * 0.75, "walk", +speed, -arc_yaw),  # forward-right arc
    (settle, "hold", 0.0, 0.0),
  ]


class ScheduledCommand:
  def __init__(self, schedule):
    self.schedule = schedule
    self.total = sum(d for d, *_ in schedule)
    self.tick = 0
    self.stop_requested = False
    self._last = None

  def snapshot(self):
    t = self.tick / RATE_HZ
    self.tick += 1
    elapsed = 0.0
    for i, (dur, mode, vx, wz) in enumerate(self.schedule):
      if t < elapsed + dur:
        if self._last != i:
          self._last = i
          print(f"\n  [{t:6.1f}s] phase {i + 1}/{len(self.schedule)}: "
                f"{mode:5s} vx={vx:+.3f} wz={wz:+.2f}  ({dur:.0f}s)")
        return mode, vx, wz, self.stop_requested
      elapsed += dur
    return "hold", 0.0, 0.0, True


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--policy", required=True)
  ap.add_argument("--bus", default="/dev/ttyAMA0")
  ap.add_argument("--speed", type=float, default=0.018,
                  help="forward/backward magnitude, m/s (default 0.018). NOTE: "
                       "0.025 diverged laterally on hardware 2026-08-21 -- roll "
                       "grew 10->25 deg p2p over 7 s of backward walking at the "
                       "0.75 Hz gait frequency, and forward peaked at 13.98 deg "
                       "tilt against a 14 deg cutoff. Raise this one step at a "
                       "time and watch roll, not pitch.")
  ap.add_argument("--start-phase", type=int, default=1,
                  help="resume the schedule from this 1-based phase, to retest "
                       "after a stop without repeating what already passed")
  ap.add_argument("--yaw-rate", type=float, default=0.10,
                  help="turn-in-place yaw magnitude, rad/s (default 0.10)")
  ap.add_argument("--arc-yaw", type=float, default=0.07,
                  help="yaw magnitude while also translating (default 0.07)")
  ap.add_argument("--phase", type=float, default=20.0,
                  help="seconds per measured phase (default 20)")
  ap.add_argument("--settle", type=float, default=5.0)
  ap.add_argument("--approach", type=float, default=5.0)
  ap.add_argument("--fall-angle", type=float, default=14.0)
  ap.add_argument("--slew", type=float, default=1.0)
  ap.add_argument("--balance-trim-right", type=float, default=0.0)
  ap.add_argument("--joint-trim", default=None)
  ap.add_argument("--log", default="protocol.csv")
  ap.add_argument("--dry-run", action="store_true",
                  help="print the schedule and exit without touching the robot")
  ap.add_argument("--yes", action="store_true")
  args = ap.parse_args(argv)

  if not 0.0 < args.speed <= 0.05:
    raise SystemExit("--speed must be in (0, 0.05] m/s")
  if not 0.0 <= args.yaw_rate <= 0.25:
    raise SystemExit("--yaw-rate must be in [0, 0.25] rad/s")
  if not 0.0 <= args.arc_yaw <= 0.25:
    raise SystemExit("--arc-yaw must be in [0, 0.25] rad/s")

  schedule = build_schedule(args.speed, args.yaw_rate, args.arc_yaw,
                            args.settle, args.phase)
  if not 1 <= args.start_phase <= len(schedule):
    raise SystemExit(f"--start-phase must be in [1, {len(schedule)}]")
  if args.start_phase > 1:
    skipped = args.start_phase - 1
    # Always re-run a hold first so the robot settles before it is asked to move.
    schedule = [(max(args.settle, 4.0), "hold", 0.0, 0.0)] + schedule[skipped:]
    print(f"resuming from phase {args.start_phase} "
          f"({skipped} phases skipped, settle prepended)")
  total = sum(d for d, *_ in schedule)
  print(f"Schedule ({total:.0f} s of policy control after a "
        f"{args.approach:g} s approach):")
  elapsed = 0.0
  for i, (dur, mode, vx, wz) in enumerate(schedule):
    print(f"  {elapsed:6.1f}-{elapsed + dur:6.1f}s  {mode:5s} "
          f"vx={vx:+.3f} wz={wz:+.2f}")
    elapsed += dur
  if args.dry_run:
    return 0

  session = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
  (default_pos, action_scale, gait_freq, checkpoint, _shift, gait_dim,
   heading_dim, transition_time, velocity_ramp,
   yaw_ramp) = _read_metadata(session)
  if gait_dim != 5:
    raise SystemExit(f"{args.policy} is not a turning policy (gait dim {gait_dim})")
  joint_trim = parse_joint_trim(args.joint_trim)
  print(f"\nPolicy: {args.policy}  checkpoint {checkpoint}  "
        f"gait {gait_freq:g} Hz  yaw ramp {yaw_ramp:g} rad/s^2")
  print("The robot drives ITSELF through the schedule. Spot it; do not steer.")
  if not args.yes and input("Type 'go' to enable servo torque: ").strip().lower() != "go":
    print("aborted")
    return 1

  controller = ScheduledCommand(schedule)
  bus = k2_bus.open_bus(args.bus)

  def request_stop(*_):
    controller.stop_requested = True

  signal.signal(signal.SIGINT, request_stop)
  started = time.time()
  try:
    bus.torque(True)
    run(bus, session, default_pos, action_scale, "hold",
        total + 5.0, args.slew, 0.0, gait_freq,
        approach_s=args.approach, fall_angle_deg=args.fall_angle,
        log=args.log, heading_observation_dim=heading_dim,
        balance_trim_right_deg=args.balance_trim_right,
        gait_transition_time_s=transition_time,
        command_source=controller.snapshot,
        gait_velocity_ramp_rate_mps2=velocity_ramp,
        gait_yaw_rate_ramp_rate_rps2=yaw_ramp,
        joint_trim=joint_trim)
  except UnsafeTiltError as exc:
    print(f"\nSAFETY STOP after {time.time() - started:.1f}s: {exc}")
    return 2
  finally:
    try:
      bus.torque(False)
      print("servo torque released")
    finally:
      bus.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
