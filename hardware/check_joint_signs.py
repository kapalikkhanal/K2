#!/usr/bin/env python3
"""Verify any subset of the 12 calibration signs, both directions.

One tool for every calibration-sign check: same procedure, any joint list.

    # the two groups a squat/crouch never proves:
    python -m hardware.check_joint_signs --joints hip_pitch --bus /dev/ttyAMA0
    # everything:
    python -m hardware.check_joint_signs --joints all --amp 10

HOLD THE ROBOT UP so both legs hang free. Each joint moves to +amp, home, -amp,
home. Compare what you SEE to the "+expected" line; if it does the opposite,
that joint's sign is inverted -> flip it in calibration.json.

WHY hip_pitch matters even though the robot stands fine: the default crouch only
holds it at -0.15 rad (8.6 deg), small enough that an inverted sign still lets
the robot stand. It swings +-5..11 deg during a march, so the error only appears
as fore/aft jerk once stepping starts.

Expectations come from POS_DESC, which is measured off the compiled MJCF. They
are deliberately MIRROR-SYMMETRIC in wording ("outward", "pigeon-toed", "toe
up") -- the MJCF's `right_*` bodies sit at +y, which is the robot's own LEFT
(Fusion names them from the viewer's side), so absolute left/right wording in a
hardware test is unreliable. Do not "fix" these strings into absolute terms.
"""

from __future__ import annotations

import argparse
import signal
import sys

import numpy as np

from . import k2_bus
from . import k2_conventions as C
from .k2_ctrl import _calibration, slew

RATE = 50.0

GROUPS = {
  "hip_pitch": ("hip_pitch_R", "hip_pitch_L"),
  "hip_roll": ("hip_roll_R", "hip_roll_L"),
  "hip_yaw": ("hip_yaw_R", "hip_yaw_L"),
  "knee": ("knee_R", "knee_L"),
  "ankle_pitch": ("ankle_pitch_R", "ankle_pitch_L"),
  "ankle_roll": ("ankle_roll_R", "ankle_roll_L"),
  "lateral": ("hip_roll_R", "hip_roll_L", "hip_yaw_R", "hip_yaw_L"),
  "ankles": ("ankle_pitch_R", "ankle_roll_R", "ankle_pitch_L", "ankle_roll_L"),
  "all": tuple(C.SIM_ORDER),
}


def resolve(tokens):
    out = []
    for tok in tokens:
        if tok in GROUPS:
            out.extend(GROUPS[tok])
        elif tok in C.SIM_ORDER:
            out.append(tok)
        else:
            raise SystemExit(
                f"unknown joint/group {tok!r}. Groups: "
                f"{', '.join(sorted(GROUPS))}. Joints: {', '.join(C.SIM_ORDER)}")
    seen, uniq = set(), []
    for j in out:
        if j not in seen:
            seen.add(j)
            uniq.append(j)
    return uniq


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", default="/dev/ttyAMA0")
    ap.add_argument("--joints", nargs="+", default=["hip_pitch"],
                    help="joint names and/or groups: " + ", ".join(sorted(GROUPS)))
    ap.add_argument("--amp", type=float, default=10.0, help="degrees")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)

    joints = resolve(args.joints)
    if not 0.0 < args.amp <= 20.0:
        raise SystemExit("--amp must be between 0 and 20 degrees")
    amp = np.radians(args.amp)

    # Both directions must fit inside the calibrated limits. The knees are
    # one-sided ([-90, 0]), so they can only be driven negative.
    plan = []
    for j in joints:
        lo, hi = C.LIMITS[j]
        dirs = [d for d in (+1, -1) if lo <= d * amp <= hi]
        if not dirs:
            raise SystemExit(
                f"{j} limits {np.degrees([lo, hi]).round(1)} deg cannot take "
                f"+-{args.amp:g} deg; lower --amp")
        plan.append((j, dirs))

    print("Will test, in order:")
    for j, dirs in plan:
        ds = "/".join(f"{d*args.amp:+g}" for d in dirs)
        print(f"  {j:16s} {ds} deg   +{args.amp:g} => {C.POS_DESC[j]}")
    if not args.yes:
        print("\nHOLD THE ROBOT IN THE AIR, both legs hanging free.")
        if input("Type 'go' to continue: ").strip().lower() != "go":
            return 1

    dt = 1.0 / RATE
    bus = k2_bus.open_bus(args.bus)
    calib = _calibration(bus)

    def limp(*_):
        print("\nreleasing torque")
        try:
            bus.torque(False)
        finally:
            bus.close()
        sys.exit(130)

    signal.signal(signal.SIGINT, limp)

    idx = {j: k for k, j in enumerate(C.SIM_ORDER)}
    home = np.zeros(len(C.SIM_ORDER), np.float32)
    cmd = C.q_vector(bus.read_pos_speed()[0], calib)

    def goto(vec, hold_s=1.5):
        nonlocal cmd
        for _ in range(int(hold_s / dt)):
            cmd = slew(cmd, vec, dt, 0.5)
            bus.write_goals(C.goal_counts(cmd, calib))
            bus.tick(dt)

    try:
        bus.torque(True)
        goto(home, 2.5)
        for j, dirs in plan:
            print(f"\n=== {j} ===", flush=True)
            for d in dirs:
                if d > 0:
                    print(f"  +{args.amp:g} deg expected: {C.POS_DESC[j]}",
                          flush=True)
                else:
                    print(f"  -{args.amp:g} deg: the OPPOSITE of the above",
                          flush=True)
                v = home.copy()
                v[idx[j]] = d * amp
                goto(v)
                goto(home)
    finally:
        bus.torque(False)
        bus.close()
        print("\ntorque released")

    print("If any joint moved opposite to its '+expected' line, flip its sign:")
    print("  edit hardware/calibration.json -> sign[<joint>] *= -1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
