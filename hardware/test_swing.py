#!/usr/bin/env python3
"""Single-step / one-foot-balance test for K2 (sim or real).

Flow:
  1. ease into the DEFAULT CROUCH (hardware/crouch_pose.json, captured by hand)
  2. wait for you to type 'go'
  3. SHIFT the CoM over the stance foot, LIFT the swing foot, hold, come back
  4. return to the DEFAULT CROUCH and hold it

If it tips toward the LIFTING foot in step 3, the CoM did not get far enough over
the stance foot. Stand it on the floor, keep a hand near it. Ctrl-C = limp.

    python -m hardware.test_swing --bus /dev/ttyACM0            # right stance (lift left)
    python -m hardware.test_swing --bus /dev/ttyACM0 --stance L # left stance (lift right)
    python -m hardware.test_swing --bus sim --viewer            # preview in sim
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

from . import k2_bus
from . import k2_conventions as C
from .k2_walk_ik import WalkIK, DT

CROUCH_PATH = Path(__file__).with_name("crouch_pose.json")


def load_default_crouch(model_centre):
    """The hand-captured crouch, or the model crouch if none has been saved."""
    if CROUCH_PATH.exists():
        q = np.array(json.load(open(CROUCH_PATH))["q"], np.float32)
        if len(q) == len(C.SIM_ORDER):
            return q
    return model_centre


def build_cycle(stance, stance_y, shift, lift, hold_s, ramp_s=2.5, base_roll=0.0):
    """CoM-shift cycle, starting and ending at the model crouch centre."""
    ik = WalkIK()
    ik.base_roll = base_roll
    yR, yL = abs(stance_y), -abs(stance_y)
    swing = "L" if stance == "R" else "R"
    sy = +abs(shift) if stance == "R" else -abs(shift)

    def pose(com_y, sw_lift):
        tgt = {"com": (0.0, com_y), "R": (0.0, yR, 0.0), "L": (0.0, yL, 0.0)}
        tgt[swing] = (0.0, tgt[swing][1], sw_lift)
        q, _, _, ok = ik.solve(tgt, None)
        if not ok:
            raise SystemExit(f"IK failed (com_y={com_y*1000:.0f} lift={sw_lift*1000:.0f})")
        return q

    centre, shifted, lifted = pose(0.0, 0.0), pose(sy, 0.0), pose(sy, lift)

    def ramp(a, b, secs=ramp_s):
        n = int(secs / DT); f = 0.5 * (1 - np.cos(np.pi * np.arange(n) / n))
        return a + (b - a) * f[:, None]

    def hold(q, secs):
        return np.repeat(q[None], int(secs / DT), 0)

    cycle = np.vstack([
        ramp(centre, shifted), hold(shifted, hold_s),      # shift weight
        ramp(shifted, lifted), hold(lifted, hold_s + 1),   # lift, balance
        ramp(lifted, shifted), hold(shifted, 0.5),         # set down
        ramp(shifted, centre),
    ]).astype(np.float32)
    return centre, cycle, swing


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", default="/dev/ttyACM0")
    ap.add_argument("--stance", choices=("R", "L"), default="R",
                    help="which foot STAYS planted (the other lifts)")
    ap.add_argument("--stance-y", type=float, default=0.055, help="foot spacing, m")
    ap.add_argument("--shift", type=float, default=0.039,
                    help="lateral CoM shift toward the stance foot, m")
    ap.add_argument("--lift", type=float, default=0.020, help="swing-foot lift, m")
    ap.add_argument("--hold", type=float, default=3.0, help="dwell at each phase, s")
    ap.add_argument("--base-roll", type=float, default=5.0,
                    help="fixed base trim in degrees (+ leans toward the right foot)")
    ap.add_argument("--viewer", action="store_true", help="sim only")
    ap.add_argument("--yes", action="store_true", help="skip the go prompt")
    args = ap.parse_args(argv)

    centre, cycle, swing = build_cycle(args.stance, args.stance_y, args.shift,
                                       args.lift, args.hold,
                                       base_roll=np.radians(args.base_roll))
    crouch = load_default_crouch(centre)
    calib = (C.default_calibration() if args.bus == "sim" else C.load_calibration())
    bus = k2_bus.open_bus(args.bus, viewer=args.viewer)

    def limp(*_):
        print("\ninterrupted -> releasing torque")
        try:
            bus.torque(False)
        finally:
            bus.close()
        sys.exit(130)
    signal.signal(signal.SIGINT, limp)

    def goto(cmd, target, secs):
        n = int(secs / DT)
        for i in range(n):
            f = 0.5 * (1 - np.cos(np.pi * (i + 1) / n))
            bus.write_goals(C.goal_counts(cmd + (target - cmd) * f, calib))
            bus.tick(DT)
        return target.copy()

    def play(traj):
        for q in traj:
            bus.write_goals(C.goal_counts(q, calib))
            bus.tick(DT)

    print(f"single step: stance={args.stance} (lift {swing}), shift {args.shift*1000:.0f} mm, "
          f"lift {args.lift*1000:.0f} mm")
    try:
        bus.torque(True)
        # 1. ease into the default crouch
        cmd = C.q_vector(bus.read_pos_speed()[0], calib)
        print("easing to DEFAULT CROUCH ...")
        cmd = goto(cmd, crouch, 3.0)
        # 2. wait for go (hardware only)
        if args.bus != "sim" and not args.yes:
            print("robot is in the default crouch on both feet. Keep a hand near it.")
            if input("Type 'go' to run the shift/lift cycle: ").strip().lower() != "go":
                print("aborted (staying in crouch)")
                return 1
        # 3. crouch -> cycle -> crouch
        cmd = goto(cmd, centre, 2.0)
        print(f"  SHIFT onto {args.stance} foot, LIFT {swing} foot ...")
        play(cycle); cmd = centre.copy()
        print("returning to DEFAULT CROUCH ...")
        cmd = goto(cmd, crouch, 2.0)
        # 4. hold the crouch (torque stays on so it does not collapse)
        for _ in range(int(1.0 / DT)):
            bus.write_goals(C.goal_counts(crouch, calib)); bus.tick(DT)
        print("done -- holding the default crouch (torque ON). Ctrl-C or go_home to relax.")
    except SystemExit:
        raise
    finally:
        if args.bus == "sim":
            bus.torque(False)   # sim goes limp; real keeps holding the crouch
        bus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
