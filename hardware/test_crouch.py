#!/usr/bin/env python3
"""Crouch + gentle lateral sway test for K2 (sim or real).

Eases into the walk's crouch, then shifts the CoM a small amount toward the
RIGHT foot, back to centre, toward the LEFT foot, back to centre. BOTH feet stay
planted the whole time (no lift), so the robot can only lean -- it cannot step
off balance. Use it to confirm the crouch is symmetric and the lateral
hip_roll + ankle_roll go the right way before trying the IK walk.

    python -m hardware.test_crouch --bus sim --viewer          # preview in sim
    python -m hardware.test_crouch --bus /dev/ttyACM0          # real robot

The same slew-limited, torque-safe loop as every other K2 motion (k2_ctrl.run):
it eases from wherever the robot is into the crouch, and always releases torque
on exit, including Ctrl-C.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import k2_bus
from . import k2_conventions as C
from . import k2_ctrl
from .k2_walk_ik import WalkIK, DT


def build(shift, stance, cycles, hold_s, ramp_s=2.5):
    ik = WalkIK()
    yR = np.sign(ik.y_nom["R"]) * stance
    yL = np.sign(ik.y_nom["L"]) * stance

    def pose(com_y):
        q, _, _, ok = ik.solve(
            {"com": (0.0, com_y), "R": (0.0, yR, 0.0), "L": (0.0, yL, 0.0)}, None)
        if not ok:
            raise SystemExit(f"crouch IK failed at com_y={com_y*1000:.0f} mm")
        return q

    centre, right, left = pose(0.0), pose(+shift), pose(-shift)

    def ramp(a, b, secs=ramp_s):
        n = int(secs / DT)
        f = 0.5 * (1 - np.cos(np.pi * np.arange(n) / n))
        return a + (b - a) * f[:, None]

    def hold(q, secs):
        return np.repeat(q[None], int(secs / DT), 0)

    cycle = [ramp(centre, right), hold(right, hold_s), ramp(right, centre),
             ramp(centre, left), hold(left, hold_s), ramp(left, centre)]
    traj = np.vstack(cycle * int(cycles) + [hold(centre, 1.0)]).astype(np.float32)
    return traj, np.zeros(len(traj))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", default="/dev/ttyACM0",
                    help="'sim' for MuJoCo, or a device path (default /dev/ttyACM0)")
    ap.add_argument("--shift", type=float, default=0.020,
                    help="lateral CoM shift each way, m (default 0.020 = 20 mm)")
    ap.add_argument("--stance", type=float, default=0.048,
                    help="lateral foot placement, m (default 0.048)")
    ap.add_argument("--cycles", type=int, default=1, help="right/left sway cycles")
    ap.add_argument("--hold", type=float, default=2.0, help="dwell at each lean, s")
    ap.add_argument("--viewer", action="store_true", help="sim only")
    ap.add_argument("--yes", action="store_true", help="skip the hardware prompt")
    args = ap.parse_args(argv)

    traj, heights = build(args.shift, args.stance, args.cycles, args.hold)
    print(f"crouch + sway: +-{args.shift*1000:.0f} mm CoM, stance {args.stance*1000:.0f} mm, "
          f"{args.cycles} cycle(s), {len(traj)} ticks (~{len(traj)*DT:.0f}s of motion)")
    print("WATCH: symmetric crouch (both feet planted), then lean toward the RIGHT")
    print("       foot, back to centre, lean LEFT, back to centre. Feet stay flat.")

    real = args.bus != "sim"
    if real and not args.yes:
        print(f"\nAbout to MOVE THE REAL ROBOT on {args.bus}. Stand it on both feet "
              f"with room, keep a hand near it.")
        if input("Type 'go' to continue: ").strip().lower() != "go":
            print("aborted")
            return 1

    bus = k2_bus.open_bus(args.bus, viewer=args.viewer)
    try:
        bus.torque(True)
        k2_ctrl.run(bus, traj, heights, DT)
    finally:
        bus.torque(False)      # always go limp at the end / on Ctrl-C
        bus.close()
    print("done -- torque released.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
