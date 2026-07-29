#!/usr/bin/env python3
"""Safely test the model-derived Robot_v4 default crouch.

The target is solved fresh from the MJCF relative to calibrated home (q=0):
both soles flat, feet aligned, and the total CoM centered between the feet.
Hardware motion is stopped before the next command if IMU tilt reaches 10 deg.
Torque is always released on exit.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from . import k2_bus
from . import k2_conventions as C
from . import k2_ctrl
from .k2_attitude import ImuAttitude, SimAttitude
from .k2_motion import SquatIK
from .k2_stabilizer import TiltStabilizer, UnsafeTiltError


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bus", default="/dev/ttyAMA0")
    ap.add_argument("--depth", type=float, default=35.0,
                    help="knee bend in degrees (default: 35)")
    ap.add_argument("--base-shift", type=float, default=-0.010,
                    help="base_link fore/aft shift relative to feet, m "
                         "(negative is backward; default: -0.010)")
    ap.add_argument("--hold", type=float, default=3.0)
    ap.add_argument("--approach", type=float, default=5.0)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--log")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)

    q, height = SquatIK().solve(args.depth, args.base_shift)
    print(f"model crouch: knee={args.depth:.1f} deg, base={height*1000:.1f} mm, "
          f"base shift={args.base_shift*1000:+.1f} mm")
    print("target (deg):",
          dict(zip(C.SIM_ORDER, np.degrees(q).round(2))))

    real = args.bus != "sim"
    if real and not args.yes:
        print("About to move the real robot. Keep a hand close.")
        if input("Type 'go' to continue: ").strip().lower() != "go":
            return 1

    bus = k2_bus.open_bus(args.bus, viewer=args.viewer)
    attitude = ImuAttitude() if real else SimAttitude(bus)
    # Zero response vectors make this an IMU safety monitor, not a posture
    # correction: the robot receives the exact model-derived crouch target.
    monitor = TiltStabilizer(
        np.zeros(len(C.SIM_ORDER)), np.zeros(len(C.SIM_ORDER)),
        kp=0.0, kd=0.0, max_correction=0.0,
        fall_angle=math.radians(10.0),
    )
    ticks = max(1, int(args.hold * k2_ctrl.RATE_HZ))
    traj = np.repeat(q[None], ticks, axis=0)
    heights = np.full(ticks, height)

    try:
        bus.torque(True)
        k2_ctrl.run(
            bus, traj, heights, 1.0 / k2_ctrl.RATE_HZ,
            log=args.log, attitude=attitude, stabilizer=monitor,
            approach_s=args.approach,
        )
    except UnsafeTiltError as exc:
        print(f"SAFETY STOP: {exc}")
        return 2
    finally:
        bus.torque(False)
        attitude.close()
        bus.close()
        print("torque released")
    return 0


if __name__ == "__main__":
    sys.exit(main())
