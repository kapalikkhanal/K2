#!/usr/bin/env python3
"""Ease every K2 joint to its calibrated home (q = 0, the straight pose).

Home is the calibrated straight pose, i.e. each servo at its home_raw count --
not the IK stand pose, which centres the CoM and so is slightly bent. Motion is
eased in over a few seconds and slew-limited, and torque is left ON at the end
so the robot holds home. Ctrl-C releases torque.

    python -m hardware.go_home --bus /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

import numpy as np

from . import k2_conventions as C
from . import k2_bus
from .k2_ctrl import _calibration, slew

RATE_HZ = 50.0
EASE_S = 3.0
SLEW_RAD_S = 1.0   # gentle


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", default="/dev/ttyACM0")
    ap.add_argument("--ease", type=float, default=EASE_S)
    ap.add_argument("--hold", type=float, default=1e9,
                    help="seconds to keep holding home after arriving")
    args = ap.parse_args(argv)

    dt = 1.0 / RATE_HZ
    bus = k2_bus.open_bus(args.bus)
    calib = _calibration(bus)

    def limp(*_):
        print("\ninterrupted -> releasing torque")
        try:
            bus.torque(False)
        finally:
            bus.close()
        sys.exit(130)

    signal.signal(signal.SIGINT, limp)

    home = np.zeros(len(C.SIM_ORDER), np.float32)   # q = 0 for every joint
    pos, _ = bus.read_pos_speed()
    q0 = C.q_vector(pos, calib)
    print("starting from (deg):",
          dict(zip(C.SIM_ORDER, np.degrees(q0).round(1))))
    print(f"easing to home over {args.ease:g}s ...")

    try:
        bus.torque(True)
        cmd = q0.copy()
        n = int(args.ease / dt)
        for i in range(n):
            frac = 0.5 * (1 - np.cos(np.pi * i / n))     # cosine ease 0 -> 1
            target = q0 + (home - q0) * frac
            cmd = slew(cmd, target, dt, SLEW_RAD_S)
            bus.write_goals(C.goal_counts(cmd, calib))
            bus.tick(dt)

        # Settle at home and report.
        for _ in range(int(0.5 / dt)):
            bus.write_goals(C.goal_counts(home, calib))
            bus.tick(dt)
        pos, _ = bus.read_pos_speed()
        qf = C.q_vector(pos, calib)
        print("arrived at home. residual (deg):",
              dict(zip(C.SIM_ORDER, np.degrees(qf).round(2))))
        print(f"max residual: {np.degrees(np.abs(qf)).max():.2f} deg")
        print("holding home (torque ON). Ctrl-C to release.")

        held = 0.0
        while held < args.hold:
            bus.write_goals(C.goal_counts(home, calib))
            bus.tick(dt)
            held += dt
    finally:
        pass  # leave torque on; Ctrl-C (limp) or process end releases the port


if __name__ == "__main__":
    sys.exit(main())
