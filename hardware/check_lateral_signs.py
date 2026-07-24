#!/usr/bin/env python3
"""Verify the hip_roll and hip_yaw calibration signs -- the four joints the
squat never moved (it held them at 0), so they were never checked in motion.
The walk policy is the first thing to drive them; a wrong sign makes the gait
go sideways instead of forward.

    python -m hardware.check_lateral_signs --bus /dev/ttyACM0

HOLD THE ROBOT UP so both legs hang free. Each joint moves to +0.3 rad, pauses,
then -0.3, then home. Compare what you SEE to the "+expected" line. If it does
the opposite, that joint's sign is inverted -> flip it in calibration.json.
"""

from __future__ import annotations

import argparse
import signal
import sys

import numpy as np

from . import k2_conventions as C
from . import k2_bus
from .k2_ctrl import _calibration, slew

RATE = 50.0
LATERAL = ("hip_roll_R", "hip_roll_L", "hip_yaw_R", "hip_yaw_L")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", default="/dev/ttyACM0")
    ap.add_argument("--amp", type=float, default=0.3)
    args = ap.parse_args(argv)
    dt = 1.0 / RATE

    bus = k2_bus.open_bus(args.bus)
    calib = _calibration(bus)

    def limp(*_):
        print("\nreleasing torque"); bus.torque(False); bus.close(); sys.exit(130)
    signal.signal(signal.SIGINT, limp)

    idx = {j: k for k, j in enumerate(C.SIM_ORDER)}
    bus.torque(True)
    home = np.zeros(len(C.SIM_ORDER), np.float32)
    cmd = C.q_vector(bus.read_pos_speed()[0], calib)

    def goto(vec, hold_s=1.2):
        nonlocal cmd
        for _ in range(int(hold_s / dt)):
            cmd = slew(cmd, vec, dt, 1.0)
            bus.write_goals(C.goal_counts(cmd, calib)); bus.tick(dt)

    goto(home)
    for j in LATERAL:
        print(f"\n=== {j} ===")
        print(f"  +0.3 rad expected: {C.POS_DESC[j]}")
        v = home.copy(); v[idx[j]] = args.amp
        goto(v); goto(home)
        v[idx[j]] = -args.amp
        print("  now -0.3 rad: the OPPOSITE of the above")
        goto(v); goto(home)
    bus.torque(False); bus.close()
    print("\nIf any joint moved opposite to its '+expected' line, flip its sign:")
    print("  edit hardware/calibration.json -> sign[<joint>] *= -1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
