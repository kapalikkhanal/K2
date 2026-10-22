#!/usr/bin/env python3
"""Capture the held straight pose as the 12-servo home calibration.

Torque is disabled before sampling. Direction signs are preserved.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import k2_conventions as C
from .k2_bus import SerialBus


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bus", default="/dev/ttyAMA0")
    ap.add_argument("--samples", type=int, default=25)
    args = ap.parse_args()
    if args.samples < 5:
        raise SystemExit("--samples must be at least 5")

    bus = SerialBus(args.bus)
    readings = []
    try:
        bus.torque(False)
        for _ in range(args.samples):
            pos, _ = bus.read_pos_speed()
            readings.append(
                [pos[C.JOINT_TO_ID[joint]] for joint in C.SIM_ORDER])
            time.sleep(0.02)
    finally:
        bus.torque(False)
        bus.close()

    values = np.asarray(readings, dtype=int)
    home = np.rint(np.median(values, axis=0)).astype(int)
    calib = C.load_calibration() or C.default_calibration()
    old = dict(calib["home_raw"])
    for i, joint in enumerate(C.SIM_ORDER):
        calib["home_raw"][joint] = int(home[i])
    C.save_calibration(calib)

    for i, joint in enumerate(C.SIM_ORDER):
        print(
            f"{joint:16} id={C.JOINT_TO_ID[joint]:2d} "
            f"old={old[joint]:4d} new={home[i]:4d} "
            f"span={values[:, i].min()}..{values[:, i].max()} "
            f"sign={calib['sign'][joint]:+d}")
    print(f"saved {C.CALIB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
