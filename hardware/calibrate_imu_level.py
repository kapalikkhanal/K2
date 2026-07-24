#!/usr/bin/env python3
"""Calibrate the fixed IMU mounting tilt while the K2 base is level.

Place a spirit level on the base, keep the robot completely still, then run:

    python -m hardware.calibrate_imu_level
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from .k2_attitude import LEVEL_CALIB_PATH, R_IMU_TO_ROOT


def rotation_to_down(g: np.ndarray) -> np.ndarray:
    """Smallest rotation mapping measured unit gravity to [0, 0, -1]."""
    a = np.asarray(g, dtype=float) / np.linalg.norm(g)
    b = np.array([0.0, 0.0, -1.0])
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s2 = float(np.dot(v, v))
    if s2 < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0.0, -v[2], v[1]],
                  [v[2], 0.0, -v[0]],
                  [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k * ((1.0 - c) / s2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--bus-number", type=int, default=1)
    args = ap.parse_args(argv)
    if args.samples < 50:
        raise SystemExit("use at least 50 samples")

    sys.path.insert(0, "/home/pi/IMU")
    from visualize_imu import LSM6DS3

    imu = LSM6DS3(args.bus_number)
    readings = []
    try:
        for _ in range(args.samples):
            accel, _ = imu.read()
            readings.append(-np.asarray(accel, dtype=float))
            time.sleep(0.01)
    finally:
        imu.close()

    g_chip = np.mean(readings, axis=0)
    g_chip /= np.linalg.norm(g_chip)
    g_root = R_IMU_TO_ROOT @ g_chip
    correction = rotation_to_down(g_root)
    corrected = correction @ g_root
    payload = {
        "root_correction": correction.tolist(),
        "measured_g_chip": g_chip.tolist(),
        "measured_g_root_before": g_root.tolist(),
    }
    with open(LEVEL_CALIB_PATH, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    def angles(g):
        return (np.degrees(np.arctan2(g[0], -g[2])),
                np.degrees(np.arctan2(g[1], -g[2])))

    p0, r0 = angles(g_root)
    p1, r1 = angles(corrected)
    print(f"before: pitch={p0:+.2f} deg roll={r0:+.2f} deg")
    print(f"after:  pitch={p1:+.2f} deg roll={r1:+.2f} deg")
    print(f"wrote {LEVEL_CALIB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
