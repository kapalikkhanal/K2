#!/usr/bin/env python3
"""Validate K2 gyro orientation against measured gravity-vector motion.

Keep the robot level while the gyro bias is captured.  After ``READY`` is
printed, smoothly tilt the unpowered torso and return it toward level.  The
test checks the frame-invariant rigid-body relation

    gravity_dot_body = -omega_body x gravity_body

after converting the policy's MJCF-site gyro back into the root frame.
Nothing in this command opens the servo bus or changes torque.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .k2_attitude import ImuAttitude, R_SITE_TO_ROOT


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--start-angle", type=float, default=3.0)
    ap.add_argument("--wait", type=float, default=30.0)
    args = ap.parse_args(argv)
    dt = 1.0 / args.rate

    imu = ImuAttitude(bias_samples=300)
    try:
        print("READY: smoothly tilt the torso now", flush=True)
        deadline = time.monotonic() + args.wait
        while True:
            _, gravity = imu.attitude(dt)
            tilt = np.degrees(np.arccos(np.clip(-gravity[2], -1.0, 1.0)))
            if tilt >= args.start_angle:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("no tilt detected before timeout")
            time.sleep(dt)

        gyros, gravities, times = [], [], []
        end = time.monotonic() + args.duration
        while time.monotonic() < end:
            start = time.monotonic()
            gyro_site, gravity = imu.attitude(dt)
            gyros.append(gyro_site)
            gravities.append(gravity)
            times.append(start)
            time.sleep(max(0.0, dt - (time.monotonic() - start)))
    finally:
        imu.close()

    gyro_site = np.asarray(gyros)
    gravity = np.asarray(gravities)
    times = np.asarray(times)
    gravity_dot = np.gradient(gravity, times, axis=0)
    omega_root = gyro_site @ R_SITE_TO_ROOT.T
    predicted = -np.cross(omega_root, gravity)

    # Ignore filter/finite-difference edge samples and near-stationary samples.
    valid = np.linalg.norm(predicted, axis=1) > 0.02
    valid[:3] = False
    valid[-3:] = False
    if np.count_nonzero(valid) < 20:
        raise RuntimeError("not enough smooth motion captured")
    x = predicted[valid].reshape(-1)
    y = gravity_dot[valid].reshape(-1)
    correlation = float(np.corrcoef(x, y)[0, 1])
    gain = float(np.dot(x, y) / np.dot(x, x))
    nrmse = float(np.sqrt(np.mean((x - y) ** 2)) / np.sqrt(np.mean(y ** 2)))
    max_tilt = float(np.degrees(np.max(np.arccos(np.clip(-gravity[:, 2], -1, 1)))))
    print(f"samples={len(times)} moving_samples={np.count_nonzero(valid)}")
    print(f"max_tilt={max_tilt:.2f} deg")
    print(f"gravity_dot correlation={correlation:.3f} gain={gain:.3f} nrmse={nrmse:.3f}")
    if correlation < 0.8 or gain < 0.5:
        print("FAIL: gyro frame/sign does not agree with gravity motion")
        return 2
    print("PASS: gyro frame/sign agrees with gravity motion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
