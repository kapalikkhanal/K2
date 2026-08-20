#!/usr/bin/env python3
"""Compare K2's physical-left and physical-right static support response.

The two test targets are exact sagittal mirrors.  Both soles remain planted;
the base shifts about 13 mm toward one side in the validated MuJoCo twin.
No RL policy runs during this test.  Torque is always released on exit.
"""

from __future__ import annotations

import argparse
import csv
import math
import signal
import sys
import time

import numpy as np

from . import k2_bus
from . import k2_conventions as C
from .k2_attitude import ImuAttitude
from .k2_ctrl import _calibration, slew
from .k2_motion import SquatIK
from .k2_stabilizer import UnsafeTiltError


RATE_HZ = 50.0
MIRROR = np.r_[6:12, 0:6]

# Half of a simulation-validated fixed-phase support pose relative to the
# symmetric 35-degree crouch.  At full scale the twin rolls about 7 degrees;
# this 0.5 scale keeps the diagnostic near 3.1 degrees with both feet planted.
LEFT_DELTA_DEG = np.array([
    -1.395, 2.125, 0.080, -4.550, 2.690, -4.860,
     0.350, -0.015, 0.040,  1.195, -1.360, 3.880,
], dtype=np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bus", default="/dev/ttyAMA0")
    ap.add_argument("--hold", type=float, default=5.0)
    ap.add_argument("--approach", type=float, default=5.0)
    ap.add_argument("--transition", type=float, default=2.0)
    ap.add_argument("--center", type=float, default=3.0)
    ap.add_argument("--fall-angle", type=float, default=8.0)
    ap.add_argument("--log", default="lateral_balance_audit.csv")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)
    if not 0.0 < args.hold <= 10.0:
        raise SystemExit("--hold must be in (0, 10] seconds")
    if not 5.0 <= args.fall_angle <= 10.0:
        raise SystemExit("--fall-angle must be between 5 and 10 degrees")

    center, _ = SquatIK().solve(35.0, -0.010)
    left = center + np.radians(LEFT_DELTA_DEG)
    right = left[MIRROR]
    for name, target in (("left", left), ("right", right)):
        for joint, value in zip(C.SIM_ORDER, target, strict=True):
            lo, hi = C.LIMITS[joint]
            if not lo <= value <= hi:
                raise SystemExit(f"{name} target exceeds {joint} limit")

    print("Static mirrored test: LEFT 5 s -> CENTER -> RIGHT 5 s -> CENTER")
    print("Both feet must remain planted. Keep both hands ready to catch it.")
    if not args.yes and input("Type 'go' to continue: ").strip().lower() != "go":
        return 1

    bus = k2_bus.open_bus(args.bus)
    calib = _calibration(bus)
    imu = ImuAttitude()
    dt = 1.0 / RATE_HZ
    fall_angle = math.radians(args.fall_angle)
    fields = ["t", "stage", "roll_deg", "pitch_deg", "tilt_deg"]
    fields += [f"cmd_{j}" for j in C.SIM_ORDER]
    fields += [f"q_{j}" for j in C.SIM_ORDER]
    fields += [f"load_{j}" for j in C.SIM_ORDER]
    rows = []
    stopped = False

    def stop(*_args):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    q = C.q_vector(bus.read_pos_speed()[0], calib).astype(np.float32)
    t0 = time.monotonic()

    def execute(stage: str, target: np.ndarray, duration: float) -> None:
        nonlocal q
        ticks = max(1, int(round(duration * RATE_HZ)))
        start = q.copy()
        for k in range(ticks):
            if stopped:
                raise KeyboardInterrupt
            frac = min(1.0, (k + 1) / ticks)
            eased = 0.5 - 0.5 * math.cos(math.pi * frac)
            desired = start + (target - start) * eased
            q = slew(q, desired, dt, 0.5)
            bus.write_goals(C.goal_counts(q, calib))
            bus.tick(dt)
            pos, _ = bus.read_pos_speed()
            measured = C.q_vector(pos, calib).astype(np.float32)
            gyro, gravity = imu.attitude(dt)
            del gyro
            gravity = gravity / max(float(np.linalg.norm(gravity)), 1e-9)
            roll = math.atan2(float(gravity[1]), float(-gravity[2]))
            pitch = math.atan2(float(gravity[0]), float(-gravity[2]))
            tilt = math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
            if tilt > fall_angle:
                raise UnsafeTiltError(
                    f"tilt {math.degrees(tilt):.1f} exceeds {args.fall_angle:.1f} deg"
                )
            telemetry = bus.latest_telemetry() or {}
            loads = [float(telemetry.get(C.JOINT_TO_ID[j], {}).get(
                "load_signed", math.nan)) for j in C.SIM_ORDER]
            rows.append([
                time.monotonic() - t0, stage, math.degrees(roll),
                math.degrees(pitch), math.degrees(tilt), *q, *measured, *loads,
            ])
            if k % 25 == 0:
                print(f"{stage:>12s}  roll={math.degrees(roll):+5.2f} "
                      f"pitch={math.degrees(pitch):+5.2f} "
                      f"tilt={math.degrees(tilt):4.2f}", flush=True)

    try:
        bus.torque(True)
        execute("approach", center, args.approach)
        execute("load_left", left, args.transition)
        execute("hold_left", left, args.hold)
        execute("center_1", center, args.center)
        execute("load_right", right, args.transition)
        execute("hold_right", right, args.hold)
        execute("center_2", center, args.center)
    except UnsafeTiltError as exc:
        print(f"SAFETY STOP: {exc}")
        return 2
    finally:
        try:
            bus.torque(False)
        finally:
            imu.close()
            bus.close()
            with open(args.log, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            print(f"torque released; wrote {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
