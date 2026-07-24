#!/usr/bin/env python3
"""K2 controller. Same code drives MuJoCo or the real servo chain.

    python -m hardware.k2_ctrl --bus sim --motion squat --viewer
    python -m hardware.k2_ctrl --bus /dev/ttyACM0 --motion squat

The only thing --bus changes is which object implements the count-level servo
interface. Observation building, calibration, clamping, slew limiting and the
control rate are identical on both paths, so a motion that behaves in sim is
running byte-identical logic on hardware.

Safety on the real robot: goals are clamped to the joint limits and to counts
[40, 4055], slew-limited, and the run always ends with torque released.
Ctrl-C goes limp rather than freezing at the last goal.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

import numpy as np

from . import k2_conventions as C
from . import k2_bus, k2_motion

RATE_HZ = 50.0
SLEW_RAD_S = 1.5        # max commanded joint speed
APPROACH_S = 3.0        # time to ease from the current pose into the start pose


def build_obs(q, qd, last_action, extra=()):
    """The observation vector, built identically for sim and real."""
    return np.concatenate([q, qd, np.asarray(extra, np.float32), last_action])


def slew(prev, target, dt, limit=SLEW_RAD_S):
    step = limit * dt
    return prev + np.clip(target - prev, -step, step)


def _calibration(bus):
    """Sim is perfectly calibrated by construction; hardware must be measured."""
    if isinstance(bus, k2_bus.SimBus):
        return C.default_calibration()
    calib = C.load_calibration()
    if calib is None:
        # Never silently fall back to home_raw=2048 on hardware. The real home
        # counts run from 1370 to 2337, so the default would command every
        # joint tens of degrees off -- up to 60 on hip_roll_L.
        raise SystemExit(
            f"No calibration at {C.CALIB_PATH}\n"
            f"Run:  python -m hardware.k2_calibrate\n"
            f"(a calibration saved under Robot_v2_single_leg is NOT used)")
    missing = [j for j in C.SIM_ORDER
               if j not in calib.get("home_raw", {})
               or j not in calib.get("sign", {})]
    if missing:
        raise SystemExit(f"calibration is missing joints: {missing}")
    return calib


def run(bus, traj, heights, dt, verbose=True, log=None,
        attitude=None, stabilizer=None):
    """Play a joint trajectory, closing the loop through the bus each tick."""
    calib = _calibration(bus)

    pos, _ = bus.read_pos_speed()
    q_now = C.q_vector(pos, calib)

    # Ease from wherever the robot actually is into the first pose, so nothing
    # snaps when torque comes on.
    n_appr = int(APPROACH_S / dt)
    approach = np.stack([q_now + (traj[0] - q_now) * (0.5 * (1 - np.cos(np.pi * i / n_appr)))
                         for i in range(n_appr)])

    cmd = q_now.copy()
    rows = []
    feedback_ticks = 0
    t_start = time.perf_counter()
    for phase, seq in (("approach", approach), ("motion", traj)):
        for i, target in enumerate(seq):
            pos, spd = bus.read_pos_speed()
            q = C.q_vector(pos, calib)
            qd = C.qd_vector(spd, calib)

            if stabilizer is not None:
                if attitude is None:
                    raise ValueError("stabilizer requires an attitude source")
                gyro, gravity = (attitude.attitude()
                                 if isinstance(bus, k2_bus.SimBus)
                                 else attitude.attitude(dt))
                # Fade feedback in over one second instead of stepping the
                # servos when the complementary filter first initializes.
                gain_scale = min(1.0, (feedback_ticks + 1) * dt)
                target = stabilizer.update(
                    target, gyro, gravity, dt, gain_scale=gain_scale)
                feedback_ticks += 1

            cmd = slew(cmd, target, dt)
            bus.write_goals(C.goal_counts(cmd, calib))
            bus.tick(dt)

            if log is not None:
                # Log wall-clock, not the nominal tick count: a loop that
                # cannot hold the rate is otherwise invisible in the CSV.
                row = [time.perf_counter() - t_start]
                row.extend(cmd)
                row.extend(q)
                row.extend(qd)
                if stabilizer is not None:
                    s = stabilizer.last_state
                    row.extend([s["roll"], s["pitch"],
                                s["roll_rate"], s["pitch_rate"],
                                s["roll_correction"], s["pitch_correction"]])
                rows.append(row)
            if verbose and phase == "motion" and i % 25 == 0:
                err = np.abs(q - cmd).max()
                extra = ""
                if isinstance(bus, k2_bus.SimBus):
                    extra = (f"  base {bus.base_height()*1000:6.1f}mm "
                             f"(want {heights[i]*1000:6.1f})")
                print(f"  t={i*dt:5.2f}s  knee_R={np.degrees(q[3]):+6.1f}deg  "
                      f"max|q-cmd|={np.degrees(err):5.2f}deg{extra}")
                if stabilizer is not None and stabilizer.last_state is not None:
                    s = stabilizer.last_state
                    print(f"       IMU roll={np.degrees(s['roll']):+5.1f}deg "
                          f"pitch={np.degrees(s['pitch']):+5.1f}deg  "
                          f"correction=({np.degrees(s['roll_correction']):+4.1f},"
                          f"{np.degrees(s['pitch_correction']):+4.1f})deg")

    n_ticks = len(approach) + len(traj)
    elapsed = time.perf_counter() - t_start
    print(f"  {n_ticks} ticks in {elapsed:.2f}s "
          f"= {n_ticks/elapsed:.1f} Hz achieved (target {1/dt:.0f} Hz, "
          f"nominal {n_ticks*dt:.2f}s)")
    if elapsed > n_ticks * dt * 1.05:
        print("  NOTE: the loop could not hold the target rate")

    if log is not None and rows:
        header = "t," + ",".join(f"cmd_{j}" for j in C.SIM_ORDER) + "," \
                 + ",".join(f"q_{j}" for j in C.SIM_ORDER) + "," \
                 + ",".join(f"qd_{j}" for j in C.SIM_ORDER)
        if stabilizer is not None:
            header += (",imu_roll,imu_pitch,imu_roll_rate,imu_pitch_rate,"
                       "feedback_roll,feedback_pitch")
        np.savetxt(log, np.array(rows), delimiter=",", header=header, comments="")
        print(f"wrote {log}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", default="sim",
                    help="'sim' for MuJoCo, or a device path like /dev/ttyACM0")
    ap.add_argument("--motion", default="squat", choices=("squat", "hold"))
    ap.add_argument("--depth", type=float, default=30.0,
                    help=f"squat depth in knee degrees (max {k2_motion.MAX_KNEE_DEG:.0f})")
    ap.add_argument("--period", type=float, default=4.0, help="seconds per squat")
    ap.add_argument("--cycles", type=float, default=3.0)
    ap.add_argument("--viewer", action="store_true", help="sim only")
    ap.add_argument("--fast", action="store_true",
                    help="sim only: run as fast as possible instead of "
                         "wall-clock (sim matches hardware timing by default)")
    ap.add_argument("--kp", type=float, default=k2_bus.DEFAULT_KP)
    ap.add_argument("--kd", type=float, default=k2_bus.DEFAULT_KD)
    ap.add_argument("--log", help="write a CSV of commanded vs measured")
    ap.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt on hardware")
    args = ap.parse_args(argv)

    dt = 1.0 / RATE_HZ
    if args.motion == "squat":
        traj, heights = k2_motion.squat_trajectory(
            args.depth, args.period, args.cycles, RATE_HZ)
    else:
        ik = k2_motion.SquatIK()
        q, h = ik.solve(args.depth)
        n = int(args.period * args.cycles * RATE_HZ)
        traj, heights = np.repeat(q[None], n, 0), np.full(n, h)

    real = args.bus != "sim"
    if real and not args.yes:
        print(f"About to MOVE THE REAL ROBOT on {args.bus}:")
        print(f"  {args.motion}, depth {args.depth:.0f} deg knee, "
              f"{args.cycles:g} cycles of {args.period:g}s")
        print("  Make sure it is standing on both feet with room to move.")
        if input("Type 'go' to continue: ").strip().lower() != "go":
            print("aborted")
            return 1

    bus = k2_bus.open_bus(args.bus, kp=args.kp, kd=args.kd,
                          viewer=args.viewer, realtime=not args.fast)

    def limp(*_):
        print("\ninterrupted -> releasing torque")
        try:
            bus.torque(False)
        finally:
            bus.close()
        sys.exit(130)

    signal.signal(signal.SIGINT, limp)

    try:
        bus.torque(True)
        print(f"[{args.bus}] {args.motion}: depth {args.depth:.0f} deg, "
              f"{args.cycles:g} x {args.period:g}s at {RATE_HZ:.0f} Hz")
        run(bus, traj, heights, dt, log=args.log)
    finally:
        bus.torque(False)
        bus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
