#!/usr/bin/env python3
"""Run a trained K2 RL policy (ONNX) over the Bus. Same code, sim or real.

    # digital twin (MuJoCo):
    python -m hardware.k2_policy_run --bus sim --viewer --mode march --policy <policy.onnx>
    # real robot (Pi):
    python -m hardware.k2_policy_run --bus /dev/ttyACM0 --mode hold  --policy <policy.onnx>

The policy is the in-place hold/march net from `k2_rl` (see k2_rl/README.md).
The 45-D observation is rebuilt here byte-for-byte the way the mjlab env built
it at training time:

    obs = [ base_ang_vel(3)      # gyro, IMU/site frame
            projected_gravity(3) # gravity dir, k2_root frame
            joint_pos_rel(12)    # q - default_pose, SIM_ORDER
            joint_vel_rel(12)    # qdot, SIM_ORDER
            last_action(12)      # previous RAW policy output
            gait(3) ]            # [march, march*sin(phase), march*cos(phase)]

Joint order, default pose and action scale are read from the ONNX metadata, so
this stays correct if the policy is retrained. Actions are position targets:
    q_target = default_pose + action_scale * action
clamped to joint limits (and counts [40,4055] in the bus). The IMU obs comes
from SimAttitude (sim) or ImuAttitude (real LSM6DS3), which share one interface.

Safety (real robot): torque always released on exit incl. Ctrl-C; optional
--slew caps per-tick joint motion; the robot is eased into the default crouch
before the policy takes over.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time

import numpy as np
import onnxruntime as ort

from . import k2_conventions as C
from . import k2_bus
from .k2_attitude import SimAttitude, ImuAttitude
from .k2_stabilizer import UnsafeTiltError

RATE_HZ = 50.0
APPROACH_S = 3.0
LEGACY_GAIT_FREQ_HZ = 0.75  # checkpoint 2000 predates gait metadata


def _read_metadata(sess: ort.InferenceSession):
  """Pull the deployment contract out of the ONNX."""
  md = sess.get_modelmeta().custom_metadata_map
  mjcf_names = md["joint_names"].split(",")
  default_pos = np.array([float(x) for x in md["default_joint_pos"].split(",")],
                         dtype=np.float32)
  action_scale = float(md["action_scale"])
  gait_freq_hz = float(md.get("gait_frequency_hz", LEGACY_GAIT_FREQ_HZ))
  # Map the policy's MJCF joint order to our SIM_ORDER short names.
  mjcf_to_short = {v: k for k, v in C.JOINT_TO_MJCF.items()}
  policy_short = [mjcf_to_short[n] for n in mjcf_names]
  if tuple(policy_short) != tuple(C.SIM_ORDER):
    raise SystemExit(
      "policy joint order != SIM_ORDER; permutation not implemented.\n"
      f"  policy: {policy_short}\n  sim   : {list(C.SIM_ORDER)}")
  return default_pos, action_scale, gait_freq_hz


def run(bus, sess, default_pos, action_scale, mode, duration, slew_rad_s,
        march_after, gait_freq_hz, fall_angle_deg=12.0, log=None):
  dt = 1.0 / RATE_HZ
  iname = sess.get_inputs()[0].name

  calib = (C.default_calibration() if isinstance(bus, k2_bus.SimBus)
           else _hw_calibration())
  att = SimAttitude(bus) if isinstance(bus, k2_bus.SimBus) else ImuAttitude()
  fall_angle = math.radians(fall_angle_deg)

  def attitude():
    gyro, gravity = (att.attitude() if isinstance(bus, k2_bus.SimBus)
                     else att.attitude(dt))
    gravity = np.asarray(gravity, np.float32)
    gravity /= max(float(np.linalg.norm(gravity)), 1e-6)
    tilt = math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
    if tilt > fall_angle:
      raise UnsafeTiltError(
        f"torso tilt {math.degrees(tilt):.1f} deg exceeds "
        f"{fall_angle_deg:.1f} deg")
    return gyro, gravity, tilt

  rows = []
  try:
    # 1) Ease from the current pose into the default crouch. The policy is not
    # active yet, but the IMU safety cutoff is.
    pos, _ = bus.read_pos_speed()
    q_now = C.q_vector(pos, calib)
    n_appr = int(APPROACH_S / dt)
    for i in range(n_appr):
      attitude()
      a = 0.5 * (1 - math.cos(math.pi * i / n_appr))
      cmd = q_now + (default_pos - q_now) * a
      bus.write_goals(C.goal_counts(cmd, calib))
      bus.tick(dt)

    # 2) Policy loop.
    last_action = np.zeros(12, dtype=np.float32)
    phase = 0.0
    cmd = default_pos.copy()
    n_steps = int(duration * RATE_HZ)
    t0 = time.perf_counter()
    for k in range(n_steps):
      pos, spd = bus.read_pos_speed()
      q = C.q_vector(pos, calib).astype(np.float32)
      qd = C.qd_vector(spd, calib).astype(np.float32)
      gyro, proj_grav, tilt = attitude()

      # Gait command: march from the start, or after `march_after` seconds.
      marching = (mode == "march") and (k * dt >= march_after)
      m = 1.0 if marching else 0.0
      if marching:
        phase = (phase + 2 * math.pi * gait_freq_hz * dt) % (2 * math.pi)
      else:
        phase = 0.0
      gait = np.array([m, m * math.sin(phase), m * math.cos(phase)], np.float32)

      obs = np.concatenate([
        gyro.astype(np.float32),          # base_ang_vel
        proj_grav.astype(np.float32),     # projected_gravity
        q - default_pos,                  # joint_pos_rel
        qd,                               # joint_vel_rel
        last_action,                      # actions
        gait,                             # gait command
      ]).astype(np.float32)[None]         # [1,45]

      action = sess.run(None, {iname: obs})[0][0].astype(np.float32)
      if not np.all(np.isfinite(action)):
        raise UnsafeTiltError("policy returned a non-finite action")
      last_action = action

      target = default_pos + action_scale * action
      target = np.array([C.clamp_q(j, float(target[i]))
                         for i, j in enumerate(C.SIM_ORDER)], np.float32)
      if slew_rad_s > 0:
        step = slew_rad_s * dt
        target = cmd + np.clip(target - cmd, -step, step)
      cmd = target

      bus.write_goals(C.goal_counts(cmd, calib))
      bus.tick(dt)

      if log is not None:
        rows.append(np.concatenate([
          [time.perf_counter() - t0, float(m), tilt],
          gyro, proj_grav, action, cmd, q, qd,
        ]))

      if k % 25 == 0:
        extra = ""
        if isinstance(bus, k2_bus.SimBus):
          extra = f"  base={bus.base_height()*1000:5.1f}mm"
        print(f"  t={k*dt:5.2f}s  {'MARCH' if marching else 'hold '}  "
              f"tilt={math.degrees(tilt):4.1f}deg  "
              f"|act|={np.abs(action).max():.2f}{extra}")

    hz = n_steps / (time.perf_counter() - t0)
    print(f"  ran {n_steps} policy ticks at {hz:.1f} Hz (target {RATE_HZ:.0f})")
  finally:
    att.close()

  if log is not None and rows:
    header = ["t", "march", "tilt",
              "gyro_x", "gyro_y", "gyro_z",
              "gravity_x", "gravity_y", "gravity_z"]
    header += [f"action_{j}" for j in C.SIM_ORDER]
    header += [f"cmd_{j}" for j in C.SIM_ORDER]
    header += [f"q_{j}" for j in C.SIM_ORDER]
    header += [f"qd_{j}" for j in C.SIM_ORDER]
    np.savetxt(log, np.asarray(rows), delimiter=",",
               header=",".join(header), comments="")
    print(f"wrote {log}")


def _hw_calibration():
  calib = C.load_calibration()
  if calib is None:
    raise SystemExit(
      f"No calibration at {C.CALIB_PATH}. Run: python -m hardware.k2_calibrate")
  return calib


def main(argv=None):
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--bus", default="sim",
                  help="'sim' for MuJoCo, or a device path like /dev/ttyACM0")
  ap.add_argument("--policy", required=True, help="path to policy.onnx")
  ap.add_argument("--mode", default="hold", choices=("hold", "march"))
  ap.add_argument("--march-after", type=float, default=0.0,
                  help="seconds to hold before marching (mode=march)")
  ap.add_argument("--gait-freq", type=float,
                  help="override policy gait clock frequency (normally ONNX metadata)")
  ap.add_argument("--duration", type=float, default=20.0,
                  help="seconds of policy control")
  ap.add_argument("--slew", type=float, default=1.0,
                  help="max joint rad/s per tick (default 1.0, 0 disables)")
  ap.add_argument("--fall-angle", type=float, default=12.0,
                  help="release torque above this torso tilt, degrees")
  ap.add_argument("--log", help="write policy, IMU, command and encoder CSV")
  ap.add_argument("--viewer", action="store_true", help="sim only")
  ap.add_argument("--fast", action="store_true", help="sim only: no wall-clock")
  ap.add_argument("--yes", action="store_true",
                  help="skip the confirmation prompt on hardware")
  args = ap.parse_args(argv)

  sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
  default_pos, action_scale, policy_gait_freq = _read_metadata(sess)
  gait_freq_hz = (args.gait_freq if args.gait_freq is not None
                  else policy_gait_freq)
  if not 0.1 <= gait_freq_hz <= 3.0:
    raise SystemExit(f"unsafe gait frequency: {gait_freq_hz:g} Hz")
  print(f"Policy gait frequency: {gait_freq_hz:g} Hz")

  real = args.bus != "sim"
  if real and not args.yes:
    print(f"About to RUN AN RL POLICY ON THE REAL ROBOT ({args.bus}), "
          f"mode={args.mode}, {args.duration:g}s.")
    print("  Stand it on both feet with room to move, keep a hand near it.")
    if input("Type 'go' to continue: ").strip().lower() != "go":
      print("aborted")
      return 1

  bus = k2_bus.open_bus(args.bus, viewer=args.viewer, realtime=not args.fast)

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
    print(f"[{args.bus}] policy={args.policy}  mode={args.mode}  "
          f"scale={action_scale}  slew={args.slew or 'off'}")
    try:
      run(bus, sess, default_pos, action_scale, args.mode, args.duration,
          args.slew, args.march_after, gait_freq_hz, args.fall_angle, args.log)
    except UnsafeTiltError as exc:
      print(f"SAFETY STOP: {exc}")
      return 2
  finally:
    bus.torque(False)
    bus.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
