#!/usr/bin/env python3
"""Run a trained K2 RL policy (ONNX) over the Bus. Same code, sim or real.

    # digital twin (MuJoCo):
    python -m hardware.k2_policy_run --bus sim --viewer --mode march --policy <policy.onnx>
    # real robot (Pi):
    python -m hardware.k2_policy_run --bus /dev/ttyAMA0 --mode hold --policy <policy.onnx>

The policy is a hold/march or command-conditioned walking net from `k2_rl`.
Its 45-D (legacy march), 46-D (initial forward), 48-D (heading-aware forward /
bidirectional), or 49-D (turning) observation is rebuilt byte-for-byte the way
the training environment built it:

    obs = [ base_ang_vel(3)      # gyro, IMU/site frame
            projected_gravity(3) # gravity dir, k2_root frame
            joint_pos_rel(12)    # q - default_pose, SIM_ORDER
            joint_vel_rel(12)    # qdot, SIM_ORDER
            last_action(12)      # previous RAW policy output
            heading(0 or 2)      # optional [sin, cos] of the HEADING ERROR
            gait(3, 4 or 5) ]    # [march, march*sin, march*cos, vx, wz]

On a 5-D (turning) policy the heading channel is the error against a reference
heading obtained by integrating the yaw rate this runner itself commanded --
exactly what `k2_rl.mdp.GaitCommand` integrates during training. With no yaw
command that reference stays at zero and the channel is the plain relative
heading a 48-D policy expects, so the older policies run unchanged.

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
APPROACH_S = 5.0
LEGACY_GAIT_FREQ_HZ = 0.75  # checkpoint 2000 predates gait metadata


def _read_metadata(sess: ort.InferenceSession):
  """Pull the deployment contract out of the ONNX."""
  md = sess.get_modelmeta().custom_metadata_map
  mjcf_names = md["joint_names"].split(",")
  default_pos = np.array([float(x) for x in md["default_joint_pos"].split(",")],
                         dtype=np.float32)
  action_scale = float(md["action_scale"])
  gait_freq_hz = float(md.get("gait_frequency_hz", LEGACY_GAIT_FREQ_HZ))
  gait_transition_time_s = float(md.get("gait_transition_time_s", "0.0"))
  gait_velocity_ramp_rate_mps2 = float(
    md.get("gait_velocity_ramp_rate_mps2", "0.08")
  )
  gait_yaw_rate_ramp_rate_rps2 = float(
    md.get("gait_yaw_rate_ramp_rate_rps2", "0.30")
  )
  checkpoint_iteration = md.get("checkpoint_iteration", "unknown")
  neutral_base_shift_m = float(md.get("neutral_base_shift_m", "nan"))
  obs_dim = int(sess.get_inputs()[0].shape[-1])
  if obs_dim == 45:
    inferred_heading_dim, inferred_gait_dim = 0, 3
  elif obs_dim == 46:
    inferred_heading_dim, inferred_gait_dim = 0, 4
  elif obs_dim == 48:
    inferred_heading_dim, inferred_gait_dim = 2, 4
  elif obs_dim == 49:
    inferred_heading_dim, inferred_gait_dim = 2, 5
  else:
    raise SystemExit(f"unsupported policy observation size: {obs_dim}")
  gait_command_dim = int(md.get("gait_command_dim", inferred_gait_dim))
  heading_observation_dim = int(
    md.get("heading_observation_dim", inferred_heading_dim)
  )
  # Map the policy's MJCF joint order to our SIM_ORDER short names.
  mjcf_to_short = {v: k for k, v in C.JOINT_TO_MJCF.items()}
  policy_short = [mjcf_to_short[n] for n in mjcf_names]
  if tuple(policy_short) != tuple(C.SIM_ORDER):
    raise SystemExit(
      "policy joint order != SIM_ORDER; permutation not implemented.\n"
      f"  policy: {policy_short}\n  sim   : {list(C.SIM_ORDER)}")
  if default_pos.shape != (len(C.SIM_ORDER),) or not np.all(np.isfinite(default_pos)):
    raise SystemExit("policy metadata has an invalid default_joint_pos")
  outside = [
    name for i, name in enumerate(C.SIM_ORDER)
    if not C.LIMITS[name][0] <= float(default_pos[i]) <= C.LIMITS[name][1]
  ]
  if outside:
    raise SystemExit(f"policy default pose exceeds joint limits: {outside}")
  if not 0.0 < action_scale <= 0.5:
    raise SystemExit(f"unsafe policy action scale: {action_scale:g}")
  if gait_command_dim not in (3, 4, 5) or gait_command_dim != inferred_gait_dim:
    raise SystemExit(
      f"unsupported/inconsistent gait interface: metadata={gait_command_dim}, "
      f"observation implies {inferred_gait_dim}"
    )
  if (heading_observation_dim not in (0, 2)
      or heading_observation_dim != inferred_heading_dim):
    raise SystemExit(
      f"unsupported/inconsistent heading interface: "
      f"metadata={heading_observation_dim}, observation implies "
      f"{inferred_heading_dim}"
    )
  return (default_pos, action_scale, gait_freq_hz,
          checkpoint_iteration, neutral_base_shift_m, gait_command_dim,
          heading_observation_dim, gait_transition_time_s,
          gait_velocity_ramp_rate_mps2, gait_yaw_rate_ramp_rate_rps2)


SWAP_LEGS = np.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5])
MAX_JOINT_TRIM_DEG = 10.0


def parse_joint_trim(spec: str | None) -> np.ndarray:
  """Parse "hip_yaw_R=+2.5,hip_yaw_L=-2.5" into a SIM_ORDER vector in radians.

  A mechanical asymmetry -- a foot mounted at a slight yaw, one leg built a
  fraction long -- forces the policy to hold a permanent differential just to
  stand square, and it has to spend control authority doing so every tick. The
  trim moves that differential into the wiring: the policy is shown a SYMMETRIC
  robot, and the offset is added back on the way out to the servos.

      observation : joint_pos_rel = (q - trim) - default_pose
      command     : q_target      = default_pose + scale * action + trim

  which is the same shape as the `encoder_bias` randomization the policy trained
  against, so it is in-distribution by construction. Reverts by omission.
  """
  trim = np.zeros(len(C.SIM_ORDER), dtype=np.float32)
  if not spec:
    return trim
  index = {name: i for i, name in enumerate(C.SIM_ORDER)}
  for item in spec.split(","):
    item = item.strip()
    if not item:
      continue
    if "=" not in item:
      raise SystemExit(f"--joint-trim needs NAME=DEGREES, got '{item}'")
    name, _, value = item.partition("=")
    name = name.strip()
    if name not in index:
      raise SystemExit(
        f"unknown joint '{name}' in --joint-trim; expected one of "
        f"{', '.join(C.SIM_ORDER)}")
    try:
      degrees = float(value)
    except ValueError:
      raise SystemExit(f"--joint-trim value for {name} is not a number: {value}")
    if abs(degrees) > MAX_JOINT_TRIM_DEG:
      raise SystemExit(
        f"--joint-trim {name}={degrees:g} exceeds the +/-{MAX_JOINT_TRIM_DEG:g} "
        "degree guard; a trim that large is a calibration problem, not a trim")
    trim[index[name]] = math.radians(degrees)
  return trim


def run(bus, sess, default_pos, action_scale, mode, duration, slew_rad_s,
        march_after, gait_freq_hz, approach_s=APPROACH_S,
        fall_angle_deg=12.0, log=None, swap_legs=False, forward_speed=0.03,
        heading_observation_dim=0, balance_trim_right_deg=0.0,
        gait_transition_time_s=0.0, command_source=None,
        gait_velocity_ramp_rate_mps2=0.08, yaw_rate=0.0,
        gait_yaw_rate_ramp_rate_rps2=0.30, joint_trim=None):
  dt = 1.0 / RATE_HZ
  iname = sess.get_inputs()[0].name
  # Route the policy's right-leg block to the other physical leg. Equivalent to
  # permuting JOINT_TO_ID and calibration.json together (each servo keeps its
  # own home_raw/sign, and every POS_DESC is mirror-symmetric so the signs stay
  # valid) -- but as a flag, so it reverts by omission. Diagnostic only.
  perm = SWAP_LEGS if swap_legs else np.arange(len(C.SIM_ORDER))
  # Physical, per-servo quantity: applied in SIM_ORDER, never permuted. NOTE the
  # name: `trim` below is the scalar balance-trim ANGLE, a different thing.
  joint_trim_rad = (np.zeros(len(C.SIM_ORDER), np.float32) if joint_trim is None
                    else np.asarray(joint_trim, np.float32))
  if np.any(joint_trim_rad):
    shown = ", ".join(f"{C.SIM_ORDER[i]}={math.degrees(v):+.2f}"
                      for i, v in enumerate(joint_trim_rad) if v)
    print(f"  joint trim (deg): {shown}")

  calib = (C.default_calibration() if isinstance(bus, k2_bus.SimBus)
           else _hw_calibration())
  att = SimAttitude(bus) if isinstance(bus, k2_bus.SimBus) else ImuAttitude()
  fall_angle = math.radians(fall_angle_deg)
  # A positive user trim must move the learned support response toward -Y
  # (robot-right). The policy's response sign was verified in the twin with a
  # +10 mm robot-left base CoM perturbation, hence the negative virtual roll.
  trim = -math.radians(balance_trim_right_deg)
  trim_cos, trim_sin = math.cos(trim), math.sin(trim)

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
  tracking_errors = []
  telemetry_samples = []
  sim_diag = {
    "swing_spread": [],
    "apex_clearance": [],
    "inner_gap": [],
    "foot_contacts": 0,
  }

  def sample_sim_feet(sinp, marching):
    if not isinstance(bus, k2_bus.SimBus):
      return
    model, data = bus.model, bus.data
    points = {}
    for side in ("left", "right"):
      points[side] = np.array([
        data.site_xpos[model.site(f"{side}_foot_{part}").id].copy()
        for part in ("heel", "toe", "inner", "outer")
      ])
    inner_l = data.site_xpos[model.site("left_foot_inner").id]
    inner_r = data.site_xpos[model.site("right_foot_inner").id]
    sim_diag["inner_gap"].append(abs(float(inner_l[1] - inner_r[1])))
    if marching and abs(sinp) > 0.18:
      # SceneEntityCfg resolves the MJCF site order (right, then left).
      side = "right" if sinp > 0 else "left"
      z = points[side][:, 2]
      sim_diag["swing_spread"].append(float(np.max(z) - np.min(z)))
      if abs(sinp) > 0.90:
        sim_diag["apex_clearance"].append(float(np.min(z)))

    left_gid = model.geom("left_foot_collision").id
    right_gid = model.geom("right_foot_collision").id
    for contact in data.contact:
      if {int(contact.geom1), int(contact.geom2)} == {left_gid, right_gid}:
        sim_diag["foot_contacts"] += 1

  def print_sim_diagnostics():
    if not isinstance(bus, k2_bus.SimBus) or not sim_diag["inner_gap"]:
      return
    spread = np.asarray(sim_diag["swing_spread"])
    apex = np.asarray(sim_diag["apex_clearance"])
    gap = np.asarray(sim_diag["inner_gap"])
    print("  sole diagnostics:")
    if len(spread):
      print(f"    max swing sole height spread: {spread.max()*1000:.1f} mm")
    if len(apex):
      print(f"    minimum apex clearance (all sole points): {apex.min()*1000:.1f} mm")
    print(f"    minimum inner-edge gap: {gap.min()*1000:.1f} mm")
    print(f"    foot-to-foot collision contacts: {sim_diag['foot_contacts']}")

  def write_log():
    if log is None or not rows:
      return
    header = ["t", "t_nominal", "loop_dt", "march", "phase",
              "forward_command", "yaw_command", "heading_reference",
              "base_x", "base_y", "base_z", "base_heading",
              "heading_sin", "heading_cos", "tilt",
              "gyro_x", "gyro_y", "gyro_z",
              "gravity_x", "gravity_y", "gravity_z",
              "policy_gravity_x", "policy_gravity_y", "policy_gravity_z"]
    header += [f"action_{j}" for j in C.SIM_ORDER]
    header += [f"cmd_{j}" for j in C.SIM_ORDER]
    header += [f"q_{j}" for j in C.SIM_ORDER]
    header += [f"qd_{j}" for j in C.SIM_ORDER]
    header += [f"error_{j}" for j in C.SIM_ORDER]
    for field in ("load", "current_raw", "voltage", "temperature", "status",
                  "telemetry_outlier"):
      header += [f"{field}_{j}" for j in C.SIM_ORDER]
    np.savetxt(log, np.asarray(rows), delimiter=",",
               header=",".join(header), comments="")
    print(f"wrote {log}")

  try:
    # 1) Ease from the current pose into the default crouch. The policy is not
    # active yet, but the IMU safety cutoff is.
    pos, _ = bus.read_pos_speed()
    q_now = C.q_vector(pos, calib)
    n_appr = int(approach_s / dt)
    for i in range(n_appr):
      attitude()
      a = 0.5 * (1 - math.cos(math.pi * i / n_appr))
      cmd = q_now + (default_pos - q_now) * a
      bus.write_goals(C.goal_counts(cmd, calib))
      bus.tick(dt)

    # 2) Policy loop.
    last_action = np.zeros(12, dtype=np.float32)
    phase = 0.0
    gait_activation = 0.0
    interactive_forward_speed = 0.0
    interactive_yaw_rate = 0.0
    # Reference heading the commanded yaw rate asks the robot to face. Training
    # integrates exactly this from the same emitted command, so a 5-D policy
    # sees the same near-zero heading error here that it saw in simulation.
    heading_reference = 0.0
    cmd = default_pos.copy()
    n_steps = int(duration * RATE_HZ)
    t0 = time.perf_counter()
    previous_tick_time = t0
    ran_steps = 0
    for k in range(n_steps):
      active_mode = mode
      active_forward_speed = forward_speed
      active_yaw_rate = yaw_rate
      if command_source is not None:
        (active_mode, active_forward_speed, active_yaw_rate,
         stop_requested) = command_source()
        if stop_requested:
          print("  keyboard quit requested")
          break
        if active_mode not in ("hold", "march", "walk"):
          raise UnsafeTiltError(f"invalid interactive mode: {active_mode}")
        # Training rate-limits signed command resamples. Do the same here so
        # W<->S and A<->D never ask the policy or the robot for an
        # instantaneous reversal.
        speed_target = active_forward_speed if active_mode == "walk" else 0.0
        speed_step = gait_velocity_ramp_rate_mps2 * dt
        interactive_forward_speed += float(np.clip(
          speed_target - interactive_forward_speed, -speed_step, speed_step
        ))
        yaw_target = (active_yaw_rate
                      if active_mode in ("march", "walk") else 0.0)
        yaw_step = gait_yaw_rate_ramp_rate_rps2 * dt
        interactive_yaw_rate += float(np.clip(
          yaw_target - interactive_yaw_rate, -yaw_step, yaw_step
        ))
      ran_steps += 1
      tick_time = time.perf_counter()
      loop_dt = tick_time - previous_tick_time
      previous_tick_time = tick_time
      pos, spd = bus.read_pos_speed()
      q = C.q_vector(pos, calib).astype(np.float32)
      qd = C.qd_vector(spd, calib).astype(np.float32)
      telemetry = bus.latest_telemetry()
      if telemetry is None:
        telemetry_arrays = [np.full(12, np.nan, np.float32) for _ in range(6)]
      else:
        telemetry_arrays = [
          np.array([telemetry[C.JOINT_TO_ID[j]][field] for j in C.SIM_ORDER],
                   np.float32)
          for field in ("load_signed", "current_raw", "voltage_v",
                        "temperature_c", "error_status", "temperature_outlier")
        ]
        telemetry_samples.append(np.stack(telemetry_arrays))
      gyro, proj_grav, tilt = attitude()
      # Positive trim shifts the policy's closed-loop support response toward
      # the robot's right, countering a repeatable robot-left lean.
      # Safety and the raw gravity columns always retain the true IMU estimate.
      policy_grav = proj_grav.copy()
      policy_grav[1] = trim_cos * proj_grav[1] - trim_sin * proj_grav[2]
      policy_grav[2] = trim_sin * proj_grav[1] + trim_cos * proj_grav[2]
      # Gait command: march from the start, or after `march_after` seconds.
      marching = active_mode in ("march", "walk") and k * dt >= march_after
      gait_target = 1.0 if marching else 0.0
      if gait_transition_time_s > 0.0:
        activation_step = dt / gait_transition_time_s
        gait_activation += float(np.clip(
          gait_target - gait_activation, -activation_step, activation_step
        ))
      else:
        gait_activation = gait_target
      m = gait_activation
      # Match training: finish the smooth activation in double support, then
      # begin the first swing from phase zero.
      if marching and m >= 1.0 - 1e-6:
        phase = (phase + 2 * math.pi * gait_freq_hz * dt) % (2 * math.pi)
      elif m <= 0.0:
        phase = 0.0
      gait_values = [m, m * math.sin(phase), m * math.cos(phase)]
      if command_source is None:
        forward_command = (m * active_forward_speed
                           if active_mode == "walk" else 0.0)
        yaw_command = (m * active_yaw_rate
                       if active_mode in ("march", "walk") else 0.0)
      else:
        forward_command = m * interactive_forward_speed
        yaw_command = m * interactive_yaw_rate
      expected_obs = sess.get_inputs()[0].shape[-1]
      if expected_obs in (46, 48, 49):
        gait_values.append(forward_command)
      if expected_obs == 49:
        gait_values.append(yaw_command)
      else:
        # A policy without a yaw channel cannot be steered; keep the reference
        # pinned so its heading observation stays the plain relative heading.
        yaw_command = 0.0
      gait = np.asarray(gait_values, np.float32)

      # Advance the reference heading with the command just emitted, then read
      # the heading error against it -- the same order as training, where
      # GaitCommand integrates the reference at the end of its command update
      # and the observation manager runs immediately afterwards.
      heading_reference += yaw_command * dt
      heading_reference = (heading_reference + math.pi) % (2 * math.pi) - math.pi
      if heading_observation_dim == 2:
        heading_error = att.relative_heading() - heading_reference
        heading_obs = np.array(
          [math.sin(heading_error), math.cos(heading_error)], np.float32)
      else:
        heading_obs = np.empty(0, np.float32)

      obs_parts = [
        gyro.astype(np.float32),          # base_ang_vel
        policy_grav.astype(np.float32),   # projected_gravity with balance trim
        (q - joint_trim_rad - default_pos)[perm],  # joint_pos_rel, trimmed
        qd[perm],                         # joint_vel_rel
        last_action,                      # actions
        heading_obs,                      # optional [sin(yaw), cos(yaw)]
        gait,                             # gait command
      ]
      obs = np.concatenate(obs_parts).astype(np.float32)[None]
      if obs.shape[1] != expected_obs:
        raise UnsafeTiltError(
          f"policy expects {expected_obs} observations, built {obs.shape[1]}")

      action = sess.run(None, {iname: obs})[0][0].astype(np.float32)
      if not np.all(np.isfinite(action)):
        raise UnsafeTiltError("policy returned a non-finite action")
      last_action = action

      target = default_pos.copy()
      target[perm] = default_pos[perm] + action_scale * action
      target = target + joint_trim_rad
      target = np.array([C.clamp_q(j, float(target[i]))
                         for i, j in enumerate(C.SIM_ORDER)], np.float32)
      if slew_rad_s > 0:
        step = slew_rad_s * dt
        target = cmd + np.clip(target - cmd, -step, step)
      cmd = target
      tracking_error = cmd - q
      if marching:
        tracking_errors.append(tracking_error.copy())

      bus.write_goals(C.goal_counts(cmd, calib))
      bus.tick(dt)
      sample_sim_feet(float(gait[1]), marching)
      base_position = bus.base_position()
      if base_position is None:
        base_position = np.full(3, np.nan, np.float32)
      base_heading = bus.base_heading()
      if base_heading is None:
        base_heading = np.nan

      if log is not None:
        rows.append(np.concatenate([
          [time.perf_counter() - t0, k * dt, loop_dt, float(m), phase,
           forward_command, yaw_command, heading_reference,
           *base_position, base_heading,
           *(heading_obs if len(heading_obs) else (np.nan, np.nan)), tilt],
          gyro, proj_grav, policy_grav, action, cmd, q, qd, tracking_error,
          *telemetry_arrays,
        ]))

      if k % 25 == 0:
        extra = ""
        if isinstance(bus, k2_bus.SimBus):
          extra = f"  base={bus.base_height()*1000:5.1f}mm"
        label = ("WALK " if active_mode == "walk" and marching
                 else "MARCH" if marching else "hold ")
        turn = f"  wz={yaw_command:+.2f}" if expected_obs == 49 else ""
        print(f"  t={k*dt:5.2f}s  {label}  "
              f"tilt={math.degrees(tilt):4.1f}deg  "
              f"|act|={np.abs(action).max():.2f}{turn}{extra}")

    hz = ran_steps / max(time.perf_counter() - t0, 1e-9)
    print(f"  ran {ran_steps} policy ticks at {hz:.1f} Hz "
          f"(target {RATE_HZ:.0f})")
    if tracking_errors:
      err = np.asarray(tracking_errors)
      rms = np.sqrt(np.mean(err * err, axis=0))
      peak = np.max(np.abs(err), axis=0)
      worst_rms = int(np.argmax(rms))
      worst_peak = int(np.argmax(peak))
      print("  command tracking during march:")
      print(f"    worst RMS:  {C.SIM_ORDER[worst_rms]} "
            f"{math.degrees(rms[worst_rms]):.2f} deg")
      print(f"    worst peak: {C.SIM_ORDER[worst_peak]} "
            f"{math.degrees(peak[worst_peak]):.2f} deg")
    if telemetry_samples:
      telem = np.asarray(telemetry_samples)  # tick, field, joint
      voltage = telem[:, 2, :]
      temperature = telem[:, 3, :]
      status = telem[:, 4, :]
      outliers = telem[:, 5, :]
      vi = np.unravel_index(np.argmin(voltage), voltage.shape)
      ti = np.unravel_index(np.argmax(temperature), temperature.shape)
      print("  servo telemetry:")
      print(f"    minimum voltage: {voltage[vi]:.1f} V "
            f"({C.SIM_ORDER[vi[1]]})")
      print(f"    maximum temperature: {temperature[ti]:.0f} C "
            f"({C.SIM_ORDER[ti[1]]})")
      print(f"    nonzero error samples: {int(np.count_nonzero(status))}")
      print(f"    rejected temperature outliers: "
            f"{int(np.count_nonzero(outliers))}")
  finally:
    # Preserve the samples leading up to a safety stop; these are the most
    # important data for deciding whether a policy is safe enough for hardware.
    write_log()
    print_sim_diagnostics()
    att.close()


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
  ap.add_argument("--mode", default="hold", choices=("hold", "march", "walk"))
  ap.add_argument("--march-after", type=float, default=0.0,
                  help="seconds to hold before marching (mode=march)")
  ap.add_argument("--gait-freq", type=float,
                  help="override policy gait clock frequency (normally ONNX metadata)")
  ap.add_argument("--forward-speed", type=float, default=0.03,
                  help="commanded m/s for mode=walk (forward policy only; default 0.03)")
  ap.add_argument("--yaw-rate", type=float, default=0.0,
                  help="commanded body yaw rate in rad/s, positive = turn left; "
                       "applies in mode=march (turn in place) and mode=walk "
                       "(walk an arc). Turning policies only.")
  ap.add_argument("--duration", type=float, default=20.0,
                  help="seconds of policy control")
  ap.add_argument("--approach", type=float, default=APPROACH_S,
                  help="seconds to ease into the policy's training pose")
  ap.add_argument("--slew", type=float, default=1.0,
                  help="max joint rad/s per tick (default 1.0, 0 disables)")
  ap.add_argument("--fall-angle", type=float, default=12.0,
                  help="release torque above this torso tilt, degrees")
  ap.add_argument("--balance-trim-right", type=float, default=0.0,
                  help="degrees of rightward balance trim; positive counters "
                       "a repeatable robot-left lean (safe range +/-5)")
  ap.add_argument("--joint-trim", default=None,
                  help="per-joint mechanical trim, e.g. "
                       "'hip_yaw_R=+2.5,hip_yaw_L=-2.5' (degrees). Shows the "
                       "policy a symmetric robot and adds the offset back at "
                       "the servos; omit for no trim.")
  ap.add_argument("--log", help="write policy, IMU, command and encoder CSV")
  ap.add_argument("--viewer", action="store_true", help="sim only")
  ap.add_argument("--fast", action="store_true", help="sim only: no wall-clock")
  ap.add_argument("--yes", action="store_true",
                  help="skip the confirmation prompt on hardware")
  ap.add_argument("--swap-legs", action="store_true",
                  help="diagnostic: swap the policy's right/left observation "
                       "and action joint blocks without changing servo calibration")
  args = ap.parse_args(argv)

  sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
  (default_pos, action_scale, policy_gait_freq, checkpoint_iteration,
   neutral_base_shift_m, gait_command_dim,
   heading_observation_dim, gait_transition_time_s,
   gait_velocity_ramp_rate_mps2,
   gait_yaw_rate_ramp_rate_rps2) = _read_metadata(sess)
  gait_freq_hz = (args.gait_freq if args.gait_freq is not None
                  else policy_gait_freq)
  if not 0.1 <= gait_freq_hz <= 3.0:
    raise SystemExit(f"unsafe gait frequency: {gait_freq_hz:g} Hz")
  print(f"Policy gait frequency: {gait_freq_hz:g} Hz")
  print(f"Policy checkpoint: {checkpoint_iteration}")
  if np.isfinite(neutral_base_shift_m):
    print(f"Policy neutral base shift: {neutral_base_shift_m*1000:+.1f} mm")
  print(f"Policy gait interface: {gait_command_dim}-D")
  print(f"Policy heading interface: {heading_observation_dim}-D")
  print(f"Policy gait transition: {gait_transition_time_s:g} s")
  print(f"Policy velocity ramp: {gait_velocity_ramp_rate_mps2:g} m/s^2")
  if gait_command_dim == 5:
    print(f"Policy yaw ramp: {gait_yaw_rate_ramp_rate_rps2:g} rad/s^2")
  if not -0.10 <= args.forward_speed <= 0.10:
    raise SystemExit("--forward-speed must be between -0.10 and +0.10 m/s")
  # Same spirit as the forward guard: refuse a command well outside any trained
  # yaw range before the robot ever moves.
  if not -0.35 <= args.yaw_rate <= 0.35:
    raise SystemExit("--yaw-rate must be between -0.35 and +0.35 rad/s")
  if not -5.0 <= args.balance_trim_right <= 5.0:
    raise SystemExit("--balance-trim-right must be between -5 and +5 degrees")
  if args.mode == "walk" and gait_command_dim not in (4, 5):
    raise SystemExit("mode=walk requires a command-conditioned forward policy")
  if args.yaw_rate != 0.0 and gait_command_dim != 5:
    raise SystemExit("--yaw-rate requires a turning policy (5-D gait command)")

  if args.swap_legs:
    print("DIAGNOSTIC: swapping policy right/left joint blocks; servo IDs and "
          "per-servo calibration remain unchanged.")

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
          f"scale={action_scale}  slew={args.slew or 'off'}  "
          f"balance_trim_right={args.balance_trim_right:+g}deg")
    try:
      run(bus, sess, default_pos, action_scale, args.mode, args.duration,
          args.slew, args.march_after, gait_freq_hz,
          args.approach, args.fall_angle, args.log, args.swap_legs,
          args.forward_speed, heading_observation_dim,
          args.balance_trim_right, gait_transition_time_s, None,
          gait_velocity_ramp_rate_mps2, args.yaw_rate,
          gait_yaw_rate_ramp_rate_rps2, parse_joint_trim(args.joint_trim))
    except UnsafeTiltError as exc:
      print(f"SAFETY STOP: {exc}")
      return 2
  finally:
    try:
      bus.torque(False)
    finally:
      bus.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
