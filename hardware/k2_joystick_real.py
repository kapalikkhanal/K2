#!/usr/bin/env python3
"""Hardware backend selected by ``joystick.py --bus SERIAL_PORT``.

  Left stick  Y   walk forward / backward   (analog, up = forward)
  Left stick  X   strafe left / right       (needs a lateral channel)
  Right stick X   turn left / right         (needs a yaw channel)
  A               MARCH in place (resting mode) and clear any skill
  B               HOLD -- stop, straighten the yaw command (resting mode)
  LB / RB / Y     crouch deeper / rise / nominal height (HOLD only)
  Back            stop, release servo torque, and quit

The RIGHT TRIGGER is a dead-man and is REQUIRED BY DEFAULT: release it and the
robot parks in the learned hold. Pass ``--no-deadman`` to drive without it.

Losing the Bluetooth link parks the robot the same way -- the pad going out of
range or flat is the failure this guards, and the reason a stick beats a
latching key on hardware: nothing stays commanded that nobody is holding.

A policy trained with ``rel_march_envs = 1.0`` has NO hold: ``march == 0`` is an
input it has never seen, and on hardware feeding it is a fall, not a warning.
Such a policy parks in MARCH instead and the gait activation ramp is skipped.

Keep a hand near the robot and use a support rope for initial policy tests.
"""

from __future__ import annotations

import argparse
import math
import signal
from pathlib import Path

import onnxruntime as ort

from hardware import k2_bus
from hardware.k2_policy_run import (
  _channel5_contract,
  _read_metadata,
  parse_joint_trim,
  run,
)
from hardware.k2_stabilizer import UnsafeTiltError

from joystick import (
  JoystickCommand,
  find_pad,
  probe,
  supported_run_kwargs,
  wait_centered,
)


ROOT = Path(__file__).resolve().parents[1]


def _default_policy() -> Path:
  shipped = ROOT / "policies" / "last_policy.onnx"
  if shipped.is_file():
    return shipped
  policies = list((ROOT / "policies").glob("full*.onnx"))
  if not policies:
    policies = list((ROOT / "policies").glob("turn*.onnx"))
  if not policies:
    policies = list((ROOT / "policies").glob("bidir_symmetric*.onnx"))
  if not policies:
    policies = list((ROOT / "policies").glob("*.onnx"))
  if not policies:
    raise SystemExit("no ONNX policy found; pass --policy PATH")
  return max(policies, key=lambda path: path.stat().st_mtime)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--policy", type=Path, default=None)
  parser.add_argument("--bus", default="/dev/ttyAMA0")
  parser.add_argument("--probe", action="store_true",
                      help="print live pad values and exit; no servo torque")
  parser.add_argument("--device", default=None,
                      help="/dev/input/eventN to use instead of autodetecting")
  parser.add_argument("--name", default=None,
                      help="substring of the controller name to match")
  parser.add_argument("--deadzone", type=float, default=0.15,
                      help="stick deadzone as a fraction (default: 0.15)")
  parser.add_argument("--no-deadman", dest="deadman", action="store_false",
                      help="do not require the right trigger held to move")
  parser.set_defaults(deadman=True)
  parser.add_argument("--speed", type=float, default=0.02,
                      help="full-stick speed magnitude in m/s (default: 0.02)")
  parser.add_argument("--yaw-rate", type=float, default=0.10,
                      help="full-stick yaw magnitude in rad/s (default: 0.10). "
                      "Start well below the trained maximum on hardware.")
  parser.add_argument("--lateral-speed", type=float, default=0.01,
                      help="full-stick lateral magnitude in m/s; start at 0.01")
  parser.add_argument("--duration", type=float, default=600.0)
  parser.add_argument("--approach", type=float, default=5.0)
  parser.add_argument("--fall-angle", type=float, default=14.0)
  parser.add_argument("--slew", type=float, default=1.0)
  parser.add_argument("--balance-trim-right", type=float, default=0.0)
  parser.add_argument("--joint-trim", default=None,
                      help="per-joint mechanical trim in degrees, e.g. "
                           "'hip_yaw_R=+2.5,hip_yaw_L=-2.5'")
  parser.add_argument("--log", type=Path, default=None)
  parser.add_argument("--yes", action="store_true")
  args = parser.parse_args()
  for name in ("duration", "approach", "fall_angle", "slew"):
    if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
      raise SystemExit(f"--{name.replace('_', '-')} must be finite and positive")

  if not 0.0 <= args.deadzone < 0.9:
    raise SystemExit("--deadzone must be in [0, 0.9)")
  device = find_pad(args.name, args.device)
  if args.probe:
    return probe(device, args.deadzone)

  # Conservative hardware caps: half the twin's speed ceiling, and
  # a lateral ceiling the robot has actually been driven at.
  if not 0.0 < args.speed <= 0.05:
    raise SystemExit("--speed must be in (0, 0.05] m/s")
  if not 0.0 <= args.yaw_rate <= 0.25:
    raise SystemExit("--yaw-rate must be in [0, 0.25] rad/s")
  if not 0.0 < args.lateral_speed <= 0.025:
    raise SystemExit("--lateral-speed must be in (0, 0.025] m/s")
  if not -5.0 <= args.balance_trim_right <= 5.0:
    raise SystemExit("--balance-trim-right must be between -5 and +5 degrees")
  joint_trim = parse_joint_trim(args.joint_trim)   # validates before torque on
  policy = (args.policy or _default_policy()).resolve()
  if not policy.exists():
    raise SystemExit(f"missing policy: {policy}")

  session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
  (default_pos, action_scale, gait_freq, checkpoint, _neutral_shift,
   gait_dim, heading_dim, transition_time, velocity_ramp,
   yaw_ramp, height_ramp) = _read_metadata(session)
  if gait_dim not in (4, 5, 6, 7):
    raise SystemExit(f"policy is not signed-speed conditioned (gait dim {gait_dim})")

  meta = session.get_modelmeta().custom_metadata_map
  gait_channel5, lateral_ramp = _channel5_contract(session)
  supports_hold = float(meta.get("gait_rel_march_envs", "0.0")) < 1.0
  # On hardware, trust the exported training range
  # rather than the graph shape. A policy whose yaw range is zero was never
  # driven off-axis, whatever its tensor width allows.
  turning = max(abs(float(meta.get("yaw_rate_min_rps", "0.0"))),
                abs(float(meta.get("yaw_rate_max_rps", "0.0")))) > 0.0
  lateral = (gait_channel5 == "lateral_velocity" and int(meta.get("k2_pipeline_stage", "2")) >= 2
             and float(meta.get("gait_rel_lateral_envs", "1")) > 0)
  crouching = (gait_channel5 == "height_offset" or int(meta.get("k2_pipeline_stage", "0")) >= 3)

  print(f"Controller: {device.name} ({device.path})")
  print(f"Policy: {policy.name}; checkpoint: {checkpoint}")
  print(f"Gait: {gait_freq:g} Hz; transition: {transition_time:g} s; "
        f"velocity ramp: {velocity_ramp:g} m/s^2")
  if turning:
    print(f"Turning enabled; yaw ramp: {yaw_ramp:g} rad/s^2, "
          f"full right stick = {args.yaw_rate:g} rad/s")
  else:
    print("No yaw channel: the right stick is ignored.")
  if lateral:
    print(f"Lateral enabled; full left-stick X = {args.lateral_speed:g} m/s")
  else:
    print("No lateral channel: the left stick X axis is ignored.")
  if crouching:
    print(f"Crouch enabled; height ramp: {height_ramp:g} m/s "
          f"(LB deeper / RB rise / Y nominal, HOLD only)")
  if args.deadman:
    print("DEAD-MAN ACTIVE: hold the RIGHT TRIGGER for any motion; "
          "releasing it parks the robot.")
  else:
    print("Dead-man DISABLED (--no-deadman): the sticks move the robot "
          "with nothing held.")
  if supports_hold:
    print("Robot will start in HOLD. Keep it supported with room to move.")
  else:
    print("This policy has NO HOLD (trained with rel_march_envs = 1.0):")
    print("  it will START MARCHING IN PLACE. Support it before typing 'go'.")
  controller = JoystickCommand(
    device, args.speed, args.yaw_rate, turning,
    crouching=crouching, holding=supports_hold,
    lateral_speed=args.lateral_speed, lateral=lateral, skills=False,
    deadzone=args.deadzone, deadman=args.deadman,
  )
  if meta.get("gait_staged_height") == "1":
    controller.crouch_min = float(meta["height_offset_min_m"])
    controller.crouch_max = float(meta["height_offset_max_m"])
  # Before torque, not after: a stale off-centre axis must not become the
  # opening command.
  if not wait_centered(controller):
    print("aborted: controller not centered (or disconnected)")
    controller.device.close()
    return 1
  if not args.yes and input("Type 'go' to enable servo torque: ").strip().lower() != "go":
    print("aborted")
    controller.device.close()
    return 1
  bus = k2_bus.open_bus(args.bus)

  def request_stop(*_args) -> None:
    controller.stop_requested = True

  signal.signal(signal.SIGINT, request_stop)
  signal.signal(signal.SIGTERM, request_stop)
  try:
    controls = ["left stick Y = forward/backward"]
    if lateral:
      controls.append("left stick X = strafe")
    if turning:
      controls.append("right stick X = turn")
    controls.append("A = march" + ("  B = hold" if supports_hold else "  B = zero"))
    if crouching:
      controls.append("LB = crouch  RB = rise  Y = nominal")
    controls.append("Back / Ctrl-C = quit")
    print("Controls: " + "  |  ".join(controls))
    bus.torque(True)
    run(
      bus, session, default_pos, action_scale,
      mode="hold" if supports_hold else "march",
      duration=args.duration, slew_rad_s=args.slew,
      march_after=0.0, gait_freq_hz=gait_freq, approach_s=args.approach,
      fall_angle_deg=args.fall_angle, log=args.log,
      forward_speed=args.speed, heading_observation_dim=heading_dim,
      balance_trim_right_deg=args.balance_trim_right,
      # No hold means no handover to smooth, and the policy never saw an
      # intermediate 0 < march < 1 -- snap the activation instead.
      gait_transition_time_s=transition_time if supports_hold else 0.0,
      command_source=controller.snapshot,
      gait_velocity_ramp_rate_mps2=velocity_ramp,
      gait_yaw_rate_ramp_rate_rps2=yaw_ramp,
      gait_height_ramp_rate_mps=height_ramp,
      gait_lateral_ramp_rate_mps2=lateral_ramp,
      joint_trim=joint_trim,
      # Filtered rather than passed blind: the Pi's deployed k2_policy_run
      # predates gait_command_dim and infers the gait width from the
      # observation width instead (49 -> forward+yaw), which is the same
      # answer. Let the signature decide rather than this file guessing.
      **supported_run_kwargs({
        "gait_channel5": gait_channel5,
        "gait_command_dim": gait_dim,
      }),
    )
  except UnsafeTiltError as exc:
    print(f"\nSAFETY STOP: {exc}")
    return 2
  finally:
    try:
      bus.torque(False)
      print("Servo torque released.")
    finally:
      try:
        bus.close()
      finally:
        try:
          controller.device.close()
        except OSError:
          pass
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
