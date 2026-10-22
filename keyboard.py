#!/usr/bin/env python3
"""Keyboard-drive an ONNX K2 policy in the deployment digital twin.

Keys are captured by the MuJoCo viewer window:

  H  learned stable two-foot hold (also straightens the yaw command)
     -- unavailable on a policy trained with no hold mode; H falls back to M
  M  march in place
  W  walk forward
  S  walk backward
  A  strafe left    (from hold this starts lateral stepping in place)
  D  strafe right
  Q  turn left      (from hold this starts a turn in place)
  E  turn right
  O  rotate body left from a two-foot stand (one learned IK-guided pulse)
  P  rotate body right from a two-foot stand (one learned IK-guided pulse)
  X  cancel the yaw and lateral commands, keep the current mode
  C  crouch deeper (hold only)   V  rise      Z  back to nominal height
     -- these need a 6-D gait policy; ignored otherwise
  Escape  quit

A/D require a policy with a lateral-velocity channel. Q/E require a policy with
a yaw channel. Unsupported commands are ignored rather than being fed to the
policy.

A policy trained with ``rel_march_envs = 1.0`` has NO hold: ``march == 0`` is an
input it has never seen, so asking for it is out of distribution and the robot
is likely to collapse. Such a policy starts in MARCH instead of HOLD, H maps to
M, and the gait activation ramp is skipped -- there is no hold<->walk handover
left to smooth, and the intermediate ``0 < march < 1`` is itself untrained.

This deliberately uses hardware.k2_policy_run.run, SimBus, encoder
quantization, servo gains/limits, latency, observation construction, and ONNX
Runtime.  It therefore exercises the deployment path rather than mjlab's
training viewer.
"""

from __future__ import annotations

import argparse
import math
import os
import select
import signal
import sys
import termios
import time
from pathlib import Path

import onnxruntime as ort

from hardware import k2_bus
from hardware.k2_policy_run import (_channel5_contract, _read_metadata,
                                    height_active_in_march, parse_joint_trim, run)
from hardware.k2_stabilizer import UnsafeTiltError


ROOT = Path(__file__).resolve().parent
class KeyboardCommand:
  def __init__(self, speed: float, yaw_rate: float, turning: bool,
               crouching: bool = False, holding: bool = True,
               lateral_speed: float = 0.015, lateral: bool = False,
               crouch_range: tuple[float, float] = (-0.045, 0.006),
               crouch_in_march: bool = False,
               standing_rotation: bool = False,
               standing_rotate_yaw_rate: float = 0.003):
    self.holding = holding
    self.mode = "hold" if holding else "march"
    self.speed = speed
    self.yaw_magnitude = abs(yaw_rate)
    self.turning = turning
    self.crouching = crouching
    self.lateral = lateral
    self.lateral_magnitude = abs(lateral_speed)
    self.yaw = 0.0
    self.height = 0.0
    self.lateral_speed = 0.0
    self.crouch_step = 0.015
    # Clamp to what the policy was actually TRAINED on rather than a fixed
    # historical -45 mm: the merged walk/turn/height task deliberately trains a
    # shallower range so the knee keeps flexion headroom for the swing, and
    # driving past it asks for a height the policy has never seen.
    self.crouch_min, self.crouch_max = crouch_range
    self.crouch_in_march = crouch_in_march
    self.standing_rotation = standing_rotation
    self.standing_rotate_magnitude = abs(standing_rotate_yaw_rate)
    self.standing_yaw = 0.0
    # A monotonically increasing token makes each O/P key press a distinct
    # one-shot request even when the same direction is pressed twice.
    self.rotation_request_id = 0
    self.stop_requested = False

  def _cancel_standing_rotation(self) -> None:
    if self.standing_yaw:
      self.standing_yaw = 0.0
      self.rotation_request_id += 1

  def _set_standing_rotation(self, sign: int, label: str) -> None:
    if not self.standing_rotation:
      print("\n[!] this policy has no standing-rotation skill; ignoring")
      return
    if not self.holding:
      print("\n[!] this policy has no trained hold; ignoring standing rotation")
      return
    # Queue a one-shot pulse behind the hold transition. The deployment loop
    # will not expose its rotation clock or yaw until gait activation is zero.
    self.mode = "hold"
    self.yaw = 0.0
    self.lateral_speed = 0.0
    self.standing_yaw = sign * self.standing_rotate_magnitude
    self.rotation_request_id += 1
    print(f"\n[{label}] STANDING ROTATE "
          f"{'LEFT' if sign > 0 else 'RIGHT'} (queued after full HOLD)")

  def _set_yaw(self, sign: int, label: str) -> None:
    if not self.turning:
      print("\n[!] this policy has no yaw channel; ignoring")
      return
    # Switching from a lateral command to yaw should test turning alone. When
    # already walking, retain WALK so Q/E can still produce a forward arc.
    self.lateral_speed = 0.0
    self.yaw = sign * self.yaw_magnitude
    if self.mode == "hold":
      # Turning is only meaningful once the gait is active; step in place.
      self.mode = "march"
    print(f"\n[{label}] YAW {self.yaw:+.2f} rad/s ({self.mode.upper()})")

  def _set_lateral(self, sign: int, label: str) -> None:
    if not self.lateral:
      print("\n[!] this policy has no lateral channel; ignoring")
      return
    # Lateral keys are cardinal commands, not additions to the previous walk
    # command: MARCH gives vx=0 while the lateral channel supplies vy.
    self.mode = "march"
    self.yaw = 0.0
    self.lateral_speed = sign * self.lateral_magnitude
    print(f"\n[{label}] LATERAL {self.lateral_speed:+.3f} m/s")

  def on_key(self, keycode: int) -> None:
    key = chr(keycode).lower() if 0 <= keycode < 256 else ""
    if key == "h":
      self._cancel_standing_rotation()
      if not self.holding:
        # march == 0 is an untrained input for this policy; do not emit it.
        self.yaw = 0.0
        self.lateral_speed = 0.0
        self.mode = "march"
        print("\n[!] this policy has no hold mode; using MARCH instead")
        return
      self.mode = "hold"
      self.yaw = 0.0
      self.lateral_speed = 0.0
      print("\n[H] HOLD")
    elif key == "m":
      self._cancel_standing_rotation()
      self.mode = "march"
      print("\n[M] MARCH")
    elif key == "w":
      self._cancel_standing_rotation()
      self.mode = "walk"
      self.speed = abs(self.speed)
      print(f"\n[W] FORWARD {self.speed:+.3f} m/s")
    elif key == "s":
      self._cancel_standing_rotation()
      self.mode = "walk"
      self.speed = -abs(self.speed)
      print(f"\n[S] BACKWARD {self.speed:+.3f} m/s")
    elif key == "a":
      self._cancel_standing_rotation()
      self._set_lateral(+1, "A")
    elif key == "d":
      self._cancel_standing_rotation()
      self._set_lateral(-1, "D")
    elif key == "q":
      self._cancel_standing_rotation()
      self._set_yaw(+1, "Q")
    elif key == "e":
      self._cancel_standing_rotation()
      self._set_yaw(-1, "E")
    elif key == "o":
      self._set_standing_rotation(+1, "O")
    elif key == "p":
      self._set_standing_rotation(-1, "P")
    elif key == "x":
      self._cancel_standing_rotation()
      self.yaw = 0.0
      self.lateral_speed = 0.0
      print("\n[X] ZERO YAW / LATERAL")
    elif key in ("c", "v", "z") and not self.crouching:
      print("\n[!] this policy has no height channel; ignoring")
    elif key == "c":
      self.height = max(self.height - self.crouch_step, self.crouch_min)
      print(f"\n[C] CROUCH {self.height*1000:+.0f} mm"
            + ("" if (self.crouch_in_march or self.mode == "hold")
               else "  (applies in HOLD)"))
    elif key == "v":
      self.height = min(self.height + self.crouch_step, self.crouch_max)
      print(f"\n[V] RISE {self.height*1000:+.0f} mm"
            + ("" if (self.crouch_in_march or self.mode == "hold")
               else "  (applies in HOLD)"))
    elif key == "z":
      self.height = 0.0
      print("\n[Z] NOMINAL HEIGHT")
    elif keycode == 256:  # GLFW_KEY_ESCAPE
      self.stop_requested = True
      print("\n[ESC] QUIT")

  def snapshot(self) -> tuple[str, float, float, float, float, bool, float, int]:
    return (self.mode, self.speed, self.yaw, self.height,
            self.lateral_speed, self.stop_requested,
            self.standing_yaw, self.rotation_request_id)


class TerminalKeyboard(KeyboardCommand):
  """Use the same command state with single-key terminal input on hardware."""
  def __enter__(self):
    if not sys.stdin.isatty():
      raise SystemExit("hardware mode requires an interactive terminal")
    self._fd = sys.stdin.fileno()
    self._saved = termios.tcgetattr(self._fd)
    raw = termios.tcgetattr(self._fd)
    raw[3] &= ~(termios.ICANON | termios.ECHO)
    raw[6][termios.VMIN] = raw[6][termios.VTIME] = 0
    termios.tcsetattr(self._fd, termios.TCSANOW, raw)
    return self

  def __exit__(self, *_args):
    termios.tcsetattr(self._fd, termios.TCSANOW, self._saved)

  def snapshot(self):
    while select.select([self._fd], [], [], 0)[0]:
      value = os.read(self._fd, 1).decode('ascii', errors='ignore')
      if not value:
        self.stop_requested = True
        break
      self.on_key(256 if value == "\x1b" else ord(value))
    return super().snapshot()


def _default_policy() -> Path:
  shipped = ROOT / "policies" / "last_policy.onnx"
  if shipped.is_file():
    return shipped
  # Prefer the newest exported bidirectional policy. Fall back to any ONNX so
  # the script remains useful for explicitly compatible future policies.
  policies = list((ROOT / "policies").glob("turn*.onnx"))
  if not policies:
    policies = list((ROOT / "policies").glob("bidir*.onnx"))
  if not policies:
    policies = list((ROOT / "policies").glob("*.onnx"))
  if not policies:
    raise SystemExit("no ONNX policy found; pass --policy PATH")
  return max(policies, key=lambda path: path.stat().st_mtime)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--policy", type=Path, default=None)
  parser.add_argument("--bus", default="sim", help="sim or serial port, e.g. /dev/ttyAMA0")
  parser.add_argument("--yes", action="store_true", help="skip hardware torque confirmation")
  parser.add_argument("--approach", type=float, default=5.0)
  parser.add_argument("--log", type=Path)
  parser.add_argument("--balance-trim-right", type=float, default=0.0)
  parser.add_argument("--joint-trim")
  parser.add_argument("--speed", type=float, default=0.02,
                      help="W/S speed magnitude in m/s (default: 0.02)")
  parser.add_argument("--yaw-rate", type=float, default=0.15,
                      help="Q/E yaw magnitude in rad/s (default: 0.15)")
  parser.add_argument("--lateral-speed", type=float, default=0.015,
                      help="A/D lateral magnitude in m/s")
  parser.add_argument("--duration", type=float, default=600.0,
                      help="maximum runtime in seconds (default: 600)")
  parser.add_argument("--fall-angle", type=float, default=None,
                      help="tilt cutoff: default 45 degrees in sim, 14 on hardware")
  parser.add_argument("--slew", type=float, default=1.0)
  args = parser.parse_args()
  real = args.bus != "sim"
  if args.fall_angle is None:
    args.fall_angle = 14.0 if real else 45.0
  for name in ("duration", "approach", "fall_angle", "slew"):
    if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
      raise SystemExit(f"--{name.replace('_', '-')} must be finite and positive")
  if not -5 <= args.balance_trim_right <= 5:
    raise SystemExit("--balance-trim-right must be between -5 and +5 degrees")
  joint_trim = parse_joint_trim(args.joint_trim)
  if real and (args.speed > .05 or args.yaw_rate > .30 or args.lateral_speed > .015):
    raise SystemExit("Hardware limits: speed <= .05, yaw <= .30, lateral <= .015")
  if real and not sys.stdin.isatty():
    raise SystemExit("hardware mode requires an interactive terminal")

  if not 0.0 < args.speed <= 0.10:
    raise SystemExit("--speed must be in (0, 0.10] m/s")
  if not 0.0 <= args.yaw_rate <= 0.35:
    raise SystemExit("--yaw-rate must be in [0, 0.35] rad/s")
  if not 0.0 < args.lateral_speed <= 0.05:
    raise SystemExit("--lateral-speed must be in (0, 0.05] m/s")
  policy = (args.policy or _default_policy()).resolve()
  if not policy.exists():
    raise SystemExit(f"missing policy: {policy}")

  session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
  (default_pos, action_scale, gait_freq, checkpoint, neutral_shift,
   gait_dim, heading_dim, transition_time, velocity_ramp,
   yaw_ramp, height_ramp) = _read_metadata(session)
  if gait_dim not in (4, 5, 6, 7):
    raise SystemExit(
      f"{policy.name} is not signed-speed conditioned (gait dimension {gait_dim})"
    )

  # Read straight from the ONNX rather than widening `_read_metadata`, whose
  # tuple is unpacked positionally in four different scripts. Absent on every
  # policy exported before this flag existed, all of which did train a hold.
  meta = session.get_modelmeta().custom_metadata_map
  gait_channel5, lateral_ramp = _channel5_contract(session)
  # A walk+turn+height policy commands height in BOTH modes; without
  # this the runner reproduces the historical (1 - march) gate and the
  # C/V keys go dead the instant the robot starts marching.
  height_in_march = height_active_in_march(session)
  supports_hold = float(meta.get("gait_rel_march_envs", "0.0")) < 1.0
  standing_rotation = meta.get("gait_standing_rotation", "0") == "1"
  standing_rotate_period = float(
    meta.get("gait_standing_rotate_period_s", "0.0")
  )
  standing_rotate_yaw_rate = float(
    meta.get("gait_standing_rotate_yaw_rate_rps", "0.0")
  )
  # The lateral policy is widened from the turning policy and still has the
  # full yaw-conditioned observation contract. Its lateral fine-tuning config
  # pins wz=0, so the honest exported training range is zero even though the
  # inherited policy still accepts the yaw input. Use the graph contract here;
  # the metadata range describes fine-tuning coverage, not tensor capability.
  turning = gait_dim in (5, 6, 7) and heading_dim == 2
  lateral = (gait_channel5 == "lateral_velocity" and int(meta.get("k2_pipeline_stage", "2")) >= 2
             and float(meta.get("gait_rel_lateral_envs", "1")) > 0)
  crouching = (gait_channel5 == "height_offset" or int(meta.get("k2_pipeline_stage", "0")) >= 3)
  crouch_range = (float(meta.get("height_offset_min_m", "-0.045")),
                  float(meta.get("height_offset_max_m", "0.006")))
  controller = KeyboardCommand(args.speed, args.yaw_rate,
                                turning, crouching,
                                holding=supports_hold,
                                lateral_speed=args.lateral_speed,
                                lateral=lateral,
                                crouch_range=crouch_range,
                                crouch_in_march=height_in_march,
                                standing_rotation=standing_rotation,
                                standing_rotate_yaw_rate=standing_rotate_yaw_rate)
  if real:
    controller = TerminalKeyboard(args.speed, args.yaw_rate, turning, crouching,
      holding=supports_hold, lateral_speed=args.lateral_speed, lateral=lateral,
      crouch_range=crouch_range, crouch_in_march=height_in_march,
      standing_rotation=standing_rotation,
      standing_rotate_yaw_rate=standing_rotate_yaw_rate)
    if not args.yes and input("Type 'go' to enable servo torque: ").strip().lower() != "go":
      return 1
    bus = k2_bus.open_bus(args.bus)
  else:
    bus = k2_bus.open_bus("sim", viewer=True, realtime=True, key_callback=controller.on_key)

  def request_stop(*_args) -> None:
    controller.stop_requested = True

  signal.signal(signal.SIGINT, request_stop)
  signal.signal(signal.SIGTERM, request_stop)
  print(f"Policy: {policy}")
  print(f"Checkpoint: {checkpoint}; gait: {gait_freq:g} Hz; "
        f"transition: {transition_time:g} s; ramp: {velocity_ramp:g} m/s^2")
  keys = ("H=hold  M=march" if supports_hold else "M=march (no hold mode)")
  keys += "  W=forward  S=backward"
  if lateral:
    keys += "  A=strafe-left  D=strafe-right"
  if turning:
    keys += "  Q=turn-left  E=turn-right"
  if standing_rotation:
    keys += "  O=stand-rotate-left  P=stand-rotate-right"
  if lateral or turning:
    keys += "  X=zero steering"
  if crouching:
    keys += ("  C=crouch  V=rise  Z=nominal"
             + ("" if height_in_march else " (hold only)"))
  print(f"{'Terminal keys' if real else 'Click the MuJoCo window, then press'}: {keys}  Esc=quit")
  if supports_hold:
    print("Starting safely in HOLD.")
  else:
    print("This policy has NO HOLD (trained with rel_march_envs = 1.0):")
    print("  starting in MARCH, and the gait activation ramp is skipped.")

  context = controller if real else None
  entered = False
  try:
    if context:
      context.__enter__()
      entered = True
    bus.torque(True)
    deadline = time.monotonic() + args.duration
    while not controller.stop_requested and time.monotonic() < deadline:
      try:
        run(
          bus, session, default_pos, action_scale,
          mode="hold" if supports_hold else "march",
          duration=max(0.0, deadline - time.monotonic()),
          slew_rad_s=args.slew, march_after=0.0, gait_freq_hz=gait_freq,
          approach_s=args.approach if real else 2.0,
          fall_angle_deg=args.fall_angle,
          log=args.log, balance_trim_right_deg=args.balance_trim_right,
          joint_trim=joint_trim,
          forward_speed=args.speed, heading_observation_dim=heading_dim,
          # With no hold there is no handover to smooth, and the policy never
          # saw an intermediate 0 < march < 1 -- snap the activation instead.
          gait_transition_time_s=transition_time if supports_hold else 0.0,
          command_source=controller.snapshot,
          gait_velocity_ramp_rate_mps2=velocity_ramp,
          gait_yaw_rate_ramp_rate_rps2=yaw_ramp,
          gait_height_ramp_rate_mps=height_ramp,
          gait_lateral_ramp_rate_mps2=lateral_ramp,
          gait_channel5=gait_channel5,
          gait_height_active_in_march=height_in_march,
          standing_rotate_period_s=standing_rotate_period,
        )
      except UnsafeTiltError as exc:
        if real:
          print(f"\nSAFETY STOP: {exc}")
          return 2
        print(f"\nSIM FALL: {exc}")
        if controller.stop_requested:
          break
        print("Auto-resetting twin; keys stay live during the 2 s approach.")
        bus.reset()
  finally:
    try:
      bus.torque(False)
    finally:
      try:
        bus.close()
      finally:
        if entered:
          context.__exit__(None, None, None)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
