#!/usr/bin/env python3
"""Safely keyboard-drive a real K2 with a signed-speed ONNX policy.

Keys are read directly from the terminal (no Enter required):

  H  learned stable two-foot hold (also straightens the yaw command)
  M  march in place
  W  walk forward
  S  walk backward
  A  turn left      (from hold this starts a turn in place)
  D  turn right
  X  straighten -- cancel the yaw command, keep the current mode
  Q  stop, release servo torque, and quit

A/D/X are accepted only by a turning policy (5-D gait command).

The controller always starts in HOLD, eases into the policy crouch, applies
the ONNX transition/ramp metadata, and releases torque on exit or excess tilt.
Keep a hand near the robot and use a support rope for initial policy tests.
"""

from __future__ import annotations

import argparse
import select
import signal
import sys
import termios
from pathlib import Path

import onnxruntime as ort

from hardware import k2_bus
from hardware.k2_policy_run import _read_metadata, run
from hardware.k2_stabilizer import UnsafeTiltError


ROOT = Path(__file__).resolve().parent


class TerminalCommand:
  def __init__(self, speed: float, yaw_rate: float, turning: bool):
    self.mode = "hold"
    self.speed = abs(speed)
    self.yaw_magnitude = abs(yaw_rate)
    self.turning = turning
    self.yaw = 0.0
    self.stop_requested = False
    self._fd = sys.stdin.fileno()
    self._saved = None

  def __enter__(self):
    if not sys.stdin.isatty():
      raise SystemExit("real_keyboard.py requires an interactive terminal")
    self._saved = termios.tcgetattr(self._fd)
    raw = termios.tcgetattr(self._fd)
    raw[3] &= ~(termios.ICANON | termios.ECHO)
    raw[6][termios.VMIN] = 0
    raw[6][termios.VTIME] = 0
    termios.tcsetattr(self._fd, termios.TCSANOW, raw)
    return self

  def __exit__(self, *_args):
    if self._saved is not None:
      termios.tcsetattr(self._fd, termios.TCSANOW, self._saved)

  def _set_yaw(self, sign: int, label: str) -> None:
    if not self.turning:
      print("\n[!] this policy has no yaw channel; ignoring")
      return
    self.yaw = sign * self.yaw_magnitude
    if self.mode == "hold":
      # Turning is only meaningful once the gait is active; step in place.
      self.mode = "march"
    print(f"\n[{label}] YAW {self.yaw:+.2f} rad/s ({self.mode.upper()})")

  def _poll_key(self) -> None:
    while select.select([self._fd], [], [], 0.0)[0]:
      key = sys.stdin.read(1).lower()
      if key == "h":
        self.mode = "hold"
        self.yaw = 0.0
        print("\n[H] HOLD")
      elif key == "m":
        self.mode = "march"
        print("\n[M] MARCH")
      elif key == "w":
        self.mode = "walk"
        self.speed = abs(self.speed)
        print(f"\n[W] FORWARD {self.speed:+.3f} m/s")
      elif key == "s":
        self.mode = "walk"
        self.speed = -abs(self.speed)
        print(f"\n[S] BACKWARD {self.speed:+.3f} m/s")
      elif key == "a":
        self._set_yaw(+1, "A")
      elif key == "d":
        self._set_yaw(-1, "D")
      elif key == "x":
        self.yaw = 0.0
        print("\n[X] STRAIGHT")
      elif key in ("q", "\x03"):
        self.stop_requested = True
        print("\n[Q] STOP")

  def snapshot(self) -> tuple[str, float, float, bool]:
    self._poll_key()
    return self.mode, self.speed, self.yaw, self.stop_requested


def _default_policy() -> Path:
  policies = list((ROOT / "policies").glob("turn*.onnx"))
  if not policies:
    policies = list((ROOT / "policies").glob("bidir_symmetric*.onnx"))
  if not policies:
    policies = list((ROOT / "policies").glob("bidir*.onnx"))
  if not policies:
    raise SystemExit("no bidirectional ONNX policy found; pass --policy PATH")
  return max(policies, key=lambda path: path.stat().st_mtime)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--policy", type=Path, default=None)
  parser.add_argument("--bus", default="/dev/ttyAMA0")
  parser.add_argument("--speed", type=float, default=0.02,
                      help="W/S speed magnitude in m/s (default: 0.02)")
  parser.add_argument("--yaw-rate", type=float, default=0.10,
                      help="A/D yaw magnitude in rad/s (default: 0.10). Start "
                           "well below the trained maximum on hardware.")
  parser.add_argument("--duration", type=float, default=600.0)
  parser.add_argument("--approach", type=float, default=5.0)
  parser.add_argument("--fall-angle", type=float, default=14.0)
  parser.add_argument("--slew", type=float, default=1.0)
  parser.add_argument("--balance-trim-right", type=float, default=0.0)
  parser.add_argument("--log", type=Path, default=None)
  parser.add_argument("--yes", action="store_true")
  args = parser.parse_args()

  if not 0.0 < args.speed <= 0.05:
    raise SystemExit("--speed must be in (0, 0.05] m/s")
  if not 0.0 <= args.yaw_rate <= 0.25:
    raise SystemExit("--yaw-rate must be in [0, 0.25] rad/s")
  if not -5.0 <= args.balance_trim_right <= 5.0:
    raise SystemExit("--balance-trim-right must be between -5 and +5 degrees")
  policy = (args.policy or _default_policy()).resolve()
  if not policy.exists():
    raise SystemExit(f"missing policy: {policy}")

  session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
  (default_pos, action_scale, gait_freq, checkpoint, _neutral_shift,
   gait_dim, heading_dim, transition_time, velocity_ramp,
   yaw_ramp) = _read_metadata(session)
  if gait_dim not in (4, 5):
    raise SystemExit(f"policy is not signed-speed conditioned (gait dim {gait_dim})")

  print(f"Policy: {policy.name}; checkpoint: {checkpoint}")
  print(f"Gait: {gait_freq:g} Hz; transition: {transition_time:g} s; "
        f"velocity ramp: {velocity_ramp:g} m/s^2")
  if gait_dim == 5:
    print(f"Turning enabled; yaw ramp: {yaw_ramp:g} rad/s^2, "
          f"A/D command {args.yaw_rate:g} rad/s")
  print("Robot will start in HOLD. Keep it supported with room to move.")
  if not args.yes and input("Type 'go' to enable servo torque: ").strip().lower() != "go":
    print("aborted")
    return 1

  controller = TerminalCommand(args.speed, args.yaw_rate, gait_dim == 5)
  bus = k2_bus.open_bus(args.bus)

  def request_stop(*_args) -> None:
    controller.stop_requested = True

  signal.signal(signal.SIGINT, request_stop)
  try:
    with controller:
      bus.torque(True)
      keys = "H=hold  M=march  W=forward  S=backward"
      if gait_dim == 5:
        keys += "  A=left  D=right  X=straight"
      print(f"Keys: {keys}  Q=quit")
      run(
        bus, session, default_pos, action_scale,
        mode="hold", duration=args.duration, slew_rad_s=args.slew,
        march_after=0.0, gait_freq_hz=gait_freq, approach_s=args.approach,
        fall_angle_deg=args.fall_angle, log=args.log,
        forward_speed=args.speed, heading_observation_dim=heading_dim,
        balance_trim_right_deg=args.balance_trim_right,
        gait_transition_time_s=transition_time,
        command_source=controller.snapshot,
        gait_velocity_ramp_rate_mps2=velocity_ramp,
        gait_yaw_rate_ramp_rate_rps2=yaw_ramp,
      )
  except UnsafeTiltError as exc:
    print(f"\nSAFETY STOP: {exc}")
    return 2
  finally:
    try:
      bus.torque(False)
      print("Servo torque released.")
    finally:
      bus.close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
