#!/usr/bin/env python3
"""Keyboard-drive an ONNX K2 policy in the deployment digital twin.

Keys are captured by the MuJoCo viewer window:

  H  learned stable two-foot hold
  M  march in place
  W  walk forward
  S  walk backward
  Q / Escape  quit

This deliberately uses hardware.k2_policy_run.run, SimBus, encoder
quantization, servo gains/limits, latency, observation construction, and ONNX
Runtime.  It therefore exercises the deployment path rather than mjlab's
training viewer.
"""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import onnxruntime as ort

from hardware import k2_bus
from hardware.k2_policy_run import _read_metadata, run
from hardware.k2_stabilizer import UnsafeTiltError


ROOT = Path(__file__).resolve().parent
class KeyboardCommand:
  def __init__(self, speed: float):
    self.mode = "hold"
    self.speed = speed
    self.stop_requested = False

  def on_key(self, keycode: int) -> None:
    key = chr(keycode).lower() if 0 <= keycode < 256 else ""
    if key == "h":
      self.mode = "hold"
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
    elif key == "q" or keycode == 256:  # GLFW_KEY_ESCAPE
      self.stop_requested = True
      print("\n[Q] QUIT")

  def snapshot(self) -> tuple[str, float, bool]:
    return self.mode, self.speed, self.stop_requested


def _default_policy() -> Path:
  # Prefer the newest exported bidirectional policy. Fall back to any ONNX so
  # the script remains useful for explicitly compatible future policies.
  policies = list((ROOT / "policies").glob("bidir*.onnx"))
  if not policies:
    policies = list((ROOT / "policies").glob("*.onnx"))
  if not policies:
    raise SystemExit("no ONNX policy found; pass --policy PATH")
  return max(policies, key=lambda path: path.stat().st_mtime)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--policy", type=Path, default=None)
  parser.add_argument("--speed", type=float, default=0.02,
                      help="W/S speed magnitude in m/s (default: 0.02)")
  parser.add_argument("--duration", type=float, default=600.0,
                      help="maximum runtime in seconds (default: 600)")
  parser.add_argument("--fall-angle", type=float, default=45.0,
                      help="simulation safety stop angle (default: 45 deg)")
  parser.add_argument("--slew", type=float, default=1.0)
  args = parser.parse_args()

  if not 0.0 < args.speed <= 0.10:
    raise SystemExit("--speed must be in (0, 0.10] m/s")
  policy = (args.policy or _default_policy()).resolve()
  if not policy.exists():
    raise SystemExit(f"missing policy: {policy}")

  session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
  (default_pos, action_scale, gait_freq, checkpoint, neutral_shift,
   gait_dim, heading_dim, transition_time, velocity_ramp) = _read_metadata(session)
  if gait_dim != 4:
    raise SystemExit(
      f"{policy.name} is not signed-speed conditioned (gait dimension {gait_dim})"
    )

  controller = KeyboardCommand(args.speed)
  bus = k2_bus.open_bus(
    "sim", viewer=True, realtime=True, key_callback=controller.on_key
  )

  def request_stop(*_args) -> None:
    controller.stop_requested = True

  signal.signal(signal.SIGINT, request_stop)
  print(f"Policy: {policy}")
  print(f"Checkpoint: {checkpoint}; gait: {gait_freq:g} Hz; "
        f"transition: {transition_time:g} s; ramp: {velocity_ramp:g} m/s^2")
  print("Click the MuJoCo window, then press: "
        "H=hold  M=march  W=forward  S=backward  Q/Esc=quit")
  print("Starting safely in HOLD.")

  try:
    bus.torque(True)
    deadline = time.monotonic() + args.duration
    while not controller.stop_requested and time.monotonic() < deadline:
      try:
        run(
          bus, session, default_pos, action_scale,
          mode="hold", duration=max(0.0, deadline - time.monotonic()),
          slew_rad_s=args.slew, march_after=0.0, gait_freq_hz=gait_freq,
          approach_s=2.0, fall_angle_deg=args.fall_angle,
          forward_speed=args.speed, heading_observation_dim=heading_dim,
          gait_transition_time_s=transition_time,
          command_source=controller.snapshot,
          gait_velocity_ramp_rate_mps2=velocity_ramp,
        )
      except UnsafeTiltError as exc:
        print(f"\nSIM FALL: {exc}")
        if controller.stop_requested:
          break
        print("Auto-resetting twin; press H/M/W/S during the 2 s approach.")
        bus.reset()
  finally:
    try:
      bus.torque(False)
    finally:
      bus.close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
