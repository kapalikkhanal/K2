#!/usr/bin/env python3
"""Joystick-drive an ONNX K2 policy in the deployment digital twin.

An Xbox (or any evdev) gamepad replaces the keyboard of ``keyboard.py``. The
sticks are analog and momentary; the buttons latch a resting mode:

  Left stick  Y   walk forward / backward   (analog, up = forward)
  Left stick  X   strafe left / right       (analog, needs a lateral channel)
  Right stick X   turn left / right         (analog, needs a yaw channel)
  A               MARCH in place (resting mode) and clear any skill
  B               HOLD -- stop, straighten the yaw command (resting mode)
  LB / RB         crouch deeper / rise      Y-button: nominal height
  D-pad L/R/U/D   skills 1/2/3/4 (balance left, balance right, kick, kick)
  Back / View     quit

Centering the sticks returns to the resting mode, so letting go of the pad
stops the robot -- the reason a stick is safer than a latching key. Everything
else matches ``keyboard.py``: the same command tuple, the same magnitude caps,
and the same ``hardware.k2_policy_run.run`` deployment path, so this exercises
encoder quantization, servo gains, latency and ONNX Runtime rather than
mjlab's training viewer.

``run`` already rate-limits every command target (velocity, yaw, lateral and
height ramps come from the ONNX metadata), so the raw analog value is fed
straight in as the target and the ramp does the smoothing.

Lateral and yaw are kept mutually exclusive -- whichever stick is pushed
further wins -- because that combination is outside what the policies were
trained on; ``keyboard.py`` enforces the same exclusivity between A/D and Q/E.

Run ``--probe`` first to see the raw axis and button values with no policy and
no simulation, which is also how you confirm the pad is connected and readable.

A policy trained with ``rel_march_envs = 1.0`` has NO hold: ``march == 0`` is an
input it has never seen, so B rests in MARCH instead and the gait activation
ramp is skipped.
"""

from __future__ import annotations

import argparse
import inspect
import math
import select
import signal
import sys
import time
from pathlib import Path

import onnxruntime as ort

from hardware import k2_bus
from hardware.k2_policy_run import (_channel5_contract, _read_metadata,
                                    height_active_in_march, run)
from hardware.k2_stabilizer import UnsafeTiltError


ROOT = Path(__file__).resolve().parent

# Reopening costs one directory scan; at 50 Hz a tighter retry would spam it.
RECONNECT_PERIOD_S = 1.0
# A light squeeze should arm: the dead-man exists to notice a dropped pad, not
# to demand a firm pull before the robot will answer the stick at all.
TRIGGER_ARM = 0.15
NAG_PERIOD_S = 2.0
SKILL_LABELS = ("LOCOMOTION", "LIFT LEFT / BALANCE ON RIGHT",
                "LIFT RIGHT / BALANCE ON LEFT", "KICK LEFT", "KICK RIGHT")


def _evdev():
  try:
    import evdev
  except ImportError:  # pragma: no cover - environment guard
    raise SystemExit(
      "python-evdev is required: pip install evdev\n"
      "(the Pi's venv does not have it yet; it does have pygame)"
    )
  return evdev


def find_pad(name_hint: str | None = None, path: str | None = None):
  """Return an open InputDevice for the first gamepad, or raise SystemExit."""
  evdev = _evdev()
  ecodes = evdev.ecodes
  if path is not None:
    candidates = [path]
  else:
    candidates = evdev.list_devices()
  found, rejected = [], []
  for dev_path in sorted(candidates):
    try:
      dev = evdev.InputDevice(dev_path)
    except PermissionError:
      rejected.append(dev_path)
      continue
    except OSError:
      continue
    caps = dev.capabilities()
    abs_codes = {code for code, _ in caps.get(ecodes.EV_ABS, [])}
    key_codes = set(caps.get(ecodes.EV_KEY, []))
    # A gamepad is the thing with a left stick and a south face button. This
    # rejects the touchpad (no BTN_SOUTH) and the keyboard (no ABS_X).
    is_pad = ecodes.ABS_X in abs_codes and ecodes.BTN_SOUTH in key_codes
    if is_pad and (name_hint is None or name_hint.lower() in dev.name.lower()):
      found.append(dev)
    else:
      dev.close()
  if not found:
    detail = ""
    if rejected:
      detail = (f"\n{len(rejected)} input device(s) were unreadable "
                "(permission denied).\nAdd yourself to the input group:\n"
                "  sudo usermod -aG input $USER\n"
                "then log out and back in (or: newgrp input).")
    raise SystemExit(
      "no gamepad found on /dev/input.\n"
      "Pair and connect the controller to THIS machine first:\n"
      "  bluetoothctl -- scan on ; pair <MAC> ; trust <MAC> ; connect <MAC>\n"
      "Then check that it appears:  ls /dev/input/event*" + detail
    )
  for extra in found[1:]:
    extra.close()
  return found[0]


class PadLayout:
  """Which absolute axes carry the right stick and the triggers.

  Xbox pads enumerate differently depending on the driver that claims them:
  ``xpad``/``xpadneo`` expose the right stick as ABS_RX/ABS_RY and the triggers
  as ABS_Z/ABS_RZ, while a plain Bluetooth HID binding shifts the right stick
  down onto ABS_Z/ABS_RZ and puts the triggers on ABS_BRAKE/ABS_GAS. Detect it
  from the reported capabilities rather than guessing from the device name.
  """

  def __init__(self, device):
    ecodes = _evdev().ecodes
    self.info = {code: info
                 for code, info in device.capabilities().get(ecodes.EV_ABS, [])}
    codes = set(self.info)
    if ecodes.ABS_RX in codes:
      self.name = "xpad"
      self.right_x, self.right_y = ecodes.ABS_RX, ecodes.ABS_RY
      self.trigger_l, self.trigger_r = ecodes.ABS_Z, ecodes.ABS_RZ
    else:
      self.name = "bluetooth-hid"
      self.right_x, self.right_y = ecodes.ABS_Z, ecodes.ABS_RZ
      self.trigger_l, self.trigger_r = ecodes.ABS_BRAKE, ecodes.ABS_GAS
    self.left_x, self.left_y = ecodes.ABS_X, ecodes.ABS_Y
    self.hat_x, self.hat_y = ecodes.ABS_HAT0X, ecodes.ABS_HAT0Y

  def normalize(self, code: int, raw: int, deadzone: float) -> float:
    """Map a raw axis reading to [-1, 1] with the deadzone scaled out."""
    info = self.info.get(code)
    if info is None or info.max == info.min:
      return 0.0
    center = 0.5 * (info.max + info.min)
    half = 0.5 * (info.max - info.min)
    value = (raw - center) / half
    if abs(value) <= deadzone:
      return 0.0
    # Rescale so the command starts at zero at the deadzone edge instead of
    # jumping to `deadzone` worth of speed the moment the stick breaks out.
    return math.copysign((abs(value) - deadzone) / (1.0 - deadzone), value)

  def trigger(self, code: int, raw: int) -> float:
    """Map a trigger to [0, 1]; unpressed rests at its axis minimum."""
    info = self.info.get(code)
    if info is None or info.max == info.min:
      return 0.0
    return float(min(1.0, max(0.0, (raw - info.min) / (info.max - info.min))))


class JoystickCommand:
  """Turn pad state into the command tuple `k2_policy_run.run` consumes."""

  def __init__(self, device, speed: float, yaw_rate: float, turning: bool,
               crouching: bool = False, holding: bool = True,
               lateral_speed: float = 0.015, lateral: bool = False,
               skills: bool = False, deadzone: float = 0.15,
               deadman: bool = False, verbose: bool = True):
    self.device = device
    self.layout = PadLayout(device)
    self.holding = holding
    self.turning = turning
    self.crouching = crouching
    self.lateral = lateral
    self.skills = skills
    self.deadzone = deadzone
    self.deadman = deadman
    self.verbose = verbose
    self.speed_magnitude = abs(speed)
    self.yaw_magnitude = abs(yaw_rate)
    self.lateral_magnitude = abs(lateral_speed)
    self.crouch_step = 0.015
    self.crouch_min = -0.045
    self.crouch_max = 0.006
    # Resting mode: what the robot does with every stick centered.
    self.base_mode = "hold" if holding else "march"
    self.mode = self.base_mode
    self.speed = 0.0
    self.yaw = 0.0
    self.lateral_speed = 0.0
    self.height = 0.0
    self.skill = 0
    self.stop_requested = False
    self.connected = True
    self.armed = not deadman
    self._raw: dict[int, int] = {}
    self._next_reconnect = 0.0
    self._next_nag = 0.0
    self._warned_yaw = False
    self._warned_lateral = False
    for code, info in self.layout.info.items():
      self._raw[code] = info.value

  # -- device I/O -----------------------------------------------------------

  def _drain(self) -> None:
    """Absorb every pending event without blocking the 50 Hz control loop."""
    evdev = _evdev()
    ecodes = evdev.ecodes
    try:
      while select.select([self.device.fd], [], [], 0.0)[0]:
        for event in self.device.read():
          if event.type == ecodes.EV_ABS:
            self._raw[event.code] = event.value
          elif event.type == ecodes.EV_KEY and event.value == 1:
            self._on_button(event.code)
    except BlockingIOError:
      pass
    except OSError as exc:
      self._on_disconnect(exc)

  def _on_disconnect(self, exc: BaseException) -> None:
    """Losing the pad must stop the robot, not freeze the last command."""
    if self.connected:
      print(f"\n[!] controller disconnected ({exc}); stopping and holding")
    self.connected = False
    self.mode = self.base_mode = "hold" if self.holding else "march"
    self.speed = self.yaw = self.lateral_speed = 0.0
    self.skill = 0
    try:
      self.device.close()
    except OSError:
      pass

  def _try_reconnect(self) -> None:
    now = time.monotonic()
    if now < self._next_reconnect:
      return
    self._next_reconnect = now + RECONNECT_PERIOD_S
    evdev = _evdev()
    for dev_path in sorted(evdev.list_devices()):
      try:
        dev = evdev.InputDevice(dev_path)
      except OSError:
        continue
      caps = dev.capabilities()
      abs_codes = {code for code, _ in caps.get(evdev.ecodes.EV_ABS, [])}
      if (evdev.ecodes.ABS_X in abs_codes
          and evdev.ecodes.BTN_SOUTH in set(caps.get(evdev.ecodes.EV_KEY, []))):
        self.device = dev
        self.layout = PadLayout(dev)
        self._raw = {code: info.value for code, info in self.layout.info.items()}
        self.connected = True
        print(f"\n[+] controller reconnected: {dev.name} ({dev.path})")
        return
      dev.close()

  # -- button edges ---------------------------------------------------------

  def _say(self, message: str) -> None:
    if self.verbose:
      print(f"\n{message}")

  def _on_button(self, code: int) -> None:
    ecodes = _evdev().ecodes
    if code == ecodes.BTN_SOUTH:  # A
      self.skill = 0
      self.base_mode = "march"
      self._say("[A] MARCH")
    elif code == ecodes.BTN_EAST:  # B
      self.skill = 0
      self.height = 0.0
      if self.holding:
        self.base_mode = "hold"
        self._say("[B] HOLD")
      else:
        # march == 0 is an untrained input for this policy; never emit it.
        self.base_mode = "march"
        self._say("[B] ZERO / MARCH (this policy has no hold mode)")
    elif code in (ecodes.BTN_NORTH, ecodes.BTN_WEST):  # X / Y face buttons
      if not self.crouching:
        return
      self.height = 0.0
      self._say("[Y] NOMINAL HEIGHT")
    elif code == ecodes.BTN_TL:  # LB
      if not self.crouching:
        self._say("[!] this policy has no height channel; ignoring")
        return
      self.height = max(self.height - self.crouch_step, self.crouch_min)
      self._say(f"[LB] CROUCH {self.height * 1000:+.0f} mm"
                + ("" if self.mode == "hold" else "  (applies in HOLD)"))
    elif code == ecodes.BTN_TR:  # RB
      if not self.crouching:
        self._say("[!] this policy has no height channel; ignoring")
        return
      self.height = min(self.height + self.crouch_step, self.crouch_max)
      self._say(f"[RB] RISE {self.height * 1000:+.0f} mm"
                + ("" if self.mode == "hold" else "  (applies in HOLD)"))
    elif code in (ecodes.BTN_SELECT, ecodes.BTN_MODE):
      self.stop_requested = True
      self._say("[BACK] QUIT")

  def _poll_hat_skill(self) -> None:
    """D-pad selects a skill; it is a hat axis, not a button, on every pad."""
    if not self.skills:
      return
    hat_x = self._raw.get(self.layout.hat_x, 0)
    hat_y = self._raw.get(self.layout.hat_y, 0)
    skill = 0
    if hat_x < 0:
      skill = 1
    elif hat_x > 0:
      skill = 2
    elif hat_y < 0:
      skill = 3
    elif hat_y > 0:
      skill = 4
    if skill and skill != self.skill:
      self.skill = skill
      self.yaw = 0.0
      self.lateral_speed = 0.0
      self._say(f"[D-PAD] {SKILL_LABELS[skill]}")

  # -- command assembly -----------------------------------------------------

  def _update(self) -> None:
    layout, dz = self.layout, self.deadzone
    forward = -layout.normalize(layout.left_y, self._raw.get(layout.left_y, 0), dz)
    strafe = -layout.normalize(layout.left_x, self._raw.get(layout.left_x, 0), dz)
    turn = -layout.normalize(layout.right_x, self._raw.get(layout.right_x, 0), dz)

    self.armed = True
    if self.deadman:
      pull = layout.trigger(layout.trigger_r,
                            self._raw.get(layout.trigger_r, 0))
      self.armed = pull > TRIGGER_ARM
      if not self.armed:
        # Say so rather than swallowing the command: a stick that does nothing
        # and never explains itself reads as a broken controller, not as a
        # safety interlock doing its job.
        if (forward or strafe or turn) and time.monotonic() >= self._next_nag:
          self._next_nag = time.monotonic() + NAG_PERIOD_S
          self._say("[dead-man] hold the RIGHT TRIGGER to move "
                    "(or restart with --no-deadman)")
        forward = strafe = turn = 0.0

    if not self.lateral:
      if strafe and not self._warned_lateral:
        self._warned_lateral = True
        self._say("[!] this policy has no lateral channel; ignoring the X axis")
      strafe = 0.0
    if not self.turning:
      if turn and not self._warned_yaw:
        self._warned_yaw = True
        self._say("[!] this policy has no yaw channel; ignoring the right stick")
      turn = 0.0
    # Lateral and yaw together is an untrained command combination; let the
    # stick that is pushed further win, as keyboard.py's A/D vs Q/E does.
    if strafe and turn:
      if abs(strafe) >= abs(turn):
        turn = 0.0
      else:
        strafe = 0.0

    self._poll_hat_skill()
    self.speed = forward * self.speed_magnitude
    self.yaw = turn * self.yaw_magnitude
    self.lateral_speed = strafe * self.lateral_magnitude

    if self.skill:
      # A skill is a standing behaviour; the stick does not drive it.
      self.mode = "hold"
      self.speed = self.yaw = self.lateral_speed = 0.0
      return
    if not self.armed:
      # A released dead-man must park the robot, not merely stop steering it:
      # a latched MARCH would otherwise keep stepping with nobody holding the
      # trigger. `march == 0` stays unreachable on a no-hold policy.
      self.mode = "hold" if self.holding else "march"
    elif forward:
      self.mode = "walk"
    elif turn or strafe:
      # Turning and strafing are only meaningful once the gait is active.
      self.mode = "march"
    else:
      self.mode = self.base_mode

  def snapshot(self) -> tuple:
    """Return the command tuple, sized to what the running `run` understands.

    The 7-element form carrying a skill is only emitted for a skill policy.
    Deployed copies of `k2_policy_run` predating skills unpack 6, 5 or 4
    elements and fall through to the 4-element branch for anything longer --
    a `ValueError` raised after servo torque is already on. A non-skill policy
    has no skill to report, so send the six-element locomotion tuple
    sends and stay compatible with both.
    """
    if self.connected:
      self._drain()
    else:
      self._try_reconnect()
    if self.connected:
      self._update()
    if self.skills:
      return (self.mode, self.speed, self.yaw, self.height,
              self.lateral_speed, self.skill, self.stop_requested)
    return (self.mode, self.speed, self.yaw, self.height,
            self.lateral_speed, self.stop_requested)


def wait_centered(controller, timeout_s: float = 2.0) -> bool:
  """Drain startup events and confirm every stick really is centered.

  The axis state is seeded from the kernel's last-known values, which can be
  stale: a pad that was mid-flick when it last reported leaves an off-centre
  reading sitting in `absinfo` until its next event. Feeding that to the policy
  as an opening command is exactly the surprise a robot on a support rope does
  not need, so drain first and check before any torque is enabled.
  """
  layout = controller.layout
  axes = (layout.left_x, layout.left_y, layout.right_x, layout.right_y)
  deadline = time.monotonic() + timeout_s
  while time.monotonic() < deadline:
    controller._drain()
    if not controller.connected:
      return False
    offsets = [abs(layout.normalize(code, controller._raw.get(code, 0), 0.0))
               for code in axes]
    if max(offsets) <= controller.deadzone:
      return True
    time.sleep(0.05)
  ecodes = _evdev().ecodes
  worst = max(axes, key=lambda code: abs(
    layout.normalize(code, controller._raw.get(code, 0), 0.0)))
  print(f"[!] {ecodes.ABS[worst]} reads "
        f"{layout.normalize(worst, controller._raw.get(worst, 0), 0.0):+.2f} "
        f"(deadzone {controller.deadzone:.2f}) -- release the sticks")
  return False


def probe(device, deadzone: float) -> int:
  """Print live axis and button values. No policy, no simulation."""
  ecodes = _evdev().ecodes
  layout = PadLayout(device)
  raw = {code: info.value for code, info in layout.info.items()}
  print(f"Device: {device.name}  ({device.path})")
  print(f"Axis layout: {layout.name}")
  print(f"  left stick   X={ecodes.ABS[layout.left_x]}  "
        f"Y={ecodes.ABS[layout.left_y]}")
  print(f"  right stick  X={ecodes.ABS[layout.right_x]}  "
        f"Y={ecodes.ABS[layout.right_y]}")
  print(f"  triggers     L={ecodes.ABS[layout.trigger_l]}  "
        f"R={ecodes.ABS[layout.trigger_r]}")
  print("Move the sticks and press buttons. Ctrl-C to quit.\n")
  stop = False

  def request_stop(*_args) -> None:
    nonlocal stop
    stop = True

  signal.signal(signal.SIGINT, request_stop)
  pressed: set[int] = set()
  while not stop:
    try:
      while select.select([device.fd], [], [], 0.05)[0]:
        for event in device.read():
          if event.type == ecodes.EV_ABS:
            raw[event.code] = event.value
          elif event.type == ecodes.EV_KEY:
            name = ecodes.BTN.get(event.code, event.code)
            if isinstance(name, list):
              name = name[0]
            if event.value:
              pressed.add(name)
            else:
              pressed.discard(name)
    except BlockingIOError:
      pass
    except OSError as exc:
      print(f"\ncontroller disconnected: {exc}")
      return 1
    lx = layout.normalize(layout.left_x, raw.get(layout.left_x, 0), deadzone)
    ly = layout.normalize(layout.left_y, raw.get(layout.left_y, 0), deadzone)
    rx = layout.normalize(layout.right_x, raw.get(layout.right_x, 0), deadzone)
    ry = layout.normalize(layout.right_y, raw.get(layout.right_y, 0), deadzone)
    lt = layout.trigger(layout.trigger_l, raw.get(layout.trigger_l, 0))
    rt = layout.trigger(layout.trigger_r, raw.get(layout.trigger_r, 0))
    hat = (raw.get(layout.hat_x, 0), raw.get(layout.hat_y, 0))
    print(f"\rLX{lx:+.2f} LY{ly:+.2f} | RX{rx:+.2f} RY{ry:+.2f} | "
          f"LT{lt:.2f} RT{rt:.2f} | hat{hat} | "
          f"{' '.join(sorted(pressed)) or '-':<40}", end="", flush=True)
  print()
  return 0


def supported_run_kwargs(candidate: dict) -> dict:
  """Drop keyword arguments this copy of `k2_policy_run.run` does not accept.

  The deployed `hardware/` on the robot lags the training checkout, and a
  checkout pinned to an older commit lags it too. Passing a keyword the local
  `run` predates is a `TypeError` -- and the sibling failure, handing it a
  command tuple wider than it unpacks, is a `ValueError` raised after servo
  torque is already on. Ask the function what it accepts instead of assuming.
  """
  params = set(inspect.signature(run).parameters)
  dropped = sorted(set(candidate) - params)
  if dropped:
    print(f"[i] this k2_policy_run predates {', '.join(dropped)}; omitting")
  return {name: value for name, value in candidate.items() if name in params}


def run_supports(name: str) -> bool:
  return name in inspect.signature(run).parameters


def _default_policy() -> Path:
  shipped = ROOT / "policies" / "last_policy.onnx"
  if shipped.is_file():
    return shipped
  # Prefer a `full*` export: it is the only family carrying the lateral
  # channel, so it is the one that uses all three stick axes.
  policies = list((ROOT / "policies").glob("full*.onnx"))
  if not policies:
    policies = list((ROOT / "policies").glob("turn*.onnx"))
  if not policies:
    policies = list((ROOT / "policies").glob("bidir*.onnx"))
  if not policies:
    policies = list((ROOT / "policies").glob("*.onnx"))
  if not policies:
    raise SystemExit("no ONNX policy found; pass --policy PATH")
  return max(policies, key=lambda path: path.stat().st_mtime)


def main(require_skills: bool = False) -> int:
  # Keep one public launcher. Hardware-specific safety defaults live in the
  # hardware package, while controller parsing and command semantics stay here.
  selector = argparse.ArgumentParser(add_help=False)
  selector.add_argument("--bus", default="sim")
  selected, _ = selector.parse_known_args()
  if selected.bus != "sim":
    from hardware.k2_joystick_real import main as real_main
    return real_main()
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--policy", type=Path, default=None)
  parser.add_argument("--bus", default="sim")
  parser.add_argument("--probe", action="store_true",
                      help="print live pad values and exit; no policy is run")
  parser.add_argument("--device", default=None,
                      help="/dev/input/eventN to use instead of autodetecting")
  parser.add_argument("--name", default=None,
                      help="substring of the controller name to match")
  parser.add_argument("--deadzone", type=float, default=0.15,
                      help="stick deadzone as a fraction (default: 0.15)")
  parser.add_argument("--deadman", action="store_true",
                      help="require the right trigger held to move at all")
  parser.add_argument("--speed", type=float, default=0.02,
                      help="full-stick speed magnitude in m/s (default: 0.02)")
  parser.add_argument("--yaw-rate", type=float, default=0.15,
                      help="full-stick yaw magnitude in rad/s (default: 0.15)")
  parser.add_argument("--lateral-speed", type=float, default=0.015,
                      help="full-stick lateral magnitude in m/s")
  parser.add_argument("--duration", type=float, default=600.0,
                      help="maximum runtime in seconds (default: 600)")
  parser.add_argument("--fall-angle", type=float, default=45.0,
                      help="simulation safety stop angle (default: 45 deg)")
  parser.add_argument("--slew", type=float, default=1.0)
  args = parser.parse_args()

  if not 0.0 <= args.deadzone < 0.9:
    raise SystemExit("--deadzone must be in [0, 0.9)")
  device = find_pad(args.name, args.device)
  if args.probe:
    return probe(device, args.deadzone)

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
  meta = session.get_modelmeta().custom_metadata_map
  skill_dim = int(meta.get("skill_command_dim", "0"))
  if require_skills and skill_dim != 7:
    raise SystemExit(
      f"{policy.name} is not a K2 skill policy (skill dimension {skill_dim})"
    )
  gait_channel5, lateral_ramp = _channel5_contract(session)
  # A walk+turn+height policy commands height in BOTH modes; without
  # this the runner reproduces the historical (1 - march) gate and the
  # crouch control goes dead the instant the robot starts marching.
  height_in_march = height_active_in_march(session)
  supports_hold = float(meta.get("gait_rel_march_envs", "0.0")) < 1.0
  turning = gait_dim in (5, 6, 7) and heading_dim == 2
  lateral = (gait_channel5 == "lateral_velocity" and int(meta.get("k2_pipeline_stage", "2")) >= 2
             and float(meta.get("gait_rel_lateral_envs", "1")) > 0)
  crouching = (gait_channel5 == "height_offset" or int(meta.get("k2_pipeline_stage", "0")) >= 3)
  # A skill policy is only drivable if this `run` understands the 7-element
  # command tuple; otherwise treat it as a plain locomotion policy.
  skills = skill_dim == 7 and run_supports("skill_command_dim")
  if skill_dim == 7 and not skills:
    print("[i] this k2_policy_run predates skills; the D-pad is inactive")

  controller = JoystickCommand(
    device, args.speed, args.yaw_rate, turning,
    crouching=crouching, holding=supports_hold,
    lateral_speed=args.lateral_speed, lateral=lateral, skills=skills,
    deadzone=args.deadzone, deadman=args.deadman,
  )
  if meta.get("gait_staged_height") == "1":
    controller.crouch_min = float(meta["height_offset_min_m"])
    controller.crouch_max = float(meta["height_offset_max_m"])
  bus = k2_bus.open_bus(
    "sim", viewer=True, realtime=True, training_ball=skills,
  )

  def request_stop(*_args) -> None:
    controller.stop_requested = True

  signal.signal(signal.SIGINT, request_stop)
  print(f"Controller: {device.name} ({device.path}), "
        f"layout {controller.layout.name}")
  print(f"Policy: {policy}")
  print(f"Checkpoint: {checkpoint}; gait: {gait_freq:g} Hz; "
        f"transition: {transition_time:g} s; ramp: {velocity_ramp:g} m/s^2")
  controls = ["left stick Y = forward/backward"]
  if lateral:
    controls.append("left stick X = strafe")
  if turning:
    controls.append("right stick X = turn")
  controls.append("A = march" + ("  B = hold" if supports_hold else "  B = zero"))
  if crouching:
    controls.append("LB = crouch  RB = rise  Y = nominal"
                    + ("" if height_in_march else " (hold only)"))
  if skill_dim:
    controls.append("D-pad = skills 1-4")
  controls.append("Back = quit")
  print("Controls: " + "  |  ".join(controls))
  if args.deadman:
    print("Dead-man enabled: hold the RIGHT TRIGGER for any motion.")
  if supports_hold:
    print("Starting safely in HOLD.")
  else:
    print("This policy has NO HOLD (trained with rel_march_envs = 1.0):")
    print("  starting in MARCH, and the gait activation ramp is skipped.")

  try:
    bus.torque(True)
    deadline = time.monotonic() + args.duration
    while not controller.stop_requested and time.monotonic() < deadline:
      try:
        run(
          bus, session, default_pos, action_scale,
          mode="hold" if supports_hold else "march",
          duration=max(0.0, deadline - time.monotonic()),
          slew_rad_s=args.slew, march_after=0.0, gait_freq_hz=gait_freq,
          approach_s=2.0, fall_angle_deg=args.fall_angle,
          forward_speed=args.speed, heading_observation_dim=heading_dim,
          gait_transition_time_s=transition_time if supports_hold else 0.0,
          command_source=controller.snapshot,
          gait_velocity_ramp_rate_mps2=velocity_ramp,
          gait_yaw_rate_ramp_rate_rps2=yaw_ramp,
          gait_height_ramp_rate_mps=height_ramp,
          gait_lateral_ramp_rate_mps2=lateral_ramp,
          **supported_run_kwargs({
            "gait_channel5": gait_channel5,
            "gait_height_active_in_march": height_in_march,
            "skill_command_dim": skill_dim,
            "gait_command_dim": gait_dim,
          }),
        )
      except UnsafeTiltError as exc:
        print(f"\nSIM FALL: {exc}")
        if controller.stop_requested:
          break
        print("Auto-resetting twin; the pad stays live during the 2 s approach.")
        bus.reset()
  finally:
    try:
      bus.torque(False)
    finally:
      bus.close()
      try:
        controller.device.close()
      except OSError:
        pass
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
