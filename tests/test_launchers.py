"""Deployment entry-point regressions; never open a physical bus."""
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import keyboard
import joystick
from hardware import k2_joystick_real
from hardware.k2_stabilizer import UnsafeTiltError


class LauncherTests(unittest.TestCase):
  def test_shipped_policy_defaults(self):
    expected = Path(keyboard.__file__).parent / "policies/last_policy.onnx"
    for module in (keyboard, joystick, k2_joystick_real):
      self.assertEqual(module._default_policy(), expected)
      self.assertTrue(expected.is_file())

  def test_serial_dispatch_accepts_both_arg_forms(self):
    for args in (["--bus", "/dev/ttyAMA0"], ["--bus=/dev/ttyAMA0"]):
      with patch("sys.argv", ["joystick.py", *args]), patch.object(
          k2_joystick_real, "main", return_value=7) as main:
        self.assertEqual(joystick.main(), 7)
        main.assert_called_once()

  def keyboard_hardware(self, run_error=None, enter_error=None, exit_error=None):
    bus = Mock()
    with patch("sys.argv", ["keyboard.py", "--bus", "/dev/ttyAMA0", "--yes"]), \
         patch("sys.stdin.isatty", return_value=True), \
         patch.object(keyboard.k2_bus, "open_bus", return_value=bus), \
         patch.object(keyboard.TerminalKeyboard, "__enter__", side_effect=enter_error), \
         patch.object(keyboard.TerminalKeyboard, "__exit__", side_effect=exit_error) as restore, \
         patch.object(keyboard, "run", side_effect=run_error) as run:
      if enter_error or exit_error:
        with self.assertRaises(RuntimeError):
          keyboard.main()
      else:
        self.assertEqual(keyboard.main(), 2)
        self.assertEqual(run.call_args.kwargs["fall_angle_deg"], 14)
      bus.reset.assert_not_called()
      bus.torque.assert_called_with(False)
      bus.close.assert_called_once()
      if enter_error:
        restore.assert_not_called()

  def test_real_fall_releases_torque_without_sim_reset(self):
    self.keyboard_hardware(run_error=UnsafeTiltError("test tilt"))

  def test_terminal_entry_failure_still_closes_bus(self):
    self.keyboard_hardware(enter_error=RuntimeError("terminal setup"))

  def test_terminal_restore_failure_still_releases_torque(self):
    self.keyboard_hardware(run_error=UnsafeTiltError("test tilt"),
                           exit_error=RuntimeError("terminal restore"))

  def test_bad_trim_rejected_before_bus_opens(self):
    with patch("sys.argv", ["keyboard.py", "--joint-trim", "bad"]), \
         patch.object(keyboard.k2_bus, "open_bus") as bus:
      with self.assertRaises((ValueError, SystemExit)):
        keyboard.main()
      bus.assert_not_called()

  def test_nonfinite_trim_rejected(self):
    with self.assertRaises(SystemExit):
      keyboard.parse_joint_trim("hip_yaw_R=nan")

  def test_real_joystick_fall_cleans_up(self):
    bus, controller = Mock(), Mock()
    with patch("sys.argv", ["joystick.py", "--bus=/dev/ttyAMA0", "--yes"]), \
         patch.object(k2_joystick_real, "find_pad", return_value=controller.device), \
         patch.object(k2_joystick_real, "JoystickCommand", return_value=controller), \
         patch.object(k2_joystick_real, "wait_centered", return_value=True), \
         patch.object(k2_joystick_real.k2_bus, "open_bus", return_value=bus), \
         patch.object(k2_joystick_real, "run", side_effect=UnsafeTiltError("test tilt")), \
         patch.object(k2_joystick_real, "supported_run_kwargs", side_effect=lambda value: value):
      self.assertEqual(joystick.main(), 2)
      bus.torque.assert_called_with(False)
      bus.close.assert_called_once()
      controller.device.close.assert_called_once()

  def test_terminal_drains_keys_and_stops_on_eof(self):
    read_fd, write_fd = os.pipe()
    try:
      command = keyboard.TerminalKeyboard(.02, .1, True, lateral=True)
      command._fd = read_fd
      os.write(write_fd, b"wad")
      command.snapshot()
      self.assertLess(command.lateral_speed, 0)
      os.close(write_fd)
      write_fd = None
      self.assertTrue(command.snapshot()[5])
    finally:
      os.close(read_fd)
      if write_fd is not None:
        os.close(write_fd)


if __name__ == "__main__":
  unittest.main()
