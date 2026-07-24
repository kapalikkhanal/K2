"""Bounded IMU feedback for the scripted K2 walk.

The IMU can measure torso tilt and angular rate, but not the centre of mass
position relative to a foot.  This controller is therefore a conservative
ankle/hip posture stabilizer, not a substitute for contact sensing or a trained
locomotion controller.
"""

from __future__ import annotations

import math

import numpy as np

from . import k2_conventions as C
from .k2_attitude import R_IMU_TO_ROOT


class UnsafeTiltError(RuntimeError):
    """Raised before commanding another tick when the robot is falling."""


class TiltStabilizer:
    """PD torso-angle correction mapped through IK-derived joint responses."""

    def __init__(self, roll_response, pitch_response, *, roll_target=0.0,
                 pitch_target=0.0, kp=0.45, kd=0.10,
                 max_correction=math.radians(3.0),
                 fall_angle=math.radians(12.0), filter_tau=0.16,
                 deadband=math.radians(0.35)):
        self.roll_response = np.asarray(roll_response, dtype=np.float32)
        self.pitch_response = np.asarray(pitch_response, dtype=np.float32)
        if self.roll_response.shape != (len(C.SIM_ORDER),):
            raise ValueError("roll_response has the wrong shape")
        if self.pitch_response.shape != (len(C.SIM_ORDER),):
            raise ValueError("pitch_response has the wrong shape")
        self.roll_target = float(roll_target)
        self.pitch_target = float(pitch_target)
        self.kp = float(kp)
        self.kd = float(kd)
        self.max_correction = float(max_correction)
        self.fall_angle = float(fall_angle)
        self.filter_tau = float(filter_tau)
        self.deadband = float(deadband)
        self.correction = np.zeros(2, dtype=np.float64)
        self.last_state = None

    @staticmethod
    def state(gyro_site, gravity_root):
        g = np.asarray(gravity_root, dtype=np.float64)
        if g.shape != (3,) or not np.all(np.isfinite(g)):
            raise UnsafeTiltError("invalid gravity vector from IMU")
        n = float(np.linalg.norm(g))
        if not 0.5 < n < 1.5:
            raise UnsafeTiltError(f"invalid gravity magnitude {n:.2f}")
        g /= n
        # Same convention used by calibrate_imu_level.py.
        pitch = math.atan2(float(g[0]), float(-g[2]))
        roll = math.atan2(float(g[1]), float(-g[2]))
        gyro_root = R_IMU_TO_ROOT @ np.asarray(gyro_site, dtype=np.float64)
        if not np.all(np.isfinite(gyro_root)):
            raise UnsafeTiltError("invalid gyro vector from IMU")
        return roll, pitch, float(gyro_root[0]), float(gyro_root[1])

    def update(self, nominal, gyro_site, gravity_root, dt, gain_scale=1.0):
        roll, pitch, roll_rate, pitch_rate = self.state(gyro_site, gravity_root)
        gravity_root = np.asarray(gravity_root, dtype=np.float64)
        gravity_root /= max(float(np.linalg.norm(gravity_root)), 1e-9)
        tilt = math.acos(float(np.clip(-gravity_root[2], -1.0, 1.0)))
        if tilt > self.fall_angle:
            raise UnsafeTiltError(
                f"torso tilt {math.degrees(tilt):.1f} deg exceeds "
                f"{math.degrees(self.fall_angle):.1f} deg")

        def outside_deadband(error):
            return math.copysign(
                max(0.0, abs(error) - self.deadband), error)

        wanted = np.array([
            -self.kp * outside_deadband(roll - self.roll_target)
            - self.kd * roll_rate,
            -self.kp * outside_deadband(pitch - self.pitch_target)
            - self.kd * pitch_rate,
        ])
        wanted = np.clip(wanted, -self.max_correction, self.max_correction)
        # Avoid feeding accelerometer noise directly into servo positions.
        a = 1.0 - math.exp(-max(dt, 1e-4) / max(self.filter_tau, 1e-4))
        self.correction += a * (wanted - self.correction)
        corr = self.correction * float(np.clip(gain_scale, 0.0, 1.0))

        target = (np.asarray(nominal, dtype=np.float32)
                  + self.roll_response * corr[0]
                  + self.pitch_response * corr[1])
        target = np.array(
            [C.clamp_q(name, float(target[i]))
             for i, name in enumerate(C.SIM_ORDER)],
            dtype=np.float32)
        self.last_state = {
            "roll": roll, "pitch": pitch,
            "roll_rate": roll_rate, "pitch_rate": pitch_rate,
            "roll_correction": float(corr[0]),
            "pitch_correction": float(corr[1]),
            "tilt": tilt,
        }
        return target
