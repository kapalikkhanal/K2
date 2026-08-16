"""Base attitude for the walking policy, matching the walk env's observation.

The walk observation is (verified against the env):
  * base_ang_vel     = the raw gyro, in the IMU/site frame (NOT transformed).
    The env reads the imu_ang_vel sensor directly, so the real LSM6DS3 gyro
    goes in as-is (converted to rad/s).
  * projected_gravity = gravity direction in the k2_root frame (world-aligned,
    Z up). On hardware this is estimated from the IMU: filter gravity in the
    chip frame, then rotate to the root frame with R_IMU_TO_ROOT.

    attitude() -> (gyro[3] rad/s (chip frame), proj_grav_b[3] unit (root frame))

  * SimAttitude  -- true values from MuJoCo (a perfect IMU): what the policy
    trained against.
  * ImuAttitude  -- the mounted LSM6DS3 + an adaptive complementary filter.
    It trusts gyro propagation during dynamic acceleration and smoothly restores
    accelerometer correction near 1 g.

R_IMU_TO_ROOT (chip->root) is measured from held physical tilts and rotates the
filtered gravity estimate into the root frame. It is a fixed rotation because
the IMU is rigidly attached to the base.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# chip frame -> k2_root frame. root_x = chip_y, root_y = -chip_x, root_z = chip_z.
# Re-measured on the physical Robot_v4 on 2026-08-15 using held static tilts:
#   physical forward  -> old policy pitch -12.5 deg (must be positive)
#   physical backward -> old policy pitch +13.9 deg (must be negative)
#   physical right    -> old policy roll  +12.8 deg (must be negative)
# Both horizontal axes were inverted.  Flipping X and Y together is a proper
# 180-degree rotation about root Z (determinant +1), not a reflection.
R_IMU_TO_ROOT = np.array([[0.0, 1.0, 0.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0]])

# MJCF `imu` site frame -> k2_root frame, read straight off the compiled model
# (mjcf/k2_physics.xml, site quat "0 0 -0.707107 0.707107") and confirmed by
# driving the free joint at a known angular velocity. This is NOT the same as
# R_IMU_TO_ROOT, and the difference is the bug below.
R_SITE_TO_ROOT = np.array([[0.0, -1.0, 0.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, 0.0, -1.0]])
LEVEL_CALIB_PATH = Path(__file__).with_name("imu_level_calibration.json")


def _level_correction(path: Path = LEVEL_CALIB_PATH) -> np.ndarray:
    """Actual root frame -> level-calibrated root frame."""
    if not path.exists():
        return np.eye(3)
    with open(path) as f:
        matrix = np.asarray(json.load(f)["root_correction"], dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"invalid IMU level calibration at {path}")
    return matrix


class SimAttitude:
    """True base state from the SimBus's MuJoCo model (perfect IMU)."""

    def __init__(self, sim_bus):
        self.bus = sim_bus
        self.initial_heading = float(sim_bus.base_heading() or 0.0)

    def attitude(self):
        return self.bus.true_base_state()

    def heading_sin_cos(self) -> np.ndarray:
        heading = float(self.bus.base_heading() or 0.0) - self.initial_heading
        return np.array([np.sin(heading), np.cos(heading)], dtype=np.float32)

    def close(self):
        pass


class ImuAttitude:
    """LSM6DS3 -> root-frame angular velocity + complementary-filtered gravity.

    The gyro bias is captured and removed at construction (robot must be still).
    Static gravity/Z is verified; in-plane signs were checked and are fine
    ([-0.004,-0.011,-1.0] at level stand).
    """

    def __init__(self, bus_number: int = 1, alpha: float = 0.02,
                 alpha_min: float = 0.001, bias_samples: int = 200):
        from .lsm6ds3 import LSM6DS3

        self.imu = LSM6DS3(bus_number)
        self.alpha = alpha            # maximum accel trust per tick
        self.alpha_min = alpha_min    # retain slow bias correction in motion
        self.g_chip = None            # gravity estimate, chip frame
        self._deg2rad = np.pi / 180.0
        self.root_correction = _level_correction()
        self.heading = 0.0
        # The actor's gyro observation is the gyro in the MJCF `imu` SITE frame
        # -- that is what mjlab's imu_ang_vel sensor reports at training time.
        # Take the chip gyro to the root frame (R_IMU_TO_ROOT, validated by the
        # gravity path), apply the level correction there, then go back out to
        # the site frame with R_SITE_TO_ROOT.T.
        #
        # The chip frame and the MJCF site frame are not the same. With the
        # measured mounting rotation, chip -> site is
        # R_SITE_TO_ROOT.T @ R_IMU_TO_ROOT = diag(+1, -1, -1). The same proper
        # frame chain is used for gyro and gravity; dynamic gyro signs are
        # validated separately against differentiated gravity after any mount
        # transform change.
        self.gyro_correction = (
            R_SITE_TO_ROOT.T @ self.root_correction @ R_IMU_TO_ROOT
        )

        # Capture and remove the gyro bias while the robot is still. The raw
        # LSM6DS3 has a real bias (~[0.4, -3.4, -1.0] dps measured); left in, it
        # feeds a false angular velocity into base_ang_vel AND drifts the
        # complementary-filter gravity estimate (~0.05 over 8 s), so the policy
        # slowly leans. THE ROBOT MUST BE STILL at construction.
        import time
        samples = []
        for _ in range(bias_samples):
            _, g = self.imu.read()
            samples.append(g)
            time.sleep(0.002)
        self.gyro_bias = np.mean(samples, axis=0) * self._deg2rad  # rad/s

    def attitude(self, dt: float = 0.02):
        accel, gyro_dps = self.imu.read()           # g, deg/s (chip frame)
        gyro = np.asarray(gyro_dps) * self._deg2rad - self.gyro_bias  # rad/s, debiased
        g_meas = -np.asarray(accel)
        n = np.linalg.norm(g_meas)
        accel_valid = np.isfinite(n) and 0.2 < n < 2.5
        if accel_valid:
            g_meas = g_meas / n

        if self.g_chip is None:
            self.g_chip = g_meas.copy() if accel_valid else np.array([0., 0., -1.])
        else:
            # Predict: a world-fixed vector rotates as -omega x g in the body.
            self.g_chip = self.g_chip - dt * np.cross(gyro, self.g_chip)
            self.g_chip /= max(np.linalg.norm(self.g_chip), 1e-6)
            if accel_valid:
                # Linear body acceleration corrupts the apparent gravity vector.
                # Trust accel fully only when its magnitude is near 1 g AND its
                # direction agrees with the gyro-propagated estimate. Both
                # confidence terms vary smoothly to avoid control discontinuities.
                norm_conf = np.exp(-((n - 1.0) / 0.12) ** 2)
                agreement = float(np.clip(np.dot(self.g_chip, g_meas), -1.0, 1.0))
                innovation = np.arccos(agreement)
                direction_conf = np.exp(-(innovation / 0.25) ** 2)
                confidence = float(norm_conf * direction_conf)
                alpha = self.alpha_min + (self.alpha - self.alpha_min) * confidence
                self.g_chip = (1 - alpha) * self.g_chip + alpha * g_meas
                self.g_chip /= max(np.linalg.norm(self.g_chip), 1e-6)

        # base_ang_vel obs is the raw gyro (chip frame); gravity is rotated to
        # the root frame to match projected_gravity.
        proj_grav_b = self.root_correction @ R_IMU_TO_ROOT @ self.g_chip
        gyro_root = self.root_correction @ R_IMU_TO_ROOT @ gyro
        self.heading = (self.heading + float(gyro_root[2]) * dt + np.pi) % (
            2.0 * np.pi
        ) - np.pi
        gyro_nominal = R_SITE_TO_ROOT.T @ gyro_root
        return gyro_nominal.astype(np.float32), proj_grav_b.astype(np.float32)

    def heading_sin_cos(self) -> np.ndarray:
        return np.array(
            [np.sin(self.heading), np.cos(self.heading)], dtype=np.float32
        )

    def close(self):
        try:
            self.imu.close()
        except Exception:
            pass
