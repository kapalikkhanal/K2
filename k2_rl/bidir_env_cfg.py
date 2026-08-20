"""Conservative forward/backward walking task for K2.

This is intentionally a separate from-scratch task. It inherits the validated
  forward gait, proven learned two-foot hold, smooth hold/walk transition, sole
randomization, and balance rewards, while changing only the signed velocity
distribution. Keeping it separate preserves the known-good forward policies.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg

from k2_rl.forward_env_cfg import make_k2_forward_env_cfg
from k2_rl.mdp import GaitCommandCfg


TRAIN_BIDIR_RANGE = (-0.05, 0.05)
PLAY_BIDIR_SPEED = 0.03


def make_k2_bidir_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_k2_forward_env_cfg(play=play)

  gait = cfg.commands["gait"]
  assert isinstance(gait, GaitCommandCfg)
  # Keep 25% hold samples so the stable hold and both hold<->walk transitions
  # remain well represented. Signed commands are rate-limited by GaitCommand,
  # so a forward/backward resample cannot demand an instantaneous reversal.
  gait.rel_march_envs = 1.0 if play else 0.75
  gait.forward_velocity_range = (
    (PLAY_BIDIR_SPEED, PLAY_BIDIR_SPEED) if play else TRAIN_BIDIR_RANGE
  )
  # v5 sampled the full signed interval uniformly. Preserve that exact command
  # distribution for compatibility with the preferred iteration-2500 policy.
  gait.resampling_time_range = (4.0, 7.0)
  # A slightly longer gait blend and signed-speed ramp remove the visible
  # walk->hold/reversal snap while remaining responsive on hardware.
  gait.transition_time_s = 0.8
  gait.velocity_ramp_rate_mps2 = 0.06

  if not play:
    # Robust but still representative of indoor floors. Independent +/-2.5 mm
    # sole offsets create at most 5 mm support-height mismatch. Seven mm would
    # consume most of the policy's measured 10--12 mm swing clearance and is
    # better introduced only after this first bidirectional policy succeeds.
    cfg.events["sole_height"].params["ranges"] = (-0.0025, 0.0025)
    cfg.events["sole_tilt"].params["roll_range"] = (
      -math.radians(1.5), math.radians(1.5)
    )
    cfg.events["sole_tilt"].params["pitch_range"] = (
      -math.radians(1.5), math.radians(1.5)
    )

    # `push_by_setting_velocity` is an instantaneous velocity disturbance, so
    # small values are already meaningful on a 1.28 kg robot. Keep these light
    # enough that learning locomotion, rather than surviving impacts, dominates.
    cfg.events["push_robot"].interval_range_s = (4.0, 7.0)
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.10, 0.10),
      "y": (-0.10, 0.10),
      "roll": (-0.10, 0.10),
      "pitch": (-0.10, 0.10),
      "yaw": (-0.15, 0.15),
    }

  return cfg
