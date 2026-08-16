"""Low-speed forward-walking extension of the validated K2 march task.

The proven hold/march task remains unchanged. This variant adds one observable
command channel (desired forward speed), permits fore/aft foot motion, and
continues to penalize lateral drift and yaw. Initial speed is intentionally
limited to 1--4 cm/s for the first sim-to-real walking policy.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from k2_rl import mdp
from k2_rl.inplace_env_cfg import make_k2_inplace_env_cfg
from k2_rl.k2_constants import FOOT_GEOMS, FOOT_SITES
from k2_rl.mdp import GaitCommandCfg


# Stride length at the fixed 0.75 Hz cadence is commanded velocity / 1.5 steps
# per second, so doubling the top of the range takes the longest trained step
# from ~2.7 cm to ~5.3 cm. Raised from (0.01, 0.04) after iteration 2000 walked
# on hardware but covered only 12--18 cm in 30 s.
TRAIN_FORWARD_RANGE = (0.02, 0.08)
PLAY_FORWARD_SPEED = 0.05
# The learned policy realizes roughly 73% of this shaped target once velocity
# and balance terms are included.  Fifteen millimetres therefore produces the
# required 10--12 mm measured full-sole clearance on the digital twin.
FORWARD_SWING_HEIGHT = 0.015


def make_k2_forward_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_k2_inplace_env_cfg(play=play)

  gait = cfg.commands["gait"]
  assert isinstance(gait, GaitCommandCfg)
  gait.rel_march_envs = 1.0 if play else 0.8
  gait.forward_velocity_range = (
    (PLAY_FORWARD_SPEED, PLAY_FORWARD_SPEED) if play else TRAIN_FORWARD_RANGE
  )

  # Gyro rate alone cannot reveal accumulated yaw. Add the relative heading
  # available from integrated gyro on hardware, immediately before gait so the
  # deployment observation order remains explicit and deterministic.
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    gait_obs = terms.pop("gait")
    terms["heading"] = ObservationTermCfg(
      func=mdp.heading_sin_cos,
      noise=Unoise(n_min=-0.02, n_max=0.02),
      delay_max_lag=0 if play else 1,
    )
    terms["gait"] = gait_obs

  # Replace rewards that intentionally prevented translation in the march task.
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=mdp.track_forward_velocity_exp,
    weight=2.5,
    params={"command_name": "gait", "std": 0.025},
  )
  cfg.rewards["base_lin_vel_xy"] = RewardTermCfg(
    func=mdp.base_lateral_velocity_l2,
    weight=-8.0,
  )
  cfg.rewards["feet_xy_vel"] = RewardTermCfg(
    func=mdp.feet_lateral_vel_l2,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", site_names=list(FOOT_SITES))},
  )
  cfg.rewards["base_xy_position"] = RewardTermCfg(
    func=mdp.base_lateral_position_l2,
    weight=-20.0,
  )
  cfg.rewards["yaw_rate"] = RewardTermCfg(
    func=mdp.base_yaw_rate_l2,
    weight=-1.0,
  )
  cfg.rewards["heading"] = RewardTermCfg(
    func=mdp.base_heading_error,
    weight=-5.0,
  )
  cfg.rewards["track_angular_velocity"].weight = 1.5
  # Keep the support polygon near the body, but allow deliberate step length.
  # Halved alongside the longer stride range: at -0.5 this term charges four
  # times as much for a step twice as long and would cancel the new range.
  cfg.rewards["feet_under_base"].weight = -0.25
  # Iteration 2000 learned a lopsided gait: left hip pitch swung at 0.45x the
  # right in simulation and 0.31x on hardware, which is the likely source of
  # the persistent physical-left lean that needs a runtime balance trim.
  cfg.rewards["gait_symmetry"] = RewardTermCfg(
    func=mdp.gait_antisymmetry_l2,
    # At the iteration-2000 asymmetry the error is rms 0.05 rad, so the whole
    # term is worth ~0.0026 before weighting. -1.5 left it three orders below
    # gait_swing and the policy simply ignored it; -6.0 makes removing the
    # asymmetry worth about as much as the heading term.
    weight=-6.0,
    params={
      "command_name": "gait",
      "right_cfg": SceneEntityCfg(
        "robot", joint_names=[r"base_link_hip_pitch_right_joint"]
      ),
      "left_cfg": SceneEntityCfg(
        "robot", joint_names=[r"base_link_hip_pitch_left_joint"]
      ),
    },
  )
  cfg.rewards["base_tilt"].weight = -2.0
  cfg.rewards["roll_pitch_rate"].weight = -0.08

  # Walking must lift rather than exploit low-friction simulation by shuffling.
  # Track all five sole sites, enforce the phase contact schedule, and make
  # loaded-foot slip expensive enough that a planted foam sole is viable.
  cfg.rewards["gait_swing"].weight = 6.0
  cfg.rewards["gait_swing"].params["target_height"] = FORWARD_SWING_HEIGHT
  cfg.rewards["gait_swing"].params["tracking_sigma"] = 0.004
  cfg.rewards["gait_contact"].weight = 2.0
  cfg.rewards["feet_slip"].weight = -2.0
  cfg.rewards["swing_sole_flat"].weight = -0.5
  cfg.events["foot_friction"].params["ranges"] = (0.7, 1.4)

  if not play:
    # Generator terrains trigger rare MuJoCo-Warp NaNs at large batch sizes.
    # Instead, independently vary sole thickness and mounting angle per env.
    # On a flat plane this produces 0--3 mm unequal support and up to 1 degree
    # sole/floor mismatch, directly exercising the observed hardware failure.
    feet_cfg = SceneEntityCfg("robot", geom_names=list(FOOT_GEOMS))
    cfg.events["sole_height"] = EventTermCfg(
      mode="startup",
      func=dr.geom_pos,
      params={
        "asset_cfg": feet_cfg,
        "operation": "add",
        "ranges": (-0.0015, 0.0015),
        "axes": [2],
        "shared_random": False,
      },
    )
    cfg.events["sole_tilt"] = EventTermCfg(
      mode="startup",
      func=dr.geom_quat,
      params={
        "asset_cfg": feet_cfg,
        "roll_range": (-math.radians(1.0), math.radians(1.0)),
        "pitch_range": (-math.radians(1.0), math.radians(1.0)),
      },
    )

  # Training retains the base task's validated randomized 0--10 ms substep
  # delay and dynamics ranges. An experimental 20--80 ms whole-step queue and
  # simultaneously widened dynamics created rare long-horizon divergence.

  # A world-frame lateral corridor is meaningful only when every episode starts
  # facing world +X. Yaw-rate pushes remain, so the policy still learns recovery.
  cfg.events["reset_base"].params["pose_range"]["yaw"] = (-0.08, 0.08)

  # The old curriculum ramps penalties whose purpose was zero translation.
  for name in ("cur_base_lin_vel_xy", "cur_feet_under_base", "cur_feet_xy_vel"):
    cfg.curriculum.pop(name, None)

  return cfg
