"""Stage two: add sideways walking to the validated turn task (Mjlab-Lateral-K2).

Built strictly on top of :func:`make_k2_turn_env_cfg`, so every hold, gait,
smoothness and domain-randomization decision lives in ONE place
(``turn_env_cfg.py``) and this file only says what changes when ``vy != 0``.

Command sampling: 30% pure lateral (of which 25% pinned at the range endpoint),
10% exact in-place march/turn, the remaining 60% the turn task's signed vx/wz
mixture. The inherited hold share (``1 - REL_MARCH_ENVS``) applies to all of
them. Observation layout: ``[march, sin, cos, vx, wz, vy]`` -- actor 50-D,
critic 55-D; ``k2_rl.prepare_lateral`` widens a 49-D turn checkpoint to it.

Every inherited reward keeps its exact value whenever the emitted vy is zero
(``LateralRewardBlend`` returns the legacy branch bit-for-bit there), so the
straight and turning gaits are scored precisely as in stage one. Only while vy
is emitted do the terms that would forbid a side step (sideways body/foot
velocity, lateral corridor, leg symmetry, tilt envelope, pose roll tolerance,
foot clearance floors) blend into their lateral form.
"""
from __future__ import annotations

from copy import deepcopy
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from k2_rl import mdp
from k2_rl.k2_constants import JOINT_PATTERNS, FOOT_SITES
from k2_rl.mdp.lateral import (
  LateralPlanarTracking,
  LateralGaitCommandCfg,
  LateralRewardBlend,
  lateral_yaw_pitch_tracking,
  LateralFootSmoothness,
  lateral_swing_velocity_l2,
)
from k2_rl.turn_env_cfg import make_k2_turn_env_cfg


# At 0.58 Hz, 0.030 m/s asks ~26 mm of body progress per half-cycle. Hardware
# is driven at <= 0.015 m/s (keyboard.py / firmware) -- the trained range is
# deliberately twice the deployed one so the deployed command sits well inside
# the distribution rather than at its edge.
TRAIN_LATERAL_RANGE = (-0.030, 0.030)
PLAY_LATERAL_SPEED = 0.025
LATERAL_COMMAND_INDEX = 5
LATERAL_TRACK_STD = 0.015
LATERAL_RAMP_RATE_MPS2 = 0.06
REL_LATERAL_ENVS = 0.30
REL_IN_PLACE_ENVS = 0.10
REL_LATERAL_EXTREME_ENVS = 0.25
# Roll the body may use for weight transfer during a side step before the
# tilt penalty charges it. Measured (2026-09-14): tightening this to 5 deg did
# NOT change tilt or roll-rate costs at all; lateral tracking was limited by
# training length, not by this envelope. Leave it alone and train longer.
MAX_FREE_ROLL = math.radians(10.0)


def make_k2_lateral_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_k2_turn_env_cfg(play=play)

  ##
  # Command: same GaitCommand, plus the vy channel and the sampling mix above.
  ##
  gait = LateralGaitCommandCfg(**vars(cfg.commands["gait"]))
  cfg.commands["gait"] = gait
  gait.lateral_velocity_range = (
    (PLAY_LATERAL_SPEED, PLAY_LATERAL_SPEED) if play else TRAIN_LATERAL_RANGE
  )
  gait.lateral_ramp_rate_mps2 = LATERAL_RAMP_RATE_MPS2
  gait.rel_lateral_envs = 1.0 if play else REL_LATERAL_ENVS
  gait.rel_in_place_envs = 0.0 if play else REL_IN_PLACE_ENVS
  gait.rel_lateral_extreme_envs = 0.0 if play else REL_LATERAL_EXTREME_ENVS
  # Phase 0 leads with the +Y (robot-left) foot for +vy; exported as metadata
  # and mirrored in the firmware's CommandProfile.
  gait.lateral_positive_phase = 0.0
  # LateralFootSmoothness reads per-substep contact force history.
  for sensor in cfg.scene.sensors:
    if sensor.name == "feet_ground_contact":
      sensor.history_length = cfg.decimation

  ##
  # Rewards: blend, never replace. Legacy branch is exact at vy == 0.
  ##
  def blend(name, lateral=None):
    legacy = cfg.rewards[name]
    cfg.rewards[name] = RewardTermCfg(
      func=LateralRewardBlend,
      weight=legacy.weight,
      params={"legacy": legacy, "lateral": lateral, "command_name": "gait"},
    )

  # A side step needs sideways body and foot travel and unequal leg strokes.
  # These terms forbid exactly that, so they fade to zero while vy is emitted.
  # Slip, height, contact schedule, swing lift, shortfall, heading, the
  # longitudinal anchor and every hold term stay untouched.
  for name in (
    "base_lin_vel_xy", "feet_xy_vel", "base_xy_position",
    "gait_symmetry", "gait_symmetry_yaw", "leg_symmetry",
  ):
    blend(name)

  # Tilt: pitch stays pinned, roll is free inside the weight-transfer envelope.
  blend("base_tilt", RewardTermCfg(
    func=mdp.lateral_balance_envelope_l2, weight=1.0,
    params={"max_roll": MAX_FREE_ROLL},
  ))
  blend("roll_pitch_rate", RewardTermCfg(func=mdp.base_pitch_rate_l2, weight=1.0))
  blend("track_angular_velocity", RewardTermCfg(
    func=lateral_yaw_pitch_tracking, weight=1.0,
    params={"command_name": "gait", "std": math.sqrt(0.5)},
  ))
  # Hip and ankle roll are the side-step joints; widen only their tolerance.
  lateral_pose = deepcopy(cfg.rewards["pose"])
  lateral_pose.params["std_walking"][JOINT_PATTERNS["hip_roll"]] = 0.25
  lateral_pose.params["std_walking"][JOINT_PATTERNS["ankle_roll"]] = 0.18
  blend("pose", lateral_pose)

  # The turn task's 60 mm heel floor exceeds the nominal crouch heel gap
  # (~49 mm), so the trailing foot of a side step would be charged for closing
  # to a gap it already stands at. Reachable floors while vy is emitted only.
  for name, floor in (
    ("feet_center_clearance", 0.050),
    ("feet_heel_clearance", 0.038),
    ("feet_toe_clearance", 0.055),
  ):
    clearance = deepcopy(cfg.rewards[name])
    clearance.params["minimum_separation"] = floor
    blend(name, clearance)

  # Velocity tracking SHARES the original +4 budget (75% lateral while vy is
  # emitted) rather than adding to it, so a side step is never worth more per
  # step than a straight walk and the hold/gait ratio stays where stage one
  # set it. Tracks the cycle mean: instantaneous vy is dominated by the
  # weight-transfer rock, which is periodic and averages out over one cycle.
  blend("track_linear_velocity", RewardTermCfg(
    func=LateralPlanarTracking, weight=1.0,
    params={
      "command_name": "gait", "std": LATERAL_TRACK_STD,
      "forward_std": cfg.rewards["track_linear_velocity"].params["std"],
      "lateral_share": 0.75,
    },
  ))

  ##
  # Lateral-only shaping (all gated by the emitted vy, exactly zero otherwise).
  ##
  foot_cfg = SceneEntityCfg("robot", site_names=list(FOOT_SITES))
  cfg.rewards["lateral_landing"] = RewardTermCfg(
    func=LateralFootSmoothness, weight=-0.30,   # impact, loaded slide, late drop
    params={"command_name": "gait", "asset_cfg": foot_cfg,
            "sensor_name": "feet_ground_contact"},
  )
  cfg.rewards["lateral_stride"] = RewardTermCfg(
    func=lateral_swing_velocity_l2, weight=-0.25,   # swing foot must travel
    params={"command_name": "gait", "asset_cfg": deepcopy(foot_cfg), "std": 0.08},
  )
  return cfg
