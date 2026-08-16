"""Robust low-speed omnidirectional locomotion for K2.

This task deliberately lives beside, rather than replacing, the proven
forward-only task.  It keeps the same crouch, gait clock, full-sole lift,
high-friction feet, actuator latency, and millimetre-scale rough terrain while
commanding body-frame forward/backward velocity, lateral velocity, and yaw
rate through mjlab's standard ``twist`` command.
"""

from __future__ import annotations

import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from k2_rl import mdp
from k2_rl.inplace_env_cfg import make_k2_inplace_env_cfg
from k2_rl.k2_constants import FOOT_SITES, JOINT_PATTERNS
from k2_rl.mdp import GaitCommandCfg, UniformVelocityCommandCfg


TRAIN_VX_RANGE = (-0.04, 0.04)
TRAIN_VY_RANGE = (-0.02, 0.02)
TRAIN_YAW_RATE_RANGE = (-0.35, 0.35)
PLAY_COMMAND = (0.02, 0.0, 0.0)
FULL_SWING_HEIGHT = 0.015


def _robust_terrain_cfg() -> TerrainEntityCfg:
  return TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      seed=43,
      curriculum=False,
      size=(2.0, 2.0),
      border_width=2.0,
      num_rows=5,
      num_cols=8,
      difficulty_range=(0.0, 1.0),
      sub_terrains={
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.35),
        "rough_2mm": terrain_gen.HfRandomUniformTerrainCfg(
          proportion=0.30,
          noise_range=(-0.002, 0.002),
          noise_step=0.001,
          horizontal_scale=0.04,
          vertical_scale=0.001,
          downsampled_scale=0.12,
          border_width=0.20,
        ),
        "rough_5mm": terrain_gen.HfRandomUniformTerrainCfg(
          proportion=0.20,
          noise_range=(-0.005, 0.005),
          noise_step=0.001,
          horizontal_scale=0.04,
          vertical_scale=0.001,
          downsampled_scale=0.12,
          border_width=0.20,
        ),
        "gentle_wave": terrain_gen.HfWaveTerrainCfg(
          proportion=0.15,
          amplitude_range=(0.002, 0.008),
          num_waves=2,
          horizontal_scale=0.04,
          vertical_scale=0.001,
          border_width=0.20,
        ),
      },
    ),
  )


def make_k2_full_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_k2_inplace_env_cfg(play=play)

  if not play:
    cfg.scene.terrain = _robust_terrain_cfg()

  # Standard body-frame velocity command: +x forward, +y robot-left, +yaw CCW.
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.resampling_time_range = (4.0, 8.0)
  twist.rel_standing_envs = 0.10 if not play else 0.0
  twist.heading_command = False
  twist.debug_vis = play
  if play:
    vx, vy, wz = PLAY_COMMAND
    twist.ranges = UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(vx, vx), lin_vel_y=(vy, vy), ang_vel_z=(wz, wz)
    )
  else:
    twist.ranges = UniformVelocityCommandCfg.Ranges(
      lin_vel_x=TRAIN_VX_RANGE,
      lin_vel_y=TRAIN_VY_RANGE,
      ang_vel_z=TRAIN_YAW_RATE_RANGE,
    )

  # Every non-standing command uses the same alternating low-frequency gait.
  gait = cfg.commands["gait"]
  assert isinstance(gait, GaitCommandCfg)
  gait.rel_march_envs = 1.0
  gait.forward_velocity_range = None

  # Actor observes the requested body-frame motion immediately before the gait
  # clock.  No absolute/relative heading observation is needed for yaw-rate
  # control, which keeps the task rotation-invariant and avoids gyro drift.
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    gait_obs = terms.pop("gait")
    terms["twist_command"] = ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    )
    terms["gait"] = gait_obs

  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=mdp.track_linear_velocity,
    weight=3.0,
    params={"command_name": "twist", "std": 0.04},
  )
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=mdp.track_angular_velocity,
    weight=1.5,
    params={"command_name": "twist", "std": 0.25},
  )
  # Translation and yaw are commanded in this task; only unsafe tilt, contact
  # slip, foot crossing, and excessive limb motion should be penalized.
  cfg.rewards["base_lin_vel_xy"].weight = 0.0
  cfg.rewards["base_xy_position"].weight = 0.0
  cfg.rewards["feet_xy_vel"].weight = 0.0
  cfg.rewards["feet_under_base"].weight = -0.25
  cfg.rewards["base_tilt"].weight = -2.5
  cfg.rewards["roll_pitch_rate"].weight = -0.10
  cfg.rewards["gait_swing"].weight = 6.0
  cfg.rewards["gait_swing"].params["target_height"] = FULL_SWING_HEIGHT
  cfg.rewards["gait_swing"].params["tracking_sigma"] = 0.004
  cfg.rewards["gait_contact"].weight = 2.0
  cfg.rewards["feet_slip"].weight = -2.0
  cfg.rewards["swing_sole_flat"].weight = -0.5

  # Permit the hip roll/yaw excursions needed for lateral and turning steps.
  walking_std = cfg.rewards["pose"].params["std_walking"]
  walking_std[JOINT_PATTERNS["hip_roll"]] = 0.25
  walking_std[JOINT_PATTERNS["hip_yaw"]] = 0.25
  walking_std[JOINT_PATTERNS["ankle_roll"]] = 0.18

  cfg.events["foot_friction"].params["ranges"] = (0.7, 1.4)
  cfg.events["reset_base"].params["pose_range"]["yaw"] = (-3.1416, 3.1416)

  if not play:
    action = cfg.actions["joint_pos"]
    assert isinstance(action, mdp.DelayedJointPositionActionCfg)
    action.min_delay_steps = 1
    action.max_delay_steps = 4

    cfg.events["body_inertia"].params["alpha_range"] = (-0.25, 0.25)
    cfg.events["pd_gains"].params["kp_range"] = (0.55, 1.15)
    cfg.events["pd_gains"].params["kd_range"] = (0.70, 1.30)
    cfg.events["encoder_bias"].params["bias_range"] = (-0.03, 0.03)
    cfg.events["base_com"].params["ranges"] = {
      0: (-0.015, 0.015),
      1: (-0.018, 0.018),
      2: (-0.025, 0.025),
    }
    cfg.events["push_robot"].interval_range_s = (2.0, 4.0)
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.22, 0.22),
      "y": (-0.22, 0.22),
      "roll": (-0.22, 0.22),
      "pitch": (-0.22, 0.22),
      "yaw": (-0.30, 0.30),
    }
    cfg.events["reset_base"].params["pose_range"].update({
      "roll": (-0.04, 0.04),
      "pitch": (-0.04, 0.04),
    })
    cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)

  # These curricula target zero translation and would fight commanded motion.
  for name in ("cur_base_lin_vel_xy", "cur_feet_under_base", "cur_feet_xy_vel"):
    cfg.curriculum.pop(name, None)

  return cfg
