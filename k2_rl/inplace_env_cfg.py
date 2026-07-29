"""K2 in-place velocity task: hold (stand still) + march (step in place).

One environment, one policy, a runtime switch. The :class:`GaitCommand` emits a
per-episode ``march`` bit (a fraction ``rel_march_envs`` of envs march, the rest
hold) plus a phase clock; the policy observes it and learns both behaviours.
Zero commanded translation in both modes -> the robot stays on the spot.

Flat ground, proprio-only actor observations (gyro + projected gravity + joints
+ last action + gait), matching what the Pi 5 can supply on the real robot
(``robot.md`` sec.9). Domain randomization covers the exact sim-to-real gaps
chased during bring-up: mass, friction, actuator gains, encoder/home offsets,
CoM, and pushes.

Built from scratch (not the rough velocity template) to keep it minimal and
readable, per the RL note in ``robot.md`` sec.10 (build fresh, never edit mjlab
site-packages).
"""

from __future__ import annotations

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from k2_rl import mdp
from k2_rl.k2_constants import (
  CROUCH_BASE_HEIGHT,
  FOOT_BODIES,
  FOOT_GEOMS,
  FOOT_HEEL_SITES,
  FOOT_INNER_SITES,
  FOOT_OUTER_SITES,
  FOOT_SITES,
  FOOT_TOE_SITES,
  JOINT_PATTERNS,
  K2_ACTION_SCALE,
  get_k2_robot_cfg,
)
from k2_rl.mdp import GaitCommandCfg
from k2_rl.mdp import UniformVelocityCommandCfg

# Keep the reward target synchronized with the model-derived reset crouch.
TARGET_BASE_HEIGHT = CROUCH_BASE_HEIGHT
# Swing-foot lift target while marching. Higher = a cleaner, unambiguous lift
# (foot fully clears the floor) so it can't skim forward into a walk.
SWING_HEIGHT = 0.025  # 2.5 cm: clear lift without overloading the support leg
# Stepping cadence (Hz). Gentler than the first attempt so the real STS3215s
# can actually complete each lift/plant under load.
GAIT_FREQ_HZ = 0.75


def make_k2_inplace_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  hardware_finetune = os.environ.get("K2_HARDWARE_FINETUNE") == "1"
  ##
  # Scene: flat plane + K2 robot + feet contact sensor.
  ##
  feet_ground = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(Feet_left|Feet_right)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  feet_self_contact = ContactSensorCfg(
    name="feet_self_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern="Feet_left",
      entity="robot",
    ),
    secondary=ContactMatch(
      mode="subtree",
      pattern="Feet_right",
      entity="robot",
    ),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
  )

  scene = SceneCfg(
    num_envs=1 if play else 4096,
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"robot": get_k2_robot_cfg()},
    sensors=(feet_ground, feet_self_contact),
    extent=2.0,
  )

  ##
  # Observations.
  ##
  # Actor: proprioceptive only (what the Pi can produce). No base linear
  # velocity -- the IMU velocimeter isn't trustworthy on hardware.
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      delay_max_lag=0 if play else 1,
      delay_hold_prob=0.9,
      delay_update_period=5,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      # Slightly wider noise than G1: on the real robot projected gravity is
      # estimated from the accelerometer, which is corrupted by the body's own
      # linear acceleration while stepping -- train the policy to tolerate that.
      # (Too wide, e.g. 0.1, and it can't learn balance.)
      noise=Unoise(n_min=-0.07, n_max=0.07),
      delay_max_lag=0 if play else 1,
      delay_hold_prob=0.9,
      delay_update_period=5,
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      delay_max_lag=0 if play else 1,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      delay_max_lag=0 if play else 1,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "gait": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "gait"},
    ),
  }
  # Critic: privileged extras (no corruption).
  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }
  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=not play
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False
    ),
  }

  ##
  # Actions.
  ##
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": mdp.DelayedJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=K2_ACTION_SCALE,
      use_default_offset=True,
      max_delay_substeps=0 if play else 5,
      max_target_rate=1.0,
    )
  }

  ##
  # Commands: zero twist (stay in place) + gait (hold/march switch).
  ##
  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(1e9, 1e9),  # never resample; always zero
      heading_command=False,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 0.0),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
      ),
    ),
    "gait": GaitCommandCfg(
      entity_name="robot",
      gait_freq=GAIT_FREQ_HZ,
      rel_march_envs=0.0 if play else 0.65,
      resampling_time_range=(4.0, 6.0),
      debug_vis=False,
    ),
  }

  ##
  # Events (reset + domain randomization).
  ##
  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.05, 0.05),
          "y": (-0.05, 0.05),
          "z": (0.0, 0.02),
          "yaw": (-0.2, 0.2),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(3.0, 5.0),
      params={
        "velocity_range": {
          "x": (-0.18, 0.18),
          "y": (-0.18, 0.18),
          "roll": (-0.18, 0.18),
          "pitch": (-0.18, 0.18),
          "yaw": (-0.25, 0.25),
        }
      },
    ),
    # --- Domain randomization (sim-to-real gaps from bring-up) ---
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=list(FOOT_GEOMS)),
        "operation": "abs",
        "ranges": (0.4, 1.1),
        "shared_random": True,
      },
    ),
    "body_inertia": EventTermCfg(
      mode="startup",
      func=dr.pseudo_inertia,
      params={
        # Only the real (massive) link bodies -- NOT the massless k2_root
        # free-joint carrier, whose zero inertia makes the mass matrix singular.
        "asset_cfg": SceneEntityCfg(
          "robot",
          body_names=(r"base_link", r"hip_.*", r"Knee_.*", r"ankle_.*", r"Feet_.*"),
        ),
        "alpha_range": (-0.15, 0.15),  # ~+-15% mass, inertia scaled consistently
      },
    ),
    "pd_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.8, 1.2),
        "kd_range": (0.8, 1.2),
        "operation": "scale",
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.02, 0.02),  # ~+-1.1 deg home/calibration offset
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "operation": "add",
        "ranges": {
          0: (-0.01, 0.01),
          1: (-0.01, 0.01),
          2: (-0.02, 0.02),
        },
      },
    ),
  }

  ##
  # Rewards.
  ##
  # Per-joint pose tolerance. Legs (pitch/knee/ankle_pitch) loosen while
  # marching so the policy can step; roll/yaw stay tight for lateral stability.
  std_standing = {p: 0.1 for p in JOINT_PATTERNS.values()}
  std_walking = {
    JOINT_PATTERNS["hip_pitch"]: 0.35,
    JOINT_PATTERNS["hip_roll"]: 0.12,
    JOINT_PATTERNS["hip_yaw"]: 0.12,
    JOINT_PATTERNS["knee"]: 0.4,
    JOINT_PATTERNS["ankle_pitch"]: 0.3,
    JOINT_PATTERNS["ankle_roll"]: 0.1,
  }

  rewards = {
    # Stay in place. Proportional (quadratic) penalties do the real work here --
    # a wide exp kernel on velocity is flat near zero and lets the march drift
    # into a walk. Keep a little exp shaping on top.
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=0.75,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    # In-place shaping penalties. These are unbounded quadratics that blow up
    # during early random flailing and stop the robot ever learning to stand,
    # so they START AT ZERO and are ramped in by the curriculum below (after the
    # policy can already stand + march). Targets: base_lin_vel_xy -1.0,
    # feet_under_base -2.0, feet_xy_vel -1.5. They are ~0 for a clean in-place
    # march (feet move only vertically, body doesn't translate).
    "base_lin_vel_xy": RewardTermCfg(
      func=mdp.base_lin_vel_xy_l2,
      weight=-10.0 if hardware_finetune else 0.0,
    ),
    "feet_under_base": RewardTermCfg(
      func=mdp.feet_under_base_l2,
      weight=-6.0 if hardware_finetune else 0.0,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=list(FOOT_SITES))},
    ),
    # Whole-foot clearance. A center-only term missed inward foot yaw: the
    # centers stayed separated while the rear edges collided. Shape center,
    # heel, and toe independently, then penalize exact mesh contact below.
    "feet_center_clearance": RewardTermCfg(
      func=mdp.feet_lateral_clearance_l2,
      weight=-1.0,
      params={
        "minimum_separation": 0.060,
        "asset_cfg": SceneEntityCfg("robot", site_names=list(FOOT_SITES)),
      },
    ),
    "feet_heel_clearance": RewardTermCfg(
      func=mdp.feet_lateral_clearance_l2,
      weight=-2.0,
      params={
        "minimum_separation": 0.060,
        "asset_cfg": SceneEntityCfg("robot", site_names=list(FOOT_HEEL_SITES)),
      },
    ),
    "feet_toe_clearance": RewardTermCfg(
      func=mdp.feet_lateral_clearance_l2,
      weight=-2.0,
      params={
        "minimum_separation": 0.060,
        "asset_cfg": SceneEntityCfg("robot", site_names=list(FOOT_TOE_SITES)),
      },
    ),
    "feet_inner_clearance": RewardTermCfg(
      func=mdp.feet_lateral_clearance_l2,
      weight=-4.0,
      params={
        "minimum_separation": 0.012,
        "asset_cfg": SceneEntityCfg(
          "robot", site_names=list(FOOT_INNER_SITES)
        ),
      },
    ),
    "feet_self_collision": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-6.0,
      params={"sensor_name": "feet_self_contact"},
    ),
    # Clean vertical marching: feet only move up/down, never skim forward.
    "feet_xy_vel": RewardTermCfg(
      func=mdp.feet_xy_vel_l2,
      weight=-8.0 if hardware_finetune else 0.0,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=list(FOOT_SITES))},
    ),
    "upright": RewardTermCfg(
      func=mdp.flat_orientation,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      },
    ),
    "roll_pitch_rate": RewardTermCfg(
      func=mdp.base_ang_vel_xy_l2,
      weight=-0.12 if hardware_finetune else -0.03,
    ),
    "base_tilt": RewardTermCfg(
      func=mdp.base_tilt_l2,
      weight=-3.0 if hardware_finetune else 0.0,
    ),
    "base_xy_position": RewardTermCfg(
      func=mdp.base_xy_position_l2,
      weight=-8.0 if hardware_finetune else 0.0,
    ),
    "base_height": RewardTermCfg(
      func=mdp.base_height_l2,
      weight=-25.0,
      params={"target_height": TARGET_BASE_HEIGHT},
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=0.5,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "gait",
        "std_standing": std_standing,
        "std_walking": std_walking,
        "std_running": std_walking,
        "walking_threshold": 0.05,
        "running_threshold": 5.0,
      },
    ),
    # Gait: march schedule + swing lift (marching envs only).
    "gait_contact": RewardTermCfg(
      func=mdp.gait_contact,
      weight=1.0,
      params={"sensor_name": "feet_ground_contact", "command_name": "gait"},
    ),
    "gait_swing": RewardTermCfg(
      func=mdp.gait_swing,
      weight=2.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "gait",
        "target_height": SWING_HEIGHT,
        "center_cfg": SceneEntityCfg("robot", site_names=list(FOOT_SITES)),
        "heel_cfg": SceneEntityCfg("robot", site_names=list(FOOT_HEEL_SITES)),
        "toe_cfg": SceneEntityCfg("robot", site_names=list(FOOT_TOE_SITES)),
        "inner_cfg": SceneEntityCfg("robot", site_names=list(FOOT_INNER_SITES)),
        "outer_cfg": SceneEntityCfg("robot", site_names=list(FOOT_OUTER_SITES)),
      },
    ),
    "swing_sole_flat": RewardTermCfg(
      func=mdp.swing_sole_height_spread_l2,
      weight=-0.25,
      params={
        "command_name": "gait",
        "heel_cfg": SceneEntityCfg("robot", site_names=list(FOOT_HEEL_SITES)),
        "toe_cfg": SceneEntityCfg("robot", site_names=list(FOOT_TOE_SITES)),
        "inner_cfg": SceneEntityCfg("robot", site_names=list(FOOT_INNER_SITES)),
        "outer_cfg": SceneEntityCfg("robot", site_names=list(FOOT_OUTER_SITES)),
        "tolerance": 0.005,
      },
    ),
    # Hold: keep both feet planted (holding envs only).
    "feet_planted": RewardTermCfg(
      func=mdp.feet_planted,
      weight=0.5,
      params={"sensor_name": "feet_ground_contact", "command_name": "gait"},
    ),
    # Penalties (both modes).
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-2.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.15),
    "action_rate_limit": RewardTermCfg(
      func=mdp.action_rate_soft_limit_l2,
      weight=-0.2 if hardware_finetune else 0.0,
      params={"action_scale": K2_ACTION_SCALE, "max_rate": 0.9},
    ),
    "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.01),
    "joint_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=-2e-3),
    "feet_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "gait",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=list(FOOT_SITES)),
      },
    ),
  }

  ##
  # Terminations.
  ##
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(45.0)},
    ),
    "collapsed": TerminationTermCfg(
      func=mdp.root_height_below_minimum,
      params={"minimum_height": 0.18},
    ),
  }

  ##
  # Metrics.
  ##
  metrics = {"mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc)}

  cfg = ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    # Ramp the in-place shaping penalties in AFTER the robot learns to stand +
    # march (~iter 250), reaching full strength by ~iter 800. step = iter * 24
    # (num_steps_per_env). Without this the penalties block initial learning.
    curriculum={} if play or hardware_finetune else {
      "cur_base_lin_vel_xy": CurriculumTermCfg(
        func=mdp.reward_weight,
        params={"reward_name": "base_lin_vel_xy", "weight_stages": [
          {"step": 250 * 24, "weight": -0.3},
          {"step": 500 * 24, "weight": -0.6},
          {"step": 800 * 24, "weight": -1.0},
        ]},
      ),
      "cur_feet_under_base": CurriculumTermCfg(
        func=mdp.reward_weight,
        params={"reward_name": "feet_under_base", "weight_stages": [
          {"step": 250 * 24, "weight": -0.6},
          {"step": 500 * 24, "weight": -1.2},
          {"step": 800 * 24, "weight": -2.0},
        ]},
      ),
      "cur_feet_xy_vel": CurriculumTermCfg(
        func=mdp.reward_weight,
        params={"reward_name": "feet_xy_vel", "weight_stages": [
          {"step": 250 * 24, "weight": -0.4},
          {"step": 500 * 24, "weight": -0.9},
          {"step": 800 * 24, "weight": -1.5},
        ]},
      ),
      "cur_action_rate_limit": CurriculumTermCfg(
        func=mdp.reward_weight,
        params={"reward_name": "action_rate_limit", "weight_stages": [
          {"step": 150 * 24, "weight": -0.01},
          {"step": 300 * 24, "weight": -0.03},
          {"step": 600 * 24, "weight": -0.06},
        ]},
      ),
    },
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=1.2,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      njmax=300,
      nconmax=None,
      contact_sensor_maxmatch=64,
      mujoco=MujocoCfg(
        timestep=0.002,  # K2 twin is validated at 2 ms (5 ms bounces)
        iterations=10,
        ls_iterations=20,
        ccd_iterations=50,
      ),
    ),
    decimation=10,  # 2 ms x 10 = 50 Hz control (matches hardware)
    episode_length_s=20.0,
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg
