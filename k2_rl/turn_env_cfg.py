"""Turn/walk task and its shared robot, standing and walking foundations."""
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
from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg
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
      # IMU DR: per-step noise PLUS a per-episode constant gyro bias. The
      # deployed LSM6DS3 keeps a residual bias after the boot capture (measured
      # ~0.06 deg/s on the ESP32 log, up to ~1 deg/s if the robot is touched
      # during capture). Training against a fixed offset stops the policy from
      # treating a small constant rate as real motion to be corrected.
      noise=NoiseModelWithAdditiveBiasCfg(
        noise_cfg=Unoise(n_min=-0.2, n_max=0.2),
        bias_noise_cfg=Unoise(n_min=-0.02, n_max=0.02),  # +/-1.1 deg/s
      ),
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
      # Per-episode constant gravity-direction bias on top of the per-step
      # noise: an IMU mounting/level error of up to ~2 deg. Without it the
      # policy chases the gravity vector exactly, so a 1-2 deg level offset on
      # the real robot becomes a permanent lean and, in hold, a slow shuffle
      # as it keeps "correcting" a tilt that is not there. With it the policy
      # has to stand on its feet (flat soles at the default crouch), not on the
      # gravity reading, which is what survives a re-mounted IMU or a battery.
      noise=NoiseModelWithAdditiveBiasCfg(
        noise_cfg=Unoise(n_min=-0.07, n_max=0.07),
        bias_noise_cfg=Unoise(n_min=-0.035, n_max=0.035),  # ~2 deg
      ),
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
        # base_link is Y-up (see `upright`): local x -> world Y (lateral),
        # local y -> world Z (vertical), local z -> world X (fore/aft). The
        # battery pack's exact position is the uncertain part, so the fore/aft
        # and vertical axes get the wider ranges.
        "ranges": {
          0: (-0.010, 0.010),  # lateral
          1: (-0.020, 0.020),  # vertical
          2: (-0.015, 0.015),  # fore/aft
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
        # ROOT LINK (k2_root, Z-up). NOT base_link: that body carries
        # quat="0.5 0.5 0.5 0.5" in the MJCF, so its local Y is world-up and a
        # perfectly level robot scored exp(-1/0.2) = 0.0067 -- the constant
        # 0.0068 seen in every run's Episode_Reward/upright. The term was dead.
        # (Must be NAMED: an unnamed SceneEntityCfg selects every body.)
        "asset_cfg": SceneEntityCfg("robot", body_names=("k2_root",)),
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
    # Do not force the captured crouch during hold. Hardware/digital-twin
    # validation showed that constraint can defeat the policy's previously
    # stable, slightly wider learned support stance. Feet-planted plus the
    # existing balance/posture terms reproduce the proven hold behavior.
    "hold_pair_symmetry": RewardTermCfg(
      func=mdp.hold_joint_pair_symmetry_l2,
      weight=-3.0,
      params={
        "command_name": "gait",
        "right_cfg": SceneEntityCfg(
          "robot",
          joint_names=[
            "base_link_hip_pitch_right_joint",
            "hip_pitch_hip_roll_right_joint",
            "hip_roll_hip_yaw_right_joint",
            "hip_yaw_knee_right_joint",
            "knee_ankle_pitch_right_joint",
            "ankle_roll_foot_right_joint",
          ],
        ),
        "left_cfg": SceneEntityCfg(
          "robot",
          joint_names=[
            "base_link_hip_pitch_left_joint",
            "hip_pitch_hip_roll_left_joint",
            "hip_roll_hip_yaw_left_joint",
            "hip_yaw_knee_left_joint",
            "knee_ankle_pitch_left_joint",
            "ankle_roll_foot_left_joint",
          ],
        ),
      },
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



import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from k2_rl import mdp
from k2_rl.k2_constants import FOOT_GEOMS, FOOT_SITES
from k2_rl.mdp import GaitCommandCfg


# Checkpoint 2000 was visibly cleaner than the longer-stride checkpoint 2300.
# Keep the proven conservative range: at 0.75 Hz this asks for no more than
# roughly 2.7 cm per alternating step.
TRAIN_FORWARD_RANGE = (0.01, 0.04)
PLAY_FORWARD_SPEED = 0.03
# Hardware checkpoint 1800 lifted aggressively enough to unload the stance
# side and provoke large ankle-roll corrections. Twelve millimetres retains
# useful clearance over the 5 mm support-height DR while reducing single-leg
# leverage. Validate the realized full-sole clearance before hardware export.
FORWARD_SWING_HEIGHT = 0.012  # base tasks keep 12 mm


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
    weight=4.0,
    params={"command_name": "gait", "std": 0.015},
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
  # Keep the support polygon near the body while permitting the conservative
  # step length above.
  cfg.rewards["feet_under_base"].weight = -0.25
  # Iteration 2000 learned a lopsided gait: left hip pitch swung at 0.45x the
  # right in simulation and 0.31x on hardware, which is the likely source of
  # the persistent physical-left lean that needs a runtime balance trim.
  cfg.rewards["gait_symmetry"] = RewardTermCfg(
    func=mdp.gait_antisymmetry_l2,
    # Exact value used by bidir_symmetric_swing12mm_iter2500.onnx (v5).
    weight=-4.0,
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
    cfg.events["sole_height"] = EventTermCfg(
      mode="startup",
      func=dr.geom_pos,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=list(FOOT_GEOMS)
        ),
        "operation": "add",
        "ranges": (-0.0015, 0.0015),
        # Feet_* bodies are Y-up: body axis 1 is world-vertical. Axis 2 is
        # fore/aft and only slid the soles along the foot.
        "axes": [1],
        "shared_random": False,
      },
    )
    cfg.events["sole_tilt"] = EventTermCfg(
      mode="startup",
      func=dr.geom_quat,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=list(FOOT_GEOMS)
        ),
        # Y-up foot body: rotations about body x and z tilt the sole against
        # the floor; "pitch" (about body y) only spins it about the vertical.
        "roll_range": (-math.radians(1.0), math.radians(1.0)),
        "yaw_range": (-math.radians(1.0), math.radians(1.0)),
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



import math

from mjlab.envs import ManagerBasedRlEnvCfg

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
    cfg.events["sole_tilt"].params["yaw_range"] = (
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



import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from k2_rl import mdp
from k2_rl.k2_constants import FOOT_SITES, JOINT_PATTERNS
from k2_rl.mdp import GaitCommandCfg


# Deliberately conservative for a first turning policy, in the same spirit as
# the forward range that was walked back from 0.08 to 0.04 m/s. At the 0.75 Hz
# cadence (0.67 s per step) 0.25 rad/s asks for 9.5 degrees of yaw per step,
# which the hip-yaw joints can supply without a lateral weight-shift strategy.
# Hardware realizes roughly half of a commanded rate, so expect ~7 deg/s.
TRAIN_YAW_RATE_RANGE = (-0.25, 0.25)
PLAY_YAW_RATE = 0.15
# Share of resamples that receive a nonzero yaw command. The remaining 40%
# reproduce the bidirectional task exactly.
REL_TURNING_ENVS = 0.6
# Share of sampled TIME spent marching; the rest is a two-foot standstill.
# Deliberately high: walking is the expensive behaviour and must not lose data
# to standing, which is far cheaper to learn. 0.90 leaves 10% for the
# standstill, the same share `full_env_cfg.py` gives its standing mode.
#
# This is the ONLY knob that trades gait data against standstill quality --
# every other standstill term is gated by `1 - march` and cannot touch the gait
# at all. Lower it if the standstill comes out weak; 0.60 was tried and cost
# the gait too much. 0.85: the hold now carries four shaping terms (sole
# alignment, stagger, stillness, pair symmetry) and needs a little more data
# than the bare feet_planted hold did at 0.90.
REL_MARCH_ENVS = 0.85
# Swing apex for every gait (walk, turn, side step). 12 mm was chosen when the
# robot weighed 1.28 kg; the 375 g battery assembly makes the stance side work
# harder and the observed forward/backward gait skims. 15 mm with the same
# relative tracking tolerance (a third of the target) gives an unambiguous
# lift. `swing_shortfall` below keeps a gradient alive when a foot drags, which
# the exponential tracking kernel loses entirely 2-3 sigma from its target.
SWING_HEIGHT_M = 0.015
SWING_TRACKING_SIGMA = 0.005
# Hold stance geometry. Fore/aft foot offset (stagger) and lateral excess width
# are measured on the FEET, not the joints, so they stay valid under the
# per-foot sole randomization where level joints are not symmetric joints.
HOLD_STAGGER_TOLERANCE_M = 0.020
# Nominal lateral foot-centre gap at the crouch, measured on the compiled MJCF.
# `hold_stance_width` charges only EXCESS over this, and the clearance floors
# sit at 50-55 mm, leaving a dead band where neither term is active.
NOMINAL_STANCE_WIDTH = 0.0676
# Swing-onset threshold for the contact schedule. See `gait_contact`: at the
# inherited 0.18 the schedule demands the foot be off the ground while
# `gait_swing`'s target for it is still 0.4 mm, so the two terms very nearly
# cancel and the policy gets almost no gradient about WHEN to break contact.
# 0.35 is the first value whose net is positive regardless of how much
# clearance actually breaks contact (1, 2 or 4 mm), at the cost of widening
# double support from 11.5% to 22.8% of the cycle.
SWING_THRESHOLD = 0.35
# Full-scale reversal (0.5 rad/s) in 1.7 s, matching the time the 0.06 m/s^2
# linear ramp takes to reverse across its own +/-0.05 m/s range.
YAW_RATE_RAMP_RPS2 = 0.30
# Yaw command at which the straight-gait symmetry penalty has faded to ~13%.
SYMMETRY_YAW_FADE_STD = 0.12
# Index of the yaw channel in the 5-D gait command [march, sin, cos, vx, wz].
YAW_COMMAND_INDEX = 4


def make_k2_turn_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_k2_bidir_env_cfg(play=play)

  gait = cfg.commands["gait"]
  assert isinstance(gait, GaitCommandCfg)
  gait.yaw_rate_range = (
    (PLAY_YAW_RATE, PLAY_YAW_RATE) if play else TRAIN_YAW_RATE_RANGE
  )
  # A MINIMAL STANDSTILL, bought with as little gait data as possible.
  #
  # `rel_march_envs` is a fraction of TIME, not of environments: `march_target`
  # is re-drawn on every resample, i.e. every 4-7 s, so each env switches mode
  # several times per 20 s episode. That is why a small hold share still gets
  # ample hold<->walk transition practice.
  #
  # The previous hold-restored attempt used 0.60, spending 40% of the sampled
  # time standing -- and the walking gait paid for it, because the gait is the
  # hard behaviour and it lost nearly half its data. Standing still is much
  # cheaper to learn than walking, so it gets the smaller slice: 15% here,
  # matching the 0.10 that `full_env_cfg.py` uses for its standing share.
  # Everything about the gait itself is untouched.
  gait.rel_march_envs = 1.0 if play else REL_MARCH_ENVS
  gait.rel_turning_envs = 1.0 if play else REL_TURNING_ENVS
  gait.yaw_rate_ramp_rate_rps2 = YAW_RATE_RAMP_RPS2

  # The actor's heading feature must stay near zero at every commanded yaw rate,
  # otherwise turning walks the warm-started policy straight out of the input
  # range it was trained on. Reassigning the existing key preserves the term
  # order the deployment observation layout depends on (heading, then gait).
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    assert list(terms)[-2:] == ["heading", "gait"], list(terms)
    terms["heading"] = ObservationTermCfg(
      func=mdp.heading_error_sin_cos,
      params={"command_name": "gait"},
      noise=Unoise(n_min=-0.02, n_max=0.02),
      delay_max_lag=0 if play else 1,
    )

  # Follow the commanded heading rather than world +X, and FADE THE TERM WITH
  # COMMANDED YAW. `heading_ref` integrates the yaw command, so this is the
  # time-integral of the same error `yaw_rate` and `track_angular_velocity`
  # already measure directly -- a stacked P+P+I in which only the I can wind
  # up. Any persistent rate deficit (hardware realizes ~half a commanded rate)
  # accumulates here unopposed; at 0.25 rad/s over a 20 s episode that is 2.5
  # rad of uncatchable reference, worth about -9/step, larger than the whole
  # positive budget. Fading with |wz| removes the windup exactly where it lives
  # and leaves the straight-line signal untouched at wz == 0.
  cfg.rewards["heading"] = RewardTermCfg(
    func=mdp.heading_reference_error,
    weight=-5.0,
    params={
      "command_name": "gait",
      "yaw_command_index": YAW_COMMAND_INDEX,
      "yaw_fade_std": SYMMETRY_YAW_FADE_STD,
    },
  )
  # Nothing constrained position ALONG the commanded path -- the lateral term
  # takes only the perpendicular component. Harmless while walking (the
  # reference moves with the command), but with no hold mode left, "stand
  # still" has to mean something, and this is the term that says it. Gated to
  # ~zero commanded speed so it can never wind up the way `heading` could.
  cfg.rewards["path_longitudinal"] = RewardTermCfg(
    func=mdp.longitudinal_path_deviation_l2,
    weight=-20.0,
    params={"command_name": "gait"},
  )
  # Stay in the corridor around the commanded path rather than the spawn line.
  cfg.rewards["base_xy_position"] = RewardTermCfg(
    func=mdp.lateral_path_deviation_l2,
    weight=-20.0,
    params={"command_name": "gait"},
  )
  # Damp only the yaw rate the command did not ask for.
  cfg.rewards["yaw_rate"] = RewardTermCfg(
    func=mdp.yaw_rate_error_l2,
    weight=-1.0,
    params={"command_name": "gait", "yaw_command_index": YAW_COMMAND_INDEX},
  )
  # Same kernel and std as the inherited term; the yaw target now comes from the
  # gait command instead of the permanently-zero twist command.
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=mdp.track_yaw_rate_exp,
    weight=1.5,
    params={
      "command_name": "gait",
      "std": math.sqrt(0.5),
      "yaw_command_index": YAW_COMMAND_INDEX,
    },
  )
  # "Sideways" is a body-frame direction once the robot may face anywhere.
  cfg.rewards["feet_xy_vel"] = RewardTermCfg(
    func=mdp.feet_lateral_vel_body_l2,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", site_names=list(FOOT_SITES))},
  )
  # Do not charge a turning gait for the stride asymmetry that turning requires.
  cfg.rewards["gait_symmetry"].params["yaw_command_index"] = YAW_COMMAND_INDEX
  cfg.rewards["gait_symmetry"].params["yaw_fade_std"] = SYMMETRY_YAW_FADE_STD

  # NOTE (2026-08-21): a +/-0.06 rad encoder_bias widening was tried here and
  # REVERTED. It rested on the belief that the robot stood ~5 deg more
  # asymmetric than the twin -- but comparing the SAME policy's hold pose in sim
  # vs hardware showed the twin's own hip-yaw split is +6.15 deg against the
  # robot's +4.99, i.e. the real robot is slightly MORE symmetric, and every
  # joint agrees within 1.5 deg. There was no asymmetry to be robust to.
  # Worse, telling the policy to distrust its encoders raised loop gain: on
  # hardware, action-rate RMS +51%, jerk +53..87%, and gyro energy in the known
  # 2.5-8 Hz closed-loop resonance band went from 54% to 72%. The twin cannot
  # see that failure -- it has no resonance -- so the 128-robot score IMPROVED
  # (5.9 -> 11.8) while the real robot got visibly worse. Do not widen this
  # without a hardware jitter metric in the loop.

  # Hardware root cause (2026-08-21): marching in place with ZERO yaw
  # commanded, the real robot yaws left at +0.61 deg/s while the twin yaws
  # +0.06 deg/s -- a 10x gap that matches the +0.54 deg/s additive offset
  # measured during commanded turns, and fully explains "left turns fast, right
  # turns slow" (it adds to +wz and subtracts from -wz). It is NOT sensor bias:
  # the static residual after ImuAttitude debiasing is +0.024 deg/s, 5% of it
  # (`hardware/test_imu_yaw_bias.py`).
  #
  # The cause is the gait's own left/right asymmetry. Hip yaw is by far the
  # worst joint -- swing amplitude R:L is 0.41 in sim and 0.58 on hardware, so
  # the left leg twists roughly twice as far as the right. Equal and opposite
  # leg twists cancel their reaction torque on the body; unequal ones ratchet it
  # around. The twin hides this because its feet do not slip or comply.
  #
  # `gait_antisymmetry_l2` already penalizes exactly this (u_R + u_L)^2 form and
  # is only applied to hip pitch today. Hip yaw wants the same treatment, faded
  # with commanded yaw so a real turn -- which is asymmetric on purpose -- is
  # not charged for it.
  cfg.rewards["gait_symmetry_yaw"] = RewardTermCfg(
    func=mdp.gait_antisymmetry_l2,
    weight=-4.0,
    params={
      "command_name": "gait",
      "right_cfg": SceneEntityCfg(
        "robot", joint_names=[r"hip_roll_hip_yaw_right_joint"]
      ),
      "left_cfg": SceneEntityCfg(
        "robot", joint_names=[r"hip_roll_hip_yaw_left_joint"]
      ),
      "yaw_command_index": YAW_COMMAND_INDEX,
      "yaw_fade_std": SYMMETRY_YAW_FADE_STD,
    },
  )

  # STRIDE LENGTH. At a fixed cadence, stride is not a free parameter:
  #   stride = commanded_speed / (2 * gait_freq)
  # so `feet_under_base` (below) only PERMITS a longer step, it cannot demand
  # one -- easing it from -0.25 to -0.15 measurably changed nothing. Raising the
  # commanded speed is the other arithmetic option and is the wrong lever (it
  # was walked back once already). That leaves cadence.
  #
  # 0.75 -> 0.58 Hz is a 30% longer step at the same speed, which takes the
  # measured forward hip-pitch swing from ~6.7 deg toward ~8.7 deg (+2 deg).
  # TRADE-OFF: a slower cycle means longer single support, and lateral rocking
  # at gait frequency is already what caps this robot's speed (it fell at
  # 0.025 m/s, and touched 13.8 of 14 deg at 0.018). Slower may give the policy
  # more time to correct, or may give the rocking more time to build. Watch
  # rollp2p and tilt in evaluation, not just the stride number.
  # 0.75 -> 0.58 was validated on hardware (backward roll p2p 19.8 -> 14.3 deg,
  # tilt max 13.4 -> 8.0, airborne-leg velocity RMS down 20-40%), so the worry
  # that longer single support would worsen the lateral rocking was wrong -- it
  # damped it. Operator asked for a further 25% step: 0.58 / 1.25 = 0.46 Hz.
  # That is a 1.09 s swing per step, 63% longer than the original 0.67 s.
  # 0.75 -> 0.58 Hz, validated on hardware: backward roll p2p 19.8 -> 14.3 deg,
  # tilt max 13.4 -> 8.0, airborne-leg velocity RMS down 20-40%, and it survived
  # 0.025 m/s -- the speed at which the 0.75 Hz gait fell. Longer single support
  # DAMPED the lateral rocking rather than feeding it.
  #
  # This is the exact cadence that produced turn_v5_iter2900. Two later
  # experiments were tried and REVERTED at the operator's request:
  #   0.46 Hz (+25% stride) -- worked, but visibly too slow.
  #   0.70 Hz with vx +/-0.07 -- faster legs, untested on hardware.
  # 0.58 Hz -- the exact cadence of the v13_nohold run, restored at the
  # operator's request. It is 0.75 / 1.30, i.e. the arithmetic answer to "30%
  # more stride", and 0.75 -> 0.58 is the one cadence change measured on the
  # real robot: backward roll p2p 19.8 -> 14.3 deg, tilt max 13.4 -> 8.0, and
  # it survived 0.025 m/s, the speed at which the 0.75 Hz gait fell.
  cfg.commands["gait"].gait_freq = 0.58

  # `feet_under_base` is the one term whose PURPOSE is to keep
  # the feet under the body, so it is what caps step length -- the forward task
  # already halved it from -0.5 to -0.25 when the speed range was raised. Ease
  # it a little further for a slightly longer step. Deliberately a small move:
  # this term is also what stops the swing foot landing far ahead of the body,
  # and the robot is already close to a lateral-rocking limit at 0.018 m/s.
  cfg.rewards["feet_under_base"].weight = -0.15

  # DOMAIN RANDOMIZATION -- inherited from the bidirectional task, unchanged.
  # This is the turn_v5_iter2900 recipe. Widenings tried and reverted:
  #   encoder_bias +/-0.06 BACKFIRED on hardware (action rate +51%, jerk +87%,
  #     gyro energy in the 2.5-8 Hz resonance band 54% -> 72%) while the twin
  #     scored it two points HIGHER, because the twin has no resonance.
  #   friction 0.2-1.5 / inertia +/-20% / sole 6 mm / vx +/-0.07 (run v8) --
  #     scored well (14.58 vs 13.85) but the operator judged the resulting gait
  #     a limp on hardware. Reverted in favour of fixing the limp directly.

  # Lower overall loop gain -- this is what carries into the hardware buzz, and
  # unlike the removed hold-only damping term it has real gradient because the
  # policy moves plenty while walking.
  # overall loop gain, which is what actually carries into the hold buzz.
  cfg.rewards["action_rate_l2"].weight = -0.30
  cfg.rewards["action_acc_l2"].weight = -0.03
  # The hinge on target rates above 0.9 rad/s is the one smoothness term that
  # charges only the jerky part of a motion; the inherited curriculum topped
  # out at -0.06 and the trained episodes still paid -0.25 of it, i.e. the
  # ankle-roll corrections were sitting on the rate limiter. Top out higher.
  if "cur_action_rate_limit" in cfg.curriculum:  # absent in play / finetune
    cfg.curriculum["cur_action_rate_limit"].params["weight_stages"] = [
      {"step": 150 * 24, "weight": -0.02},
      {"step": 300 * 24, "weight": -0.05},
      {"step": 600 * 24, "weight": -0.10},
    ]

  # LEG SYMMETRY. Operator observation on hardware: the robot limps -- and the
  # logs agree, hip-pitch swing measured 8.3 deg right against 9.7 left in
  # forward walking (~15% mismatch), with the same split backward.
  #
  # `gait_symmetry` above already penalizes (u_R + u_L)^2 on hip pitch, which
  # does see amplitude mismatch, but weakly: a 1.4 deg mismatch is 0.024 rad, so
  # the whole term is worth 6e-4 before weighting and the policy can afford to
  # ignore it. And that form is simply WRONG for knee and ankle pitch, whose
  # antiphase sum is a nonzero constant (see the gait_antisymmetry_l2 docstring)
  # -- penalizing it there fights their normal shape.
  #
  # This instead samples each leg at ITS OWN swing apex and compares like with
  # like, so it reads amplitude mismatch directly on every sagittal joint.
  #
  # FADED WITH COMMANDED YAW. At -25 this is the largest penalty in the task,
  # and stride-amplitude mismatch between the legs is exactly HOW a turn is
  # produced -- the inside leg takes the shorter step. `gait_symmetry` and
  # `gait_symmetry_yaw` were both faded for precisely that reason; this one was
  # not, so it charged the heaviest penalty in the task for executing the turn
  # `track_angular_velocity` pays to produce.
  cfg.rewards["leg_symmetry"] = RewardTermCfg(
    func=mdp.swing_amplitude_symmetry_l2,
    weight=-25.0,
    params={
      "command_name": "gait",
      "yaw_command_index": YAW_COMMAND_INDEX,
      "yaw_fade_std": SYMMETRY_YAW_FADE_STD,
      "right_cfg": SceneEntityCfg(
        "robot",
        joint_names=[
          "base_link_hip_pitch_right_joint",
          "hip_yaw_knee_right_joint",
          "knee_ankle_pitch_right_joint",
        ],
      ),
      "left_cfg": SceneEntityCfg(
        "robot",
        joint_names=[
          "base_link_hip_pitch_left_joint",
          "hip_yaw_knee_left_joint",
          "knee_ankle_pitch_left_joint",
        ],
      ),
    },
  )

  # ---------------------------------------------------------------------
  # FALL ECONOMICS. Nothing in the inherited reward set referenced the
  # termination, so falling carried no explicit price -- only forfeited future
  # reward, which a value function has to learn before it means anything. These
  # two terms survive the removal of hold because they were never about hold:
  # they are what makes ending the episode strictly worse than continuing it.
  cfg.rewards["alive"] = RewardTermCfg(func=mdp.is_alive, weight=1.0)
  cfg.rewards["terminated"] = RewardTermCfg(func=mdp.is_terminated, weight=-200.0)

  # The `upright` 2.0 / `pose` 1.0 / `track_angular_velocity` 2.0 bumps existed
  # ONLY to make a holding environment worth defending against the 8.0/step of
  # gait reward it forfeited. With no hold mode there is no forfeit and no
  # imbalance to correct, and those terms are ungated -- they lifted both modes
  # equally and never moved the ratio anyway. Left raised they would just add
  # pressure against stride (`pose`) and against yaw authority. Back to the
  # values the bidirectional gait was validated with.

  # THE STANDSTILL, built from hold-gated terms only. Every one multiplies by
  # `1 - march`, so none of them can touch the gait: the walking reward set is
  # bit-identical to the forever-marching config that produced the stride being
  # kept. That is the whole design constraint here.
  #
  # `feet_planted` is the only hold-gated POSITIVE, so it alone decides whether
  # standing is worth defending against the gait_swing + gait_contact (8.0) a
  # holding step forfeits.
  cfg.rewards["feet_planted"].weight = 3.5

  # "Standing with the legs apart" has two separate causes and needs both fixed.
  #
  # Laterally: nothing stopped it. Every clearance term is a FLOOR that charges
  # feet coming too close and is exactly zero above its minimum, so a wider
  # stance was free -- and it is genuinely easier to balance on. This charges
  # only excess width over nominal, so it cannot push against those floors:
  # between the 50-55 mm floors and the 67.6 mm nominal both terms are
  # identically zero. It constrains the FEET, not the joints, because forcing a
  # prescribed hold pose is what defeated a stable stance before.
  cfg.rewards["hold_stance_width"] = RewardTermCfg(
    func=mdp.hold_stance_width_l2,
    weight=-3.0,
    params={
      "command_name": "gait",
      "max_separation": NOMINAL_STANCE_WIDTH,
      "asset_cfg": SceneEntityCfg("robot", site_names=list(FOOT_SITES)),
    },
  )
  # Fore/aft: a staggered stance, one leg ahead of the other, which is what
  # `hold_pair_symmetry` was built for. Left at the inherited -3.0 and NOT
  # raised to -8.0: at -8 it overwhelmed a levelling term about 6:1 under the
  # per-foot sole randomization, where holding the base level genuinely
  # requires asymmetric legs.
  cfg.rewards["hold_pair_symmetry"].weight = -3.0

  # Still out: `double_support_base_tilt` (it is the term -8.0 above fought)
  # and `hold_joint_vel` (sim hold is already still at action-rate 0.01, so it
  # has no gradient to learn from).
  for dead in ("double_support_base_tilt", "hold_joint_vel"):
    cfg.rewards.pop(dead, None)

  # CROUCH COMMAND REMOVED. The `dh` channel is gated by `(1 - march)`, so with
  # no hold it would read exactly zero forever -- a dead observation input, a
  # dead -150 weight, and dead C/V/Z keys in `keyboard.py`, all silently. It is
  # dropped outright, which takes the gait command back to 5-D
  # [march, sin, cos, vx, wz] and the actor observation to 49-D.
  #
  # That also dissolves the sharpest conflict in the task. `variable_posture`
  # scores the joints against the DEFAULT crouch with std_standing = 0.1 rad,
  # while a -45 mm height command needs knee -70 deg against that default's
  # -35 deg. Measured: obeying the crouch drove `pose` from 1.0000 to 0.000002,
  # so it cost the full 1.0 of `pose` to gain 150 * 0.045^2 = 0.30 of
  # `base_height` -- a net -0.70 per step. The policy was correct to refuse,
  # and that is exactly the logged symptom ("a 45 mm command moved the robot
  # 0.8 mm"). Raising -25 -> -150 could never have fixed it; break-even was
  # -494, at which point height would have dominated the whole task.
  cfg.rewards["base_height"] = RewardTermCfg(
    func=mdp.base_height_l2,
    weight=-25.0,
    params={"target_height": TARGET_BASE_HEIGHT},
  )

  # Make the contact schedule agree with the swing-height reward; see
  # SWING_THRESHOLD above and the `gait_contact` docstring.
  cfg.rewards["gait_contact"].params["swing_threshold"] = SWING_THRESHOLD
  cfg.rewards["gait_swing"].params["target_height"] = SWING_HEIGHT_M
  cfg.rewards["gait_swing"].params["tracking_sigma"] = SWING_TRACKING_SIGMA
  # Fresh SceneEntityCfg objects PER TERM: the manager resolves them in place
  # and a second resolve of a shared object fails its consistency check.
  def sole_points():
    return {
      key: SceneEntityCfg("robot", site_names=list(sites))
      for key, sites in (
        ("heel_cfg", FOOT_HEEL_SITES), ("toe_cfg", FOOT_TOE_SITES),
        ("inner_cfg", FOOT_INNER_SITES), ("outer_cfg", FOOT_OUTER_SITES),
      )
    }
  # Hinge below the lift trajectory, march-gated, every gait. Zero once the
  # sole follows the target, so it cannot ask for more lift than gait_swing.
  cfg.rewards["swing_shortfall"] = RewardTermCfg(
    func=mdp.swing_shortfall_l2,
    weight=-1.0,
    params={"command_name": "gait", "target_height": SWING_HEIGHT_M,
            **sole_points()},
  )

  # THE HOLD STANCE, shaped on the feet. Every term multiplies by `1 - march`.
  # Observed on hardware with the battery: standing still, one foot slid
  # forward flat while the other slid back with its heel lifted, and the robot
  # crept until it fell. Nothing in the inherited hold set could see that:
  # `feet_planted` pays for ANY contact (a toe touching counts), and
  # `hold_pair_symmetry` reads joints, which the sole randomization legitimately
  # de-symmetrizes. These read the sole geometry directly.
  # Weights START LOW and ramp in by curriculum (below): under a random
  # policy these read -3 to -5 per step, and shaping that heavy from iteration
  # zero has made a cold start unlearnable before. feet_planted + alive carry
  # the hold until the policy can stand; the geometry is tightened afterwards.
  cfg.rewards["hold_sole_alignment"] = RewardTermCfg(
    func=mdp.hold_sole_alignment_l2,   # all eight sole corners on the floor,
    weight=-0.25,                      # both feet pointing the same way
    params={"command_name": "gait", **sole_points()},
  )
  cfg.rewards["hold_stance_stagger"] = RewardTermCfg(
    func=mdp.hold_stance_stagger_l2,   # feet side by side, not one ahead
    weight=-0.5,
    params={"command_name": "gait", "tolerance": HOLD_STAGGER_TOLERANCE_M,
            "asset_cfg": SceneEntityCfg("robot", site_names=list(FOOT_SITES))},
  )
  cfg.rewards["hold_feet_still"] = RewardTermCfg(
    func=mdp.hold_feet_still_l2,       # no sliding or pivoting while planted
    weight=-0.1,
    params={"command_name": "gait",
            "asset_cfg": SceneEntityCfg("robot", site_names=list(FOOT_SITES))},
  )
  if cfg.curriculum:  # training only; play/finetune carry no curriculum
    for name, stages in (
      ("hold_sole_alignment", (-0.5, -0.75, -1.0)),
      ("hold_stance_stagger", (-1.0, -1.5, -2.0)),
      ("hold_feet_still", (-0.25, -0.4, -0.5)),
    ):
      cfg.curriculum[f"cur_{name}"] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={"reward_name": name, "weight_stages": [
          {"step": 150 * 24, "weight": stages[0]},
          {"step": 300 * 24, "weight": stages[1]},
          {"step": 500 * 24, "weight": stages[2]},
        ]},
      )

  # Hip yaw is the steering joint. The bidirectional 0.12 rad tolerance charges
  # nearly the whole per-joint pose reward for the ~0.17 rad excursion a step of
  # the commanded turn needs. Widen it to the value the omnidirectional task
  # already uses; foot crossing stays guarded by the (unchanged) inner-edge
  # clearance and self-collision penalties, not by this pose term.
  cfg.rewards["pose"].params["std_walking"][JOINT_PATTERNS["hip_yaw"]] = 0.25

  return cfg
