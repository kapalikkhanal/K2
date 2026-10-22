"""Smooth turn + lateral task; warm up turn first, then enable lateral rehearsal."""
import math
from copy import deepcopy

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from k2_rl.lateral_env_cfg import make_k2_lateral_env_cfg
from k2_rl.mdp.full_command import FullCommandCfg
from k2_rl.mdp.lateral import LateralFootSmoothness
from k2_rl.k2_constants import FOOT_SITES


FULL_SWING_HEIGHT = 0.015
TRAIN_YAW_RATE_RANGE = (-0.30, 0.30)


def swing_vertical_tracking_l2(env, command_name, asset_cfg, height, speed_std=.08):
  """Damp airborne bounce by tracking the existing smooth lift derivative."""
  cmd = env.command_manager.get_command(command_name)
  phase = env.command_manager.get_term(command_name).phase
  frequency = env.command_manager.get_term(command_name).cfg.gait_freq
  sinp, cosp = phase.sin(), phase.cos()
  swing = torch.stack((sinp > 0, sinp < 0), dim=1)
  desired = (4 * math.pi * frequency * height * sinp * cosp)[:, None] * swing
  robot = env.scene[asset_cfg.name]
  actual = robot.data.site_lin_vel_w[:, asset_cfg.site_ids, 2]
  angular = robot.data.site_ang_vel_w[:, asset_cfg.site_ids]
  cost = ((actual - desired) / speed_std).square()
  cost += 0.10 * (angular / 1.0).square().sum(dim=-1)
  return (cost * swing).sum(dim=1) * cmd[:, 0]


def pitch_envelope_l2(env, command_name, free_pitch=math.radians(3)):
  # Small inclination is useful while accelerating. Charge sustained large
  # backward/forward lean without forcing an instantaneous neutral posture.
  gravity = env.scene["robot"].data.projected_gravity_b
  pitch = torch.asin(gravity[:, 0].clamp(-1, 1)).abs()
  return ((pitch - free_pitch).clamp(min=0) / free_pitch).square()


def make_k2_full_env_cfg(play=False):
  cfg = make_k2_lateral_env_cfg(play)
  cfg.commands["gait"] = FullCommandCfg(**vars(cfg.commands["gait"]))
  gait = cfg.commands["gait"]
  gait.gait_freq = .58
  gait.transition_time_s = 1.0
  gait.yaw_rate_range = TRAIN_YAW_RATE_RANGE
  gait.lateral_velocity_range = (-.03, .03)
  gait.rel_march_envs = 0.90
  # Warm up for at least 1000 PPO iterations; enable .35 when resuming.
  gait.rel_lateral_envs = 0.0
  gait.rel_in_place_envs = .15
  if play:
    # Same controllable modes in the viewer; a preview must not pin hold forever.
    gait.rel_march_envs = 1.0
    gait.rel_lateral_envs = 0.0
    gait.rel_in_place_envs = 0.0
    gait.forward_velocity_range = (.02, .02)
    gait.yaw_rate_range = TRAIN_YAW_RATE_RANGE
  cfg.rewards["gait_swing"].params["target_height"] = FULL_SWING_HEIGHT
  feet = SceneEntityCfg("robot", site_names=list(FOOT_SITES))
  cfg.rewards.pop("lateral_landing")
  cfg.rewards["smooth_landing"] = RewardTermCfg(
    func=LateralFootSmoothness, weight=-.30,
    params={"command_name": "gait", "asset_cfg": feet,
            "sensor_name": "feet_ground_contact", "lateral_only": False})
  cfg.rewards["swing_velocity"] = RewardTermCfg(
    func=swing_vertical_tracking_l2, weight=-.15,
    params={"command_name": "gait", "asset_cfg": deepcopy(feet), "height": FULL_SWING_HEIGHT})
  cfg.rewards["pitch_envelope"] = RewardTermCfg(
    func=pitch_envelope_l2, weight=-.30, params={"command_name": "gait"})
  # Keep the inherited plane. Height-field contacts caused rare non-finite
  # physics at 4096 envs; a small smoke test cannot establish their stability.
  # Sole height/tilt, friction, inertia and actuator DR remain inherited.
  # This approximates mounting/support variation, NOT spatial rough terrain.
  return cfg
