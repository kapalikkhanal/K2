"""Command sampling and reward isolation for the combined lateral task.

Kept separate so the validated turn task and its shared command are unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import inspect

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from k2_rl.mdp.gait_command import GaitCommand, GaitCommandCfg


class LateralGaitCommand(GaitCommand):
  def _resample_command(self, env_ids):
    super()._resample_command(env_ids)
    lateral = self.lateral_velocity_target[env_ids] != 0.0
    # Sideways practice is initially pure vy. Keep the inherited vx/wz mixture
    # on the remaining samples, including its exact-zero yaw share.
    self.forward_velocity_target[env_ids] *= ~lateral
    self.yaw_rate_target[env_ids] *= ~lateral
    # Reserve an unconditional share for exact marching / turning in place.
    # Conditional probability keeps this share independent of lateral's share.
    available = 1.0 - self.cfg.rel_lateral_envs
    probability = self.cfg.rel_in_place_envs / available if available > 0 else 0.0
    in_place = (~lateral) & (
      torch.rand(len(env_ids), device=self.device) < probability
    )
    self.forward_velocity_target[env_ids] *= ~in_place
    # Half of in-place samples are pure march; half retain the yaw draw.
    march = in_place & (torch.rand(len(env_ids), device=self.device) < 0.5)
    self.yaw_rate_target[env_ids] *= ~march
    # Avoid near-zero lateral targets with negligible learning signal.
    target = self.lateral_velocity_target[env_ids]
    self.lateral_velocity_target[env_ids] = target.sign() * target.abs().clamp(
      min=self.cfg.min_lateral_speed
    )
    extreme = lateral & (
      torch.rand(len(env_ids), device=self.device) < self.cfg.rel_lateral_extreme_envs
    )
    lo, hi = self.cfg.lateral_velocity_range
    endpoint = torch.where(target > 0, hi, lo)
    self.lateral_velocity_target[env_ids] = torch.where(
      extreme, endpoint, self.lateral_velocity_target[env_ids]
    )


@dataclass(kw_only=True)
class LateralGaitCommandCfg(GaitCommandCfg):
  rel_in_place_envs: float = 0.10
  min_lateral_speed: float = 0.008
  rel_lateral_extreme_envs: float = 0.0

  def build(self, env):
    if not 0 <= self.rel_lateral_extreme_envs <= 1:
      raise ValueError("lateral endpoint share must be in [0, 1]")
    if not 0 <= self.rel_in_place_envs <= 1 - self.rel_lateral_envs:
      raise ValueError("in-place and lateral sample shares must sum to at most 1")
    lo, hi = self.lateral_velocity_range
    if not 0 < self.min_lateral_speed <= max(abs(lo), abs(hi)):
      raise ValueError("minimum lateral speed must lie inside the lateral range")
    return LateralGaitCommand(self, env)


def lateral_blend(env, command_name="gait"):
  """Exactly zero in hold/legacy modes; smooth during the emitted vy ramp."""
  vy = env.command_manager.get_command(command_name)[:, 5]
  return (vy.abs() / 0.005).clamp(max=1.0)


class LateralRewardBlend:
  """Blend complete reward terms, including stateful terms and their resets.

  Nested SceneEntityCfg objects need explicit resolution: the manager resolves
  only top-level parameters. Always evaluate both branches to keep histories
  current across mode switches. Weights live on the outer RewardTermCfg.
  """
  def __init__(self, cfg, env):
    self.legacy = self._build(cfg.params["legacy"], env)
    self.lateral = self._build(cfg.params["lateral"], env)

  @staticmethod
  def _build(term, env):
    if term is None:
      return None
    term = deepcopy(term)
    for value in term.params.values():
      if isinstance(value, SceneEntityCfg):
        value.resolve(env.scene)
    if inspect.isclass(term.func):
      term.func = term.func(cfg=term, env=env)
    return term

  def reset(self, env_ids=None):
    for term in (self.legacy, self.lateral):
      if term is not None and hasattr(term.func, "reset"):
        term.func.reset(env_ids=env_ids)
    return {}

  def __call__(self, env, legacy, lateral, command_name="gait"):
    blend = lateral_blend(env, command_name)
    old = self.legacy.func(env, **self.legacy.params)
    new = (
      torch.zeros_like(old) if self.lateral is None
      else self.lateral.func(env, **self.lateral.params)
    )
    # Preserve the original value exactly on every emitted vy=0 sample.
    return torch.where(blend == 0, old, old * (1 - blend) + new * blend)


def lateral_yaw_pitch_tracking(env, command_name, std):
  actual = env.scene["robot"].data.root_link_ang_vel_b
  cmd = env.command_manager.get_command(command_name)
  error = (actual[:, 2] - cmd[:, 4]).square() + actual[:, 1].square()
  return torch.exp(-error / std**2)


class LateralCycleTracking:
  """Track cycle-mean error against the SAME window of emitted commands.

  Averaging instantaneous tracking error cancels periodic body rocking and
  handles acceleration/reversal without comparing old velocity to a new target.
  Hold changes clear history too, even if the sampled vy target is unchanged.
  """
  def __init__(self, cfg, env):
    gait = env.command_manager.get_term(cfg.params["command_name"])
    self.steps = max(1, round(1 / (gait.cfg.gait_freq * env.step_dt)))
    self.history = torch.zeros(self.steps, env.num_envs, device=env.device)
    self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.last = torch.full((env.num_envs, 2), float("nan"), device=env.device)
    self.index = 0

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.history[:, ids] = 0
    self.count[ids] = 0
    self.last[ids] = float("nan")
    return {}

  def __call__(self, env, command_name, std):
    gait = env.command_manager.get_term(command_name)
    target = torch.stack((gait.lateral_velocity_target, gait.march_target), dim=1)
    changed = (target != self.last).any(dim=1)
    self.history[:, changed] = 0
    self.count[changed] = 0
    self.last.copy_(target)
    cmd = env.command_manager.get_command(command_name)
    self.history[self.index] = (
      env.scene["robot"].data.root_link_lin_vel_b[:, 1] - cmd[:, 5]
    ).detach()
    self.index = (self.index + 1) % self.steps
    self.count.add_(1).clamp_(max=self.steps)
    mean_error = self.history.mean(dim=0)
    return (
      torch.exp(-mean_error.square() / std**2)
      * (self.count == self.steps)
      * lateral_blend(env, command_name)
    )


class LateralPlanarTracking(LateralCycleTracking):
  """Share the original velocity reward budget between forward and lateral.

  Keep forward tracking during window initialization. Once ready, split the
  same budget across both axes (75% lateral in the longer-stride config), so
  sideways motion does not increase the
  maximum gait reward relative to hold or the original walking modes.
  """
  def __call__(self, env, command_name, std, forward_std, lateral_share=0.5):
    from k2_rl.mdp.rewards import track_forward_velocity_exp

    sideways = super().__call__(env, command_name, std)
    forward = track_forward_velocity_exp(env, command_name, forward_std)
    return torch.where(
      self.count == self.steps,
      (1 - lateral_share) * forward + lateral_share * sideways,
      forward,
    )


class LateralFootSmoothness:
  """Penalize impact, loaded sliding, and late-swing downward arrival.

  Force history covers every physics substep, so a short impact cannot hide
  between policy ticks. Costs are normalized to useful physical scales.
  """
  def __init__(self, cfg, env):
    robot = env.scene["robot"]
    # Nominal body weight, in newtons. EntityData exposes no mass field, so
    # read it from the compiled model: get_default_field returns the values
    # from BEFORE mass randomization, which is what the normalizer wants --
    # a force limit expressed in body weights must not move with the draw.
    mass = env.sim.get_default_field("body_mass")[robot.indexing.body_ids].sum()
    self.weight = (mass * 9.81).clamp(min=1e-6)

  def __call__(self, env, command_name, asset_cfg, sensor_name,
               force_limit_bw=1.5, slip_speed=0.04, landing_speed=0.04,
               lateral_only=True):
    robot = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_name]
    history = sensor.data.force_history
    if history is None:
      raise RuntimeError("Smooth landing requires physics-substep force history")
    peak = history[..., 2].abs().amax(dim=2) / self.weight
    impact = (peak - force_limit_bw).clamp(min=0).square().sum(dim=1)
    velocity = robot.data.site_lin_vel_w[:, asset_cfg.site_ids]
    contact = sensor.data.found > 0
    slip = (velocity[..., :2] / slip_speed).square().sum(dim=2)
    slip = (slip * contact).sum(dim=1)
    height = robot.data.site_pos_w[:, asset_cfg.site_ids, 2]
    near_floor = ((0.008 - height) / 0.008).clamp(0, 1)
    arrival = ((-velocity[..., 2] - landing_speed).clamp(min=0) / landing_speed).square()
    cost = impact + 0.5 * slip + (arrival * near_floor).sum(dim=1)
    gate = (lateral_blend(env, command_name) if lateral_only else
            env.command_manager.get_command(command_name)[:, 0])
    return cost * gate


def lateral_swing_velocity_l2(env, command_name, asset_cfg, std=0.08):
  """Ask each swing foot to advance vy/f per cycle with eased endpoints.

  With a half-cycle swing, 2*vy is the mean swing speed. The quintic position
  profile's derivative is 30*u^2*(1-u)^2, whose integral is one. This demands
  useful travel without increasing cadence, roll allowance or foot height.
  The contact/clearance rewards still require that the moving foot be lifted.
  """
  command = env.command_manager.get_command(command_name)
  phase = torch.atan2(command[:, 1], command[:, 2]).remainder(2 * torch.pi)
  u = torch.stack((phase / torch.pi, (phase - torch.pi) / torch.pi), dim=1)
  swing = (u > 0) & (u < 1)
  u = u.clamp(0, 1)
  desired = 2 * command[:, 5:6] * 30 * u.square() * (1 - u).square()
  robot = env.scene[asset_cfg.name]
  velocity = robot.data.site_lin_vel_w[:, asset_cfg.site_ids]
  heading = robot.data.heading_w
  actual = -velocity[..., 0] * heading.sin()[:, None] + velocity[..., 1] * heading.cos()[:, None]
  return (((actual - desired) / std).square() * swing).sum(dim=1) * lateral_blend(env, command_name)

