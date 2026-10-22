"""Gait command extension for IK-guided rotation from a two-foot hold."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from k2_rl.mdp.gait_command import GaitCommand, GaitCommandCfg


class StandingRotateCommand(GaitCommand):
  """Reuse the gait clock and yaw channel for a mutually-exclusive hold skill.

  Marching behavior is inherited directly from :class:`GaitCommand`. Only
  after ``march`` has reached exactly zero does a nonzero yaw command expose a
  slow rotation clock and yaw target. A pulse is allowed to finish before the
  command can be resampled, so its loaded foot cannot change side halfway
  through the IK motion.
  """

  cfg: "StandingRotateCommandCfg"

  def __init__(self, cfg: "StandingRotateCommandCfg", env):
    super().__init__(cfg, env)
    self.rotate_phase = torch.zeros(self.num_envs, device=self.device)
    self.rotate_elapsed = torch.zeros(self.num_envs, device=self.device)
    self.standing_rotate_active = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.metrics["standing_rotate_fraction"] = torch.zeros(
      self.num_envs, device=self.device
    )

  def reset(self, env_ids) -> dict[str, float]:
    self.rotate_phase[env_ids] = 0.0
    self.rotate_elapsed[env_ids] = 0.0
    self.standing_rotate_active[env_ids] = False
    return super().reset(env_ids)

  def _resample(self, env_ids: torch.Tensor) -> None:
    # The ordinary 4--7 s locomotion resampling schedule is unchanged. A
    # standing rotation is the sole exception: once begun, finish its full IK
    # pulse before choosing another direction or returning to gait.
    if len(env_ids) == 0:
      return
    unfinished = (
      self.standing_rotate_active[env_ids]
      & (self.rotate_elapsed[env_ids] < self.cfg.standing_rotate_period_s)
    )
    protected = env_ids[unfinished]
    if len(protected):
      self.time_left[protected] = torch.clamp(
        self.cfg.standing_rotate_period_s - self.rotate_elapsed[protected],
        min=self._env.step_dt,
      )
    super()._resample(env_ids[~unfinished])

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    holding = self.march_target[env_ids] < 0.5
    rotate = holding & (
      torch.rand(len(env_ids), device=self.device)
      < self.cfg.rel_standing_rotate_envs
    )
    signs = torch.where(
      torch.rand(len(env_ids), device=self.device) < 0.5,
      -torch.ones(len(env_ids), device=self.device),
      torch.ones(len(env_ids), device=self.device),
    )
    # Preserve the original sampled yaw command for every marching env.
    self.yaw_rate_target[env_ids] = torch.where(
      holding,
      signs * self.cfg.standing_rotate_yaw_rate_rps * rotate.float(),
      self.yaw_rate_target[env_ids],
    )
    self.rotate_phase[env_ids] = 0.0
    self.rotate_elapsed[env_ids] = 0.0
    self.standing_rotate_active[env_ids] = False

  def _update_metrics(self) -> None:
    super()._update_metrics()
    self.metrics["standing_rotate_fraction"] = (
      self.standing_rotate_active.float()
    )
    if self.cfg.yaw_rate_range is not None:
      self.metrics["mean_abs_yaw_command"] = torch.abs(self._command[:, 4])

  def _update_command(self) -> None:
    super()._update_command()
    dt = self._env.step_dt
    active = (
      (self.march_target < 0.5)
      & (self.march <= 1e-6)
      & (torch.abs(self.yaw_rate_target) > 1e-9)
      & (self.rotate_elapsed < self.cfg.standing_rotate_period_s)
    )
    self.standing_rotate_active.copy_(active)
    self.rotate_elapsed.add_(active.float() * dt)
    self.rotate_phase.copy_(
      2.0 * math.pi
      * torch.clamp(
        self.rotate_elapsed / self.cfg.standing_rotate_period_s, 0.0, 1.0
      )
    )

    # In hold the original command is [0, 0, 0, 0, 0]. Replace only those
    # zero clock/yaw entries. Marching entries remain as GaitCommand made them.
    self._command[:, 1] = torch.where(
      active, torch.sin(self.rotate_phase), self._command[:, 1]
    )
    self._command[:, 2] = torch.where(
      active, torch.cos(self.rotate_phase), self._command[:, 2]
    )
    standing_yaw = torch.where(active, self.yaw_rate_target, 0.0)
    self._command[:, 4] += standing_yaw

    # GaitCommand integrated marching yaw before hold yaw was inserted.
    heading = self.heading_ref + standing_yaw * dt
    self.heading_ref.copy_((heading + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass(kw_only=True)
class StandingRotateCommandCfg(GaitCommandCfg):
  rel_standing_rotate_envs: float = 0.8
  """Fraction of hold samples assigned an in-place rotation pulse."""

  standing_rotate_yaw_rate_rps: float = 0.003
  """Signed target magnitude for the slow IK pulse (rad/s)."""

  standing_rotate_period_s: float = 9.0
  """neutral->twist->neutral cycle time, including both IK dwells."""

  def build(self, env) -> StandingRotateCommand:
    return StandingRotateCommand(self, env)
