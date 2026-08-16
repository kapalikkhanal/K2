"""K2 action terms with reset-safe randomized command latency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class DelayedJointPositionActionCfg(JointPositionActionCfg):
  """Position targets delayed across control steps and physics substeps."""

  max_delay_substeps: int = 0
  min_delay_steps: int = 0
  max_delay_steps: int = 0
  """Random whole-control-step delay range (one step is 20 ms for K2)."""
  max_target_rate: float | None = None
  """Maximum applied position-target rate in rad/s. None disables limiting."""

  def build(self, env: "ManagerBasedRlEnv") -> "DelayedJointPositionAction":
    return DelayedJointPositionAction(self, env)


class DelayedJointPositionAction(JointPositionAction):
  """Hold the previous valid target until the new command reaches the servo."""

  cfg: DelayedJointPositionActionCfg

  def __init__(
    self, cfg: DelayedJointPositionActionCfg, env: "ManagerBasedRlEnv"
  ) -> None:
    super().__init__(cfg=cfg, env=env)
    self._max_delay = int(cfg.max_delay_substeps)
    if not 0 <= self._max_delay < env.cfg.decimation:
      raise ValueError(
        f"max_delay_substeps must be in [0, {env.cfg.decimation - 1}], "
        f"got {self._max_delay}"
      )
    self._min_delay_steps = int(cfg.min_delay_steps)
    self._max_delay_steps = int(cfg.max_delay_steps)
    if not 0 <= self._min_delay_steps <= self._max_delay_steps:
      raise ValueError(
        "control-step delay must satisfy 0 <= min_delay_steps <= "
        f"max_delay_steps, got {self._min_delay_steps}..{self._max_delay_steps}"
      )
    self._default_target = self._entity.data.default_joint_pos[
      :, self._target_ids
    ].clone()
    self._processed_actions[:] = self._default_target
    self._prev_target = self._default_target.clone()
    self._commanded_target = self._default_target.clone()
    self._delay = torch.zeros(self.num_envs, 1, device=self.device)
    self._step_delay = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._target_history = self._default_target.unsqueeze(0).repeat(
      self._max_delay_steps + 1, 1, 1
    )
    self._history_index = 0
    self._substep = 0

  def process_actions(self, actions: torch.Tensor) -> None:
    self._prev_target = self._processed_actions.clone()
    super().process_actions(actions)
    if self.cfg.max_target_rate is not None:
      max_delta = float(self.cfg.max_target_rate) * self._env.step_dt
      delta = torch.clamp(
        self._processed_actions - self._commanded_target,
        min=-max_delta,
        max=max_delta,
      )
      self._processed_actions = self._commanded_target + delta
    self._commanded_target = self._processed_actions.clone()
    if self._max_delay_steps:
      self._target_history[self._history_index] = self._commanded_target
      self._step_delay.random_(
        self._min_delay_steps, self._max_delay_steps + 1
      )
      history_ids = (
        self._history_index - self._step_delay
      ) % self._target_history.shape[0]
      env_ids = torch.arange(self.num_envs, device=self.device)
      self._processed_actions = self._target_history[history_ids, env_ids]
      self._history_index = (
        self._history_index + 1
      ) % self._target_history.shape[0]
    if self._max_delay:
      self._delay = torch.randint(
        0,
        self._max_delay + 1,
        (self.num_envs, 1),
        device=self.device,
      )
    self._substep = 0

  def apply_actions(self) -> None:
    if not self._max_delay:
      super().apply_actions()
      return
    use_new = self._substep >= self._delay
    target = torch.where(use_new, self._processed_actions, self._prev_target)
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      target - encoder_bias, joint_ids=self._target_ids
    )
    self._substep += 1

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    self._prev_target[env_ids] = self._default_target[env_ids]
    self._processed_actions[env_ids] = self._default_target[env_ids]
    self._commanded_target[env_ids] = self._default_target[env_ids]
    self._target_history[:, env_ids] = self._default_target[env_ids]
