"""Phase-synchronized gait command for K2 hold, march, and forward tasks.

Emits a per-environment command the policy observes:

    command = [march, clock_sin, clock_cos]                 (legacy/in-place)
    command = [march, clock_sin, clock_cos, forward_speed]  (walking tasks)

  * ``march`` ramps smoothly in [0, 1]: 0 = stable two-foot hold and 1 =
    active gait. A binary target selects the behavior, while the observable ramp
    avoids a command discontinuity. A fraction ``rel_march_envs`` of environments
    target gait each episode; the rest target hold.
  * ``clock_sin/cos``: a phase clock that advances at ``gait_freq`` Hz ONLY while
    marching (frozen at 0 while holding, so the hold command is exactly
    ``[0, 0, 0]``). It gives the policy a cadence reference and lets the reward
    score an alternating L/R contact schedule (see ``mdp.rewards.gait_contact``).
  * ``forward_speed`` is optional and may be signed. It is rate-limited when
    resampled and multiplied by gait activation, so it is zero in hold. The
    commanded speed is known on hardware, so this adds no unobservable state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class GaitCommand(CommandTerm):
  cfg: "GaitCommandCfg"

  def __init__(self, cfg: "GaitCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    # `march_target` is binary; `march` is the smoothly ramped value exposed to
    # the policy and rewards. This trains the same gentle hold->walk handover
    # used by deployment instead of an instantaneous command discontinuity.
    self.march_target = torch.zeros(self.num_envs, device=self.device)
    self.march = torch.zeros(self.num_envs, device=self.device)
    self.phase = torch.zeros(self.num_envs, device=self.device)
    self.forward_velocity = torch.zeros(self.num_envs, device=self.device)
    self.forward_velocity_target = torch.zeros(self.num_envs, device=self.device)
    command_dim = 4 if cfg.forward_velocity_range is not None else 3
    self._command = torch.zeros(self.num_envs, command_dim, device=self.device)
    self.metrics["march_fraction"] = torch.zeros(self.num_envs, device=self.device)
    if cfg.forward_velocity_range is not None:
      self.metrics["mean_forward_command"] = torch.zeros(
        self.num_envs, device=self.device
      )
    # Play-mode GUI override.
    self._gui_force: "viser.GuiDropdownHandle | None" = None
    self._gui_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _update_metrics(self) -> None:
    self.metrics["march_fraction"] = self.march
    if self.cfg.forward_velocity_range is not None:
      self.metrics["mean_forward_command"] = self.march * self.forward_velocity

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.rand(len(env_ids), device=self.device)
    self.march_target[env_ids] = (r < self.cfg.rel_march_envs).float()
    # v5 always began at the same double-support crossing.
    self.phase[env_ids] = 0.0
    if self.cfg.forward_velocity_range is not None:
      lo, hi = self.cfg.forward_velocity_range
      self.forward_velocity_target[env_ids] = lo + (hi - lo) * torch.rand(
        len(env_ids), device=self.device
      )

  def _update_command(self) -> None:
    dt = self._env.step_dt
    if self.cfg.transition_time_s <= 0.0:
      self.march.copy_(self.march_target)
    else:
      max_delta = dt / self.cfg.transition_time_s
      self.march.add_(
        torch.clamp(self.march_target - self.march, -max_delta, max_delta)
      )
    # Hold the clock at the double-support phase throughout activation. The
    # first foot starts lifting only after the hold posture has faded smoothly.
    phase_active = (self.march_target > 0.5) & (self.march >= 1.0 - 1e-6)
    self.phase = torch.where(
      phase_active,
      (self.phase + 2.0 * math.pi * self.cfg.gait_freq * dt) % (2.0 * math.pi),
      self.phase,
    )
    if self.cfg.forward_velocity_range is not None:
      max_velocity_delta = self.cfg.velocity_ramp_rate_mps2 * dt
      self.forward_velocity.add_(torch.clamp(
        self.forward_velocity_target - self.forward_velocity,
        -max_velocity_delta,
        max_velocity_delta,
      ))
    self.phase = torch.where(
      (self.march_target < 0.5) & (self.march <= 0.0),
      torch.zeros_like(self.phase),
      self.phase,
    )
    self._command[:, 0] = self.march
    self._command[:, 1] = self.march * torch.sin(self.phase)
    self._command[:, 2] = self.march * torch.cos(self.phase)
    if self.cfg.forward_velocity_range is not None:
      self._command[:, 3] = self.march * self.forward_velocity

  # GUI (play mode): force hold / march / auto for the viewed env.
  def create_gui(self, name, server, get_env_idx) -> None:  # noqa: ANN001
    with server.gui.add_folder(name.capitalize()):
      self._gui_force = server.gui.add_dropdown(
        "Mode", options=("auto", "hold", "march"), initial_value="auto"
      )
    self._gui_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._gui_force is not None and self._gui_get_env_idx is not None:
      mode = self._gui_force.value
      if mode != "auto":
        idx = self._gui_get_env_idx()
        self.march_target[idx] = 1.0 if mode == "march" else 0.0


@dataclass(kw_only=True)
class GaitCommandCfg(CommandTermCfg):
  entity_name: str
  gait_freq: float = 1.5
  """Stepping frequency (Hz) while marching."""
  rel_march_envs: float = 0.5
  """Fraction of environments commanded to march (rest hold)."""
  forward_velocity_range: tuple[float, float] | None = None
  """Optional forward-speed range in m/s; adds a fourth command channel."""
  transition_time_s: float = 0.5
  """Time for the observable gait command to ramp between hold and walk."""
  velocity_ramp_rate_mps2: float = 0.08
  """Maximum change rate for signed forward commands, in m/s^2."""

  def build(self, env: "ManagerBasedRlEnv") -> GaitCommand:
    return GaitCommand(self, env)
