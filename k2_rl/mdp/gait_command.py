"""Phase-synchronized gait command for K2 hold, march, and walking tasks.

Emits a per-environment command the policy observes:

    command = [march, clock_sin, clock_cos]                      (legacy/in-place)
    command = [march, clock_sin, clock_cos, vx]                  (walking tasks)
    command = [march, clock_sin, clock_cos, vx, wz]              (turning task)

  * ``march`` ramps smoothly in [0, 1]: 0 = stable two-foot hold and 1 =
    active gait. A binary target selects the behavior, while the observable ramp
    avoids a command discontinuity. A fraction ``rel_march_envs`` of environments
    target gait each episode; the rest target hold.
  * ``clock_sin/cos``: a phase clock that advances at ``gait_freq`` Hz ONLY while
    marching (frozen at 0 while holding, so the hold command is exactly
    ``[0, 0, 0]``). It gives the policy a cadence reference and lets the reward
    score an alternating L/R contact schedule (see ``mdp.rewards.gait_contact``).
  * ``vx`` is optional and may be signed. It is rate-limited when resampled and
    multiplied by gait activation, so it is zero in hold. The commanded speed is
    known on hardware, so this adds no unobservable state.
  * ``wz`` is optional and requires ``vx``. It is the commanded body yaw rate,
    rate-limited and gated by gait activation the same way. Only a fraction
    ``rel_turning_envs`` of marching environments receive a nonzero yaw command,
    so straight walking stays a well-represented mode rather than a measure-zero
    slice of a uniform range.

Whenever ``vx`` is configured the term also integrates a **reference pose** --
``heading_ref`` and ``pos_ref`` -- from the commands it emits. That reference is
what ``rewards.heading_reference_error``, ``rewards.lateral_path_deviation_l2``
and ``observations.heading_error_sin_cos`` measure against, which is what makes
those terms mean "follow the commanded path" instead of "face world +X". With no
yaw channel the reference never rotates and never leaves the environment origin
line, so every one of those terms reduces exactly to its straight-line
predecessor. Deployment reproduces the same reference by integrating the same
emitted numbers -- no extra sensing.
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
    if cfg.yaw_rate_range is not None and cfg.forward_velocity_range is None:
      raise ValueError(
        "GaitCommandCfg.yaw_rate_range requires forward_velocity_range so the "
        "command channel indices stay fixed (vx is channel 3, wz is channel 4)"
      )
    # `march_target` is binary; `march` is the smoothly ramped value exposed to
    # the policy and rewards. This trains the same gentle hold->walk handover
    # used by deployment instead of an instantaneous command discontinuity.
    self.march_target = torch.zeros(self.num_envs, device=self.device)
    self.march = torch.zeros(self.num_envs, device=self.device)
    self.phase = torch.zeros(self.num_envs, device=self.device)
    self.forward_velocity = torch.zeros(self.num_envs, device=self.device)
    self.forward_velocity_target = torch.zeros(self.num_envs, device=self.device)
    self.yaw_rate = torch.zeros(self.num_envs, device=self.device)
    self.yaw_rate_target = torch.zeros(self.num_envs, device=self.device)
    # Reference pose the emitted command asks the robot to follow.
    self.heading_ref = torch.zeros(self.num_envs, device=self.device)
    self.pos_ref = torch.zeros(self.num_envs, 2, device=self.device)
    command_dim = 3
    if cfg.forward_velocity_range is not None:
      command_dim = 4
    if cfg.yaw_rate_range is not None:
      command_dim = 5
    self._command = torch.zeros(self.num_envs, command_dim, device=self.device)
    self.metrics["march_fraction"] = torch.zeros(self.num_envs, device=self.device)
    if cfg.forward_velocity_range is not None:
      self.metrics["mean_forward_command"] = torch.zeros(
        self.num_envs, device=self.device
      )
    if cfg.yaw_rate_range is not None:
      self.metrics["mean_abs_yaw_command"] = torch.zeros(
        self.num_envs, device=self.device
      )
    # Play-mode GUI override.
    self._gui_force: "viser.GuiDropdownHandle | None" = None
    self._gui_yaw: "viser.GuiSliderHandle | None" = None
    self._gui_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def reset(self, env_ids) -> dict[str, float]:
    extras = super().reset(env_ids)
    # The reference pose starts on the environment origin facing world +X, which
    # is the frame the straight-line heading and lateral-corridor rewards have
    # always used. Resetting it here (and not in `_resample_command`) keeps it
    # anchored for the whole episode across mid-episode command resamples.
    self.heading_ref[env_ids] = 0.0
    self.pos_ref[env_ids] = self._env.scene.env_origins[env_ids, :2]
    return extras

  def _update_metrics(self) -> None:
    self.metrics["march_fraction"] = self.march
    if self.cfg.forward_velocity_range is not None:
      self.metrics["mean_forward_command"] = self.march * self.forward_velocity
    if self.cfg.yaw_rate_range is not None:
      self.metrics["mean_abs_yaw_command"] = torch.abs(self.march * self.yaw_rate)

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
    if self.cfg.yaw_rate_range is not None:
      lo, hi = self.cfg.yaw_rate_range
      sampled = lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)
      # Sampling the signed range uniformly would make exactly-straight walking
      # a measure-zero event and let the proven gait decay. Give a fixed share
      # of samples a hard zero instead, the way `rel_march_envs` protects hold.
      turning = (
        torch.rand(len(env_ids), device=self.device) < self.cfg.rel_turning_envs
      ).float()
      self.yaw_rate_target[env_ids] = sampled * turning

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
    if self.cfg.yaw_rate_range is not None:
      max_yaw_delta = self.cfg.yaw_rate_ramp_rate_rps2 * dt
      self.yaw_rate.add_(torch.clamp(
        self.yaw_rate_target - self.yaw_rate, -max_yaw_delta, max_yaw_delta
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
    if self.cfg.yaw_rate_range is not None:
      self._command[:, 4] = self.march * self.yaw_rate
    if self.cfg.forward_velocity_range is not None:
      self._integrate_reference(dt)

  def _integrate_reference(self, dt: float) -> None:
    """Advance the reference pose using the commands actually emitted.

    Forward Euler, position first, so hardware can reproduce this from the same
    two numbers in the same order (see ``hardware/k2_policy_run.py``).
    """
    forward_command = self._command[:, 3]
    self.pos_ref[:, 0] += forward_command * torch.cos(self.heading_ref) * dt
    self.pos_ref[:, 1] += forward_command * torch.sin(self.heading_ref) * dt
    if self.cfg.yaw_rate_range is not None:
      heading = self.heading_ref + self._command[:, 4] * dt
      self.heading_ref = (heading + math.pi) % (2.0 * math.pi) - math.pi

  # GUI (play mode): force hold / march / auto for the viewed env.
  def create_gui(self, name, server, get_env_idx) -> None:  # noqa: ANN001
    with server.gui.add_folder(name.capitalize()):
      self._gui_force = server.gui.add_dropdown(
        "Mode", options=("auto", "hold", "march"), initial_value="auto"
      )
      if self.cfg.yaw_rate_range is not None:
        lo, hi = self.cfg.yaw_rate_range
        limit = max(abs(lo), abs(hi))
        self._gui_yaw = server.gui.add_slider(
          "Yaw rate (rad/s)", min=-limit, max=limit, step=0.01, initial_value=0.0
        )
    self._gui_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._gui_get_env_idx is None:
      return
    idx = self._gui_get_env_idx()
    if self._gui_force is not None:
      mode = self._gui_force.value
      if mode != "auto":
        self.march_target[idx] = 1.0 if mode == "march" else 0.0
    if self._gui_yaw is not None:
      self.yaw_rate_target[idx] = float(self._gui_yaw.value)


@dataclass(kw_only=True)
class GaitCommandCfg(CommandTermCfg):
  entity_name: str
  gait_freq: float = 1.5
  """Stepping frequency (Hz) while marching."""
  rel_march_envs: float = 0.5
  """Fraction of environments commanded to march (rest hold)."""
  forward_velocity_range: tuple[float, float] | None = None
  """Optional forward-speed range in m/s; adds a fourth command channel."""
  yaw_rate_range: tuple[float, float] | None = None
  """Optional body yaw-rate range in rad/s; adds a fifth command channel.

  Requires ``forward_velocity_range`` so channel indices never shift.
  """
  rel_turning_envs: float = 0.0
  """Fraction of resamples given a nonzero yaw command (rest get exactly zero)."""
  transition_time_s: float = 0.5
  """Time for the observable gait command to ramp between hold and walk."""
  velocity_ramp_rate_mps2: float = 0.08
  """Maximum change rate for signed forward commands, in m/s^2."""
  yaw_rate_ramp_rate_rps2: float = 0.30
  """Maximum change rate for signed yaw-rate commands, in rad/s^2."""

  def build(self, env: "ManagerBasedRlEnv") -> GaitCommand:
    return GaitCommand(self, env)
