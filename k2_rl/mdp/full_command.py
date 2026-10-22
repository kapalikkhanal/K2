"""Full task command: continuous phase, jerk-limited commands, settled stops."""
from dataclasses import dataclass
import torch

from hardware.k2_command_profile import RATE, JERK, STOP_SPEED, STOP_RATE, VERSION, step_profile
from k2_rl.mdp.lateral import LateralGaitCommand, LateralGaitCommandCfg


class FullCommand(LateralGaitCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.profile_value = torch.zeros(self.num_envs, 3, device=self.device)
    self.profile_rate = torch.zeros_like(self.profile_value)
    self.settling = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.profile_limits = [torch.tensor(v, device=self.device) for v in
                           (RATE, JERK, STOP_SPEED, STOP_RATE)]

  def reset(self, env_ids):
    self.profile_value[env_ids] = 0
    self.profile_rate[env_ids] = 0
    self.settling[env_ids] = False
    self.march[env_ids] = 0
    return super().reset(env_ids)

  def _resample_command(self, env_ids):
    phase = self.phase[env_ids].clone()
    walking = self.march[env_ids] > 0
    super()._resample_command(env_ids)
    self.phase[env_ids] = torch.where(walking, phase, self.phase[env_ids])

  def _update_command(self):
    dt = self._env.step_dt
    target = torch.stack((self.forward_velocity_target, self.lateral_velocity_target,
                          self.yaw_rate_target), dim=1)
    (self.profile_value, self.profile_rate, self.phase, self.march,
     self.settling) = step_profile(
       torch, self.profile_value, self.profile_rate, self.phase, self.march,
       self.settling, target, self.march_target > 0.5, dt, self.cfg.gait_freq,
       self.cfg.transition_time_s, *self.profile_limits)
    self.forward_velocity = self.profile_value[:, 0]
    self.lateral_velocity = self.profile_value[:, 1]
    self.yaw_rate = self.profile_value[:, 2]
    self._command[:, 0] = self.march
    self._command[:, 1] = self.march * self.phase.sin()
    self._command[:, 2] = self.march * self.phase.cos()
    self._command[:, 3] = self.march * self.forward_velocity
    self._command[:, 4] = self.march * self.yaw_rate
    self._command[:, 5] = self.march * self.lateral_velocity
    self._integrate_reference(dt)

  def compute(self, dt):
    # Slider updates must not reset the phase of a foot already in flight.
    slider = self._gui_lateral
    self._gui_lateral = None
    try:
      super().compute(dt)
    finally:
      self._gui_lateral = slider
    if slider is not None and self._gui_get_env_idx is not None:
      self.lateral_velocity_target[self._gui_get_env_idx()] = float(slider.value)


@dataclass(kw_only=True)
class FullCommandCfg(LateralGaitCommandCfg):
  command_profile: str = VERSION

  def build(self, env):
    return FullCommand(self, env)
