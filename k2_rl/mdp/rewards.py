"""Custom rewards for the K2 in-place (hold / march) tasks.

These complement mjlab's velocity rewards. The gait-aware terms read the
``[march, sin, cos]`` command emitted by :class:`GaitCommand`:

  * ``gait_contact``   -- reward following an alternating L/R contact schedule
                          (marching envs only).
  * ``gait_swing``     -- reward lifting whichever foot should currently swing
                          (marching envs only).
  * ``feet_planted``   -- reward keeping both feet on the ground (holding envs).
  * ``base_height_l2`` -- quadratic penalty keeping base height near target
                          (both modes). Quadratic per the reward tip in robot.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _contact(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float()  # [B, N] feet


def gait_contact(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  command_name: str,
) -> torch.Tensor:
  """Reward alternating contact with a double-support window at phase crossings."""
  command = env.command_manager.get_command(command_name)
  march = command[:, 0]  # [B]
  sinp = command[:, 1]  # [B]
  contact = _contact(env, sensor_name)  # [B, 2]
  # A foot does not become a swing foot until the smooth swing clock is clear
  # of zero. Around each zero crossing both feet should be planted. This avoids
  # the old instantaneous support-foot swap that excited lateral wobble.
  swing0 = sinp > 0.18
  swing1 = sinp < -0.18
  desired = torch.stack([~swing0, ~swing1], dim=1).float()
  match = 1.0 - torch.abs(contact - desired)  # 1 if matches schedule else 0
  return torch.mean(match, dim=1) * march


def gait_swing(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  command_name: str,
  target_height: float,
  center_cfg: SceneEntityCfg,
  heel_cfg: SceneEntityCfg,
  toe_cfg: SceneEntityCfg,
  inner_cfg: SceneEntityCfg,
  outer_cfg: SceneEntityCfg,
  tracking_sigma: float = 0.008,
) -> torch.Tensor:
  """Track the same swing height at five points across each sole.

  Center-only tracking allowed the policy to lift a tilted foot. Requiring the
  center, heel, toe, inner edge, and outer edge to follow one vertical target
  makes pitch and roll tilt lose reward while still allowing a smooth lift.
  """
  asset: Entity = env.scene[center_cfg.name]
  command = env.command_manager.get_command(command_name)
  march = command[:, 0]
  sinp = command[:, 1]
  # sin² has zero slope at lift-off, apex, and touchdown. That removes target
  # velocity discontinuities while preserving a full-height alternating step.
  lift_clock = torch.square(sinp)
  target = torch.stack(
    [
      target_height * lift_clock * (sinp > 0).float(),
      target_height * lift_clock * (sinp < 0).float(),
    ],
    dim=1,
  )
  foot_z = torch.stack(
    [
      asset.data.site_pos_w[:, center_cfg.site_ids, 2],
      asset.data.site_pos_w[:, heel_cfg.site_ids, 2],
      asset.data.site_pos_w[:, toe_cfg.site_ids, 2],
      asset.data.site_pos_w[:, inner_cfg.site_ids, 2],
      asset.data.site_pos_w[:, outer_cfg.site_ids, 2],
    ],
    dim=1,
  )  # [B, 5 sole points, 2 feet]
  # The task may tighten this tolerance when high-friction soles need the
  # commanded clearance rather than a lower, reward-equivalent shuffle.
  tracking = torch.exp(
    -torch.square(foot_z - target[:, None, :]) / (tracking_sigma**2)
  )
  return torch.mean(tracking, dim=(1, 2)) * march


def swing_sole_height_spread_l2(
  env: "ManagerBasedRlEnv",
  command_name: str,
  heel_cfg: SceneEntityCfg,
  toe_cfg: SceneEntityCfg,
  inner_cfg: SceneEntityCfg,
  outer_cfg: SceneEntityCfg,
  tolerance: float = 0.005,
) -> torch.Tensor:
  """Normalized height spread across the four swing-sole extremities.

  Zero means the complete sole is level. A 5 mm heel/toe or inner/outer
  difference costs one before weighting. Only the commanded swing foot is
  penalized, leaving the planted foot governed by contact and slip terms.
  """
  asset: Entity = env.scene[heel_cfg.name]
  command = env.command_manager.get_command(command_name)
  march, sinp = command[:, 0], command[:, 1]
  swing = torch.stack([(sinp > 0).float(), (sinp < 0).float()], dim=1)
  z = torch.stack(
    [
      asset.data.site_pos_w[:, heel_cfg.site_ids, 2],
      asset.data.site_pos_w[:, toe_cfg.site_ids, 2],
      asset.data.site_pos_w[:, inner_cfg.site_ids, 2],
      asset.data.site_pos_w[:, outer_cfg.site_ids, 2],
    ],
    dim=1,
  )
  spread = torch.amax(z, dim=1) - torch.amin(z, dim=1)
  cost = torch.sum(torch.square(spread / tolerance) * swing, dim=1)
  return cost * march


def base_ang_vel_xy_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize roll/pitch angular velocity, the direct measure of wobble."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def base_tilt_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Direct quadratic roll/pitch tilt penalty from projected gravity."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def base_xy_position_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize displacement from the environment origin for true in-place motion."""
  asset: Entity = env.scene[asset_cfg.name]
  delta = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  return torch.sum(torch.square(delta), dim=1)


def action_rate_soft_limit_l2(
  env: "ManagerBasedRlEnv",
  action_scale: float,
  max_rate: float,
) -> torch.Tensor:
  """Penalize only target rates above the real servo-friendly limit.

  The action is a normalized offset, so convert its per-control-step delta to
  rad/s before applying the squared hinge. Smooth useful motions are free;
  jerky commands receive a rapidly increasing penalty.
  """
  delta = env.action_manager.action - env.action_manager.prev_action
  rate = torch.abs(delta) * action_scale / env.step_dt
  return torch.sum(torch.square(torch.clamp(rate - max_rate, min=0.0)), dim=1)


def feet_planted(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  command_name: str,
) -> torch.Tensor:
  """Reward both feet in contact while holding (march == 0)."""
  command = env.command_manager.get_command(command_name)
  hold = 1.0 - command[:, 0]
  contact = _contact(env, sensor_name)  # [B, 2]
  return torch.mean(contact, dim=1) * hold


def hold_joint_pair_symmetry_l2(
  env: "ManagerBasedRlEnv",
  command_name: str,
  right_cfg: SceneEntityCfg,
  left_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Keep the learned hold stance paired without prescribing its exact pose.

  Corresponding K2 joint coordinates use the same physical semantics on both
  legs (bend, outward, toe-in, and so on). Their offsets from the respective
  defaults should therefore match in a symmetric two-foot hold. Unlike the
  rejected exact-default-pose term, this still permits the policy to choose a
  wider or more compliant stable stance.
  """
  asset: Entity = env.scene[right_cfg.name]
  q = asset.data.joint_pos
  d = asset.data.default_joint_pos
  u_r = q[:, right_cfg.joint_ids] - d[:, right_cfg.joint_ids]
  u_l = q[:, left_cfg.joint_ids] - d[:, left_cfg.joint_ids]
  if u_r.shape[1] != u_l.shape[1]:
    raise RuntimeError("hold symmetry requires paired right/left joint lists")
  hold = 1.0 - env.command_manager.get_command(command_name)[:, 0]
  return torch.sum(torch.square(u_r - u_l), dim=1) * hold


def base_height_l2(
  env: "ManagerBasedRlEnv",
  target_height: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Quadratic penalty on base height deviation from target (both modes)."""
  asset: Entity = env.scene[asset_cfg.name]
  base_z = asset.data.root_link_pos_w[:, 2]
  return torch.square(base_z - target_height)


def base_lin_vel_xy_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Quadratic penalty on horizontal base velocity -- the real 'stay in place'.

  A proportional penalty (not a wide exp kernel) so even a few cm/s of drift is
  penalized; without this the march learns to slow-walk instead of stepping in
  place.
  """
  asset: Entity = env.scene[asset_cfg.name]
  v_xy = asset.data.root_link_lin_vel_b[:, :2]
  return torch.sum(torch.square(v_xy), dim=1)


def track_forward_velocity_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Track the commanded body-frame forward speed without needing odometry.

  The command is the optional fourth channel of ``GaitCommand``. It is zero in
  hold environments and 1--4 cm/s in the first walking curriculum.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  if command.shape[1] < 4:
    raise RuntimeError("forward velocity reward requires a 4-D gait command")
  error = asset.data.root_link_lin_vel_b[:, 0] - command[:, 3]
  return torch.exp(-torch.square(error) / (std**2))


def base_lateral_velocity_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize sideways velocity while leaving commanded forward motion free."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 1])


def base_yaw_rate_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Direct yaw-rate penalty for straight-line walking."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_ang_vel_b[:, 2])


def gait_antisymmetry_l2(
  env: "ManagerBasedRlEnv",
  command_name: str,
  right_cfg: SceneEntityCfg,
  left_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize a left/right mismatch in sagittal swing amplitude.

  An alternating gait is antiphase, so for a near-sinusoidal joint the two
  legs' deviations from their own defaults should cancel: u_L(t) ~= -u_R(t).
  Penalizing (u_R + u_L)^2 therefore costs amplitude mismatch without
  prescribing a waveform. Measured on the iteration-2000 logs the residual is
  only 12--13% forward-lean DC, so this reads real asymmetry rather than the
  postural offset walking needs.

  Intended for hip pitch only. Knee and ankle pitch swing as once-per-step
  pulses whose antiphase sum is a nonzero constant, so the same penalty would
  fight their normal shape.
  """
  asset: Entity = env.scene[right_cfg.name]
  q = asset.data.joint_pos
  d = asset.data.default_joint_pos
  u_r = (q[:, right_cfg.joint_ids] - d[:, right_cfg.joint_ids]).sum(dim=1)
  u_l = (q[:, left_cfg.joint_ids] - d[:, left_cfg.joint_ids]).sum(dim=1)
  march = env.command_manager.get_command(command_name)[:, 0]
  return march * torch.square(u_r + u_l)


def base_heading_error(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Smooth, wrap-safe cost for deviation from the episode's forward axis."""
  asset: Entity = env.scene[asset_cfg.name]
  return 1.0 - torch.cos(asset.data.heading_w)


def feet_xy_vel_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize horizontal foot velocity at ALL times (swing AND stance).

  This is the 'clean march' signal: a foot that only moves vertically lifts
  straight up and lands in the same spot (marching in place, like a human),
  instead of skimming forward into a walk. Unlike feet_slip (contact only) this
  also shapes the swing to go up/down, not forward.
  """
  asset: Entity = env.scene[asset_cfg.name]
  v_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, 2, 2]
  return torch.sum(torch.square(v_xy), dim=(1, 2))


def feet_lateral_vel_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize sideways foot motion but permit fore/aft walking trajectories."""
  asset: Entity = env.scene[asset_cfg.name]
  lateral = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, 1]
  return torch.sum(torch.square(lateral), dim=1)


def feet_under_base_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize the feet midpoint drifting out from under the base (xy).

  Keeps the stepping in place: if the swing foot lands ahead of the body (a
  walking gait) the feet midpoint moves out from under the base and is
  penalized. Whole-body drift is handled separately by base_lin_vel_xy_l2.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # [B, 2, 2]
  mid = foot_xy.mean(dim=1)  # [B, 2]
  base_xy = asset.data.root_link_pos_w[:, :2]
  return torch.sum(torch.square(mid - base_xy), dim=1)


def base_lateral_position_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize sideways drift from the spawn line, not forward progress."""
  asset: Entity = env.scene[asset_cfg.name]
  lateral = asset.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
  return torch.square(lateral)


def feet_lateral_clearance_l2(
  env: "ManagerBasedRlEnv",
  minimum_separation: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize feet that approach or cross in the robot's lateral direction.

  A normalized squared hinge gives zero cost above the requested center-to-
  center gap and rises smoothly to one at zero gap.  Measuring in the root
  frame keeps the term correct if domain-randomization pushes yaw the robot.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_delta_w = (
    asset.data.site_pos_w[:, asset_cfg.site_ids[0]]
    - asset.data.site_pos_w[:, asset_cfg.site_ids[1]]
  )
  foot_delta_b = quat_apply_inverse(asset.data.root_link_quat_w, foot_delta_w)
  lateral_gap = torch.abs(foot_delta_b[:, 1])
  violation = torch.clamp(
    (minimum_separation - lateral_gap) / minimum_separation, min=0.0
  )
  return torch.square(violation)
