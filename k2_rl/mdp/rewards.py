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

import math
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
  swing_threshold: float = 0.18,
) -> torch.Tensor:
  """Reward alternating contact with a double-support window at phase crossings.

  ``swing_threshold`` must be read together with :func:`gait_swing`, because the
  two terms score the same foot at the same instant. This one demands the foot
  be OFF the ground once ``|sin| > swing_threshold``; that one rewards the sole
  sitting at ``target_height * sin**2``. At the historical 0.18 the swing target
  is only 0.4 mm, so the schedule asks for a lift that the height reward has not
  yet started paying for, and the two nearly cancel. Raise the threshold on a
  task that wants the two agreeing; the default preserves the validated gait.
  """
  command = env.command_manager.get_command(command_name)
  march = command[:, 0]  # [B]
  sinp = command[:, 1]  # [B]
  contact = _contact(env, sensor_name)  # [B, 2]
  # A foot does not become a swing foot until the smooth swing clock is clear
  # of zero. Around each zero crossing both feet should be planted. This avoids
  # the old instantaneous support-foot swap that excited lateral wobble.
  swing0 = sinp > swing_threshold
  swing1 = sinp < -swing_threshold
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


def base_pitch_rate_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize pitch rate while leaving lateral weight-transfer roll free."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_ang_vel_b[:, 1])


def lateral_balance_envelope_l2(
  env: "ManagerBasedRlEnv",
  max_roll: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Keep pitch level and charge roll only beyond a safe side-step envelope."""
  asset: Entity = env.scene[asset_cfg.name]
  gravity = asset.data.projected_gravity_b
  roll = torch.atan2(gravity[:, 1], -gravity[:, 2])
  pitch_cost = torch.square(gravity[:, 0])
  roll_excess = torch.clamp(torch.abs(roll) - max_roll, min=0.0)
  return pitch_cost + torch.square(roll_excess / max_roll)


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
  idle_only: bool = False,
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
  if idle_only:
    # A commanded standing twist is asymmetric by construction. Keep the
    # original hold constraint for idle stance and its original zero in gait.
    command = env.command_manager.get_command(command_name)
    hold = hold * (torch.abs(command[:, 4]) < 1e-9).float()
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
  yaw_command_index: int | None = None,
  yaw_fade_std: float = 0.12,
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

  A turning gait is genuinely asymmetric -- the inside leg takes the shorter
  step -- so pass ``yaw_command_index`` on a task that commands yaw and the
  penalty fades smoothly as the commanded yaw rate grows. Left unset (the
  default) the term is exactly the straight-walking penalty it has always been.
  """
  asset: Entity = env.scene[right_cfg.name]
  q = asset.data.joint_pos
  d = asset.data.default_joint_pos
  u_r = (q[:, right_cfg.joint_ids] - d[:, right_cfg.joint_ids]).sum(dim=1)
  u_l = (q[:, left_cfg.joint_ids] - d[:, left_cfg.joint_ids]).sum(dim=1)
  command = env.command_manager.get_command(command_name)
  march = command[:, 0]
  cost = march * torch.square(u_r + u_l)
  if yaw_command_index is not None:
    cost = cost * torch.exp(
      -torch.square(command[:, yaw_command_index]) / (yaw_fade_std**2)
    )
  return cost


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


def hold_feet_still_l2(
  env: "ManagerBasedRlEnv",
  command_name: str,
  asset_cfg: SceneEntityCfg,
  linear_scale: float = 0.03,
  angular_scale: float = 0.5,
) -> torch.Tensor:
  """Softly discourage sole translation and rotation while holding.

  Contact alone does not mean a sole is stationary: it may slide, roll, or
  pivot while remaining in contact.  This measures both linear and angular
  velocity at each foot-center site, normalized to ordinary motion scales so a
  moderate reward weight is meaningful.

  The cost is multiplied by ``1 - march``.  A policy later widened into a turn
  task inherits no restriction at all while stepping, so this cannot oppose
  the foot motion required by a turning gait.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  hold = 1.0 - command[:, 0]
  linear = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :] / linear_scale
  angular = asset.data.site_ang_vel_w[:, asset_cfg.site_ids, :] / angular_scale
  motion = torch.sum(torch.square(linear), dim=(1, 2))
  motion += torch.sum(torch.square(angular), dim=(1, 2))
  return motion * hold


class track_squat_posture_l2:
  """Track a model-derived flat-foot sagittal squat without fixing other DOFs.

  The reference is a height-indexed inverse-kinematics table.  Only hip pitch,
  knee, and ankle pitch are selected by ``asset_cfg``; roll and yaw are absent.
  This supplies the coordination that height error alone cannot express: bend
  the knee while counter-rotating hip and ankle so the torso remains vertical
  and the planted soles do not roll forward.
  """

  def __init__(self, cfg, env: "ManagerBasedRlEnv"):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    asset_cfg.resolve(env.scene)
    heights = cfg.params["reference_heights"]
    joint_pos = cfg.params["reference_joint_pos"]
    if len(heights) != len(joint_pos) or len(heights) < 2:
      raise ValueError("squat reference needs matching height/pose rows")
    if any(b <= a for a, b in zip(heights, heights[1:])):
      raise ValueError("squat reference heights must be strictly increasing")
    if any(len(row) != len(asset_cfg.joint_ids) for row in joint_pos):
      raise ValueError("squat reference width must match selected joints")
    self.heights = torch.tensor(heights, device=env.device)
    self.joint_pos = torch.tensor(joint_pos, device=env.device)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    target_height: float,
    reference_heights: tuple[float, ...],
    reference_joint_pos: tuple[tuple[float, ...], ...],
    asset_cfg: SceneEntityCfg,
    std: float,
    height_index: int = 5,
  ) -> torch.Tensor:
    del reference_heights, reference_joint_pos  # cached in __init__
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    desired = (target_height + command[:, height_index]).clamp(
      min=float(self.heights[0]), max=float(self.heights[-1])
    )
    hi = torch.searchsorted(self.heights, desired).clamp(
      min=1, max=len(self.heights) - 1
    )
    lo = hi - 1
    span = self.heights[hi] - self.heights[lo]
    alpha = ((desired - self.heights[lo]) / span).unsqueeze(1)
    reference = self.joint_pos[lo] + alpha * (
      self.joint_pos[hi] - self.joint_pos[lo]
    )
    error = (asset.data.joint_pos[:, asset_cfg.joint_ids] - reference) / std
    return torch.sum(torch.square(error), dim=1)


class track_standing_rotate_ik_exp:
  """Track the foot-locked IK stroke while a standing rotation is active."""

  def __init__(self, cfg, env: "ManagerBasedRlEnv"):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    asset_cfg.resolve(env.scene)
    left = cfg.params["left_reference"]
    right = cfg.params["right_reference"]
    if len(left) < 2 or len(left) != len(right):
      raise ValueError("standing rotation needs matching IK paths")
    if any(len(row) != len(asset_cfg.joint_ids) for row in (*left, *right)):
      raise ValueError("standing rotation IK width must match selected joints")
    self.left = torch.tensor(left, device=env.device, dtype=torch.float32)
    self.right = torch.tensor(right, device=env.device, dtype=torch.float32)

  @staticmethod
  def _stroke_position(path: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
    x = progress.clamp(0.0, 1.0) * (path.shape[0] - 1)
    lo = torch.floor(x).long().clamp(max=path.shape[0] - 2)
    alpha = (x - lo).unsqueeze(1)
    return path[lo] + alpha * (path[lo + 1] - path[lo])

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    left_reference,
    right_reference,
    asset_cfg: SceneEntityCfg,
    std: float,
    smooth_progress: bool = False,
  ) -> torch.Tensor:
    del left_reference, right_reference
    term = env.command_manager.get_term(command_name)
    if not hasattr(term, "rotate_phase"):
      raise RuntimeError("IK rotation reward requires StandingRotateCommand")
    command = env.command_manager.get_command(command_name)
    active = term.standing_rotate_active.float()
    cycle = (term.rotate_phase / (2.0 * math.pi)).clamp(0.0, 1.0)
    # Timing matches rotate_ik.py: 3 s twist, 1.5 s dwell, 3 s return,
    # 1.5 s neutral dwell. Each stroke occupies one third of the cycle.
    outbound = (cycle / (1.0 / 3.0)).clamp(0.0, 1.0)
    inbound = ((cycle - 0.5) / (1.0 / 3.0)).clamp(0.0, 1.0)
    progress = torch.where(
      cycle < 1.0 / 3.0,
      outbound,
      torch.where(
        cycle < 0.5,
        torch.ones_like(cycle),
        torch.where(cycle < 5.0 / 6.0, 1.0 - inbound, torch.zeros_like(cycle)),
      ),
    )
    if smooth_progress:
      progress = progress**3 * (progress * (6 * progress - 15) + 10)
    left_target = self._stroke_position(self.left, progress)
    right_target = self._stroke_position(self.right, progress)
    target = torch.where(
      (command[:, 4] >= 0.0).unsqueeze(1), left_target, right_target
    )
    asset: Entity = env.scene[asset_cfg.name]
    error = (asset.data.joint_pos[:, asset_cfg.joint_ids] - target) / std
    return torch.exp(-torch.mean(torch.square(error), dim=1)) * active


def track_standing_rotate_yaw_rate_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Tight yaw-rate tracking used only by the standing rotation pulse."""
  term = env.command_manager.get_term(command_name)
  if not hasattr(term, "standing_rotate_active"):
    raise RuntimeError("standing yaw reward requires StandingRotateCommand")
  asset: Entity = env.scene[asset_cfg.name]
  target = env.command_manager.get_command(command_name)[:, 4]
  error = asset.data.root_link_ang_vel_b[:, 2] - target
  return torch.exp(-torch.square(error) / (std**2)) * (
    term.standing_rotate_active.float()
  )


class turn_teacher_action_l2:
  """Keep marching actions near a frozen, hardware-proven ONNX actor.

  ``env.obs_buf['actor']`` is the exact delayed/noisy observation that produced
  the action currently being scored: rewards are computed before the next
  observation is assembled.  The penalty is march-gated, so the teacher has no
  authority over the new standing rotation skill.
  """

  def __init__(self, cfg, env: "ManagerBasedRlEnv"):
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(cfg.params["onnx_path"])
    arrays = {
      item.name: numpy_helper.to_array(item) for item in model.graph.initializer
    }
    self.mean = torch.tensor(
      np.array(arrays["obs_normalizer._mean"], copy=True), device=env.device
    )
    self.std = torch.tensor(
      np.array(arrays["onnx::Div_24"], copy=True), device=env.device
    )
    self.layers = tuple(
      (
        torch.tensor(np.array(arrays[f"mlp.{idx}.weight"], copy=True), device=env.device),
        torch.tensor(np.array(arrays[f"mlp.{idx}.bias"], copy=True), device=env.device),
      )
      for idx in (0, 2, 4, 6)
    )

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    onnx_path: str,
    std: float,
  ) -> torch.Tensor:
    del onnx_path
    obs = env.obs_buf["actor"]
    teacher = (obs - self.mean) / self.std
    for weight, bias in self.layers[:-1]:
      teacher = torch.nn.functional.elu(
        torch.nn.functional.linear(teacher, weight, bias)
      )
    teacher = torch.nn.functional.linear(
      teacher, self.layers[-1][0], self.layers[-1][1]
    )
    action = env.action_manager.action
    march = env.command_manager.get_command(command_name)[:, 0]
    return torch.mean(torch.square((action - teacher) / std), dim=1) * march


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


##
# Turning: the straight-line terms above measure against world +X and the world
# +Y spawn line, which is only correct when no yaw is commanded. Each function
# below measures the same physical quantity against the gait command's own
# reference instead, and reduces EXACTLY to its predecessor when the commanded
# yaw rate is identically zero.
##


def _gait_reference(env: "ManagerBasedRlEnv", command_name: str):
  """The gait term carrying the integrated reference pose."""
  term = env.command_manager.get_term(command_name)
  if not hasattr(term, "heading_ref"):
    raise RuntimeError(
      f"command '{command_name}' does not integrate a reference pose; "
      "these rewards require k2_rl.mdp.GaitCommand with a forward channel"
    )
  return term


def yaw_rate_error_l2(
  env: "ManagerBasedRlEnv",
  command_name: str,
  yaw_command_index: int = 4,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Quadratic penalty on yaw rate *in excess of* the commanded yaw rate.

  Drop-in replacement for :func:`base_yaw_rate_l2`: with a zero yaw command the
  two are the same expression, so straight walking is damped exactly as before.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  error = asset.data.root_link_ang_vel_b[:, 2] - command[:, yaw_command_index]
  return torch.square(error)


def track_yaw_rate_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  std: float,
  yaw_command_index: int = 4,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Reward tracking the commanded yaw rate while keeping roll/pitch rate low.

  Same kernel and same roll/pitch treatment as mjlab's ``track_angular_velocity``
  but reading the yaw target from the gait command rather than from the twist
  command, which the walking tasks pin to zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, yaw_command_index] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  return torch.exp(-(z_error + xy_error) / std**2)


def heading_reference_error(
  env: "ManagerBasedRlEnv",
  command_name: str,
  yaw_command_index: int | None = None,
  yaw_fade_std: float = 0.12,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Wrap-safe cost for deviation from the commanded (integrated) heading.

  Reduces to :func:`base_heading_error` whenever the reference never rotates.

  ANTI-WINDUP. ``heading_ref`` integrates the commanded yaw rate, so this term
  is the time-INTEGRAL of the same error that :func:`yaw_rate_error_l2` and
  :func:`track_yaw_rate_exp` measure directly. While yaw is commanded, any
  persistent rate deficit accumulates here without bound -- the robot realizes
  roughly half of a commanded rate on hardware, which over a 20 s episode at
  0.25 rad/s is 2.5 rad of reference the robot never catches, worth -9/step at
  weight -5. That is larger than the entire positive budget, and no amount of
  correct behaviour recovers it.

  Passing ``yaw_command_index`` fades the term out as yaw is commanded, which
  is exactly where the windup lives: with ``wz == 0`` the reference does not
  rotate, the error cannot wind up, and this stays the straight-line "am I
  walking where I was pointed" signal it has always been. Left unset (the
  default) the term is unchanged for the forward and bidirectional tasks.
  """
  asset: Entity = env.scene[asset_cfg.name]
  term = _gait_reference(env, command_name)
  cost = 1.0 - torch.cos(asset.data.heading_w - term.heading_ref)
  if yaw_command_index is not None:
    command = env.command_manager.get_command(command_name)
    cost = cost * torch.exp(
      -torch.square(command[:, yaw_command_index]) / (yaw_fade_std**2)
    )
  return cost


def lateral_path_deviation_l2(
  env: "ManagerBasedRlEnv",
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize drift perpendicular to the commanded path, not forward progress.

  The reference pose starts on the environment origin facing world +X, so with
  no yaw command this is identically ``(y - y_origin)**2`` -- exactly
  :func:`base_lateral_position_l2`. With yaw commanded the corridor turns with
  the command instead of pinning the robot to the world +X axis.
  """
  asset: Entity = env.scene[asset_cfg.name]
  term = _gait_reference(env, command_name)
  delta = asset.data.root_link_pos_w[:, :2] - term.pos_ref
  heading = term.heading_ref
  lateral = -delta[:, 0] * torch.sin(heading) + delta[:, 1] * torch.cos(heading)
  return torch.square(lateral)


def longitudinal_path_deviation_l2(
  env: "ManagerBasedRlEnv",
  command_name: str,
  forward_command_index: int = 3,
  command_threshold: float = 0.005,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize ALONG-path drift, but only while the commanded speed is ~zero.

  :func:`lateral_path_deviation_l2` takes only the component perpendicular to
  the commanded heading, so nothing in the task constrains position along it.
  With a nonzero speed command that is correct -- the reference moves and any
  along-path error is the integral of a speed-tracking deficit, which would
  wind up exactly like the heading term. With ZERO speed commanded the
  reference is static, so along-path error is plain drift from where the robot
  was told to stay, with no windup available.

  Gating on the commanded speed is what separates those two cases, and it is
  what makes "stand still" mean something once there is no hold mode to carry
  that meaning.
  """
  asset: Entity = env.scene[asset_cfg.name]
  term = _gait_reference(env, command_name)
  command = env.command_manager.get_command(command_name)
  delta = asset.data.root_link_pos_w[:, :2] - term.pos_ref
  heading = term.heading_ref
  along = delta[:, 0] * torch.cos(heading) + delta[:, 1] * torch.sin(heading)
  standing = (
    torch.abs(command[:, forward_command_index]) < command_threshold
  ).float()
  return torch.square(along) * standing


def feet_lateral_vel_body_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Body-frame version of :func:`feet_lateral_vel_l2`.

  The world-frame term reads ``site_lin_vel_w[..., 1]``, which stops meaning
  "sideways" as soon as the robot is allowed to turn away from world +X. Rotating
  the foot velocity into the root frame first keeps the same intent -- permit
  fore/aft swing, penalize sideways scuffing -- at any heading.
  """
  asset: Entity = env.scene[asset_cfg.name]
  v_w = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]  # [B, F, 3]
  num_sites = v_w.shape[1]
  quat = (
    asset.data.root_link_quat_w[:, None, :]
    .expand(-1, num_sites, -1)
    .reshape(-1, 4)
  )
  v_b = quat_apply_inverse(quat, v_w.reshape(-1, 3)).view(-1, num_sites, 3)
  return torch.sum(torch.square(v_b[..., 1]), dim=1)


class swing_amplitude_symmetry_l2:
  """Penalize a LIMP: unequal swing amplitude between the two legs.

  :func:`gait_antisymmetry_l2` penalizes ``(u_R + u_L)**2``, which does register
  hip-pitch amplitude mismatch, but only faintly -- the measured hardware limp
  was 1.4 deg, i.e. 0.024 rad, worth 6e-4 before weighting. And that form is
  wrong for knee and ankle pitch, whose antiphase sum is a nonzero constant, so
  driving it to zero fights the shape those joints are supposed to have.

  This samples each leg's joint deviation AT ITS OWN SWING APEX and penalizes
  the difference, comparing like with like. It needs one buffer per leg, which
  is why this is a class: mjlab instantiates class terms with ``(cfg, env)`` and
  keeps them alive across steps.

  Sagittal joints only (hip pitch, knee, ankle pitch). Both legs move the same
  way in the sagittal plane, so the deviations compare directly with no sign
  flip -- unlike roll/yaw, which are mirrored.

  MUST FADE WITH COMMANDED YAW. This measures stride-amplitude mismatch between
  the legs, which is precisely HOW a differential-stride turn is produced: the
  inside leg takes the shorter step. ``gait_antisymmetry_l2`` already fades for
  that reason; this term is a heavier (-25) and more direct measure of the same
  asymmetry, so leaving it unfaded charges the policy the largest penalty in the
  task for executing the turn it is simultaneously rewarded for. Pass
  ``yaw_command_index`` on any task that commands yaw. Left unset (the default)
  the term is the straight-walking limp penalty it has always been.
  """

  def __init__(self, cfg, env: "ManagerBasedRlEnv"):
    right_cfg: SceneEntityCfg = cfg.params["right_cfg"]
    left_cfg: SceneEntityCfg = cfg.params["left_cfg"]
    right_cfg.resolve(env.scene)
    left_cfg.resolve(env.scene)
    if len(right_cfg.joint_ids) != len(left_cfg.joint_ids):
      raise RuntimeError("leg symmetry requires paired right/left joint lists")
    n = len(right_cfg.joint_ids)
    self.peak_r = torch.zeros(env.num_envs, n, device=env.device)
    self.peak_l = torch.zeros(env.num_envs, n, device=env.device)

  def reset(self, env_ids=None):
    if env_ids is None:
      self.peak_r.zero_()
      self.peak_l.zero_()
    else:
      self.peak_r[env_ids] = 0.0
      self.peak_l[env_ids] = 0.0
    return {}

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    right_cfg: SceneEntityCfg,
    left_cfg: SceneEntityCfg,
    apex: float = 0.9,
    yaw_command_index: int | None = None,
    yaw_fade_std: float = 0.12,
  ) -> torch.Tensor:
    asset: Entity = env.scene[right_cfg.name]
    q = asset.data.joint_pos
    d = asset.data.default_joint_pos
    u_r = q[:, right_cfg.joint_ids] - d[:, right_cfg.joint_ids]
    u_l = q[:, left_cfg.joint_ids] - d[:, left_cfg.joint_ids]
    command = env.command_manager.get_command(command_name)
    march, sinp = command[:, 0], command[:, 1]
    # One leg is at its swing extreme near sin = +1, the other near sin = -1.
    # Which is which does not matter: either way each leg is sampled at the same
    # point of its own half-cycle, so the comparison stays like-for-like.
    at_r = (sinp > apex).unsqueeze(1)
    at_l = (sinp < -apex).unsqueeze(1)
    self.peak_r = torch.where(at_r, u_r.detach(), self.peak_r)
    self.peak_l = torch.where(at_l, u_l.detach(), self.peak_l)
    cost = march * torch.sum(torch.square(self.peak_r - self.peak_l), dim=1)
    if yaw_command_index is not None:
      cost = cost * torch.exp(
        -torch.square(command[:, yaw_command_index]) / (yaw_fade_std**2)
      )
    return cost


def base_height_command_l2(
  env: "ManagerBasedRlEnv",
  command_name: str,
  target_height: float,
  height_index: int = 5,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Quadratic penalty on base height versus the COMMANDED height.

  Drop-in replacement for :func:`base_height_l2`: the command channel is zero
  while walking (it is gated by ``1 - march``), so walking is scored against
  exactly the same fixed target as before and only the hold height moves.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  target = target_height + command[:, height_index]
  return torch.square(asset.data.root_link_pos_w[:, 2] - target)


def hold_joint_vel_l2(
  env: "ManagerBasedRlEnv",
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize joint motion while holding -- damping for the standing loop.

  NOTE the limitation, measured 2026-08-21: on hardware the hold loop buzzes at
  ~2.9 Hz with action-rate RMS 1.58, nearly the 1.7 it uses while WALKING. In
  the twin the same policy holds at action-rate 0.01, i.e. essentially still.
  So this term has almost no gradient to work with in simulation and cannot be
  tuned there. It is included as insurance, not as the fix; the fix that does
  have gradient is lowering overall loop gain via the action-rate penalties.
  """
  asset: Entity = env.scene[asset_cfg.name]
  hold = 1.0 - env.command_manager.get_command(command_name)[:, 0]
  return torch.sum(torch.square(asset.data.joint_vel), dim=1) * hold


def double_support_base_tilt(
  env: "ManagerBasedRlEnv",
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Quadratic roll/pitch tilt penalty applied ONLY while holding.

  Shapes the two-foot stance by what it must achieve -- a level base over a
  double-support polygon -- instead of by the joint angles that achieve it.
  This is the companion to :func:`hold_joint_pair_symmetry_l2`: that term says
  the two legs must agree with each other, this one says the pair must hold the
  body level. Neither prescribes a pose, which is the property that matters --
  forcing the exact captured crouch defeated a previously stable hold (see the
  ``hold_pair_symmetry`` note in ``inplace_env_cfg``).

  Gated by ``1 - march`` only, deliberately NOT by foot contact. A contact gate
  would let the policy dodge the penalty by lifting a foot, and the incentive to
  keep both feet down already comes from :func:`feet_planted`.

  Identical in form to :func:`base_tilt_l2`, so a task that wants tilt held
  harder in hold than in gait can carry both terms at different weights.
  """
  asset: Entity = env.scene[asset_cfg.name]
  hold = 1.0 - env.command_manager.get_command(command_name)[:, 0]
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) * hold


##
# Lateral (vy) task. The sagittal axis is what must be PINNED here, so these are
# the fore/aft mirrors of the lateral terms above. Heading is held near world +X
# by the heading reward, so world x and body forward coincide to within a few
# degrees; the velocity terms read the body frame anyway and are exact.
##


def base_longitudinal_velocity_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize fore/aft velocity while leaving commanded lateral motion free."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 0])


def base_longitudinal_position_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Penalize fore/aft displacement from the spawn line, leaving y free."""
  asset: Entity = env.scene[asset_cfg.name]
  delta = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  return torch.square(delta)


def feet_longitudinal_vel_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize fore/aft foot scuffing; sideways swing is the task here."""
  asset: Entity = env.scene[asset_cfg.name]
  v_w = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]
  num_sites = v_w.shape[1]
  quat = (
    asset.data.root_link_quat_w[:, None, :]
    .expand(-1, num_sites, -1)
    .reshape(-1, 4)
  )
  v_b = quat_apply_inverse(quat, v_w.reshape(-1, 3)).view(-1, num_sites, 3)
  return torch.sum(torch.square(v_b[..., 0]), dim=1)


def feet_midpoint_longitudinal_l2(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Fore/aft-only version of :func:`feet_under_base_l2`.

  The two-axis form cannot be used on a lateral task: it penalizes the feet
  midpoint leaving the base in ANY direction, and a side step moves that
  midpoint sideways by half the step before the body follows. Charging that is
  charging the task. Fore/aft is still worth pinning, so only that axis is read.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]
  mid = foot_xy.mean(dim=1)
  delta_w = mid - asset.data.root_link_pos_w[:, :2]
  heading = asset.data.heading_w
  longitudinal = (
    delta_w[:, 0] * torch.cos(heading) + delta_w[:, 1] * torch.sin(heading)
  )
  return torch.square(longitudinal)


class track_lateral_velocity_filtered_exp:
  """Track commanded lateral speed through a LOW-PASS FILTER, not instantly.

  This is the whole reason a naive vy reward cannot teach a side step, and it
  is geometry, not tuning. Unloading one foot moves the CoM sideways by the
  half-stance (33.8 mm on this robot, 9.9 deg of roll about a 193 mm CoM), and
  that happens once per step IN THE SAME AXIS the lateral command is measured
  in. At 0.6 Hz that rock alone is a peak lateral velocity near 0.11 m/s --
  four times a 0.025 m/s command. Forward walking never meets this problem: its
  rock is in v_y while its reward reads v_x, so the two are orthogonal. Lateral
  walking puts the disturbance and the signal on one axis.

  Scored instantaneously with the std the vx tracker uses, a PERFECT side step
  therefore reads as a huge tracking error for most of every cycle, while
  standing still reads as a small one -- the reward prefers not stepping, and
  the policy is right to obey it.

  The rock is periodic at the gait frequency and very nearly zero-mean over a
  cycle, so a first-order filter longer than one cycle removes it while leaving
  the net drift the command actually asks for. The filter runs on a detached
  buffer, which is the same pattern :class:`swing_amplitude_symmetry_l2` uses;
  PPO never differentiates a reward, so nothing is lost.

  ``filter_time_s`` trades attenuation against credit-assignment delay -- with
  gamma 0.99 at 50 Hz the discount horizon is about 2 s, so a filter far beyond
  that pushes reward past where the return can carry it. Verify both together.
  """

  def __init__(self, cfg, env: "ManagerBasedRlEnv"):
    self.v_ema = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids=None):
    if env_ids is None:
      self.v_ema.zero_()
    else:
      self.v_ema[env_ids] = 0.0
    return {}

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    std: float,
    lateral_command_index: int = 5,
    filter_time_s: float = 2.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    v_y = asset.data.root_link_lin_vel_b[:, 1].detach()
    alpha = min(env.step_dt / filter_time_s, 1.0)
    self.v_ema = self.v_ema + alpha * (v_y - self.v_ema)
    command = env.command_manager.get_command(command_name)
    error = self.v_ema - command[:, lateral_command_index]
    return torch.exp(-torch.square(error) / (std**2))


class track_lateral_velocity_cycle_exp:
  """Track body-frame lateral velocity averaged over one complete gait cycle.

  The cycle mean cancels periodic weight-transfer rocking without retaining the
  multi-second command-reversal memory of an EMA.  A target resample clears the
  affected environment's window, and reward starts after one complete cycle.
  """

  def __init__(self, cfg, env: "ManagerBasedRlEnv"):
    term = env.command_manager.get_term(cfg.params["command_name"])
    self.window_steps = max(
      1, int(round(1.0 / (float(term.cfg.gait_freq) * env.step_dt)))
    )
    self.history = torch.zeros(
      self.window_steps, env.num_envs, device=env.device
    )
    self.running_sum = torch.zeros(env.num_envs, device=env.device)
    self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.last_target = torch.full(
      (env.num_envs,), float("nan"), device=env.device
    )
    self.index = 0

  def reset(self, env_ids=None):
    if env_ids is None:
      self.history.zero_()
      self.running_sum.zero_()
      self.count.zero_()
      self.last_target.fill_(float("nan"))
      self.index = 0
    else:
      self.history[:, env_ids] = 0.0
      self.running_sum[env_ids] = 0.0
      self.count[env_ids] = 0
      self.last_target[env_ids] = float("nan")
    return {}

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    std: float,
    lateral_command_index: int = 5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    term = env.command_manager.get_term(command_name)
    target = term.lateral_velocity_target
    changed = torch.isnan(self.last_target) | ~torch.isclose(
      target, self.last_target, atol=1e-7, rtol=0.0
    )
    if bool(changed.any()):
      self.history[:, changed] = 0.0
      self.running_sum[changed] = 0.0
      self.count[changed] = 0
      self.last_target[changed] = target[changed]

    v_y = asset.data.root_link_lin_vel_b[:, 1].detach()
    old = self.history[self.index].clone()
    self.history[self.index] = v_y
    self.running_sum.add_(v_y - old)
    self.count.add_(1).clamp_(max=self.window_steps)
    self.index = (self.index + 1) % self.window_steps

    mean_vy = self.running_sum / self.count.clamp(min=1)
    command = env.command_manager.get_command(command_name)
    error = mean_vy - command[:, lateral_command_index]
    ready = (self.count >= self.window_steps).float()
    return torch.exp(-torch.square(error) / (std**2)) * ready


def hold_stance_width_l2(
  env: "ManagerBasedRlEnv",
  command_name: str,
  max_separation: float,
  asset_cfg: SceneEntityCfg,
  hold_gated: bool = True,
) -> torch.Tensor:
  """Penalize a HOLD stance wider than nominal. One-sided, and hold-gated.

  Nothing else in the task stops the feet drifting apart while standing. Every
  `feet_lateral_clearance_l2` term is a floor -- it charges feet that come too
  CLOSE and is exactly zero above its minimum -- so a wider stance is free, and
  it is also genuinely easier to balance on (bigger support polygon, less roll
  to stay inside it). A policy will find that, and the result is a robot that
  stands with its legs splayed.

  Deliberately ONE-SIDED, and deliberately not a pose constraint:

  * Only excess width is charged. Narrowing stays free and stays guarded by the
    clearance floors, so this can never push against them. With the floors at
    50-55 mm and this at the 67.6 mm nominal, there is a dead band between them
    where both terms are identically zero -- the two cannot fight by
    construction, which is the property that makes it safe to add.
  * It measures the FEET, not the joints. Forcing a prescribed hold pose is
    what defeated a previously stable stance before; this says only "do not
    splay", and leaves the policy free to choose how it stands.

  Normalized like the clearance terms so the weight is comparable to theirs.

  ``hold_gated=False`` drops the ``1 - march`` factor and charges excess width
  in BOTH modes. That is required by any task that commands base height while
  walking: lowering the base by abducting the hips is far cheaper than folding a
  knee against gravity, and NOTHING else in the reward set can see it -- the
  clearance terms are floors, and ``feet_under_base`` and the path-deviation
  terms all measure the feet MIDPOINT, which a symmetric splay leaves unchanged.
  Measured on a locked-roll comparison, the splay took the stance from 115.8 mm
  standing to 214.4 mm at full squat.

  Safe to run during the gait because it stays one-sided: a normal step moves
  the feet fore/aft, not sideways, and even 10 degrees of body yaw against a
  25 mm fore/aft stagger only lifts the body-frame gap ~3 mm above nominal,
  worth 0.002 before weighting.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_delta_w = (
    asset.data.site_pos_w[:, asset_cfg.site_ids[0]]
    - asset.data.site_pos_w[:, asset_cfg.site_ids[1]]
  )
  foot_delta_b = quat_apply_inverse(asset.data.root_link_quat_w, foot_delta_w)
  lateral_gap = torch.abs(foot_delta_b[:, 1])
  excess = torch.clamp((lateral_gap - max_separation) / max_separation, min=0.0)
  cost = torch.square(excess)
  if not hold_gated:
    return cost
  hold = 1.0 - env.command_manager.get_command(command_name)[:, 0]
  return cost * hold


##
# Hold-stance geometry and swing shortfall (turn task, inherited by lateral).
##


def hold_sole_alignment_l2(
  env: "ManagerBasedRlEnv",
  command_name: str,
  heel_cfg: SceneEntityCfg,
  toe_cfg: SceneEntityCfg,
  inner_cfg: SceneEntityCfg,
  outer_cfg: SceneEntityCfg,
  height_tolerance: float = 0.003,
  yaw_tolerance: float = 0.0872664626,
) -> torch.Tensor:
  """Keep both soles grounded and parallel in HOLD without fixing joint angles.

  Site heights use the task's flat ground at z=0. The 3 mm deadband permits
  the inherited sole offset/tilt randomization rather than fighting it.
  Toe-minus-heel vectors compare foot headings independently of robot yaw.
  """
  data = env.scene[heel_cfg.name].data
  heel = data.site_pos_w[:, heel_cfg.site_ids]
  toe = data.site_pos_w[:, toe_cfg.site_ids]
  z = torch.stack([data.site_pos_w[:, cfg.site_ids, 2]
                   for cfg in (heel_cfg, toe_cfg, inner_cfg, outer_cfg)], dim=1)
  ground = ((z.abs() - height_tolerance).clamp(min=0) / height_tolerance).square().mean((1, 2))
  axis = toe[..., :2] - heel[..., :2]
  axis = torch.nn.functional.normalize(axis, dim=-1, eps=1e-8)
  # Chord distance is wrap-safe, zero for parallel and large for opposing feet.
  yaw = (axis[:, 0] - axis[:, 1]).square().sum(-1) / yaw_tolerance**2
  hold = 1 - env.command_manager.get_command(command_name)[:, 0]
  return (ground + yaw) * hold


def hold_stance_stagger_l2(
  env: "ManagerBasedRlEnv",
  command_name: str,
  asset_cfg: SceneEntityCfg,
  tolerance: float = 0.02,
) -> torch.Tensor:
  """Penalize one foot standing ahead of the other while holding.

  Body-frame fore/aft offset between the two foot-centre sites, normalized so
  ``tolerance`` of stagger costs one before weighting. Companion of
  :func:`hold_stance_width_l2` (lateral excess) -- together they say "feet side
  by side under the body" without prescribing any joint angle, so they remain
  correct under the per-foot sole randomization where a level base genuinely
  needs unequal legs.
  """
  asset: Entity = env.scene[asset_cfg.name]
  delta_w = (
    asset.data.site_pos_w[:, asset_cfg.site_ids[0]]
    - asset.data.site_pos_w[:, asset_cfg.site_ids[1]]
  )
  delta_b = quat_apply_inverse(asset.data.root_link_quat_w, delta_w)
  hold = 1.0 - env.command_manager.get_command(command_name)[:, 0]
  return torch.square(delta_b[:, 0] / tolerance) * hold


def swing_shortfall_l2(
  env: "ManagerBasedRlEnv",
  command_name: str,
  target_height: float,
  heel_cfg: SceneEntityCfg,
  toe_cfg: SceneEntityCfg,
  inner_cfg: SceneEntityCfg,
  outer_cfg: SceneEntityCfg,
  tolerance: float = 0.002,
) -> torch.Tensor:
  """Penalize the lowest swing-sole point falling below the lift trajectory.

  Unlike the exponential :func:`gait_swing` kernel this keeps a gradient near a
  dragging foot (the kernel is ~0 and flat 3 sigma below target). It adds no
  incentive to exceed the target: the hinge is zero once the sole is on or
  above the trajectory. March-gated, so it applies to every gait direction.
  """
  command = env.command_manager.get_command(command_name)
  march, sinp = command[:, 0], command[:, 1]
  z = torch.stack([env.scene[cfg.name].data.site_pos_w[:, cfg.site_ids, 2]
                   for cfg in (heel_cfg, toe_cfg, inner_cfg, outer_cfg)], dim=1).amin(1)
  swing = torch.stack((sinp > 0, sinp < 0), dim=1)
  target = target_height * sinp.square().unsqueeze(1)
  shortfall = ((target - tolerance - z).clamp(min=0) / target_height).square()
  return (shortfall * swing).sum(1) * march
