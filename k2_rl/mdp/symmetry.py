"""Sagittal reflection for K2's RSL-RL policy symmetry loss.

Mirror commands and clock as well as the robot: a left turn is paired with a
right turn, not with an equal-stride constraint on the same turning sample.
Observations are raw (RSL-RL normalizes them inside the actor). CAD joint axes
and the IMU orientation are checked against the compiled model at startup.
"""
from __future__ import annotations

import numpy as np
import torch

from k2_rl.k2_constants import K2_XML


class K2Mirror:
  def __init__(self, env):
    import mujoco

    robot = env.scene["robot"]
    names = list(robot.joint_names)
    if len(names) != 12:
      raise ValueError("K2 symmetry expects the 12 leg joints")
    peers = [name.replace("_right_", "_left_") if "_right_" in name
             else name.replace("_left_", "_right_") for name in names]
    self.perm = torch.tensor([names.index(name) for name in peers], device=env.device)
    # Joint coordinates on this CAD model ALREADY encode the reflection sign:
    # even hip roll/yaw are swapped without negation. Verify, don't guess.
    model = mujoco.MjModel.from_xml_path(str(K2_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    polar = np.diag([1., -1., 1.])
    axial = -polar
    for name, peer in zip(names, peers, strict=True):
      np.testing.assert_allclose(
        axial @ data.xaxis[model.joint(name).id],
        data.xaxis[model.joint(peer).id], atol=2e-5,
        err_msg=f"Joint reflection sign changed: {name}",
      )
    imu_rotation = data.site_xmat[model.site("imu").id].reshape(3, 3)
    self.gyro = torch.tensor(imu_rotation.T @ axial @ imu_rotation,
                             device=env.device, dtype=torch.float32)
    self.velocity = torch.tensor(imu_rotation.T @ polar @ imu_rotation,
                                 device=env.device, dtype=torch.float32)
    default = robot.data.default_joint_pos
    torch.testing.assert_close(default, default[:1].expand_as(default))
    self.pos_bias = default[0, self.perm] - default[0]
    action = env.action_manager.get_term("joint_pos")
    if list(action._target_names) != names or not isinstance(action.scale, float):
      raise ValueError("K2 symmetry requires full joint order and a scalar action scale")
    if not action.cfg.use_default_offset or action.scale <= 0:
      raise ValueError("K2 symmetry requires positive scale and default joint offsets")
    self.action_bias = self.pos_bias / action.scale
    self.groups = env.observation_manager.active_terms
    self.dims = env.observation_manager.group_obs_term_dim
    gait = env.command_manager.get_term("gait").cfg
    self.lateral = gait.lateral_velocity_range is not None
    self.standing_rotation = hasattr(gait, "standing_rotate_period_s")
    self.staged = hasattr(gait, "pipeline_stage")
    if gait.height_offset_range is not None:
      raise ValueError("K2 mirror is enabled only for turn/lateral training")

  def actions(self, actions):
    return actions[..., self.perm] + self.action_bias

  def observations(self, obs):
    reflected = obs.clone()
    for group in obs.keys():
      start = 0
      for name, shape in zip(self.groups[group], self.dims[group], strict=True):
        width = int(np.prod(shape))
        source = obs[group][..., start:start + width]
        target = reflected[group][..., start:start + width]
        if name == "base_ang_vel":
          target.copy_(source @ self.gyro.T)
        elif name == "projected_gravity":
          target[..., 1] *= -1
        elif name == "joint_pos":
          target.copy_(source[..., self.perm] + self.pos_bias)
        elif name == "joint_vel":
          target.copy_(source[..., self.perm])
        elif name == "actions":
          target.copy_(self.actions(source))
        elif name == "base_lin_vel":
          target.copy_(source @ self.velocity.T)
        elif name == "foot_contact":
          target.copy_(source.flip(-1))
        elif name == "heading":
          target[..., 0] *= -1
        elif name == "gait":
          if width != (7 if self.staged else 6 if self.lateral else 5):
            raise ValueError(f"Unexpected gait observation width: {width}")
          # A standing pulse mirrors DIRECTION, not its progress through the
          # stroke. Only the walking clock needs a half-cycle leg swap.
          if self.standing_rotation:
            walking = source[..., 0] > 1e-6
            target[..., 1:3] *= torch.where(walking, -1., 1.)[..., None]
          else:
            target[..., 1:3] *= -1
          target[..., 4] *= -1  # reverse yaw command
          if self.lateral:
            target[..., 5] *= -1
        else:
          raise ValueError(f"No K2 reflection defined for {group}/{name}")
        start += width
      if start != obs[group].shape[-1]:
        raise ValueError("K2 symmetry does not support stacked observation history")
    return reflected


def mirror_k2(*, env, obs=None, actions=None):
  """RSL-RL contract: concatenate original samples FIRST, reflected samples next."""
  unwrapped = env.unwrapped
  if not hasattr(unwrapped, "_k2_mirror"):
    unwrapped._k2_mirror = K2Mirror(unwrapped)
  mirror = unwrapped._k2_mirror
  return (
    None if obs is None else torch.cat((obs, mirror.observations(obs)), dim=0),
    None if actions is None else torch.cat((actions, mirror.actions(actions)), dim=0),
  )
