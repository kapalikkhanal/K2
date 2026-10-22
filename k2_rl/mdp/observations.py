"""K2-specific observations that are available on the real robot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def heading_sin_cos(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Relative heading encoded continuously as ``[sin(yaw), cos(yaw)]``.

  Forward-task episodes reset close to zero heading. On hardware the same
  relative yaw is obtained by integrating the bias-corrected root-frame gyro;
  no magnetometer, camera, or linear-velocity estimate is required.
  """
  asset: Entity = env.scene[asset_cfg.name]
  heading = asset.data.heading_w
  return torch.stack((torch.sin(heading), torch.cos(heading)), dim=-1)


def heading_error_sin_cos(
  env: "ManagerBasedRlEnv",
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Heading *relative to the commanded heading*, as ``[sin, cos]``.

  :func:`heading_sin_cos` measures against world +X, which is the right frame
  only while the robot is asked to walk straight. Once a yaw rate is commanded
  that observation leaves the range the policy ever saw in training. Subtracting
  the gait command's integrated reference heading keeps this signal near zero at
  every commanded yaw rate, so it stays the same "how far off course am I"
  feature it has always been.

  With no yaw channel the reference never rotates and this is identical to
  :func:`heading_sin_cos`. On hardware the reference is the running integral of
  the commanded yaw rate -- a number the controller already has -- subtracted
  from the same integrated-gyro relative yaw. No new sensing.
  """
  asset: Entity = env.scene[asset_cfg.name]
  term = env.command_manager.get_term(command_name)
  if not hasattr(term, "heading_ref"):
    raise RuntimeError(
      f"command '{command_name}' does not integrate a reference heading; "
      "this observation requires k2_rl.mdp.GaitCommand with a forward channel"
    )
  error = asset.data.heading_w - term.heading_ref
  return torch.stack((torch.sin(error), torch.cos(error)), dim=-1)
