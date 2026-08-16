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
