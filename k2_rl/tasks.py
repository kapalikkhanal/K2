"""Register K2 in-place, forward, and full-locomotion tasks with mjlab.

Two task ids share ONE trained policy (same ``experiment_name`` -> same
checkpoint). Train on ``Mjlab-InPlace-K2``; play either id to view that policy:

  * ``Mjlab-InPlace-K2``        -- play defaults to HOLD (stand still).
  * ``Mjlab-InPlace-March-K2``  -- play defaults to MARCH (step in place).

In either, the Viser viewer's "Gait" dropdown switches hold/march live.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from k2_rl.inplace_env_cfg import make_k2_inplace_env_cfg
from k2_rl.forward_env_cfg import make_k2_forward_env_cfg
from k2_rl.full_env_cfg import make_k2_full_env_cfg
from k2_rl.mdp import GaitCommandCfg
from k2_rl.ppo_cfg import (
  k2_forward_ppo_runner_cfg,
  k2_full_ppo_runner_cfg,
  k2_inplace_ppo_runner_cfg,
)


def _play_cfg(force_march: bool):
  cfg = make_k2_inplace_env_cfg(play=True)
  gait = cfg.commands["gait"]
  assert isinstance(gait, GaitCommandCfg)
  gait.rel_march_envs = 1.0 if force_march else 0.0
  return cfg


register_mjlab_task(
  task_id="Mjlab-InPlace-K2",
  env_cfg=make_k2_inplace_env_cfg(),
  play_env_cfg=_play_cfg(force_march=False),
  rl_cfg=k2_inplace_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,  # exports ONNX on save
)

register_mjlab_task(
  task_id="Mjlab-InPlace-March-K2",
  env_cfg=make_k2_inplace_env_cfg(),
  play_env_cfg=_play_cfg(force_march=True),
  rl_cfg=k2_inplace_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Forward-K2",
  env_cfg=make_k2_forward_env_cfg(),
  play_env_cfg=make_k2_forward_env_cfg(play=True),
  rl_cfg=k2_forward_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Full-K2",
  env_cfg=make_k2_full_env_cfg(),
  play_env_cfg=make_k2_full_env_cfg(play=True),
  rl_cfg=k2_full_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
