"""Register the three supported K2 locomotion tasks."""
from mjlab.tasks.registry import register_mjlab_task
from k2_rl.runner import K2OnPolicyRunner
from k2_rl.turn_env_cfg import make_k2_turn_env_cfg
from k2_rl.lateral_env_cfg import make_k2_lateral_env_cfg
from k2_rl.full_env_cfg import make_k2_full_env_cfg
from k2_rl.ppo_cfg import (
  k2_turn_ppo_runner_cfg, k2_lateral_ppo_runner_cfg, k2_full_ppo_runner_cfg,
)

for task, factory, runner in (
  ("Mjlab-Turn-K2", make_k2_turn_env_cfg, k2_turn_ppo_runner_cfg),
  ("Mjlab-Lateral-K2", make_k2_lateral_env_cfg, k2_lateral_ppo_runner_cfg),
  ("Mjlab-Full-K2", make_k2_full_env_cfg, k2_full_ppo_runner_cfg),
):
  register_mjlab_task(task_id=task, env_cfg=factory(), play_env_cfg=factory(True),
                      rl_cfg=runner(), runner_cls=K2OnPolicyRunner)
