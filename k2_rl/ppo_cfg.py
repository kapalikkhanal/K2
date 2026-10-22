"""PPO runner configurations for the K2 locomotion tasks.

Smaller MLP than G1 (12-DOF, legs only, in-place) -- keeps inference light for
the Pi 5 while staying plenty expressive. Same PPO hyperparameters as mjlab's
G1 velocity task otherwise.
"""

from dataclasses import dataclass, field

import yaml
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


class _SymmetryCfg(dict):
  """Symmetry settings, kept dumpable after RSL-RL edits them in place.

  mjlab hands the very same dict to the runner and to dump_yaml, and RSL-RL's
  resolve_symmetry_config() stashes the live env in it under "_env" in between.
  Dumping that walks into MjSpec, which will not pickle, so hide the runtime
  keys from the YAML dumper. asdict() rebuilds dicts through type(obj), which
  preserves this class.
  """


yaml.add_representer(
  _SymmetryCfg,
  lambda dumper, cfg: dumper.represent_dict(
    {k: v for k, v in cfg.items() if not k.startswith("_")}
  ),
)


@dataclass
class K2SymmetricPpoCfg(RslRlPpoAlgorithmCfg):
  # mjlab's base config does not expose this RSL-RL extension. A dataclass
  # field is necessary: train.py uses asdict(), which drops dynamic attributes.
  symmetry_cfg: dict = field(default_factory=lambda: _SymmetryCfg({
    "use_data_augmentation": False,
    "use_mirror_loss": True,
    "data_augmentation_func": "k2_rl.mdp.symmetry:mirror_k2",
    "mirror_loss_coeff": 0.5,
  }))


def _with_symmetry(cfg):
  # Regularize the actor on mirrored observations while keeping PPO's actual
  # transitions. This does not assume perfectly mirrored simulator dynamics.
  cfg.algorithm = K2SymmetricPpoCfg(**vars(cfg.algorithm))
  return cfg


def _runner(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=2000,
  )








def k2_turn_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _with_symmetry(_runner("k2_turn_v1"))


def k2_full_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = _with_symmetry(_runner("k2_full_turn_lateral_v1"))
  cfg.algorithm.symmetry_cfg["mirror_loss_coeff"] = 1.0
  cfg.algorithm.learning_rate = 3.0e-4
  cfg.algorithm.schedule = "fixed"
  cfg.max_iterations = 1000
  return cfg


def k2_lateral_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _with_symmetry(_runner("k2_lateral_v1"))
