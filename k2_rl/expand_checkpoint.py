"""Widen a trained RSL-RL checkpoint by appended observation channels.

Adding a command channel changes the actor's input width, so ``runner.load``
refuses the old checkpoint outright -- ``load_state_dict`` rejects a shape
mismatch whether or not it is strict. This script performs the one edit that
makes a warm start possible, and does it so the widened policy is *behaviorally
identical* to the source at iteration zero:

  * ``mlp.0.weight`` gains **zero** columns for the new inputs. A zero column
    contributes nothing, so the first forward pass reproduces the source policy
    exactly -- and still receives a real gradient, because the new input varies
    across environments. Nothing is randomly initialized.
  * The input normalizer gains statistics for the new channels. This matters
    more than it looks: the running normalizer has already seen ~2.7e8 samples,
    so its update rate is ~4e-4 per iteration and a default ``var = 1`` would
    take thousands of iterations to converge. The new channel would spend that
    whole time entering the network ~10x too small. The seed is computed
    analytically from the command configuration instead.
  * The Adam moments for the widened weight are extended with zeros, so the
    optimizer keeps every moment it learned for the existing inputs.

The new channels must be at the END of the observation vector. That is checked
against the target task's observation term order, not assumed.

    python -m k2_rl.expand_checkpoint \\
      --checkpoint logs/rsl_rl/k2_bidir_v1/<run>/model_2700.pt \\
      --task Mjlab-Turn-K2
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

import k2_rl  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


def _gait_command_dim(gait_cfg) -> int:
  if getattr(gait_cfg, "yaw_rate_range", None) is not None:
    return 5
  if gait_cfg.forward_velocity_range is not None:
    return 4
  return 3


def _yaw_channel_std(gait_cfg) -> float:
  """Standard deviation of the emitted ``march * yaw_rate`` channel.

  The channel is zero unless the environment is marching (``rel_march_envs``)
  AND was given a turning command (``rel_turning_envs``); otherwise it is
  uniform over the configured range. The mean is zero for a symmetric range, so
  the variance is just the active fraction times the uniform second moment.
  """
  lo, hi = gait_cfg.yaw_rate_range
  second_moment = (lo * lo + lo * hi + hi * hi) / 3.0
  active = gait_cfg.rel_march_envs * gait_cfg.rel_turning_envs
  return math.sqrt(max(active * second_moment, 1e-12))


def _widen_linear(weight: torch.Tensor, added: int, fill: float) -> torch.Tensor:
  pad = torch.full(
    (weight.shape[0], added), fill, dtype=weight.dtype, device=weight.device
  )
  return torch.cat([weight, pad], dim=1)


def _widen_model(
  state: dict[str, torch.Tensor],
  added: int,
  new_std: float,
  label: str,
) -> int:
  """Widen one actor/critic state dict in place; return the old input width."""
  weight = state["mlp.0.weight"]
  old_dim = int(weight.shape[1])
  state["mlp.0.weight"] = _widen_linear(weight, added, 0.0)
  if "obs_normalizer._mean" in state:
    for key, fill in (
      ("obs_normalizer._mean", 0.0),
      ("obs_normalizer._var", new_std * new_std),
      ("obs_normalizer._std", new_std),
    ):
      state[key] = _widen_linear(state[key], added, fill)
  print(
    f"  {label}: input {old_dim} -> {old_dim + added} "
    f"(zero weights, seeded std {new_std:.4g})"
  )
  return old_dim


def _widen_optimizer(opt_state: dict, old_dim: int, added: int, label: str) -> None:
  """Extend the Adam moments of the single first-layer weight of width old_dim."""
  matches = [
    idx
    for idx, entry in opt_state["state"].items()
    if entry["exp_avg"].dim() == 2 and entry["exp_avg"].shape[1] == old_dim
  ]
  if len(matches) != 1:
    raise SystemExit(
      f"expected exactly one optimizer tensor with {old_dim} columns for "
      f"{label}, found {len(matches)}: {matches}"
    )
  entry = opt_state["state"][matches[0]]
  for key in ("exp_avg", "exp_avg_sq"):
    entry[key] = _widen_linear(entry[key], added, 0.0)
  print(f"  {label}: optimizer moments param[{matches[0]}] extended with zeros")


def main() -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("--checkpoint", required=True, help="source model_*.pt")
  ap.add_argument("--task", default="Mjlab-Turn-K2", help="target task id")
  ap.add_argument("--out", help="destination .pt (default: derived, see below)")
  ap.add_argument(
    "--source-gait-dim",
    type=int,
    default=4,
    help="gait command width the source was trained with (default 4: vx only)",
  )
  ap.add_argument(
    "--new-obs-std",
    type=float,
    help="override the analytic normalizer seed for the new channels",
  )
  ap.add_argument(
    "--reset-action-std",
    type=float,
    help="optionally re-inflate the Gaussian policy std (e.g. 0.3) if the warm "
         "start explores too little; omitted, the checkpoint's std is kept",
  )
  args = ap.parse_args()

  source = Path(args.checkpoint).resolve()
  if not source.exists():
    raise SystemExit(f"missing checkpoint: {source}")

  env_cfg = load_env_cfg(args.task, play=False)
  gait_cfg = env_cfg.commands["gait"]
  target_gait_dim = _gait_command_dim(gait_cfg)
  added = target_gait_dim - args.source_gait_dim
  if added <= 0:
    raise SystemExit(
      f"{args.task} has a {target_gait_dim}-D gait command; nothing to add to a "
      f"{args.source_gait_dim}-D checkpoint"
    )

  # The new channels are appended to the gait command, so they land at the very
  # end of the observation vector only if 'gait' is the last observation term.
  for group_name in ("actor", "critic"):
    terms = list(env_cfg.observations[group_name].terms)
    if terms[-1] != "gait":
      raise SystemExit(
        f"{args.task} {group_name} observation does not end with 'gait' "
        f"({terms}); the widened columns would land on the wrong inputs"
      )

  if args.new_obs_std is not None:
    new_std = args.new_obs_std
  elif added == 1 and getattr(gait_cfg, "yaw_rate_range", None) is not None:
    new_std = _yaw_channel_std(gait_cfg)
  else:
    raise SystemExit(
      "cannot derive the normalizer seed for these channels; pass --new-obs-std"
    )

  if args.out:
    destination = Path(args.out).resolve()
  else:
    experiment = load_rl_cfg(args.task).experiment_name
    # Sorts after any 'YYYY-...' run directory, so mjlab's default checkpoint
    # search picks it up for the first resume without extra flags.
    run_dir = f"warmstart_{source.parent.name}_{source.stem}"
    destination = Path("logs") / "rsl_rl" / experiment / run_dir / source.name
    destination = destination.resolve()

  checkpoint = torch.load(source, weights_only=False, map_location="cpu")
  print(f"source: {source}")
  print(f"iteration: {checkpoint['iter']}  adding {added} observation channel(s)")

  actor_dim = _widen_model(
    checkpoint["actor_state_dict"], added, new_std, "actor"
  )
  critic_dim = _widen_model(
    checkpoint["critic_state_dict"], added, new_std, "critic"
  )
  _widen_optimizer(checkpoint["optimizer_state_dict"], actor_dim, added, "actor")
  _widen_optimizer(checkpoint["optimizer_state_dict"], critic_dim, added, "critic")

  if args.reset_action_std is not None:
    std_param = checkpoint["actor_state_dict"]["distribution.std_param"]
    checkpoint["actor_state_dict"]["distribution.std_param"] = torch.full_like(
      std_param, float(args.reset_action_std)
    )
    print(f"  action std reset to {args.reset_action_std:g} (was ~{std_param.mean():.3f})")

  destination.parent.mkdir(parents=True, exist_ok=True)
  torch.save(checkpoint, destination)
  print(f"\nwrote: {destination}")
  print(
    "\nresume with:\n"
    f"  python -m k2_rl.train {args.task} --agent.resume True \\\n"
    f"    --agent.load-run {destination.parent.name} "
    f"--agent.load-checkpoint {destination.name} \\\n"
    "    --agent.max-iterations <ADDITIONAL iterations>"
  )


if __name__ == "__main__":
  main()
