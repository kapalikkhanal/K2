"""Copy a full warmup checkpoint and initialize its dormant lateral input."""
import argparse
from pathlib import Path
import torch


def prepare(checkpoint):
  if (checkpoint.get("infos") or {}).get("k2_full_lateral_share") != 0:
    raise ValueError("Expected a new full-task warmup checkpoint with lateral share zero.")
  for key, width in (("actor_state_dict", 50), ("critic_state_dict", 55)):
    state = checkpoint[key]
    if state["mlp.0.weight"].shape[1] != width:
      raise ValueError(f"Wrong input layout in {key}")
    if not all(torch.isfinite(v).all() for v in state.values() if isinstance(v, torch.Tensor)):
      raise ValueError(f"Non-finite weights in {key}")
    # The new input initially has no effect; preserve behavior on all old commands.
    state["mlp.0.weight"][:, -1] = 0
    for suffix, value in (("_mean", 0.), ("_var", .015**2), ("_std", .015)):
      state[f"obs_normalizer.{suffix}"][..., -1] = value
  checkpoint["optimizer_state_dict"]["state"] = {}
  checkpoint["infos"]["k2_full_lateral_share"] = .35
  return checkpoint


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  args = parser.parse_args()
  source = args.checkpoint.resolve()
  checkpoint = prepare(torch.load(source, map_location="cpu", weights_only=False))
  destination = Path("logs/rsl_rl/k2_full_turn_lateral_v1") / (
    f"lateral_start_{source.parent.name}_{source.stem}") / source.name
  destination.parent.mkdir(parents=True, exist_ok=True)
  with destination.open("xb") as output:
    torch.save(checkpoint, output)
  print(f"Prepared {destination}; source preserved.")
  print("python -m k2_rl.train Mjlab-Full-K2 "
        "--env.scene.num-envs 1024 --env.commands.gait.rel-lateral-envs 0.35 "
        "--agent.algorithm.learning-rate 0.0001 --agent.resume True "
        f"--agent.load-run {destination.parent.name} --agent.load-checkpoint {source.name} "
        "--agent.max-iterations 1500")


if __name__ == "__main__":
  main()
