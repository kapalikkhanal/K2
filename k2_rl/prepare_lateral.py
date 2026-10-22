"""Expand a turn checkpoint for lateral training, preserving the source policy."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import shlex

import torch
import yaml

from k2_rl.k2_constants import K2_XML, _CROUCH_JOINTS, CROUCH_BASE_HEIGHT


def prepare(checkpoint, *, allow_legacy=False):
  """Return a converted copy; legacy checkpoints require caller provenance checks."""
  infos = checkpoint.get("infos") or {}
  if "k2_stage_transfer" in infos or "k2_full_lateral_share" in infos:
    raise ValueError("Expected an unconverted turn checkpoint, not a full/lateral checkpoint.")
  design = infos.get("k2_design")
  if design is None:
    if not allow_legacy:
      raise ValueError("Missing design metadata; use the CLI with the original turn run's params/agent.yaml.")
  else:
    expected = {
      "mjcf_sha256": hashlib.sha256(K2_XML.read_bytes()).hexdigest(),
      "joint_pos": _CROUCH_JOINTS,
      "base_height_m": CROUCH_BASE_HEIGHT,
      "gait_command_dim": 5,
      "command_profile": None,
    }
    for field, value in expected.items():
      if design.get(field) != value:
        raise ValueError(f"Incompatible turn checkpoint design: {field}")

  result = deepcopy(checkpoint)
  for key, width in (("actor_state_dict", 49), ("critic_state_dict", 54)):
    state = result[key]
    weight = state["mlp.0.weight"]
    if weight.ndim != 2 or weight.shape[1] != width:
      raise ValueError(f"Expected {width} inputs in {key}; got {tuple(weight.shape)}")
    if not all(torch.isfinite(v).all() for v in state.values() if isinstance(v, torch.Tensor)):
      raise ValueError(f"Non-finite tensors in {key}")
    state["mlp.0.weight"] = torch.cat((weight, weight.new_zeros(weight.shape[0], 1)), dim=1)
    for suffix, value in (("_mean", 0.), ("_var", .015**2), ("_std", .015)):
      name = f"obs_normalizer.{suffix}"
      old = state[name]
      if old.shape != (1, width):
        raise ValueError(f"Unexpected normalizer layout: {key}.{name}")
      state[name] = torch.cat((old, old.new_full((1, 1), value)), dim=1)
  result["optimizer_state_dict"]["state"] = {}
  for group in result["optimizer_state_dict"]["param_groups"]:
    group["lr"] = 1e-4
  result["infos"] = deepcopy(infos)
  result["infos"]["k2_stage_transfer"] = {
    "source_iteration": checkpoint["iter"], "source_task": "Mjlab-Turn-K2",
    "target_task": "Mjlab-Lateral-K2", "design_verified": design is not None,
  }
  # Environment counters/curricula belong to the new training stage.
  result["infos"].pop("env_state", None)
  if design is not None:
    result["infos"]["k2_design"]["gait_command_dim"] = 6
  result["iter"] = 0
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  args = parser.parse_args()
  source = args.checkpoint.resolve()
  checkpoint = torch.load(source, map_location="cpu", weights_only=False)
  legacy = (checkpoint.get("infos") or {}).get("k2_design") is None
  if legacy:
    agent_path = source.parent / "params" / "agent.yaml"
    if not agent_path.is_file():
      parser.error("Legacy checkpoint needs its original params/agent.yaml to identify the turn task.")
    # Read strings only; mjlab dumps Python tuple tags in these settings.
    agent = yaml.load(agent_path.read_text(), Loader=yaml.BaseLoader)
    if agent.get("experiment_name") != "k2_turn_v1":
      parser.error("Expected a k2_turn_v1 source run.")
    print("Legacy turn checkpoint: task and tensor widths checked; historical MJCF/crouch cannot be verified.")
  converted = prepare(checkpoint, allow_legacy=legacy)
  converted["infos"]["k2_stage_transfer"]["source_checkpoint"] = str(source)
  destination = Path("logs/rsl_rl/k2_lateral_v1") / (
    f"lateral_start_{source.parent.name}_{source.stem}") / source.name
  destination.parent.mkdir(parents=True, exist_ok=True)
  with destination.open("xb") as output:
    torch.save(converted, output)
  print(f"Prepared {destination}; source preserved.")
  print("python -m k2_rl.train Mjlab-Lateral-K2 "
        "--env.scene.num-envs 4096 --agent.algorithm.learning-rate 0.0001 "
        "--agent.algorithm.schedule fixed --agent.resume True "
        f"--agent.load-run {shlex.quote(destination.parent.name)} "
        f"--agent.load-checkpoint {shlex.quote(source.name)} "
        "--agent.max-iterations 2000")


if __name__ == "__main__":
  main()
