"""Export a trained K2 checkpoint to policy.onnx (for sim replay + the Pi).

  python -m k2_rl.export_onnx --checkpoint logs/.../model_800.pt
  # -> writes policy.onnx next to the checkpoint (with obs/joint metadata)
"""

from __future__ import annotations

import argparse
import os
import torch
from dataclasses import asdict
from pathlib import Path

import k2_rl  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from k2_rl.onnx_metadata import attach_k2_metadata


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--task", default="Mjlab-Full-K2")
  p.add_argument("--checkpoint", required=True, help="path to model_*.pt")
  p.add_argument("--filename", default="policy.onnx")
  p.add_argument("--device", default="cpu")
  args = p.parse_args()

  configure_torch_backends()
  ckpt = Path(args.checkpoint).resolve()
  assert ckpt.exists(), f"missing checkpoint: {ckpt}"
  out_dir = ckpt.parent

  # Build a 1-env instance just to construct the policy graph.
  env_cfg = load_env_cfg(args.task, play=True)
  train_env_cfg = load_env_cfg(args.task, play=False)
  env_cfg.scene.num_envs = 1
  if args.task == "Mjlab-Full-K2":
    info = torch.load(ckpt, map_location="cpu", weights_only=False).get("infos") or {}
    if "k2_full_lateral_share" not in info:
      raise ValueError("Expected a checkpoint from the new turn/lateral full task.")
    train_env_cfg.commands["gait"].rel_lateral_envs = info["k2_full_lateral_share"]
  agent_cfg = load_rl_cfg(args.task)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task)
  runner = runner_cls(env, asdict(agent_cfg), str(out_dir), args.device)
  runner.load(str(ckpt), map_location=args.device)

  runner.export_policy_to_onnx(str(out_dir), args.filename)
  onnx_path = os.path.join(str(out_dir), args.filename)
  # Shared with `K2OnPolicyRunner.save`, so a pinned export and the rolling
  # last_policy.onnx always carry the identical deployment contract.
  attach_k2_metadata(
    onnx_path, env.unwrapped, train_env_cfg.commands["gait"], ckpt.name
  )

  env.close()
  print(f"\nExported: {onnx_path}")


if __name__ == "__main__":
  main()
