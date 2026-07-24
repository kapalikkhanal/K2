"""Headless K2 checkpoint evaluation focused on base attitude and falls."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import k2_walk_rl  # noqa: F401 - registers the task
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import euler_xyz_from_quat


TASK = "Mjlab-Velocity-Flat-K2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", type=Path)
    parser.add_argument("--vx", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=64)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = args.num_envs
    command_cfg = env_cfg.commands["twist"]
    command_cfg.heading_command = False
    command_cfg.ranges.heading = None
    command_cfg.ranges.lin_vel_x = (args.vx, args.vx)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_heading_envs = 0.0
    command_cfg.resampling_time_range = (1000.0, 1000.0)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    agent_cfg = load_rl_cfg(TASK)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

    if args.checkpoint is None:
        policy = lambda obs: torch.zeros(  # noqa: E731
            (args.num_envs, env.num_actions), device=device
        )
        label = "zero-action nominal pose"
    else:
        runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(
            str(args.checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)
        label = str(args.checkpoint)

    obs = env.get_observations()
    samples: list[torch.Tensor] = []
    velocities: list[torch.Tensor] = []
    heights: list[torch.Tensor] = []
    done_count = 0
    for _ in range(args.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        robot = raw_env.scene["robot"]
        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_link_quat_w)
        samples.append(torch.stack((roll, pitch), dim=1).cpu())
        velocities.append(robot.data.root_link_lin_vel_b[:, :2].cpu())
        heights.append(robot.data.root_link_pos_w[:, 2].cpu())
        done_count += int(dones.sum().item())

    angles = torch.cat(samples).rad2deg()
    vel = torch.cat(velocities)
    height = torch.cat(heights)
    print(f"evaluation: {label}")
    print(f"command: vx={args.vx:.3f} m/s, samples={len(angles)}")
    print(
        "roll_deg: "
        f"mean={angles[:, 0].mean():.3f} "
        f"mean_abs={angles[:, 0].abs().mean():.3f} "
        f"rms={angles[:, 0].square().mean().sqrt():.3f} "
        f"p95_abs={angles[:, 0].abs().quantile(0.95):.3f}"
    )
    print(
        "pitch_deg: "
        f"mean={angles[:, 1].mean():.3f} "
        f"mean_abs={angles[:, 1].abs().mean():.3f} "
        f"rms={angles[:, 1].square().mean().sqrt():.3f} "
        f"p95_abs={angles[:, 1].abs().quantile(0.95):.3f}"
    )
    print(
        f"fraction_abs_tilt_gt_5deg="
        f"{(angles.abs().amax(dim=1) > 5.0).float().mean():.3f}"
    )
    print(
        f"velocity_b_xy_mean=({vel[:, 0].mean():.3f}, "
        f"{vel[:, 1].mean():.3f}) m/s"
    )
    print(f"base_height_mean={height.mean():.3f} m, resets={done_count}")
    env.close()


if __name__ == "__main__":
    main()
