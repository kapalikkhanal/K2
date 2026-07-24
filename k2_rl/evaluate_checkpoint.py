"""Deterministically score K2 hold and march checkpoints without a viewer.

Example:
  python -m k2_rl.evaluate_checkpoint --checkpoint logs/.../model_800.pt
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

import k2_rl  # noqa: F401  (register tasks)
from k2_rl.k2_constants import (
  FOOT_HEEL_SITES,
  FOOT_SITES,
  FOOT_TOE_SITES,
  K2_ACTION_SCALE,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.torch import configure_torch_backends


def _percentile(x: torch.Tensor, q: float) -> float:
  return float(torch.quantile(x.float(), q).item())


def evaluate(
  checkpoint: Path,
  mode: str,
  num_envs: int,
  seconds: float,
  warmup: float,
  device: str,
) -> dict[str, float | int | str]:
  task = "Mjlab-InPlace-March-K2" if mode == "march" else "Mjlab-InPlace-K2"
  env_cfg = load_env_cfg(task, play=True)
  env_cfg.scene.num_envs = num_envs
  env_cfg.episode_length_s = seconds
  agent_cfg = load_rl_cfg(task)

  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task)
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)

  robot = env.unwrapped.scene["robot"]
  _, foot_names = robot.find_sites(list(FOOT_SITES), preserve_order=True)
  foot_cfg = SceneEntityCfg("robot", site_names=foot_names)
  foot_cfg.resolve(env.unwrapped.scene)
  _, heel_names = robot.find_sites(list(FOOT_HEEL_SITES), preserve_order=True)
  heel_cfg = SceneEntityCfg("robot", site_names=heel_names)
  heel_cfg.resolve(env.unwrapped.scene)
  _, toe_names = robot.find_sites(list(FOOT_TOE_SITES), preserve_order=True)
  toe_cfg = SceneEntityCfg("robot", site_names=toe_names)
  toe_cfg.resolve(env.unwrapped.scene)
  contact_sensor = env.unwrapped.scene["feet_ground_contact"]
  self_contact_sensor = env.unwrapped.scene["feet_self_contact"]

  obs = env.get_observations()
  env.unwrapped.command_manager.compute(dt=env.unwrapped.step_dt)
  active = torch.ones(num_envs, dtype=torch.bool, device=device)
  survived = torch.ones_like(active)
  samples = torch.zeros(num_envs, device=device)
  tilt_sum = torch.zeros_like(samples)
  tilt_max = torch.zeros_like(samples)
  ang_vel_sq_sum = torch.zeros_like(samples)
  action_rate_sq_sum = torch.zeros_like(samples)
  action_rate_max = torch.zeros_like(samples)
  command_rate_sq_sum = torch.zeros_like(samples)
  slew_saturated_sum = torch.zeros_like(samples)
  both_contact_sum = torch.zeros_like(samples)
  gait_height_err_sum = torch.zeros_like(samples)
  foot_height_max = torch.zeros(num_envs, 2, device=device)
  foot_lateral_gap_min = torch.full(
    (num_envs,), float("inf"), device=device
  )
  heel_lateral_gap_min = torch.full(
    (num_envs,), float("inf"), device=device
  )
  toe_lateral_gap_min = torch.full(
    (num_envs,), float("inf"), device=device
  )
  foot_self_contact_sum = torch.zeros_like(samples)
  start_xy = robot.data.root_link_pos_w[:, :2].clone()
  drift_max = torch.zeros_like(samples)
  final_drift_xy = torch.zeros(num_envs, 2, device=device)
  prev_action = torch.zeros(num_envs, 12, device=device)
  commanded_action = torch.zeros_like(prev_action)

  max_steps = int(math.ceil(seconds / env.unwrapped.step_dt)) + 1
  warmup_steps = int(round(warmup / env.unwrapped.step_dt))
  for step_idx in range(max_steps):
    if not active.any():
      break
    with torch.no_grad():
      action = policy(obs)

    gravity = robot.data.projected_gravity_b
    tilt = torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0))
    ang_vel_sq = torch.sum(torch.square(robot.data.root_link_ang_vel_b[:, :2]), dim=1)
    action_rate = torch.abs(action - prev_action) * K2_ACTION_SCALE / env.unwrapped.step_dt
    requested_delta = action - commanded_action
    max_action_delta = 1.0 * env.unwrapped.step_dt / K2_ACTION_SCALE
    saturated = torch.abs(requested_delta) > max_action_delta
    applied_delta = torch.clamp(
      requested_delta, min=-max_action_delta, max=max_action_delta
    )
    commanded_action += applied_delta
    command_rate = torch.abs(applied_delta) * K2_ACTION_SCALE / env.unwrapped.step_dt
    foot_z = robot.data.site_pos_w[:, foot_cfg.site_ids, 2]
    foot_delta_w = (
      robot.data.site_pos_w[:, foot_cfg.site_ids[0]]
      - robot.data.site_pos_w[:, foot_cfg.site_ids[1]]
    )
    foot_delta_b = quat_apply_inverse(robot.data.root_link_quat_w, foot_delta_w)
    foot_lateral_gap = torch.abs(foot_delta_b[:, 1])
    heel_delta_w = (
      robot.data.site_pos_w[:, heel_cfg.site_ids[0]]
      - robot.data.site_pos_w[:, heel_cfg.site_ids[1]]
    )
    heel_lateral_gap = torch.abs(
      quat_apply_inverse(robot.data.root_link_quat_w, heel_delta_w)[:, 1]
    )
    toe_delta_w = (
      robot.data.site_pos_w[:, toe_cfg.site_ids[0]]
      - robot.data.site_pos_w[:, toe_cfg.site_ids[1]]
    )
    toe_lateral_gap = torch.abs(
      quat_apply_inverse(robot.data.root_link_quat_w, toe_delta_w)[:, 1]
    )
    contact = contact_sensor.data.found > 0
    assert self_contact_sensor.data.found is not None
    foot_self_contact = self_contact_sensor.data.found.reshape(num_envs, -1).any(dim=1)
    drift = torch.linalg.vector_norm(robot.data.root_link_pos_w[:, :2] - start_xy, dim=1)
    drift_xy = robot.data.root_link_pos_w[:, :2] - start_xy

    gait = env.unwrapped.command_manager.get_command("gait")
    sinp = gait[:, 1]
    lift_clock = torch.square(sinp)
    target_z = torch.stack(
      [
        0.025 * lift_clock * (sinp > 0).float(),
        0.025 * lift_clock * (sinp < 0).float(),
      ],
      dim=1,
    )

    if step_idx == warmup_steps:
      start_xy = robot.data.root_link_pos_w[:, :2].clone()
    metric_active = active & (step_idx >= warmup_steps)
    mask = metric_active.float()
    samples += mask
    tilt_sum += tilt * mask
    tilt_max = torch.maximum(tilt_max, tilt * mask)
    ang_vel_sq_sum += ang_vel_sq * mask
    action_rate_sq_sum += torch.mean(torch.square(action_rate), dim=1) * mask
    action_rate_max = torch.maximum(action_rate_max, action_rate.max(dim=1).values * mask)
    command_rate_sq_sum += torch.mean(torch.square(command_rate), dim=1) * mask
    slew_saturated_sum += torch.mean(saturated.float(), dim=1) * mask
    both_contact_sum += contact.all(dim=1).float() * mask
    gait_height_err_sum += torch.mean(torch.abs(foot_z - target_z), dim=1) * mask
    foot_height_max = torch.maximum(foot_height_max, foot_z * mask[:, None])
    foot_lateral_gap_min = torch.where(
      metric_active,
      torch.minimum(foot_lateral_gap_min, foot_lateral_gap),
      foot_lateral_gap_min,
    )
    heel_lateral_gap_min = torch.where(
      metric_active,
      torch.minimum(heel_lateral_gap_min, heel_lateral_gap),
      heel_lateral_gap_min,
    )
    toe_lateral_gap_min = torch.where(
      metric_active,
      torch.minimum(toe_lateral_gap_min, toe_lateral_gap),
      toe_lateral_gap_min,
    )
    foot_self_contact_sum += foot_self_contact.float() * mask
    drift_max = torch.maximum(drift_max, drift * mask)
    final_drift_xy = torch.where(mask[:, None].bool(), drift_xy, final_drift_xy)

    prev_action = action
    obs, _, dones, _ = env.step(action)
    newly_done = active & dones.bool()
    survived[newly_done] = env.unwrapped.termination_manager.time_outs[newly_done]
    active &= ~newly_done

  denom = torch.clamp(samples, min=1.0)
  valid = samples > 1
  result: dict[str, float | int | str] = {
    "checkpoint": str(checkpoint),
    "mode": mode,
    "num_envs": num_envs,
    "seconds": seconds,
    "warmup_seconds": warmup,
    "survival_rate": float(survived.float().mean().item()),
    "tilt_mean_deg": float(torch.rad2deg((tilt_sum / denom)[valid]).mean().item()),
    "tilt_p95_max_deg": _percentile(torch.rad2deg(tilt_max[valid]), 0.95),
    "roll_pitch_rate_rms": float(
      torch.sqrt((ang_vel_sq_sum / denom)[valid]).mean().item()
    ),
    "drift_p95_m": _percentile(drift_max[valid], 0.95),
    "drift_p50_m": _percentile(drift_max[valid], 0.50),
    "final_drift_mean_x_m": float(final_drift_xy[valid, 0].mean().item()),
    "final_drift_mean_y_m": float(final_drift_xy[valid, 1].mean().item()),
    "action_rate_rms_rad_s": float(
      torch.sqrt((action_rate_sq_sum / denom)[valid]).mean().item()
    ),
    "action_rate_p95_max_rad_s": _percentile(action_rate_max[valid], 0.95),
    "command_rate_rms_rad_s": float(
      torch.sqrt((command_rate_sq_sum / denom)[valid]).mean().item()
    ),
    "slew_saturation_fraction": float(
      (slew_saturated_sum / denom)[valid].mean().item()
    ),
    "both_feet_contact_fraction": float(
      (both_contact_sum / denom)[valid].mean().item()
    ),
    "gait_height_mae_m": float((gait_height_err_sum / denom)[valid].mean().item()),
    "foot_height_p50_m": _percentile(foot_height_max[valid], 0.50),
    "foot_height_p95_m": _percentile(foot_height_max[valid], 0.95),
    "foot_lateral_gap_min_p05_m": _percentile(
      foot_lateral_gap_min[valid], 0.05
    ),
    "foot_lateral_gap_min_p50_m": _percentile(
      foot_lateral_gap_min[valid], 0.50
    ),
    "heel_lateral_gap_min_p05_m": _percentile(
      heel_lateral_gap_min[valid], 0.05
    ),
    "heel_lateral_gap_min_p50_m": _percentile(
      heel_lateral_gap_min[valid], 0.50
    ),
    "toe_lateral_gap_min_p05_m": _percentile(
      toe_lateral_gap_min[valid], 0.05
    ),
    "toe_lateral_gap_min_p50_m": _percentile(
      toe_lateral_gap_min[valid], 0.50
    ),
    "foot_self_contact_fraction": float(
      (foot_self_contact_sum / denom)[valid].mean().item()
    ),
  }
  env.close()
  return result


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--mode", choices=("hold", "march", "both"), default="both")
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--seconds", type=float, default=10.0)
  parser.add_argument(
    "--warmup",
    type=float,
    default=0.5,
    help="exclude initial settling from motion metrics (survival still covers it)",
  )
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()

  configure_torch_backends()
  checkpoint = Path(args.checkpoint).resolve()
  if not checkpoint.exists():
    raise FileNotFoundError(checkpoint)
  modes = ("hold", "march") if args.mode == "both" else (args.mode,)
  results = [
    evaluate(
      checkpoint,
      mode,
      args.num_envs,
      args.seconds,
      args.warmup,
      args.device,
    )
    for mode in modes
  ]
  payload = {"results": results}
  text = json.dumps(payload, indent=2)
  print(text)
  if args.output:
    args.output.write_text(text + "\n")


if __name__ == "__main__":
  main()
