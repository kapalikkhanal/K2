"""Control-rate x sensing-to-actuation latency sweep of the deployed K2 policy.

The paper's question is how little compute a learned biped loop needs. Compute
enters the loop through two knobs: how OFTEN the policy runs (control rate) and
how LATE its answer reaches the servos (latency = read + inference + write).
This drives the deployed actor (k2_lateral_v1 model_850, the ESP32 weights)
through fixed commands on N robots that each carry their own startup domain
randomization (mass, CoM, PD gains, friction, encoder bias -- the play config),
with the rate and a FIXED latency imposed exactly:

  rate     env decimation: 2 ms physics x {10, 15, 20, ...} substeps. The gait
           phase, command profile and target slew all advance by step_dt, so
           only the update rate changes. The policy is NOT retrained.
  latency  every new joint target reaches the actuators `delay_ms` after the
           observation it was computed from (whole control steps + 2 ms
           substeps), replacing training's random 0-10 ms with a fixed value.

Per robot it records survival, realized command speed (displacement over the
window, so body sway cannot inflate it), mean tilt, joint tracking RMS
(physical target - joint angle) and action-rate RMS; the summary reports the
mean and a 95% CI across robots that survived.

    python -m k2_rl.eval_compute_sweep --config 50:0,12,24 --config 25:0 \
        --out Paper/experiments/compute_sweep/results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

import k2_rl  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

TASK = "Mjlab-Lateral-K2"
DEPLOYED = "logs/rsl_rl/k2_lateral_v1/2026-09-15_08-31-17/model_850.pt"
DEPLOYED_ONNX = "logs/rsl_rl/k2_lateral_v1/2026-09-15_08-31-17/last_policy.onnx"
PHYSICS_DT = 0.002
MAX_DELAY_MS = 80

# name: (march, vx, wz, vy). Commands at the deployed limits.
MODES = {
  "hold": (0.0, 0.00, 0.00, 0.00),
  "fwd":  (1.0, 0.05, 0.00, 0.00),
  "back": (1.0, -0.05, 0.00, 0.00),
  "turn": (1.0, 0.00, 0.25, 0.00),
  "lat":  (1.0, 0.00, 0.00, 0.03),
}


class RateEnv:
  """One environment at one control rate; latency is changed per run."""

  def __init__(self, hz: float, num_envs: int, device: str, seed: int):
    decimation = round(1.0 / (hz * PHYSICS_DT))
    if abs(decimation * PHYSICS_DT * hz - 1.0) > 1e-3:
      raise ValueError(f"{hz} Hz is not a whole number of 2 ms substeps")
    cfg = load_env_cfg(TASK, play=True)
    old = cfg.decimation
    cfg.decimation = decimation
    for sensor in cfg.scene.sensors:
      if getattr(sensor, "history_length", None) == old:
        sensor.history_length = decimation
    cfg.scene.num_envs = num_envs
    cfg.seed = seed  # same seed -> the same N randomized robots at every rate
    cfg.commands["gait"].resampling_time_range = (1e9, 1e9)
    step_ms = round(decimation * PHYSICS_DT * 1000)
    act = cfg.actions["joint_pos"]
    act.max_delay_substeps = decimation - 1
    act.min_delay_steps = 0
    act.max_delay_steps = MAX_DELAY_MS // step_ms + 1
    self.env = ManagerBasedRlEnv(cfg=cfg, device=device)
    agent_cfg = load_rl_cfg(TASK)
    self.wrapped = RslRlVecEnvWrapper(self.env, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(TASK)(self.wrapped, asdict(agent_cfg), None, device)
    runner.load(DEPLOYED, map_location=device)
    self.policy = runner.get_inference_policy(device)
    self.hz, self.decimation, self.step_ms = hz, decimation, step_ms
    self.dt = self.env.step_dt
    self.device, self.num_envs = device, num_envs
    self.robot = self.env.scene["robot"]
    self.term = self.env.action_manager.get_term("joint_pos")
    self._substep_delay = 0
    original = self.term.process_actions

    def process_actions(actions):
      original(actions)
      self.term._delay.fill_(self._substep_delay)

    self.term.process_actions = process_actions

  def set_delay(self, delay_ms: int):
    if delay_ms % 2:
      raise ValueError("latency must be a multiple of the 2 ms physics step")
    steps, rem = divmod(delay_ms, self.step_ms)
    if steps > self.term._target_history.shape[0] - 1:
      raise ValueError(f"{delay_ms} ms exceeds the delay buffer")
    self.term._min_delay_steps = self.term._max_delay_steps = steps
    self._substep_delay = rem // 2

  def run(self, mode: str, settle_s: float, measure_s: float):
    march, vx, wz, vy = MODES[mode]
    env, robot, term = self.env, self.robot, self.term
    gait = env.command_manager.get_term("gait")
    obs, _ = self.wrapped.reset()
    n = self.num_envs
    settle, measure = round(settle_s / self.dt), round(measure_s / self.dt)
    alive = torch.ones(n, dtype=torch.bool, device=self.device)
    zeros = lambda: torch.zeros(n, device=self.device)
    tilt_sum, track_sq, arate_sq, steps = zeros(), zeros(), zeros(), zeros()
    heading_delta = zeros()
    prev_heading = origin = origin_heading = prev_action = None
    ids = term._target_ids
    for k in range(settle + measure):
      gait.march_target[:] = march
      gait.forward_velocity_target[:] = vx
      gait.yaw_rate_target[:] = wz
      gait.lateral_velocity_target[:] = vy
      if hasattr(gait, "staged_height_target"):
        gait.staged_height_target.zero_()
      with torch.no_grad():
        action = self.policy(obs)
      obs, _, dones, _ = self.wrapped.step(action)
      alive &= dones == 0
      d = robot.data
      h = d.heading_w
      if prev_heading is not None and k >= settle:
        heading_delta += (h - prev_heading + math.pi) % (2 * math.pi) - math.pi
      prev_heading = h.clone()
      if k == settle - 1:
        origin = d.root_link_pos_w[:, :2].clone()
        origin_heading = h.clone()
      if k >= settle:
        m = alive.float()
        tilt = torch.acos(torch.clamp(-d.projected_gravity_b[:, 2], -1, 1))
        target = term._commanded_target - d.encoder_bias[:, ids]
        err = target - d.joint_pos[:, ids]
        tilt_sum += tilt * m
        track_sq += err.square().mean(dim=1) * m
        if prev_action is not None:
          arate_sq += ((action - prev_action) / self.dt).square().mean(dim=1) * m
        steps += m
      prev_action = action.clone()
    window = measure * self.dt
    disp = robot.data.root_link_pos_w[:, :2] - origin
    c, s = torch.cos(origin_heading), torch.sin(origin_heading)
    fwd = (disp[:, 0] * c + disp[:, 1] * s) / window
    lat = (-disp[:, 0] * s + disp[:, 1] * c) / window
    yaw = heading_delta / window
    st = steps.clamp(min=1)
    per = {
      "alive": alive.float(),
      "vx": fwd, "vy": lat, "wz": yaw,
      "tilt_deg": torch.rad2deg(tilt_sum / st),
      "track_rms_deg": torch.rad2deg(torch.sqrt(track_sq / st)),
      "act_rate_rms": torch.sqrt(arate_sq / st.clamp(min=2)),
    }
    return {k: v.cpu().numpy() for k, v in per.items()}


def ci95(x):
  x = np.asarray(x, dtype=float)
  if x.size < 2:
    return float(x.mean()) if x.size else float("nan"), float("nan")
  return float(x.mean()), float(1.96 * x.std(ddof=1) / math.sqrt(x.size))


def summarize(hz, delay_ms, mode, per):
  march, vx, wz, vy = MODES[mode]
  ok = per["alive"] > 0.5
  row = {"hz": hz, "delay_ms": delay_ms, "mode": mode, "robots": len(ok),
         "survival": float(ok.mean())}
  command, realized = {"fwd": (vx, per["vx"]), "back": (vx, per["vx"]),
                       "turn": (wz, per["wz"]), "lat": (vy, per["vy"])}.get(
                         mode, (None, None))
  if command:
    row["ratio"], row["ratio_ci"] = ci95(realized[ok] / command)
  else:
    row["ratio"] = row["ratio_ci"] = float("nan")
  for key in ("tilt_deg", "track_rms_deg", "act_rate_rms"):
    row[key], row[key + "_ci"] = ci95(per[key][ok])
  return row


def check_onnx(rate_env: RateEnv):
  """The torch checkpoint must be the ESP32's weights, not a near relative."""
  import onnxruntime as ort
  sess = ort.InferenceSession(DEPLOYED_ONNX, providers=["CPUExecutionProvider"])
  g = torch.Generator().manual_seed(0)
  worst = 0.0
  for _ in range(16):
    x = torch.randn(1, 50, generator=g)
    td = TensorDict({"actor": x.to(rate_env.device)}, batch_size=[1])
    with torch.no_grad():
      y = rate_env.policy(td).cpu().numpy()
    z = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    worst = max(worst, float(np.abs(y - z).max()))
  print(f"checkpoint vs deployed ONNX: max |diff| = {worst:.2e}")
  if worst > 1e-4:
    raise SystemExit("model_850.pt does not reproduce the deployed ONNX")


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--config", action="append", required=True,
                  help="HZ:DELAY_MS[,DELAY_MS...], repeatable")
  ap.add_argument("--modes", default=",".join(MODES))
  ap.add_argument("--num-envs", type=int, default=64)
  ap.add_argument("--settle", type=float, default=3.0)
  ap.add_argument("--measure", type=float, default=10.0)
  ap.add_argument("--seed", type=int, default=20261004)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--out", required=True)
  args = ap.parse_args(argv)
  configure_torch_backends()
  modes = args.modes.split(",")
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  per_robot = out.with_name(out.stem + "_per_robot.npz")
  rows, raw = [], {}
  checked = False
  for spec in args.config:
    hz_s, delays = spec.split(":")
    hz = float(hz_s)
    rate_env = RateEnv(hz, args.num_envs, args.device, args.seed)
    if not checked:
      check_onnx(rate_env)
      checked = True
    for delay_ms in (int(d) for d in delays.split(",")):
      rate_env.set_delay(delay_ms)
      for mode in modes:
        t0 = time.time()
        per = rate_env.run(mode, args.settle, args.measure)
        row = summarize(hz, delay_ms, mode, per)
        rows.append(row)
        for k, v in per.items():
          raw[f"{hz:g}Hz_{delay_ms}ms_{mode}_{k}"] = v
        print(f"{hz:6.2f} Hz {delay_ms:3d} ms {mode:4s}  surv {row['survival']:.2f}  "
              f"ratio {row['ratio']:+.3f}±{row['ratio_ci']:.3f}  "
              f"tilt {row['tilt_deg']:.2f}  track {row['track_rms_deg']:.2f}  "
              f"arate {row['act_rate_rms']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
        with open(out, "w", newline="") as f:
          w = csv.DictWriter(f, fieldnames=list(rows[0]))
          w.writeheader()
          w.writerows(rows)
        np.savez(per_robot, **raw)
    rate_env.env.close()
    del rate_env
    torch.cuda.empty_cache()


if __name__ == "__main__":
  main()
