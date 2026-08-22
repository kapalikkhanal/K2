#!/usr/bin/env python3
"""Render a demo video of a K2 turning policy: wide crowd shot, then one robot.

    python -m scripts.record_demo --policy policies/turn_v5_iter2900.onnx \
        --out k2_demo.mp4

The first `--wide-seconds` show a field of robots all executing the same
command, then the camera dollies in and follows a single robot for the rest.
Commands are scripted: forward, backward, turn left, turn right.

Rendering reuses mjlab's OffscreenRenderer, driving its camera directly so the
shot can move. Physics runs on the same MuJoCo-Warp path training uses.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

import k2_rl  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


def ease(t: float) -> float:
  """Smoothstep, so the camera move starts and ends at rest."""
  t = min(max(t, 0.0), 1.0)
  return t * t * (3.0 - 2.0 * t)


def build_script(seconds: float, hold: float, vx: float, wz: float):
  """(label, vx, wz) per phase, filling `seconds` after the wide shot."""
  phases = [
    ("forward", vx, 0.0),
    ("backward", -vx, 0.0),
    ("turn left", 0.0, wz),
    ("turn right", 0.0, -wz),
  ]
  span = (seconds - hold) / len(phases)
  out, t = [], hold
  for label, a, b in phases:
    out.append((t, t + span, label, a, b))
    t += span
  return out


def run_live(env, wrapped, infer, args) -> None:
  """Interactive window showing EVERY robot, for screen recording.

  mjlab's own viewers render a single ``env_idx``; the rest of the batch is
  invisible. Here the first environment drives the real MjData and the others
  are injected as extra geoms, which is the same trick the offscreen renderer
  uses -- just pointed at a live window.
  """
  import mujoco
  import mujoco.viewer

  model = env.sim.mj_model
  data = mujoco.MjData(model)
  opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
  catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC.value
  script = build_script(args.seconds, 0.0, args.vx, args.wz)
  gait = env.command_manager.get_term("gait")
  dt = env.step_dt
  obs, _ = wrapped.reset()

  with mujoco.viewer.launch_passive(model, data) as viewer:
    per_robot = model.ngeom
    budget = viewer.user_scn.maxgeom
    extra = max(min(env.num_envs - 1, budget // max(per_robot, 1) - 1), 0)
    print(f"\n  window open: showing {extra + 1} of {env.num_envs} robots "
          f"({per_robot} geoms each, scene budget {budget})")
    if extra + 1 < env.num_envs:
      print(f"  NOTE: the viewer's geom budget caps this at {extra + 1}. "
            f"Pass --robots {extra + 1} to match, or the rest stay hidden.")
    print("  orbit with the mouse, then screen-record. Ctrl-C here to quit.\n")
    cam = viewer.cam
    origins = env.scene.env_origins.cpu().numpy()
    centre = origins[:, :2].mean(axis=0)
    cam.lookat[:] = [centre[0], centre[1], 0.25]
    cam.distance = float(np.abs(origins[:, :2] - centre).max()) * 2.4 + 1.5
    cam.elevation, cam.azimuth = -25.0, 90.0

    k, label = 0, ""
    while viewer.is_running():
      t = (k * dt) % args.seconds if args.loop else k * dt
      gait.march_target[:] = 1.0
      vx = wz = 0.0
      for t0, t1, lab, a, b in script:
        if t0 <= t < t1:
          vx, wz = a, b
          if lab != label:
            label = lab
            print(f"  [{t:5.1f}s] {lab}")
          break
      gait.forward_velocity_target[:] = vx
      gait.yaw_rate_target[:] = wz
      with torch.no_grad():
        action = infer(obs)
      obs, _, _, _ = wrapped.step(action)

      sim = env.sim.data
      qpos = sim.qpos.cpu().numpy()
      data.qpos[:] = qpos[0]
      mujoco.mj_forward(model, data)
      viewer.user_scn.ngeom = 0
      for i in range(1, extra + 1):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        mujoco.mjv_addGeoms(model, data, opt, pert, catmask, viewer.user_scn)
      data.qpos[:] = qpos[0]
      mujoco.mj_forward(model, data)
      viewer.sync()
      k += 1
      if not args.loop and k * dt > args.seconds:
        k = 0
        obs, _ = wrapped.reset()
  env.close()


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--policy", help="ONNX policy (or use --checkpoint)")
  ap.add_argument("--checkpoint", help="model_*.pt instead of ONNX")
  ap.add_argument("--task", default="Mjlab-Turn-K2")
  ap.add_argument("--out", default="k2_demo.mp4")
  ap.add_argument("--seconds", type=float, default=20.0)
  ap.add_argument("--wide-seconds", type=float, default=5.0)
  ap.add_argument("--robots", type=int, default=30)
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  ap.add_argument("--vx", type=float, default=0.04)
  ap.add_argument("--wz", type=float, default=0.18)
  ap.add_argument("--settle", type=float, default=2.0,
                  help="seconds of holding before recording starts")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--live", action="store_true",
                  help="open an interactive MuJoCo window with every robot "
                       "instead of writing a file -- orbit/zoom freely and "
                       "screen-record it yourself")
  ap.add_argument("--loop", action="store_true",
                  help="live mode: repeat the command script forever")
  args = ap.parse_args()

  configure_torch_backends()
  cfg = load_env_cfg(args.task, play=True)
  cfg.scene.num_envs = args.robots
  cfg.scene.extent = 1.1          # pack the field tighter for the crowd shot
  cfg.seed = 7
  cfg.commands["gait"].resampling_time_range = (1e9, 1e9)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)

  agent_cfg = load_rl_cfg(args.task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if args.checkpoint:
    runner = load_runner_cls(args.task)(
      wrapped, __import__("dataclasses").asdict(agent_cfg), None, args.device)
    runner.load(args.checkpoint, map_location=args.device)
    policy = runner.get_inference_policy(args.device)
    infer = lambda obs: policy(obs)  # noqa: E731
  else:
    import onnxruntime as ort
    sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    def infer(obs):
      a = sess.run(None, {iname: obs["actor"].cpu().numpy().astype(np.float32)})[0]
      return torch.as_tensor(a, device=args.device)

  if args.live:
    run_live(env, wrapped, infer, args)
    return

  view = ViewerConfig(
    origin_type=ViewerConfig.OriginType.WORLD,
    env_idx=0, max_extra_envs=max(args.robots - 1, 0),
    width=args.width, height=args.height,
    enable_shadows=True, enable_reflections=True,
  )
  renderer = OffscreenRenderer(model=env.sim.mj_model, cfg=view, scene=env.scene)
  renderer.initialize()
  cam = renderer._cam  # driven directly so the shot can move

  origins = env.scene.env_origins.cpu().numpy()
  centre = origins[:, :2].mean(axis=0)
  spread = float(np.abs(origins[:, :2] - centre).max()) + 1.0

  script = build_script(args.seconds, args.wide_seconds, args.vx, args.wz)
  gait = env.command_manager.get_term("gait")
  dt = env.step_dt
  fps = int(round(1.0 / dt))
  obs, _ = wrapped.reset()

  frames = []
  total = int((args.settle + args.seconds) / dt)
  label = ""
  for k in range(total):
    t = k * dt - args.settle
    if t < 0:
      gait.march_target[:] = 0.0
      gait.forward_velocity_target[:] = 0.0
      gait.yaw_rate_target[:] = 0.0
    else:
      gait.march_target[:] = 1.0
      vx = wz = 0.0
      for t0, t1, lab, a, b in script:
        if t0 <= t < t1:
          vx, wz, label = a, b, lab
          break
      gait.forward_velocity_target[:] = vx
      gait.yaw_rate_target[:] = wz
    with torch.no_grad():
      action = infer(obs)
    obs, _, _, _ = wrapped.step(action)
    if t < 0:
      continue

    # --- camera -------------------------------------------------------------
    root = env.scene["robot"].data.root_link_pos_w[0].cpu().numpy()
    zoom = ease((t - args.wide_seconds) / 2.5)   # 0 = wide, 1 = close
    wide_look = np.array([centre[0], centre[1], 0.25])
    close_look = np.array([root[0], root[1], root[2]])
    look = wide_look * (1 - zoom) + close_look * zoom
    cam.lookat[:] = look
    cam.distance = spread * 1.30 * (1 - zoom) + 1.15 * zoom
    # Keep the camera looking down enough that the horizon (which renders black,
    # there is no skybox) stays out of frame in the close shot.
    cam.elevation = -30.0 * (1 - zoom) + -20.0 * zoom
    # Slow drift through the wide shot, settling as the close-up lands.
    cam.azimuth = 90.0 + 10.0 * min(t / max(args.wide_seconds, 1e-6), 1.0) + 18.0 * zoom
    renderer._cfg.max_extra_envs = (
      max(args.robots - 1, 0) if zoom < 0.55 else 6
    )
    renderer.update(env.sim.data)
    frames.append(renderer.render())
    if k % 100 == 0:
      print(f"  t={t:5.1f}s  {label:10s} zoom={zoom:.2f}  frames={len(frames)}")

  import imageio.v2 as imageio
  out = Path(args.out).resolve()
  imageio.mimsave(out, frames, fps=fps, quality=9,
                  macro_block_size=1, codec="libx264")
  print(f"\nwrote {out}  ({len(frames)} frames, {len(frames)/fps:.1f}s @ {fps}fps, "
        f"{args.width}x{args.height})")
  renderer.close()
  env.close()


if __name__ == "__main__":
  main()
