"""Preflight for the unified full task; does not certify learned behavior.

python -m k2_rl.verify_full --device cpu
"""
import argparse
from copy import deepcopy
from pathlib import Path
import tempfile
from dataclasses import asdict

import numpy as np
import torch

from hardware.k2_command_profile import RATE, JERK, STOP_SPEED, STOP_RATE, step_profile


def check_profile():
  states = []
  for xp in (np, torch):
    zeros = lambda shape: xp.zeros(shape, dtype=xp.float32)
    states.append([zeros((2, 3)), zeros((2, 3)), zeros(2), zeros(2),
                   xp.zeros(2, dtype=xp.bool if xp is torch else bool)])
  previous_rate = np.zeros((2, 3))
  previous_value = np.zeros((2, 3))
  for i in range(1000):
    # Accelerate, reverse without resetting clock, release to hold, restart.
    speed = .025 if i < 150 else -.025 if i < 300 else .02
    moving = np.full(2, i < 300 or i >= 650)
    targets = np.array([[0., speed, 0.], [.03, 0., .25]], dtype=np.float32)
    for j, xp in enumerate((np, torch)):
      arr = np.asarray if xp is np else torch.as_tensor
      states[j] = list(step_profile(xp, *states[j], arr(targets), arr(moving),
        .02, .58, 1., *[arr(v, dtype=xp.float32) for v in (RATE, JERK, STOP_SPEED, STOP_RATE)]))
    for a, b in zip(*states):
      np.testing.assert_allclose(a, b.numpy(), atol=3e-5)
    value, rate, phase, march, settling = states[0]
    assert (abs(rate) <= np.array(RATE) + 1e-6).all()
    assert (abs(rate - previous_rate) <= np.array(JERK) * .02 + 1e-6).all()
    assert (abs(value - previous_value) <= np.array(RATE) * .02 + 1e-6).all()
    if i == 301:
      assert (march == 1).all(), "release must decelerate before hold"
    if i == 649:
      assert (march == 0).all(), "must eventually settle into hold"
    previous_rate, previous_value = rate.copy(), value.copy()
  print("profile: NumPy/Torch parity, acceleration/jerk bounds, reversals and settled stop PASS")


def check_live(device):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from k2_rl.runner import K2OnPolicyRunner
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from k2_rl.mdp.symmetry import K2Mirror
  from k2_rl.onnx_metadata import build_k2_metadata

  cfg = load_env_cfg("Mjlab-Full-K2")
  assert cfg.scene.terrain.terrain_type == "plane"
  assert "sole_height" in cfg.events and "sole_tilt" in cfg.events
  cfg.scene.num_envs = 8
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    wrapped = RslRlVecEnvWrapper(env)
    obs, _ = wrapped.reset()
    assert obs["actor"].shape == (8, 50) and obs["critic"].shape == (8, 55)
    gait = env.command_manager.get_term("gait")
    mirror = K2Mirror(env)
    torch.testing.assert_close(mirror.observations(mirror.observations(obs))["actor"], obs["actor"])
    metadata = build_k2_metadata(env, cfg.commands["gait"], "model_0.pt")
    assert metadata["gait_command_profile"] == "smooth_full_v1"
    assert "gait_standing_rotation" not in metadata
    assert cfg.commands["gait"].height_offset_range is None
    ids = torch.arange(8, device=device)
    for _ in range(100):
      gait._resample_command(ids)
      assert (gait.lateral_velocity_target == 0).all()
    gait.cfg.rel_lateral_envs = .35
    counts = torch.zeros(4, device=device)
    for _ in range(100):
      gait._resample_command(ids)
      counts += torch.stack(((gait.lateral_velocity_target > 0).sum(),
                             (gait.lateral_velocity_target < 0).sum(),
                             (gait.forward_velocity_target > 0).sum(),
                             (gait.forward_velocity_target < 0).sum()))
    assert (counts > 0).all(), counts
    # Check actual leading-foot ordering and physics-substep force history.
    foot_cfg = env.reward_manager.get_term_cfg("lateral_stride").params["asset_cfg"]
    y = env.scene["robot"].data.site_pos_w[0, foot_cfg.site_ids, 1]
    assert y[0] > y[1]
    gait.reset(torch.arange(8, device=device))
    gait.march_target.fill_(1)
    gait.lateral_velocity_target[::2] = .025
    gait.lateral_velocity_target[1::2] = -.025
    action = torch.zeros(8, env.action_manager.total_action_dim, device=device)
    for _ in range(100):
      obs, reward, *_ = env.step(action)
      assert torch.isfinite(reward).all() and torch.isfinite(obs["actor"]).all()
    agent = load_rl_cfg("Mjlab-Full-K2")
    agent.num_steps_per_env = 8
    agent.algorithm.num_learning_epochs = 1
    agent.algorithm.num_mini_batches = 1
    agent.logger = "tensorboard"
    agent.upload_model = False
    runner = K2OnPolicyRunner(wrapped, asdict(agent), log_dir=None, device=device)
    runner.learn(1)
    assert all(torch.isfinite(p).all() for p in runner.alg.actor.parameters())
    from k2_rl.enable_full_lateral import prepare
    import onnxruntime as ort
    # Save through the production runner, then exercise the real transfer.
    gait.cfg.rel_lateral_envs = 0.
    with tempfile.TemporaryDirectory() as directory:
      checkpoint_path = str(Path(directory) / "model_0.pt")
      runner.save(checkpoint_path)
      checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
      original = deepcopy(checkpoint["actor_state_dict"])
      prepared = prepare(checkpoint)
      assert torch.equal(original["mlp.0.weight"][:, :-1],
                         prepared["actor_state_dict"]["mlp.0.weight"][:, :-1])
      probe = obs["actor"].clone()
      probe[:, -1] = 0
      actor = runner.alg.actor
      actor.eval()
      with torch.inference_mode():
        before = actor.mlp(actor.obs_normalizer(probe))
        actor.load_state_dict(prepared["actor_state_dict"])
        torch.testing.assert_close(before, actor.mlp(actor.obs_normalizer(probe)))
        actor.load_state_dict(original)
      assert prepared["optimizer_state_dict"]["state"] == {}
      session = ort.InferenceSession(str(Path(directory) / "last_policy.onnx"),
                                     providers=["CPUExecutionProvider"])
      metadata = session.get_modelmeta().custom_metadata_map
      assert float(metadata["gait_rel_lateral_envs"]) == 0
      assert float(metadata["lateral_velocity_max_mps"]) == 0
      assert "gait_standing_rotation" not in metadata
      result = session.run(None, {session.get_inputs()[0].name: probe.cpu().numpy()[:1]})[0]
      assert result.shape == (1, 12) and np.isfinite(result).all()
    print("full: checkpoint transfer preserves old-command outputs; warmup ONNX contract PASS")
    print("full: plane with sole DR, 50/55 observations, force history, warmup/joint sampling, mirror and PPO update PASS")
  finally:
    env.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()
  check_profile()
  check_live(args.device)
