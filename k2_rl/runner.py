"""K2 on-policy runner: keeps ``last_policy.onnx`` current during training.

mjlab's ``VelocityOnPolicyRunner`` already exports an ONNX on every save, but
it (a) names the file after the run directory and (b) attaches only mjlab's
base metadata -- no gait frequency, no gait command dimension, no trained
ranges. Deployment reads those with defaults, so such a file runs at the wrong
cadence rather than refusing to run. See ``k2_rl.onnx_metadata``.

This writes ``last_policy.onnx`` next to the checkpoints on every save, with
the full K2 deployment contract, so the newest policy is always ready to drive:

    python keyboard.py --policy logs/rsl_rl/k2_turn_v1/<run>/last_policy.onnx
"""

from __future__ import annotations

import os

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from k2_rl.onnx_metadata import attach_k2_metadata

LAST_POLICY_FILENAME = "last_policy.onnx"


class K2OnPolicyRunner(VelocityOnPolicyRunner):
  def save(self, path: str, infos=None):
    gait_cfg = self.env.unwrapped.command_manager.get_term("gait").cfg
    if getattr(gait_cfg, "command_profile", None) == "smooth_full_v1":
      infos = dict(infos or {})
      infos["k2_full_lateral_share"] = gait_cfg.rel_lateral_envs
    super().save(path, infos)
    try:
      out_dir = os.path.dirname(path)
      self.export_policy_to_onnx(out_dir, LAST_POLICY_FILENAME)
      env = self.env.unwrapped
      # Training runs against the training config, so the env's own gait term
      # carries the trained ranges directly -- no separate cfg load needed.
      attach_k2_metadata(
        os.path.join(out_dir, LAST_POLICY_FILENAME),
        env,
        env.command_manager.get_term("gait").cfg,
        os.path.basename(path),
      )
      print(f"  -> {LAST_POLICY_FILENAME} updated from {os.path.basename(path)}")
    except Exception as e:  # noqa: BLE001
      print(f"[WARN] last_policy.onnx export failed (training continues): {e}")
