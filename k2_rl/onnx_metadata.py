"""The K2 deployment contract that gets baked into every exported ONNX.

Kept in one place because two paths write it and they MUST agree:
``k2_rl.export_onnx`` (pin a specific checkpoint) and ``K2OnPolicyRunner.save``
(the rolling ``last_policy.onnx`` written on every checkpoint).

Why this matters more than it looks: ``hardware/k2_policy_run._read_metadata``
falls back to defaults for every key it cannot find -- ``LEGACY_GAIT_FREQ_HZ``
for the cadence, ``0.0`` for the gait transition, and so on. A policy exported
with only mjlab's base metadata therefore does not fail, it runs at the WRONG
GAIT FREQUENCY, silently. mjlab's own ``VelocityOnPolicyRunner`` attaches only
that base metadata, which is exactly the trap this module exists to close.
"""

from __future__ import annotations

import re
from pathlib import Path

from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata


def build_k2_metadata(env, train_gait_cfg, checkpoint_name: str) -> dict:
  """Base metadata plus everything deployment reconstructs the command from.

  ``env`` is the unwrapped ``ManagerBasedRlEnv`` the policy was built against;
  ``train_gait_cfg`` is the TRAINING gait config, which is not the same object
  as the env's own when exporting from a play env (play pins the commands to a
  single preview value, and deployment needs the trained ranges instead).
  """
  meta = get_base_metadata(env, checkpoint_name)
  meta["gait_lateral_positive_phase"] = float(train_gait_cfg.lateral_positive_phase)
  if hasattr(train_gait_cfg, "command_profile"):
    meta["gait_command_profile"] = train_gait_cfg.command_profile

  # mjlab's base metadata does not preserve the RSL-RL iteration.
  match = re.fullmatch(r"model_(\d+)\.pt", Path(checkpoint_name).name)
  if match:
    meta["checkpoint_iteration"] = match.group(1)

  gait_term = env.command_manager.get_term("gait")
  meta["gait_frequency_hz"] = float(gait_term.cfg.gait_freq)
  meta["gait_transition_time_s"] = float(gait_term.cfg.transition_time_s)
  meta["gait_velocity_ramp_rate_mps2"] = float(
    gait_term.cfg.velocity_ramp_rate_mps2
  )
  meta["gait_yaw_rate_ramp_rate_rps2"] = float(
    gait_term.cfg.yaw_rate_ramp_rate_rps2
  )
  meta["gait_height_ramp_rate_mps"] = float(gait_term.cfg.height_ramp_rate_mps)
  meta["gait_lateral_ramp_rate_mps2"] = float(
    gait_term.cfg.lateral_ramp_rate_mps2
  )
  meta["gait_command_dim"] = int(gait_term.command.shape[1])

  actor_terms = env.cfg.observations["actor"].terms
  meta["heading_observation_dim"] = 2 if "heading" in actor_terms else 0

  # Whether a two-foot HOLD (march == 0) was ever trained. At 1.0 every
  # environment marched, so march == 0 is an input the policy has never seen
  # and deployment must not ask for it -- `keyboard.py` reads this to decide
  # whether the H key and the "start safely in hold" default are available.
  meta["gait_rel_march_envs"] = float(train_gait_cfg.rel_march_envs)

  # A standing rotation deliberately reuses the ordinary gait clock and yaw
  # tensor slots with march=0. Older turning policies must never receive this
  # otherwise out-of-distribution command, so deployment enables O/P only when
  # these explicit capability keys are present.
  if (hasattr(train_gait_cfg, "standing_rotate_period_s")
      and getattr(train_gait_cfg, "pipeline_stage", 4) >= 4):
    meta["gait_standing_rotation"] = "1"
    meta["gait_standing_rotate_period_s"] = float(
      train_gait_cfg.standing_rotate_period_s
    )
    meta["gait_standing_rotate_yaw_rate_rps"] = float(
      train_gait_cfg.standing_rotate_yaw_rate_rps
    )

  # Channel 5 is crouch depth on some tasks and lateral speed on others, and
  # NOTHING in the observation vector distinguishes them. Deployment has to be
  # told which it is; policies exported before this key existed were all crouch.
  if train_gait_cfg.lateral_velocity_range is not None:
    meta["gait_channel5"] = "lateral_velocity"
  elif train_gait_cfg.height_offset_range is not None:
    meta["gait_channel5"] = "height_offset"
    # Whether the height offset was trained UNGATED, i.e. commanded while
    # marching as well as while holding. Deployment reproduces the training
    # gate, so without this key `k2_policy_run` applies the historical
    # `(1 - march)` and feeds a walk+turn+height policy a permanently zero
    # height channel the instant a gait is requested. Absent on every policy
    # exported before the flag existed, all of which were hold-only crouch.
    if getattr(train_gait_cfg, "height_active_in_march", False):
      meta["gait_height_active_in_march"] = "1"

  # Ranges describe what was TRAINED, not the single value play pins them to.
  if train_gait_cfg.forward_velocity_range is not None:
    lo, hi = train_gait_cfg.forward_velocity_range
    meta["forward_velocity_min_mps"] = float(lo)
    meta["forward_velocity_max_mps"] = float(hi)
  if train_gait_cfg.yaw_rate_range is not None:
    lo, hi = train_gait_cfg.yaw_rate_range
    meta["yaw_rate_min_rps"] = float(lo)
    meta["yaw_rate_max_rps"] = float(hi)
  if train_gait_cfg.height_offset_range is not None:
    lo, hi = train_gait_cfg.height_offset_range
    meta["height_offset_min_m"] = float(lo)
    meta["height_offset_max_m"] = float(hi)
  if train_gait_cfg.lateral_velocity_range is not None:
    lo, hi = train_gait_cfg.lateral_velocity_range
    if train_gait_cfg.rel_lateral_envs == 0:
      lo, hi = 0., 0.
    meta["lateral_velocity_min_mps"] = float(lo)
    meta["lateral_velocity_max_mps"] = float(hi)
    meta["gait_rel_lateral_envs"] = float(train_gait_cfg.rel_lateral_envs)
  if hasattr(train_gait_cfg, "pipeline_stage"):
    meta["k2_pipeline_stage"] = int(train_gait_cfg.pipeline_stage)
    meta["gait_channel6"] = "height_offset"
    meta["gait_height_command_index"] = 6
    meta["gait_staged_height"] = "1"
    lo, hi = train_gait_cfg.staged_height_range
    meta["height_offset_min_m"] = float(lo)
    meta["height_offset_max_m"] = float(hi)
  return meta


def attach_k2_metadata(onnx_path: str, env, train_gait_cfg, checkpoint_name: str):
  """Best-effort: a metadata failure must never lose an otherwise-valid ONNX."""
  try:
    attach_metadata_to_onnx(
      onnx_path, build_k2_metadata(env, train_gait_cfg, checkpoint_name)
    )
  except Exception as e:  # noqa: BLE001
    print(f"[WARN] metadata attach failed (onnx still valid): {e}")
