#!/usr/bin/env python3
"""Measure whole-foot clearance from a k2_policy_run CSV.

Run this on the laptop in ``unitree_sim_env`` after a Bus-based digital-twin
or real-robot test.  It reconstructs both commanded and encoder joint poses in
the current K2 model and reports center, heel, toe, and exact foot-mesh contact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from hardware import k2_conventions as C
from k2_rl.k2_constants import get_spec


def _quantiles_mm(values: list[float]) -> dict[str, float]:
  a = np.asarray(values)
  return {
    "min_mm": float(np.min(a) * 1000.0),
    "p05_mm": float(np.quantile(a, 0.05) * 1000.0),
    "p50_mm": float(np.quantile(a, 0.50) * 1000.0),
  }


def analyze(log_path: Path, prefix: str) -> dict[str, object]:
  table = pd.read_csv(log_path)
  if "march" in table:
    table = table[table["march"] > 0.5]
  if table.empty:
    raise ValueError(f"{log_path} contains no marching samples")

  model = get_spec().compile()
  data = mujoco.MjData(model)
  qpos_adrs = [
    model.jnt_qposadr[
      mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, C.JOINT_TO_MJCF[joint]
      )
    ]
    for joint in C.SIM_ORDER
  ]
  site_pairs = {
    "center": ("left_foot", "right_foot"),
    "heel": ("left_foot_heel", "right_foot_heel"),
    "toe": ("left_foot_toe", "right_foot_toe"),
  }
  site_ids = {
    name: tuple(
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
      for site in pair
    )
    for name, pair in site_pairs.items()
  }
  left_geom = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision"
  )
  right_geom = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision"
  )

  gaps: dict[str, list[float]] = {name: [] for name in site_pairs}
  contact_depths: list[float] = []
  contact_frames = 0
  for _, row in table.iterrows():
    data.qpos[:] = model.qpos0
    data.qpos[qpos_adrs] = [row[prefix + joint] for joint in C.SIM_ORDER]
    mujoco.mj_forward(model, data)
    for name, (left_site, right_site) in site_ids.items():
      gaps[name].append(
        abs(float(data.site_xpos[left_site, 1] - data.site_xpos[right_site, 1]))
      )
    distances = [
      float(contact.dist)
      for contact in data.contact[: data.ncon]
      if {contact.geom1, contact.geom2} == {left_geom, right_geom}
    ]
    if distances:
      contact_frames += 1
      contact_depths.append(min(distances))

  return {
    "samples": int(len(table)),
    "pose_source": prefix.removesuffix("_"),
    "center_gap": _quantiles_mm(gaps["center"]),
    "heel_gap": _quantiles_mm(gaps["heel"]),
    "toe_gap": _quantiles_mm(gaps["toe"]),
    "mesh_contact_fraction": contact_frames / len(table),
    "worst_mesh_contact_depth_mm": (
      float(min(contact_depths) * 1000.0) if contact_depths else 0.0
    ),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("log", type=Path)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  report = {
    "log": str(args.log.resolve()),
    "commanded": analyze(args.log, "cmd_"),
    "measured": analyze(args.log, "q_"),
  }
  text = json.dumps(report, indent=2)
  print(text)
  if args.output:
    args.output.write_text(text + "\n")


if __name__ == "__main__":
  main()
