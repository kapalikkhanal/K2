"""Behavioral evaluation of a K2 turning checkpoint on a batch of robots.

Mean training reward mis-ranks these checkpoints -- it has done so twice, in
opposite directions -- so a checkpoint is chosen by what it physically does, not
by its return. This drives one policy through a scripted sequence of commands on
N independently randomized robots and measures the things that decide whether a
policy is worth putting on hardware:

  survival        did it stay up for the whole phase
  tilt            torso lean, against the 12 deg hardware safety cutoff
  sideways lean   mean SIGNED roll (a persistent lean shows here) + path offset
  feet down       both feet planted in hold, no flight phase while stepping
  foot distance   lateral foot gap and inner-edge gap in the root frame
  clearance       minimum swing-sole apex height
  vx tracking     realized / commanded forward speed, both signs
  wz tracking     realized / commanded yaw rate, both signs
  heading drift   yaw wander per 10 s when NO turn is commanded
  act rate/jerk   how hard the policy works the servos -- SEE THE WARNING BELOW
  yawsym          hip-yaw swing amplitude R:L, the gait asymmetry that ratchets
                  the real robot's heading around (1.00 = symmetric)

WARNING on act_rate. A policy that distrusts its own state raises loop gain, and
on hardware that excites K2's 2.5-8 Hz closed-loop resonance. The twin has no
such resonance, so it CANNOT score that failure: the turn_v2b policies measured
+17% action-rate here and +51% on the real robot, and scored two points HIGHER
than the policy they were worse than. Treat act_rate as a relative yellow flag
against a known-good reference (turn_v1_iter2900 = 1.86 in sim, 2.16 on
hardware), never as proof that a policy is smooth enough to deploy.

Startup domain randomization (mass, PD gains, friction, encoder bias, base CoM)
is active in the play config, so the N robots are N different machines. Pushes,
sole height/tilt randomization and observation noise are not -- this is the
nominal-twin ranking the rest of the repo uses.

    python -m k2_rl.eval_checkpoint --checkpoint logs/.../model_3400.pt
    python -m k2_rl.eval_checkpoint --sweep logs/rsl_rl/k2_turn_v1/<run> --stride 200
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

import torch

import k2_rl  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.torch import configure_torch_backends

from k2_rl.k2_constants import (
  CROUCH_BASE_HEIGHT,
  FOOT_HEEL_SITES,
  FOOT_INNER_SITES,
  FOOT_OUTER_SITES,
  FOOT_SITES,
  FOOT_TOE_SITES,
)

# Action-rate RMS of turn_v1_iter2900 in this harness: the smoothest policy that
# has actually been driven on hardware. Used as a RELATIVE baseline only.
REFERENCE_ACT_RATE = 1.86

def make_phases(vx: float, wz: float):
  """(name, march, vx, wz). Test where the policy is meant to RUN: the design
  point moves with cadence, since stride = vx / (2 * gait_freq)."""
  return (
    ("hold",    0.0,  0.00,  0.00),
    ("march",   1.0,  0.00,  0.00),
    ("fwd",     1.0,   vx,   0.00),
    ("back",    1.0,  -vx,   0.00),
    ("turn_L",  1.0,  0.00,   wz),
    ("turn_R",  1.0,  0.00,  -wz),
    ("arc_L",   1.0,   vx,  wz * 0.75),
    ("arc_R",   1.0,  -vx, -wz * 0.75),
  )


PHASES = make_phases(0.03, 0.20)


def _sites(entity, names):
  """Site ids in the given left/right order, independent of model ordering."""
  return [entity.find_sites(n)[0][0] for n in names]


class Evaluator:
  def __init__(self, task: str, num_envs: int, device: str):
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = num_envs
    cfg.seed = 20260820
    # Never resample mid-phase; the script owns the command.
    cfg.commands["gait"].resampling_time_range = (1e9, 1e9)
    self.env = ManagerBasedRlEnv(cfg=cfg, device=device)
    agent_cfg = load_rl_cfg(task)
    self.wrapped = RslRlVecEnvWrapper(self.env, clip_actions=agent_cfg.clip_actions)
    self.runner_cls = load_runner_cls(task)
    self.agent_cfg = agent_cfg
    self.device = device
    self.num_envs = num_envs
    self.dt = self.env.step_dt

    robot = self.env.scene["robot"]
    self.robot = robot
    # FOOT_* tuples are (left, right); resolve each name individually so the
    # swing-foot bookkeeping never depends on MJCF ordering.
    self.center = _sites(robot, FOOT_SITES)
    self.sole = [
      _sites(robot, names)
      for names in (FOOT_SITES, FOOT_HEEL_SITES, FOOT_TOE_SITES,
                    FOOT_INNER_SITES, FOOT_OUTER_SITES)
    ]
    self.inner = _sites(robot, FOOT_INNER_SITES)
    self.left, self.right = 0, 1  # index into the resolved (left, right) lists
    # Hip-yaw swing symmetry: unequal leg twist is what yaws the real robot when
    # no turn is commanded (+0.61 deg/s measured, vs +0.06 in the twin). Sim
    # cannot show the drift itself, but it CAN show the asymmetry that causes it.
    self.hip_yaw_r = robot.find_joints("hip_roll_hip_yaw_right_joint")[0][0]
    self.hip_yaw_l = robot.find_joints("hip_roll_hip_yaw_left_joint")[0][0]
    # Hip-pitch swing is the stride proxy: the fore/aft foot travel is roughly
    # leg length times this angle, and it is what "the stride is too short"
    # actually refers to on hardware.
    self.hip_pitch_r = robot.find_joints("base_link_hip_pitch_right_joint")[0][0]
    self.hip_pitch_l = robot.find_joints("base_link_hip_pitch_left_joint")[0][0]

  def load(self, checkpoint: Path):
    runner = self.runner_cls(
      self.wrapped, asdict(self.agent_cfg), None, self.device
    )
    runner.load(str(checkpoint), map_location=self.device)
    return runner.get_inference_policy(self.device)

  def _measure(self, acc, alive, sinp, action=None, prev=None, prev2=None):
    """Accumulate one step of physical measurements over the alive envs."""
    d = self.robot.data
    grav = d.projected_gravity_b
    tilt = torch.acos(torch.clamp(-grav[:, 2], -1.0, 1.0))
    # Signed roll: +y is the robot's left, so a persistent sign is a real lean.
    roll = torch.atan2(grav[:, 1], -grav[:, 2])
    acc["roll_max"] = torch.maximum(
      acc["roll_max"], torch.where(alive, roll, torch.full_like(roll, -9.9)))
    acc["roll_min"] = torch.minimum(
      acc["roll_min"], torch.where(alive, roll, torch.full_like(roll, 9.9)))

    contact = (self.env.scene["feet_ground_contact"].data.found > 0).float()
    both_down = (contact.sum(dim=1) >= 2).float()
    any_down = (contact.sum(dim=1) >= 1).float()

    # Foot separation in the ROOT frame, so it stays valid at any heading.
    delta_c = d.site_pos_w[:, self.center[0]] - d.site_pos_w[:, self.center[1]]
    gap_c = quat_apply_inverse(d.root_link_quat_w, delta_c)[:, 1].abs()
    delta_i = d.site_pos_w[:, self.inner[0]] - d.site_pos_w[:, self.inner[1]]
    gap_i = quat_apply_inverse(d.root_link_quat_w, delta_i)[:, 1].abs()

    self_hit = (self.env.scene["feet_self_contact"].data.found > 0).float()
    if self_hit.dim() > 1:
      self_hit = self_hit.amax(dim=1)

    m = alive.float()
    n = m.sum().clamp(min=1.0)
    acc["n"] += n
    q = d.joint_pos
    for key, jid in (("yr", self.hip_yaw_r), ("yl", self.hip_yaw_l),
                     ("pr", self.hip_pitch_r), ("pl", self.hip_pitch_l)):
      v = q[:, jid]
      big = torch.where(alive, v, torch.full_like(v, -9.9))
      small = torch.where(alive, v, torch.full_like(v, 9.9))
      acc[key + "max"] = torch.maximum(acc[key + "max"], big)
      acc[key + "min"] = torch.minimum(acc[key + "min"], small)
    if action is not None and prev is not None:
      rate = (action - prev) / self.dt
      acc["arate_sq"] += (torch.square(rate).mean(dim=1) * m).sum()
      if prev2 is not None:
        jerk = (action - 2.0 * prev + prev2) / (self.dt * self.dt)
        acc["ajerk_sq"] += (torch.square(jerk).mean(dim=1) * m).sum()
    acc["tilt_sum"] += (tilt * m).sum()
    acc["tilt_max"] = torch.maximum(acc["tilt_max"], (tilt * m).max())
    acc["tilt_hi"] += ((tilt > math.radians(10.0)).float() * m).sum()
    acc["roll_sum"] += (roll * m).sum()
    acc["roll_abs_sum"] += (roll.abs() * m).sum()
    acc["both_down"] += (both_down * m).sum()
    acc["any_down"] += (any_down * m).sum()
    acc["gap_sum"] += (gap_c * m).sum()
    acc["gap_min"] = torch.minimum(
      acc["gap_min"], torch.where(alive, gap_c, torch.full_like(gap_c, 9.9)).min())
    acc["inner_min"] = torch.minimum(
      acc["inner_min"], torch.where(alive, gap_i, torch.full_like(gap_i, 9.9)).min())
    acc["self_hit"] += (self_hit * m).sum()
    acc["vx_sum"] += (d.root_link_lin_vel_b[:, 0] * m).sum()
    acc["vy_sum"] += (d.root_link_lin_vel_b[:, 1] * m).sum()
    acc["wz_sum"] += (d.root_link_ang_vel_b[:, 2] * m).sum()
    acc["height_sum"] += (d.root_link_pos_w[:, 2] * m).sum()

    # Swing-sole apex clearance. The swing foot is whichever one is RAISED, so
    # take each foot's minimum sole height and keep the larger: no assumption
    # about MJCF site ordering or the sign convention of the phase clock.
    apex = sinp.abs() > 0.90
    if bool(apex.any()):
      z = torch.stack([
        torch.stack([d.site_pos_w[:, pair[0], 2], d.site_pos_w[:, pair[1], 2]], 1)
        for pair in self.sole
      ], dim=1)  # [B, 5 sole points, 2 feet]
      swing_z = z.amin(dim=1).amax(dim=1)  # min over sole points, max over feet
      sel = apex & alive
      if bool(sel.any()):
        acc["apex_sum"] += swing_z[sel].sum()
        acc["apex_n"] += sel.float().sum()
        acc["apex_min"] = torch.minimum(acc["apex_min"], swing_z[sel].min())

  def phase(self, policy, march, vx, wz, settle=150, measure=500):
    env, wrapped = self.env, self.wrapped
    obs, _ = wrapped.reset()
    gait = env.command_manager.get_term("gait")
    alive = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    z = lambda: torch.zeros((), device=self.device)
    acc = {k: z() for k in (
      "n", "tilt_sum", "tilt_hi", "roll_sum", "roll_abs_sum", "both_down",
      "any_down", "gap_sum", "self_hit", "vx_sum", "vy_sum", "wz_sum",
      "height_sum", "apex_sum", "apex_n", "lat_sum", "arate_sq", "ajerk_sq")}
    acc["tilt_max"] = z()
    acc["lat_max"] = z()
    big = lambda v: torch.full((self.num_envs,), v, device=self.device)
    for key in ("yrmax", "ylmax", "prmax", "plmax"):
      acc[key] = big(-9.9)
    for key in ("yrmin", "ylmin", "prmin", "plmin"):
      acc[key] = big(9.9)
    acc["roll_max"] = big(-9.9)
    acc["roll_min"] = big(9.9)
    lat0 = torch.zeros(self.num_envs, device=self.device)
    acc["gap_min"] = torch.full((), 9.9, device=self.device)
    acc["inner_min"] = torch.full((), 9.9, device=self.device)
    acc["apex_min"] = torch.full((), 9.9, device=self.device)

    heading_delta = torch.zeros(self.num_envs, device=self.device)
    prev_heading = None
    cur = prev = prev2 = None

    for step in range(settle + measure):
      # The script owns the command; ramps still carry it there smoothly.
      gait.march_target[:] = march
      gait.forward_velocity_target[:] = vx
      gait.yaw_rate_target[:] = wz
      with torch.no_grad():
        action = policy(obs)
      prev2, prev, cur = prev, cur, action.clone()
      obs, _, dones, _ = wrapped.step(action)
      alive &= dones == 0

      h = self.robot.data.heading_w
      if prev_heading is not None:
        step_delta = (h - prev_heading + math.pi) % (2 * math.pi) - math.pi
        if step >= settle:
          heading_delta += step_delta
      prev_heading = h

      delta = self.robot.data.root_link_pos_w[:, :2] - gait.pos_ref
      hd = gait.heading_ref
      lat_now = -delta[:, 0] * torch.sin(hd) + delta[:, 1] * torch.cos(hd)
      if step == settle - 1:
        # Anchor on the offset the robot already had, so this reports DRIFT and
        # not the +/-50 mm lateral scatter reset_base starts every episode with.
        lat0 = lat_now.clone()
      if step >= settle:
        sinp = gait.command[:, 1]
        self._measure(acc, alive, sinp, cur, prev, prev2)
        acc["lat_sum"] += ((lat_now - lat0).abs() * alive.float()).sum()
        acc["lat_max"] = torch.maximum(
          acc["lat_max"],
          torch.where(alive, (lat_now - lat0).abs(),
                      torch.zeros_like(lat_now)).max())

    n = float(acc["n"].clamp(min=1.0))
    window = measure * self.dt
    alive_f = float(alive.float().mean())
    live = alive.float().clamp(min=0.0)
    nlive = float(live.sum().clamp(min=1.0))
    deg = math.degrees
    out = {
      "survival": alive_f,
      "tilt_mean_deg": deg(float(acc["tilt_sum"]) / n),
      "tilt_max_deg": deg(float(acc["tilt_max"])),
      "tilt_over10_frac": float(acc["tilt_hi"]) / n,
      "roll_mean_deg": deg(float(acc["roll_sum"]) / n),
      "roll_abs_deg": deg(float(acc["roll_abs_sum"]) / n),
      "both_feet_down": float(acc["both_down"]) / n,
      "any_foot_down": float(acc["any_down"]) / n,
      "foot_gap_mean_mm": 1000.0 * float(acc["gap_sum"]) / n,
      "foot_gap_min_mm": 1000.0 * float(acc["gap_min"]),
      "inner_gap_min_mm": 1000.0 * float(acc["inner_min"]),
      "self_collision_frac": float(acc["self_hit"]) / n,
      "vx_mps": float(acc["vx_sum"]) / n,
      "vy_mps": float(acc["vy_sum"]) / n,
      "wz_rps": float(acc["wz_sum"]) / n,
      "base_height_mm": 1000.0 * float(acc["height_sum"]) / n,
      "lateral_drift_mm": 1000.0 * float(acc["lat_sum"]) / n,
      "lateral_drift_max_mm": 1000.0 * float(acc["lat_max"]),
      "heading_change_deg": deg(float((heading_delta * live).sum()) / nlive),
      "hip_pitch_swing_r_deg": deg(float(
        (acc["prmax"] - acc["prmin"])[alive].mean()) if bool(alive.any()) else 0.0),
      "hip_pitch_swing_l_deg": deg(float(
        (acc["plmax"] - acc["plmin"])[alive].mean()) if bool(alive.any()) else 0.0),
      "roll_p2p_deg": deg(float(
        (acc["roll_max"] - acc["roll_min"])[alive].mean()) if bool(alive.any()) else 0.0),
      "hip_yaw_swing_r_deg": deg(float(
        (acc["yrmax"] - acc["yrmin"])[alive].mean()) if bool(alive.any()) else 0.0),
      "hip_yaw_swing_l_deg": deg(float(
        (acc["ylmax"] - acc["ylmin"])[alive].mean()) if bool(alive.any()) else 0.0),
      "act_rate_rms": math.sqrt(max(float(acc["arate_sq"]) / n, 0.0)),
      "act_jerk_rms": math.sqrt(max(float(acc["ajerk_sq"]) / n, 0.0)),
      "cmd_vx": vx, "cmd_wz": wz, "cmd_march": march, "window_s": window,
    }
    out["hip_yaw_swing_ratio"] = (
      out["hip_yaw_swing_r_deg"] / out["hip_yaw_swing_l_deg"]
      if out["hip_yaw_swing_l_deg"] > 1e-6 else float("nan"))
    out["vx_ratio"] = out["vx_mps"] / vx if vx else float("nan")
    out["wz_ratio"] = out["wz_rps"] / wz if wz else float("nan")
    out["heading_rate_dps"] = out["heading_change_deg"] / window
    if float(acc["apex_n"]) > 0:
      out["apex_mean_mm"] = 1000.0 * float(acc["apex_sum"]) / float(acc["apex_n"])
      out["apex_min_mm"] = 1000.0 * float(acc["apex_min"])
    else:
      out["apex_mean_mm"] = float("nan")
      out["apex_min_mm"] = float("nan")
    return out

  def run(self, checkpoint: Path, settle: int, measure: int, phases=PHASES):
    policy = self.load(checkpoint)
    return {
      name: self.phase(policy, march, vx, wz, settle, measure)
      for name, march, vx, wz in phases
    }


def score(res: dict) -> tuple[float, list[str]]:
  """Explicit behavioral score. Gates first, then quality. Higher is better."""
  notes = []
  worst_survival = min(p["survival"] for p in res.values())
  if worst_survival < 0.90:
    notes.append(f"FALLS (worst survival {worst_survival:.2f})")

  def clip01(x):
    return max(0.0, min(1.0, x))

  # Command tracking, both signs, translation and rotation.
  vx_track = sum(clip01(r) for r in (res["fwd"]["vx_ratio"],
                                     res["back"]["vx_ratio"])) / 2
  wz_track = sum(clip01(r) for r in (res["turn_L"]["wz_ratio"],
                                     res["turn_R"]["wz_ratio"],
                                     res["arc_L"]["wz_ratio"],
                                     res["arc_R"]["wz_ratio"])) / 4
  # Stability, over the stepping phases.
  step_phases = [p for k, p in res.items() if k != "hold"]
  tilt = sum(p["tilt_mean_deg"] for p in step_phases) / len(step_phases)
  lean = sum(abs(p["roll_mean_deg"]) for p in step_phases) / len(step_phases)
  ground = sum(p["any_foot_down"] for p in step_phases) / len(step_phases)
  gap = min(p["inner_gap_min_mm"] for p in res.values())
  apex = min(p["apex_mean_mm"] for p in step_phases
             if not math.isnan(p["apex_mean_mm"]))
  drift = abs(res["fwd"]["heading_rate_dps"])
  hold_planted = res["hold"]["both_feet_down"]
  ratios = [p["hip_yaw_swing_ratio"] for p in step_phases
            if not math.isnan(p["hip_yaw_swing_ratio"])]
  yawsym = sum(min(r, 1.0 / r) for r in ratios) / max(len(ratios), 1) if ratios else 0.0
  arate = max(p["act_rate_rms"] for p in step_phases)
  if arate > 1.15 * REFERENCE_ACT_RATE:
    notes.append(f"JITTER act_rate {arate:.2f} vs ref {REFERENCE_ACT_RATE:.2f}")

  s = (
    3.0 * worst_survival
    + 2.5 * vx_track
    + 2.5 * wz_track
    + 1.0 * hold_planted
    + 1.0 * ground
    + 1.0 * clip01(1.0 - tilt / 8.0)          # 8 deg mean tilt -> zero credit
    + 1.0 * clip01(1.0 - lean / 4.0)          # 4 deg persistent lean -> zero
    + 0.8 * clip01(gap / 12.0)                # 12 mm inner gap -> full credit
    + 0.8 * clip01(apex / 10.0)               # 10 mm apex -> full credit
    + 0.6 * clip01(1.0 - drift / 3.0)         # 3 deg/s straight-line drift
    + 1.0 * clip01(1.0 - max(0.0, arate - REFERENCE_ACT_RATE) / 0.5)
    + 1.5 * yawsym                            # hip-yaw swing symmetry, 1 = equal
  )
  if worst_survival < 0.90:
    s -= 5.0
  return s, notes


def _fmt(name: str, p: dict) -> str:
  return (f"  {name:7s} surv {p['survival']:.2f}  tilt {p['tilt_mean_deg']:4.1f}"
          f"/{p['tilt_max_deg']:4.1f}  roll {p['roll_mean_deg']:+5.2f}  "
          f"down {p['any_foot_down']:.2f}/{p['both_feet_down']:.2f}  "
          f"gap {p['foot_gap_min_mm']:5.1f}/{p['inner_gap_min_mm']:5.1f}  "
          f"apex {p['apex_mean_mm']:5.1f}  "
          f"vx {p['vx_mps']:+.4f}({p['vx_ratio']:+.2f})  "
          f"wz {p['wz_rps']:+.3f}({p['wz_ratio']:+.2f})  "
          f"drift {p['lateral_drift_mm']:5.1f}  yaw {p['heading_rate_dps']:+5.2f}/s"
          f"  arate {p['act_rate_rms']:5.2f}"
          f"  yawsym {p['hip_yaw_swing_ratio']:4.2f}"
          f"  stride {p['hip_pitch_swing_r_deg']:4.1f}/{p['hip_pitch_swing_l_deg']:4.1f}"
          f"  rollp2p {p['roll_p2p_deg']:5.2f}")


def main() -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--task", default="Mjlab-Turn-K2")
  ap.add_argument("--checkpoint", nargs="+",
                  help="one or more model_*.pt (one env build serves all)")
  ap.add_argument("--sweep", help="run directory; evaluate many checkpoints")
  ap.add_argument("--stride", type=int, default=200)
  ap.add_argument("--min-iter", type=int, default=0)
  ap.add_argument("--num-envs", type=int, default=128)
  ap.add_argument("--settle", type=int, default=150)
  ap.add_argument("--measure", type=int, default=500)
  ap.add_argument("--vx", type=float, default=0.03,
                  help="forward/backward test speed (default 0.03)")
  ap.add_argument("--wz", type=float, default=0.20,
                  help="turn-in-place test yaw rate (default 0.20)")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--json", help="write raw results here")
  args = ap.parse_args()

  configure_torch_backends()
  if args.sweep:
    run_dir = Path(args.sweep)
    picks = []
    for f in run_dir.glob("model_*.pt"):
      it = int(re.fullmatch(r"model_(\d+)\.pt", f.name).group(1))
      if it >= args.min_iter and (it - args.min_iter) % args.stride == 0:
        picks.append((it, f))
    checkpoints = [f for _, f in sorted(picks)]
    if not checkpoints:
      raise SystemExit(f"no model_*.pt in {run_dir} matching stride {args.stride}")
  elif args.checkpoint:
    checkpoints = [Path(c) for c in args.checkpoint]
  else:
    raise SystemExit("pass --checkpoint or --sweep")

  phases = make_phases(args.vx, args.wz)
  ev = Evaluator(args.task, args.num_envs, args.device)
  print(f"\n{args.num_envs} randomized robots, {args.measure * ev.dt:.0f} s "
        f"measured per phase after a {args.settle * ev.dt:.0f} s settle\n")
  all_results = {}
  ranking = []
  for ckpt in checkpoints:
    res = ev.run(ckpt, args.settle, args.measure, phases)
    s, notes = score(res)
    all_results[ckpt.name] = res
    ranking.append((s, ckpt.name, notes))
    print(f"{ckpt.name}   SCORE {s:.3f}" + (f"   [{'; '.join(notes)}]" if notes else ""))
    for name, _, _, _ in phases:
      print(_fmt(name, res[name]))
    print()

  print("=" * 100)
  print("ranking (behavioral score, NOT training reward)")
  for s, name, notes in sorted(ranking, reverse=True):
    print(f"  {s:7.3f}  {name}" + (f"   [{'; '.join(notes)}]" if notes else ""))
  if args.json:
    Path(args.json).write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
  main()
