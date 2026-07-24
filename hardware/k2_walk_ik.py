#!/usr/bin/env python3
"""IMU-stabilized quasi-static IK walk for the 12-DOF K2 -- no RL.

This is a test gait, not a policy. It only exists because the v4 ankle roll made
single-foot support statically stable (k2_sway: CoM reaches 85 mm vs 27.5 needed).
A quasi-static walk keeps the centre of mass inside the current support foot at
every instant. On hardware, bounded LSM6DS3 roll/pitch feedback corrects measured
torso lean with coordinated ankle/hip motions derived from the same IK model.

How it works:

  * WalkIK is a whole-body inverse kinematics solver. Given a target for each
    foot (x, y, height -- flat on the ground when height=0, lifted and level when
    height>0) and a target CoM (x, y), it solves the base pose + 12 joint angles
    with the feet held flat. It is the sway/squat IK generalised to place BOTH
    feet independently, so it drives stance, swing and double support uniformly.
  * build_walk() lays out one repeatable step cycle -- shift the CoM over the
    stance foot, lift and swing the other foot forward, set it down, shift across
    -- as smooth per-tick targets, then WalkIK solves each tick. The CoM path is
    kept inside the support polygon the whole time (validated in --check).

The ankle roll is the binding joint. The current build has 20 deg inward roll,
and the gait uses feet at +/-55 mm to preserve swing clearance. Static margin is
still only a few millimetres, so feedback improves lean rejection but cannot
recover after the centre of mass has already left the support foot.

  python -m hardware.k2_walk_ik --bus sim --viewer          # watch the twin
  python -m hardware.k2_walk_ik --check                     # numeric stability only
  python -m hardware.k2_walk_ik --bus /dev/ttyACM0          # real + IMU feedback

The joints move through hardware.k2_ctrl.run, the same slew-limited, torque-safe
loop every other K2 motion uses, so sim and hardware run byte-identical logic.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys

import numpy as np

from . import k2_bus
from . import k2_conventions as C
from . import k2_ctrl
from .k2_attitude import ImuAttitude, SimAttitude
from .k2_stabilizer import TiltStabilizer, UnsafeTiltError

DT = 0.02   # 50 Hz, same as every other K2 loop


class WalkIK:
    """Whole-body IK: place both feet (flat) and the CoM, solve base + 12 joints."""

    # The ten leg joints the walk drives (hip_yaw stays 0 for a straight walk).
    NAMES = ["hip_roll_R", "hip_pitch_R", "knee_R", "ankle_pitch_R", "ankle_roll_R",
             "hip_roll_L", "hip_pitch_L", "knee_L", "ankle_pitch_L", "ankle_roll_L"]

    def __init__(self):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(C.MJCF))
        self.data = mujoco.MjData(self.model)
        self.qadr = {j: self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, m)]
            for j, m in C.JOINT_TO_MJCF.items()}

        self.foot_body = {"R": "Feet_right", "L": "Feet_left"}
        self.bodies = {b: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
                       for b in self.foot_body.values()}
        # Four sole points (heel, toe, both lateral edges) in each foot's frame,
        # read off the collision mesh so they track the CAD.
        self.sole = {
            "R": self._sole("right_foot_collision", "Feet_right"),
            "L": self._sole("left_foot_collision", "Feet_left"),
        }
        # Nominal lateral foot placement (sole-centre y at the zero pose).
        self.mj.mj_resetData(self.model, self.data)
        self.mj.mj_forward(self.model, self.data)
        self.y_nom = {s: self._sole_centre_world(s)[1] for s in ("R", "L")}
        self.z_nom = 0.30      # weak base-height regulariser (walking crouch)
        # Fixed lateral trim on the base: a right/left lean baked into every pose
        # to cancel the constant drift the open-loop (no-IMU) walk cannot correct.
        # + leans toward the RIGHT foot (+Y). Set via base_roll_deg on the tools.
        self.base_roll = 0.0   # radians
        self.base_pitch = 0.0  # radians

        jr = {n: self.model.jnt_range[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, C.JOINT_TO_MJCF[n])]
            for n in self.NAMES}
        # unknowns u: base x, y, z (the torso is kept LEVEL, so base roll/pitch
        # are NOT free -- see _set), then the 10 leg joints.
        self.lo = np.array([-.3, -.3, .15] + [jr[n][0] for n in self.NAMES])
        self.hi = np.array([+.3, +.3, .40] + [jr[n][1] for n in self.NAMES])
        # seed: slight knee bend, both knees NEGATIVE (v4), ankles level.
        self.u0 = np.array([0, 0, .30,
                            0, .10, -.30, -.20, 0,
                            0, -.10, -.30, -.20, 0], float)

    # -- geometry helpers ---------------------------------------------------
    def _sole(self, gname, bname):
        self.mj.mj_resetData(self.model, self.data)
        self.mj.mj_forward(self.model, self.data)
        gid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_GEOM, gname)
        mid = self.model.geom_dataid[gid]
        v = self.model.mesh_vert[self.model.mesh_vertadr[mid]:
                                 self.model.mesh_vertadr[mid]
                                 + self.model.mesh_vertnum[mid]]
        w = v @ self.data.geom_xmat[gid].reshape(3, 3).T + self.data.geom_xpos[gid]
        s = w[w[:, 2] < w[:, 2].min() + 0.002]
        picks = [s[s[:, 0].argmin()], s[s[:, 0].argmax()],
                 s[s[:, 1].argmin()], s[s[:, 1].argmax()]]
        R = self.data.xmat[self.bodies[bname]].reshape(3, 3)
        p = self.data.xpos[self.bodies[bname]]
        return [R.T @ (q - p) for q in picks]

    def _sole_centre_world(self, side):
        pts = self._foot_world(side)
        return pts.mean(axis=0)

    def _foot_world(self, side):
        bid = self.bodies[self.foot_body[side]]
        R = self.data.xmat[bid].reshape(3, 3)
        p = self.data.xpos[bid]
        return np.array([R @ l + p for l in self.sole[side]])

    def _set(self, u):
        self.data.qpos[:3] = u[:3]
        # base_link is level by default (root orientation identity); base_roll
        # bakes in a fixed lateral lean (a trim to cancel open-loop drift). The
        # CoM shift is still taken up by hip_roll + ankle_roll; the torso only
        # tilts by the constant trim, not per-step.
        if self.base_roll or self.base_pitch:
            q = np.zeros(4)
            self.mj.mju_euler2Quat(
                q, np.array([self.base_roll, self.base_pitch, 0.0]), "xyz")
            self.data.qpos[3:7] = q
        else:
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for k, n in enumerate(self.NAMES):
            self.data.qpos[self.qadr[n]] = u[3 + k]
        self.data.qpos[self.qadr["hip_yaw_R"]] = 0.0
        self.data.qpos[self.qadr["hip_yaw_L"]] = 0.0
        self.mj.mj_forward(self.model, self.data)

    # -- the IK -------------------------------------------------------------
    def _residual(self, u, tgt):
        self._set(u)
        r = []
        for side in ("R", "L"):
            tx, ty, tz = tgt[side]
            fw = self._foot_world(side)                     # [heel, toe, edgeY-, edgeY+]
            r += list(fw[:, 2] - tz)                        # flat & at height tz
            r += [(fw[:, 0].mean() - tx) * 2.0,             # sole centre x
                  (fw[:, 1].mean() - ty) * 2.0]             # sole centre y
        com = self.data.subtree_com[0]
        r += [(com[0] - tgt["com"][0]) * 3.0,
              (com[1] - tgt["com"][1]) * 3.0]
        r.append(0.02 * (u[2] - self.z_nom))                # weak height regulariser
        return r

    def solve(self, tgt, u_init=None):
        """Return (12-vector in SIM_ORDER, base height, u, ok)."""
        from scipy.optimize import least_squares

        u = self.u0 if u_init is None else u_init
        sol = least_squares(self._residual, u, bounds=(self.lo, self.hi),
                            args=(tgt,), xtol=1e-12, ftol=1e-12, max_nfev=200)
        ok = max(abs(np.asarray(sol.fun))) < 1e-3
        self._set(sol.x)
        q = np.zeros(len(C.SIM_ORDER), np.float32)
        idx = {j: k for k, j in enumerate(C.SIM_ORDER)}
        for k, n in enumerate(self.NAMES):
            q[idx[n]] = sol.x[3 + k]
        base_h = float(self.data.qpos[2])
        return q, base_h, sol.x, ok

    # -- static-stability check --------------------------------------------
    def support_margin(self, u, tgt):
        """Signed distance (m) from the CoM to the support polygon edge.

        Positive = CoM inside the support (statically stable). The support is the
        convex hull of every planted (height ~ 0) foot's sole points.
        """
        self._set(u)
        pts = []
        for side in ("R", "L"):
            if tgt[side][2] <= 1e-4:               # planted
                pts.append(self._foot_world(side)[:, :2])
        pts = np.vstack(pts)
        com = self.data.subtree_com[0][:2]
        return _point_in_hull_margin(com, pts)


def _point_in_hull_margin(p, pts):
    """Signed distance from p to the convex hull of pts (>0 inside, <0 outside).

    ConvexHull normalises each face to a.x + c <= 0 for interior points, with the
    normal pointing OUT, so a.x + c is the signed distance to that edge. The
    distance to the nearest edge is -max over faces: positive when p is inside.
    """
    from scipy.spatial import ConvexHull

    hull = ConvexHull(pts)
    return float(-np.max(hull.equations[:, :2] @ p + hull.equations[:, 2]))


def build_walk(ik: WalkIK, steps=6, step_len=0.025, lift=0.025, com_shift=0.039,
               t_step=4.0, dt=DT, stance_y=0.055):
    """Per-tick targets for a slow forward walk, then solve IK at each tick.

    One step = shift CoM over the stance foot (25% of t_step) -> lift & swing the
    other foot one step_len ahead of the stance foot (50%) -> settle CoM back
    toward centre (25%). Steps alternate legs; the first is a half-length step so
    both feet end up progressing together.

    Returns (traj [N,12], heights [N], margins [N]).
    """
    def ease(a, b, n):                              # cosine 0->1 ramp
        f = 0.5 * (1 - np.cos(np.pi * np.arange(n) / max(n - 1, 1)))
        return a + (b - a) * f[:, None] if np.ndim(a) else a + (b - a) * f

    # Lateral foot placement. Narrowing the stance below nominal shrinks how far
    # the CoM must travel to sit over the stance foot -- the cheapest way to buy
    # static margin when the ankle's inward (inversion) roll is limited.
    if stance_y is None:
        ynom = ik.y_nom
    else:
        ynom = {s: np.sign(ik.y_nom[s]) * stance_y for s in ("R", "L")}
    xR = xL = 0.0
    com = np.array([0.0, 0.0])
    n_seg = int(round(t_step / dt / 4))            # ticks per quarter-step

    targets = []
    for i in range(steps):
        swing = "L" if i % 2 == 0 else "R"
        stance = "R" if swing == "L" else "L"
        sy = ynom[stance] * (com_shift / abs(ynom[stance]))   # CoM shift toward stance
        # advance: first step is half so the trailing foot only catches up.
        adv = step_len * (0.5 if i == 0 else 1.0)
        new_swing_x = (xR if stance == "R" else xL) + adv

        # planted foot x/y (constant through this step)
        st_x = xR if stance == "R" else xL
        sw_x0 = xL if swing == "L" else xR

        # 1) shift CoM over the stance foot -- ease BOTH x (to the stance foot)
        #    and y here, so it joins the swing phase (which holds [st_x, sy])
        #    with no step. Easing only y left a ~7 mm x jump at this boundary
        #    from step 1 on -> the leg-switch jerk.
        for c in ease(com.copy(), np.array([st_x, sy]), n_seg):
            targets.append((stance, swing, st_x, sw_x0, 0.0, c.copy()))
        com = np.array([st_x, sy])

        # 2) swing: move the swing foot to new_swing_x with a raised-cosine lift
        for k in range(2 * n_seg):
            f = k / (2 * n_seg - 1)
            sx = sw_x0 + (new_swing_x - sw_x0) * (0.5 * (1 - np.cos(np.pi * f)))
            # Raised cosine has zero vertical velocity at lift-off and
            # touchdown.  sin(pi*f) hit the ground with non-zero velocity and
            # produced a visible kick on the real robot.
            sz = lift * 0.5 * (1 - np.cos(2 * np.pi * f))
            cx = st_x                              # keep CoM over the stance foot
            targets.append((stance, swing, st_x, sx, sz,
                            np.array([cx, sy])))
        com = np.array([st_x, sy])
        if swing == "L":
            xL = new_swing_x
        else:
            xR = new_swing_x

        # 3) settle CoM back toward the midline
        for c in ease(com.copy(), np.array([0.5 * (xR + xL), 0.0]), n_seg):
            targets.append((stance, swing, st_x,
                            (xL if swing == "L" else xR), 0.0, c.copy()))
        com = np.array([0.5 * (xR + xL), 0.0])

    # Solve IK tick by tick, warm-started, and record the stability margin.
    traj, heights, margins = [], [], []
    u = None
    n_bad = 0
    for (stance, swing, st_x, sw_x, sw_z, c) in targets:
        tgt = {"com": (c[0], c[1])}
        tgt[stance] = (st_x, ynom[stance], 0.0)
        tgt[swing] = (sw_x, ynom[swing], sw_z)
        q, h, u, ok = ik.solve(tgt, u)
        n_bad += not ok
        traj.append(q)
        heights.append(h)
        margins.append(ik.support_margin(u, tgt))
    if n_bad:
        print(f"  WARNING: IK did not fully converge on {n_bad}/{len(targets)} ticks")
    return np.array(traj, np.float32), np.array(heights), np.array(margins)


def attitude_sensitivities(ik: WalkIK, stance_y: float, eps_deg: float = 2.0):
    """Joint response that asks the torso to roll/pitch while feet stay fixed.

    These two numerical Jacobian columns turn a small desired torso-angle
    correction into a coordinated hip/ankle joint correction.  This is safer
    than guessing ankle signs independently for the mirrored legs.
    """
    eps = math.radians(eps_deg)
    y = {"R": abs(stance_y), "L": -abs(stance_y)}
    tgt = {"com": (0.0, 0.0),
           "R": (0.0, y["R"], 0.0),
           "L": (0.0, y["L"], 0.0)}
    saved = ik.base_roll, ik.base_pitch

    def at(roll, pitch):
        ik.base_roll, ik.base_pitch = roll, pitch
        q, _, _, ok = ik.solve(tgt, None)
        if not ok:
            raise RuntimeError(
                f"could not derive stabilizer response at roll={roll}, pitch={pitch}")
        return q.astype(np.float64)

    try:
        roll = (at(+eps, 0.0) - at(-eps, 0.0)) / (2 * eps)
        pitch = (at(0.0, +eps) - at(0.0, -eps)) / (2 * eps)
    finally:
        ik.base_roll, ik.base_pitch = saved
    return roll.astype(np.float32), pitch.astype(np.float32)


def smooth_retime(traj, heights, margins, dt=DT, max_speed=0.8,
                  max_accel=25.0):
    """C2-smoothly slow a joint path until speed and acceleration are bounded."""
    from scipy.interpolate import CubicSpline

    if len(traj) < 4:
        return traj, heights, margins
    t = np.arange(len(traj), dtype=float) * dt
    spline = CubicSpline(t, traj, axis=0)
    dense = np.linspace(t[0], t[-1], max(1000, len(t) * 4))
    vmax = float(np.max(np.abs(spline(dense, 1))))
    amax = float(np.max(np.abs(spline(dense, 2))))
    scale = max(1.0, vmax / max_speed, math.sqrt(amax / max_accel))
    duration = t[-1] * scale
    tout = np.linspace(0.0, duration, int(math.ceil(duration / dt)) + 1)
    source_t = np.clip(tout / scale, t[0], t[-1])
    q = spline(source_t).astype(np.float32)
    # Numerical splines may overshoot a limit by a fraction of a degree.
    for j, name in enumerate(C.SIM_ORDER):
        q[:, j] = np.clip(q[:, j], *C.LIMITS[name])
    h = np.interp(source_t, t, heights)
    m = np.interp(source_t, t, margins)
    return q, h, m


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", default="sim", help="'sim' or a serial port")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--step-len", type=float, default=0.025, help="stride, m")
    ap.add_argument("--lift", type=float, default=0.025, help="swing-foot lift, m")
    ap.add_argument("--com-shift", type=float, default=0.039,
                    help="lateral CoM shift toward the stance foot, m")
    ap.add_argument("--stance-y", type=float, default=0.055,
                    help="lateral foot placement, m (with the 20 deg ankle roll "
                         "this is wide enough to clear the feet during swing)")
    ap.add_argument("--t-step", type=float, default=4.0, help="seconds per step")
    ap.add_argument("--base-roll", type=float, default=0.0,
                    help="nominal torso roll in degrees (+ toward right)")
    ap.add_argument("--max-speed", type=float, default=0.8,
                    help="smoothed trajectory joint-speed limit, rad/s")
    ap.add_argument("--max-accel", type=float, default=25.0,
                    help="smoothed trajectory joint-acceleration limit, rad/s^2")
    ap.add_argument("--feedback", action=argparse.BooleanOptionalAction,
                    default=None, help="use IMU tilt feedback (default: on for hardware)")
    ap.add_argument("--tilt-kp", type=float, default=0.45)
    ap.add_argument("--tilt-kd", type=float, default=0.10)
    ap.add_argument("--tilt-filter", type=float, default=0.16,
                    help="feedback low-pass time constant, seconds")
    ap.add_argument("--tilt-deadband", type=float, default=0.35,
                    help="ignore smaller roll/pitch errors, degrees")
    ap.add_argument("--max-correction", type=float, default=3.0,
                    help="maximum feedback torso correction, degrees")
    ap.add_argument("--fall-angle", type=float, default=12.0,
                    help="release torque if absolute tilt exceeds this, degrees")
    ap.add_argument("--viewer", action="store_true", help="sim only")
    ap.add_argument("--fast", action="store_true", help="sim only: no wall-clock wait")
    ap.add_argument("--check", action="store_true",
                    help="generate + report stability, don't move anything")
    ap.add_argument("--log", default=None)
    ap.add_argument("--yes", action="store_true", help="skip the hardware prompt")
    args = ap.parse_args(argv)

    print("solving the walk IK ...")
    ik = WalkIK()
    ik.base_roll = math.radians(args.base_roll)
    traj, heights, margins = build_walk(
        ik, steps=args.steps, step_len=args.step_len, lift=args.lift,
        com_shift=args.com_shift, t_step=args.t_step, dt=DT, stance_y=args.stance_y)
    raw_ticks = len(traj)
    traj, heights, margins = smooth_retime(
        traj, heights, margins, DT, args.max_speed, args.max_accel)
    mm = margins * 1000
    print(f"  {len(traj)} ticks ({raw_ticks} IK samples), "
          f"{args.steps} steps of {args.step_len*100:.1f} cm, "
          f"{args.t_step:g}s each")
    print(f"  CoM support margin: min {mm.min():+.1f} mm, mean {mm.mean():+.1f} mm "
          f"({'STATICALLY STABLE throughout' if mm.min() > 0 else 'LEAVES SUPPORT -- not static'})")
    print(f"  base height: {heights.min()*1000:.0f}-{heights.max()*1000:.0f} mm")
    if args.check:
        return 0 if mm.min() > 0 else 1

    real = args.bus != "sim"
    feedback = real if args.feedback is None else args.feedback
    if real and not args.yes:
        print("About to WALK THE REAL ROBOT. Calibrate + system-ID first, and "
              "hold it ready.")
        if input("Type 'go' to continue: ").strip().lower() != "go":
            print("aborted")
            return 1

    if args.bus == "sim":
        bus = k2_bus.SimBus(viewer=args.viewer, realtime=not args.fast)
    else:
        bus = k2_bus.SerialBus(args.bus)
    att = None
    stabilizer = None
    if feedback:
        print("initialising IMU feedback -- keep the robot completely still ...")
        roll_response, pitch_response = attitude_sensitivities(ik, args.stance_y)
        att = SimAttitude(bus) if not real else ImuAttitude()
        stabilizer = TiltStabilizer(
            roll_response, pitch_response,
            roll_target=math.radians(args.base_roll),
            kp=args.tilt_kp, kd=args.tilt_kd,
            filter_tau=args.tilt_filter,
            deadband=math.radians(args.tilt_deadband),
            max_correction=math.radians(args.max_correction),
            fall_angle=math.radians(args.fall_angle))

    def limp(*_):
        print("\ninterrupted -> releasing torque")
        try:
            bus.torque(False)
        finally:
            bus.close()
            if att is not None:
                att.close()
        raise SystemExit(130)
    signal.signal(signal.SIGINT, limp)

    with bus:
        try:
            bus.torque(True)
            k2_ctrl.run(bus, traj, heights, DT, log=args.log,
                        attitude=att, stabilizer=stabilizer)
        except UnsafeTiltError as exc:
            print(f"SAFETY STOP: {exc}")
            return 2
        finally:
            bus.torque(False)
            if att is not None:
                att.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
