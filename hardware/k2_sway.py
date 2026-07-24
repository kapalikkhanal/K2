"""Lateral weight-shift test: can K2 get its CoM over one foot?

This is the make-or-break question for SLOW (quasi-static) walking. A statically
stable gait only exists if the robot can park its centre of mass inside the
stance foot's sole while the other foot is in the air. If it cannot, no amount
of slowing down helps -- the robot must fall and catch itself, which is a
dynamic gait and needs speed the STS3215 does not have.

On v3 the model said K2 could NOT: max lateral CoM shift was ~10 mm vs a stance
inner edge at 60 mm, the binding constraint being hip_roll ADDUCTION (12 deg vs
45 of abduction) on the swing leg, with no ankle roll to make up the difference.
Robot_v4 ADDS an ankle roll (+-20 deg) on each leg and widened adduction to
22 deg -- this test now includes the stance ankle roll, so re-run it to measure
whether one-foot support is finally reachable. See [[k2-cannot-walk-statically]].

This script drives that prediction on hardware so it is measured, not assumed:

    python -m hardware.k2_sway --bus sim --viewer          # twin first
    python -m hardware.k2_sway --bus /dev/ttyACM0          # then real

    --lift MM   after swaying, raise the swing foot this far (default 0 = sway
                only). HOLD THE ROBOT before using --lift; if the model is
                right it will topple toward the swing side.

Poses are solved against the MJCF with the stance foot pinned flat, so they stay
correct if the CAD changes.

The Pi has no MuJoCo (see k2_kinematics.py for why), so the IK is baked into a
small .npz on a machine that does, exactly like the kinematics npz:

    python -m hardware.k2_sway --bake sway_R.npz --stance R --lift 20   # PC
    python -m hardware.k2_sway --bus /dev/ttyACM0 --poses sway_R.npz    # Pi

--bake stores the solved trajectory AND the reach-vs-required verdict, so the
Pi reprints the same numbers it would have computed itself.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import k2_bus
from . import k2_conventions as C
from . import k2_ctrl

SWAY_S = 4.0        # time to sway out
HOLD_S = 2.0        # dwell at full sway
DT = 0.02           # 50 Hz, same as every other K2 loop


class SwayIK:
    """Lateral weight-shift poses, stance foot pinned flat on the ground."""

    def __init__(self, stance: str = "R"):
        import mujoco

        self.mj = mujoco
        self.stance = stance
        self.model = mujoco.MjModel.from_xml_path(str(C.MJCF))
        self.data = mujoco.MjData(self.model)
        self.qadr = {j: self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, m)]
            for j, m in C.JOINT_TO_MJCF.items()}

        self.st_body = "Feet_right" if stance == "R" else "Feet_left"
        self.sw_body = "Feet_left" if stance == "R" else "Feet_right"
        st_geom = f"{'right' if stance == 'R' else 'left'}_foot_collision"
        sw_geom = f"{'left' if stance == 'R' else 'right'}_foot_collision"
        self.bodies = {b: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
                       for b in (self.st_body, self.sw_body)}

        self.st_pts, st_w = self._sole(st_geom, self.st_body)
        self.sw_pts, _ = self._sole(sw_geom, self.sw_body)
        # Where the stance foot must stay, and how far the CoM may stray.
        self.pin = np.array([st_w[:, 0].mean(), st_w[:, 1].mean()])
        self.half_width = 0.5 * (st_w[:, 1].max() - st_w[:, 1].min())

        # v4: the ankle is now pitch + roll. Give the IK the new ankle_roll on
        # BOTH legs -- letting the stance ankle roll is exactly the extra DOF
        # that was added to buy the lateral CoM shift this test measures.
        self.names = ["hip_roll_R", "hip_pitch_R", "knee_R", "ankle_pitch_R", "ankle_roll_R",
                      "hip_roll_L", "hip_pitch_L", "knee_L", "ankle_pitch_L", "ankle_roll_L"]
        jr = {n: self.model.jnt_range[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, C.JOINT_TO_MJCF[n])]
            for n in self.names}
        self.lo = np.array([-.3, -.3, .15, -1.2, -.8] + [jr[n][0] for n in self.names])
        self.hi = np.array([+.3, +.3, .40, +1.2, +.8] + [jr[n][1] for n in self.names])
        # base(5): x, y, z, roll, pitch ; then the 10 leg joints in `names`
        # order. v4: both knees flex NEGATIVE (-.44), each ankle_roll seeds 0.
        self.u0 = np.array([0, 0, .32, 0, 0,
                            0, .13, -.44, -.30, 0,
                            0, -.13, -.44, -.30, 0])

    def _sole(self, gname, bname):
        """Four sole points (heel, toe, both lateral edges) in the ankle frame.

        Four points, not two: they pin the sole's roll as well as its pitch,
        which the squat never needed but a lateral shift very much does.
        """
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
        return [R.T @ (q - p) for q in picks], s

    def _set(self, u):
        self.data.qpos[:3] = u[:3]
        q = np.zeros(4)
        self.mj.mju_euler2Quat(q, np.array([u[3], u[4], 0.0]), "xyz")
        self.data.qpos[3:7] = q
        for k, n in enumerate(self.names):
            self.data.qpos[self.qadr[n]] = u[5 + k]
        self.data.qpos[self.qadr["hip_yaw_R"]] = 0.0
        self.data.qpos[self.qadr["hip_yaw_L"]] = 0.0
        self.mj.mj_forward(self.model, self.data)

    def _world(self, bname, locals_):
        R = self.data.xmat[self.bodies[bname]].reshape(3, 3)
        p = self.data.xpos[self.bodies[bname]]
        return np.array([R @ l + p for l in locals_])

    def _residual(self, u, com_y, lift):
        self._set(u)
        st = self._world(self.st_body, self.st_pts)
        r = list(st[:, 2])                                   # stance sole flat
        r += [(st[:, 0].mean() - self.pin[0]) * 3.0,         # stance foot pinned
              (st[:, 1].mean() - self.pin[1]) * 3.0]
        r.append(min(self._world(self.sw_body, self.sw_pts)[:, 2]) - lift)
        com = self.data.subtree_com[0]
        r += [(com[0] - self.pin[0]) * 3.0, (com[1] - com_y) * 3.0]
        return r

    def solve(self, com_y: float, lift: float, u_init=None):
        """Pose with the CoM at lateral `com_y` and the swing foot at `lift`.

        Returns (joint vector in SIM_ORDER, achieved CoM y, ok).
        """
        from scipy.optimize import least_squares

        u = self.u0 if u_init is None else u_init
        sol = least_squares(self._residual, u, bounds=(self.lo, self.hi),
                            args=(com_y, lift), xtol=1e-13, ftol=1e-13)
        ok = max(abs(np.asarray(sol.fun))) < 1e-4
        self._set(sol.x)
        self.base_z = float(sol.x[2])       # for the run loop's height readout
        q = np.zeros(len(C.SIM_ORDER), np.float32)
        idx = {j: k for k, j in enumerate(C.SIM_ORDER)}
        for k, n in enumerate(self.names):
            q[idx[n]] = sol.x[5 + k]
        return q, float(self.data.subtree_com[0][1]), ok, sol.x

    def max_shift(self, lift: float, step: float = 0.0025):
        """Sweep the CoM outward until the IK stops converging.

        The last feasible pose is the furthest this robot can lean, which is
        exactly the number that decides whether a slow gait exists. Everything
        is reported as a magnitude so the two stance sides are comparable: the
        left foot sits at negative y, so its shift and its target are negative.
        """
        side = 1.0 if self.pin[1] >= 0 else -1.0
        need = abs(self.pin[1]) - self.half_width      # inner edge, as a distance
        best_q, best_reach, u = None, 0.0, None
        for mag in np.arange(0.0, abs(self.pin[1]) + 0.03, step):
            q, y, ok, u_new = self.solve(side * mag, lift, u)
            if not ok:
                break
            best_q, best_reach, u = q, side * y, u_new
        return best_q, best_reach, need


def _solve(args):
    """Solve the sway trajectory. Needs MuJoCo, so it never runs on the Pi."""
    lift = args.lift / 1000.0
    ik = SwayIK(args.stance)
    q_stand, _, _, _ = ik.solve(0.0, 0.0)
    z_stand = ik.base_z
    q_sway, reach_signed, need = ik.max_shift(lift)
    z_sway = ik.base_z
    reach = abs(reach_signed)          # compare as distances toward the stance foot

    lines = [
        f"stance foot   : {args.stance}, sole centre y = {ik.pin[1]*1e3:6.1f} mm, "
        f"half-width {ik.half_width*1e3:.1f} mm",
        f"CoM must reach: {need*1e3:6.1f} mm toward the stance foot to be "
        f"statically stable on it",
        f"CoM can reach : {reach*1e3:6.1f} mm  (swing foot lift {args.lift:.0f} mm)",
    ]
    if reach >= need:
        lines.append("=> STATICALLY STABLE on one foot: a slow quasi-static gait exists.")
    else:
        lines.append(f"=> SHORT BY {(need-reach)*1e3:.1f} mm. No statically stable "
                     f"single support => no slow gait on this geometry.")
    if args.lift > 0:
        side = "left" if args.stance == "R" else "right"
        lines += ["", "!! HOLD THE ROBOT. If the verdict above says SHORT, it will",
                  f"!! topple toward the {side} as the foot leaves the ground."]

    # Ease out to the sway pose, dwell, come back -- a smooth there-and-back so
    # a failure shows up as a topple, not as a snap.
    n_out, n_hold = int(SWAY_S / DT), int(HOLD_S / DT)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(n_out) / n_out))
    traj = np.concatenate([
        q_stand + (q_sway - q_stand) * ramp[:, None],
        np.repeat(q_sway[None], n_hold, axis=0),
        q_sway + (q_stand - q_sway) * ramp[:, None],
    ])
    heights = np.concatenate([
        z_stand + (z_sway - z_stand) * ramp,
        np.full(n_hold, z_sway),
        z_sway + (z_stand - z_sway) * ramp,
    ])
    return traj, heights, "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", help="'sim' or a serial port (omit with --bake)")
    ap.add_argument("--stance", choices=("R", "L"), default="R")
    ap.add_argument("--lift", type=float, default=0.0,
                    help="swing-foot lift in MM (default 0 = sway only). "
                         "HOLD THE ROBOT if you use this.")
    ap.add_argument("--bake", metavar="NPZ",
                    help="solve the IK and write the trajectory here (needs MuJoCo)")
    ap.add_argument("--poses", metavar="NPZ",
                    help="play a baked trajectory instead of solving (for the Pi)")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)

    if args.poses:
        d = np.load(args.poses)
        traj, heights = d["traj"], d["heights"]
        verdict = str(d["verdict"])
        print(verdict)
    else:
        traj, heights, verdict = _solve(args)
        print(verdict)
        if args.bake:
            np.savez(args.bake, traj=traj, heights=heights, verdict=verdict)
            print(f"\nbaked {len(traj)} ticks -> {args.bake}")
            return 0

    if not args.bus:
        raise SystemExit("--bus is required unless you are only --bake-ing")

    if args.bus == "sim":
        bus = k2_bus.SimBus(viewer=args.viewer)
    else:
        bus = k2_bus.SerialBus(args.bus)
    with bus:
        try:
            bus.torque(True)
            k2_ctrl.run(bus, traj, heights, DT, log=args.log)
        finally:
            bus.torque(False)      # always go limp, including on Ctrl-C
    return 0


if __name__ == "__main__":
    sys.exit(main())
