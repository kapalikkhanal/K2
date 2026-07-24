"""Scripted motions for K2, generated from the model rather than hand-tuned.

The squat is solved as inverse kinematics on the MJCF: for a given knee bend,
find the hip and ankle angles that keep both soles flat on the ground, the
feet level with each other, and the centre of mass over the middle of the
support polygon. Solving against the model means the trajectory stays correct
if the CAD changes.

Only the sagittal joints move (hip_pitch, knee, ankle_pitch); hip_roll, hip_yaw
and the v4 ankle_roll all stay at zero.
"""

from __future__ import annotations

import numpy as np

from . import k2_conventions as C

# Solved with both feet planted, so this needs no IMU: the ground closes the
# kinematic chain and the encoders alone determine the base pose.
#
# Robot_v4 (2026-07-21): the squat is now limited by the ANKLE PITCH, not the
# knee. With the conservative symmetric +-30 deg ankle_pitch limit chosen for
# early RL, the shin cannot lean far enough over a flat foot, so the feet-flat
# squat binds at ~35 deg knee for only ~8 mm of base drop (v3 got ~90 mm with a
# +-60 deg ankle). To restore a deep squat, widen ANKLE_PITCH_MAX in
# scripts/phase1_mass.py (the v4 CAD clears +50 deg dorsiflexion) and re-run
# phase1 -> phase2 -> C.write_joint_limits().
MAX_KNEE_DEG = 35.0

# Reachable base-height envelope (v4, +-30 deg ankle_pitch). Re-derive if the
# ankle limit changes.
MAX_BASE_H = 0.335    # standing
MIN_BASE_H = 0.327    # deepest feet-flat squat at MAX_KNEE_DEG


def _model():
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(C.MJCF))
    return model, mujoco.MjData(model)


class SquatIK:
    """Flat-foot double-support squat poses."""

    def __init__(self):
        import mujoco
        self.mj = mujoco
        self.model, self.data = _model()
        self.qadr = {j: self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, m)]
            for j, m in C.JOINT_TO_MJCF.items()}
        self.sites = {s: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, s)
                      for s in ("right_foot", "left_foot")}
        # v4: the sole moved off the ankle onto the new ankle-roll "Feet" body.
        self.bodies = {b: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
                       for b in ("Feet_right", "Feet_left", "base_link")}
        self._heel_toe = {
            "Feet_right": self._sole_points("right_foot_collision", "Feet_right"),
            "Feet_left": self._sole_points("left_foot_collision", "Feet_left"),
        }

    def _sole_points(self, gname, bname):
        """Heel and toe of the sole, in that ankle's own frame.

        Two explicit points turn 'sole on the ground' and 'sole flat' into
        plain z==0 constraints, with no assumption about which body axis
        points forward -- the ankle's local X actually runs along world -Y.
        """
        self.mj.mj_resetData(self.model, self.data)
        self.mj.mj_forward(self.model, self.data)
        gid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_GEOM, gname)
        bid = self.bodies[bname]
        mid = self.model.geom_dataid[gid]
        v = self.model.mesh_vert[self.model.mesh_vertadr[mid]:
                                 self.model.mesh_vertadr[mid]
                                 + self.model.mesh_vertnum[mid]]
        w = v @ self.data.geom_xmat[gid].reshape(3, 3).T + self.data.geom_xpos[gid]
        sole = w[w[:, 2] < w[:, 2].min() + 0.002]
        R, p = self.data.xmat[bid].reshape(3, 3), self.data.xpos[bid]
        return (R.T @ (sole[sole[:, 0].argmin()] - p),
                R.T @ (sole[sole[:, 0].argmax()] - p))

    def _pose(self, theta, u):
        h_R, a_R, h_L, a_L, z, x = u
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[0], self.data.qpos[2] = x, z
        self.data.qpos[self.qadr["hip_pitch_R"]] = h_R
        self.data.qpos[self.qadr["knee_R"]] = -theta
        self.data.qpos[self.qadr["ankle_pitch_R"]] = a_R
        self.data.qpos[self.qadr["hip_pitch_L"]] = h_L
        self.data.qpos[self.qadr["knee_L"]] = -theta   # v4: both knees flex NEGATIVE
        self.data.qpos[self.qadr["ankle_pitch_L"]] = a_L
        self.mj.mj_forward(self.model, self.data)

    def _world_z(self, bname, local):
        bid = self.bodies[bname]
        return (self.data.xmat[bid].reshape(3, 3) @ local + self.data.xpos[bid])[2]

    def _residual(self, u, theta):
        self._pose(theta, u)
        hr, tr = self._heel_toe["Feet_right"]
        hl, tl = self._heel_toe["Feet_left"]
        xr = self.data.site_xpos[self.sites["right_foot"]][0]
        xl = self.data.site_xpos[self.sites["left_foot"]][0]
        return [
            self._world_z("Feet_right", hr), self._world_z("Feet_right", tr),
            self._world_z("Feet_left", hl), self._world_z("Feet_left", tl),
            xr - xl,                                   # feet aligned fore/aft
            self.data.subtree_com[0][0] - 0.5 * (xr + xl),   # CoM centred
        ]

    def solve(self, knee_deg: float) -> tuple[np.ndarray, float]:
        """Return (joint vector in SIM_ORDER, resulting base height)."""
        from scipy.optimize import least_squares

        theta = np.radians(knee_deg)
        u0 = np.array([0.0, 0.0, 0.0, 0.0, 0.3368, 0.0])
        sol = least_squares(self._residual, u0, args=(theta,),
                            xtol=1e-12, ftol=1e-12)
        if max(abs(np.asarray(sol.fun))) > 1e-3:
            raise RuntimeError(
                f"squat IK failed at {knee_deg} deg "
                f"(residual {max(abs(sol.fun)):.2e})")
        self._pose(theta, sol.x)
        h_R, a_R, h_L, a_L = sol.x[:4]
        q = np.zeros(len(C.SIM_ORDER), np.float32)
        idx = {j: k for k, j in enumerate(C.SIM_ORDER)}
        q[idx["hip_pitch_R"]], q[idx["knee_R"]], q[idx["ankle_pitch_R"]] = h_R, -theta, a_R
        q[idx["hip_pitch_L"]], q[idx["knee_L"]], q[idx["ankle_pitch_L"]] = h_L, -theta, a_L
        # Clamping here would silently return a pose whose feet are no longer
        # flat on the ground, which would quietly invalidate the whole
        # encoder-only base estimate. Refuse instead.
        # 1e-4 rad = 0.006 deg of slack: q is float32, so a pose solved
        # exactly at a limit can land a few ULPs outside it.
        tol = 1e-4
        over = {j: np.degrees(q[k]) for j, k in idx.items()
                if not (C.LIMITS[j][0] - tol <= q[k] <= C.LIMITS[j][1] + tol)}
        if over:
            raise RuntimeError(
                f"squat at {knee_deg:.0f} deg knee needs joints outside their "
                f"limits: {', '.join(f'{j} {v:+.1f} deg' for j, v in over.items())}. "
                f"Reduce the depth, or widen the limit if the mechanism allows it.")
        return q, float(self.data.xpos[self.bodies["base_link"]][2])


def squat_trajectory(depth_deg: float = 45.0, period_s: float = 4.0,
                     cycles: float = 3.0, rate_hz: float = 50.0,
                     samples: int = 25):
    """Smooth up/down squat as an array of joint vectors, one per control tick.

    IK is solved at `samples` knee angles and interpolated, which is far
    cheaper than solving per tick and just as accurate for a smooth motion.
    """
    depth_deg = float(np.clip(depth_deg, 0.0, MAX_KNEE_DEG))
    ik = SquatIK()
    knots = np.linspace(0.0, depth_deg, samples)
    poses = np.stack([ik.solve(k)[0] for k in knots])
    heights = np.array([ik.solve(k)[1] for k in knots])

    n = int(round(cycles * period_s * rate_hz))
    t = np.arange(n) / rate_hz
    # Cosine ease: starts and ends at rest, no velocity step at the turnaround.
    phase = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / period_s))
    knee = phase * depth_deg
    traj = np.stack([np.interp(knee, knots, poses[:, k])
                     for k in range(poses.shape[1])], axis=1)
    return traj.astype(np.float32), np.interp(knee, knots, heights)
