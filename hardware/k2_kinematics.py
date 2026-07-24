"""Pure-numpy forward kinematics for K2, so the Pi needs no MuJoCo.

The only kinematic quantity the deployed policy needs is base height above the
soles, which it computes from the encoder angles alone (valid while the feet
stay flat -- the whole premise of the no-IMU squat). Installing MuJoCo on a Pi
just to evaluate that one number is not worth it, so this module reproduces
MuJoCo's body kinematics in numpy.

The chain constants (body tree, joint axes, sole site offsets) are baked into a
small .npz by `build_kinematics_npz()` on a machine that has MuJoCo, and the
evaluator here loads only numpy. `build_kinematics_npz()` also verifies the
numpy FK against mj_forward to machine precision before writing, so a shipped
npz is known to match the simulator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import k2_conventions as C

NPZ_PATH = Path(__file__).resolve().parent / "k2_kinematics.npz"


def _quat_mul(a, b):
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ])


def _quat2mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _axis_angle_quat(axis, angle):
    h = 0.5 * angle
    s = np.sin(h)
    return np.array([np.cos(h), axis[0] * s, axis[1] * s, axis[2] * s])


class Kinematics:
    """Base height above the soles from the 10 joint angles, numpy only."""

    def __init__(self, npz_path: Path = NPZ_PATH):
        d = np.load(npz_path)
        self.parent = d["parent"]
        self.body_pos = d["body_pos"]
        self.body_quat = d["body_quat"]
        self.jnt_body = d["jnt_body"]      # body id per hinge, aligned to SIM_ORDER
        self.jnt_axis = d["jnt_axis"]
        self.jnt_pos = d["jnt_pos"]
        self.base_body = int(d["base_body"])
        self.sole_body = d["sole_body"]
        self.sole_pos = d["sole_pos"]
        self.nbody = len(self.parent)
        # SIM_ORDER index -> row in the joint arrays.
        self.order = [list(d["jnt_names"]).index(C.JOINT_TO_MJCF[j])
                      for j in C.SIM_ORDER]

    def _fk(self, q: np.ndarray):
        """World position/orientation of every body, base pinned upright."""
        xpos = np.zeros((self.nbody, 3))
        xmat = np.zeros((self.nbody, 3, 3))
        xquat = np.zeros((self.nbody, 4))
        xpos[0], xquat[0], xmat[0] = 0.0, [1, 0, 0, 0], np.eye(3)

        # Map body id -> its hinge angle (0 for bodies without one).
        angle = {}
        for k, j in enumerate(C.SIM_ORDER):
            angle[int(self.jnt_body[self.order[k]])] = float(q[k])

        for b in range(1, self.nbody):
            p = self.parent[b]
            pos = xpos[p] + xmat[p] @ self.body_pos[b]
            quat = _quat_mul(xquat[p], self.body_quat[b])
            row = np.where(self.jnt_body == b)[0]
            if len(row) and b in angle:
                r = int(row[0])
                mat = _quat2mat(quat)
                anchor = pos + mat @ self.jnt_pos[r]
                quat = _quat_mul(quat, _axis_angle_quat(self.jnt_axis[r],
                                                        angle[b]))
                mat = _quat2mat(quat)
                pos = anchor - mat @ self.jnt_pos[r]
            xpos[b], xquat[b], xmat[b] = pos, quat, _quat2mat(quat)
        return xpos, xmat

    def base_height(self, q: np.ndarray) -> float:
        xpos, xmat = self._fk(q)
        soles = [xpos[bid] + xmat[bid] @ off
                 for bid, off in zip(self.sole_body, self.sole_pos)]
        return float(xpos[self.base_body][2] - min(s[2] for s in soles))


def build_kinematics_npz(out: Path = NPZ_PATH, verify: bool = True) -> None:
    """Extract the chain from the MJCF and verify numpy FK == MuJoCo."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(C.MJCF))
    jnt_names, jnt_body, jnt_axis, jnt_pos = [], [], [], []
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        jnt_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))
        jnt_body.append(m.jnt_bodyid[j])
        jnt_axis.append(m.jnt_axis[j].copy())
        jnt_pos.append(m.jnt_pos[j].copy())

    soles = ("right_foot_heel", "right_foot_toe",
             "left_foot_heel", "left_foot_toe")
    sole_body = [m.site_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)]
                 for s in soles]
    sole_pos = [m.site_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)].copy()
                for s in soles]

    np.savez(
        out,
        parent=m.body_parentid.copy(),
        body_pos=m.body_pos.copy(),
        body_quat=m.body_quat.copy(),
        jnt_names=np.array(jnt_names),
        jnt_body=np.array(jnt_body),
        jnt_axis=np.array(jnt_axis),
        jnt_pos=np.array(jnt_pos),
        base_body=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link"),
        sole_body=np.array(sole_body),
        sole_pos=np.array(sole_pos),
    )
    print(f"wrote {out}")

    if verify:
        kin = Kinematics(out)
        data = mujoco.MjData(m)
        qadr = {j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                                   C.JOINT_TO_MJCF[j])]
                for j in C.SIM_ORDER}
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        sids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in soles]
        worst = 0.0
        rng = np.random.default_rng(0)
        for _ in range(2000):
            q = np.array([rng.uniform(*C.LIMITS[j]) for j in C.SIM_ORDER])
            mujoco.mj_resetData(m, data)
            for j in C.SIM_ORDER:
                data.qpos[qadr[j]] = q[C.SIM_ORDER.index(j)]
            mujoco.mj_forward(m, data)
            ref = data.xpos[bid][2] - min(data.site_xpos[s][2] for s in sids)
            worst = max(worst, abs(kin.base_height(q) - ref))
        print(f"numpy FK vs MuJoCo, 2000 random poses: max |err| = {worst:.3e} m")
        if worst > 1e-9:
            raise SystemExit("numpy FK does not match MuJoCo")
        print("numpy FK matches MuJoCo.")


if __name__ == "__main__":
    build_kinematics_npz()
