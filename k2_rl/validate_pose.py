"""Validate the K2 default crouch keyframe in MuJoCo.

Drops the robot (actuated, holding the default pose) and checks:
  * the soles rest flat on the floor at the pinned base height,
  * the pose is left/right symmetric,
  * it settles without the base collapsing.

Run: python -m k2_rl.validate_pose
"""

from __future__ import annotations

import numpy as np

from mjlab.entity.entity import Entity

from k2_rl import k2_constants as K


def main() -> None:
  robot = Entity(K.get_k2_robot_cfg())
  model = robot.spec.compile()
  data = __import__("mujoco").MjData(model)
  mj = __import__("mujoco")

  # Apply the default keyframe pose.
  cfg = K.CROUCH_KEYFRAME
  # Free joint: [x y z qw qx qy qz].
  data.qpos[:3] = cfg.pos
  data.qpos[3:7] = (1, 0, 0, 0)

  # Set each joint by resolving the regex against joint names.
  import re

  for jnt_i in range(model.njnt):
    name = model.joint(jnt_i).name
    if model.joint(jnt_i).type[0] == mj.mjtJoint.mjJNT_FREE:
      continue
    qadr = model.joint(jnt_i).qposadr[0]
    for pat, val in cfg.joint_pos.items():
      if re.fullmatch(pat, name):
        data.qpos[qadr] = val
        break

  mj.mj_forward(model, data)

  # Report foot heights (lowest sole point) at the pinned base height.
  def site_z(site_name: str) -> float:
    sid = model.site(site_name).id
    return float(data.site_xpos[sid][2])

  print(f"Base z (pinned)      : {K.CROUCH_BASE_HEIGHT:.4f} m")
  print(f"left_foot  site z    : {site_z('left_foot'):.4f} m")
  print(f"right_foot site z    : {site_z('right_foot'):.4f} m")

  # Foot collision geom lowest z (what actually contacts the floor).
  for gname in K.FOOT_GEOMS:
    gid = model.geom(gname).id
    # AABB-ish: geom center z minus half extent along z is approximate for a
    # mesh; use the contact-relevant site instead. Report geom center z.
    print(f"{gname:22s} center z: {float(data.geom_xpos[gid][2]):.4f} m")

  # Symmetry check on the feet (x should match, y should be mirrored).
  lz, rz = site_z("left_foot"), site_z("right_foot")
  print(f"foot z asymmetry     : {abs(lz - rz) * 1000:.2f} mm")

  # Now settle under gravity holding the default pose, report base height drift.
  # Build ctrl target = default joint pos.
  ctrl = np.zeros(model.nu)
  for act_i in range(model.nu):
    jname = model.joint(model.actuator(act_i).trnid[0]).name
    for pat, val in cfg.joint_pos.items():
      if re.fullmatch(pat, jname):
        ctrl[act_i] = val
        break
  data.ctrl[:] = ctrl

  z0 = data.qpos[2]
  for _ in range(2000):  # ~4 s at 2 ms
    mj.mj_step(model, data)
  zf = data.qpos[2]
  # Roll/pitch of base after settling.
  quat = data.qpos[3:7]
  # projected gravity tilt angle.
  R = np.zeros(9)
  mj.mju_quat2Mat(R, quat)
  R = R.reshape(3, 3)
  up = R[2, 2]
  tilt_deg = np.degrees(np.arccos(np.clip(up, -1, 1)))
  print(f"\nAfter 4 s settling:")
  print(f"  base z: {z0:.4f} -> {zf:.4f} m  (drift {abs(zf - z0) * 1000:.1f} mm)")
  print(f"  base tilt from vertical: {tilt_deg:.2f} deg")
  print(f"  nu (actuators): {model.nu},  njnt: {model.njnt}")

  ok = tilt_deg < 15 and abs(zf - z0) < 0.05
  print("\nRESULT:", "OK -- stable crouch" if ok else "CHECK -- pose unstable")


if __name__ == "__main__":
  main()
