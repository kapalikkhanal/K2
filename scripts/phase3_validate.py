#!/usr/bin/env python3
"""Phase 3: sanity-check the compiled K2 MJCF before any RL work.

Reports the mass budget, stance geometry, joint table and static torque
demand, then drops the robot under gravity to confirm it settles on its feet
instead of falling through or exploding.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MJCF = ROOT / "mjcf" / "k2_physics.xml"

STS3215_EFFORT = 2.94  # N.m


def main():
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print("=== model ===")
    print(f"nq={model.nq}  nv={model.nv}  bodies={model.nbody}  "
          f"geoms={model.ngeom}  sensors={model.nsensor}")
    total = model.body_subtreemass[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "k2_root")]
    print(f"total mass: {total * 1000:.1f} g")

    print("\n=== stance ===")
    sites = {}
    for name in ("right_foot", "left_foot", "imu"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        sites[name] = data.site_xpos[sid].copy()
        print(f"  {name:<11} {sites[name].round(4)}")
    width = abs(sites["right_foot"][1] - sites["left_foot"][1])
    print(f"  stance width : {width * 1000:.0f} mm")
    print(f"  hip height   : {sites['imu'][2] * 1000:.0f} mm")

    com = data.subtree_com[0]
    print(f"  CoM          : {com.round(4)}  "
          f"(height {com[2] * 1000:.0f} mm)")
    # For static balance the CoM must project inside the support polygon.
    span_x = sorted(s[0] for s in (sites['right_foot'], sites['left_foot']))
    print(f"  CoM offset from foot centre: "
          f"x {(com[0] - np.mean(span_x)) * 1000:+.0f} mm")

    print("\n=== joints ===")
    print(f"{'joint':<34}{'lower':>9}{'upper':>9}   axis (world)")
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        lo, hi = np.degrees(model.jnt_range[j])
        axis = data.xaxis[j].round(2)
        print(f"{name:<34}{lo:>8.1f}°{hi:>8.1f}°   {axis}")

    # The shipped MJCF has no actuators (mjlab supplies them), and with
    # feet-only collision a passive robot just folds up and drops its hip
    # through the floor. So hold the joints with the same PD gains the RL env
    # will use and watch what the servos actually have to do.
    # A free-floating biped with no balance policy topples no matter how good
    # the model is, which tells us nothing. Constrain the root to slide
    # vertically ("on a gantry") so the test isolates what we care about here:
    # do the feet make contact and carry the weight without tunnelling, and how
    # much torque does that cost.
    held = with_position_servos(MJCF, gantry=True)
    hdata = mujoco.MjData(held)

    for label in ("straight",):
        mujoco.mj_resetData(held, hdata)
        hdata.qpos[0] = 0.05  # drop from 5 cm up
        hdata.ctrl[:] = hdata.qpos[1:]
        for _ in range(int(2.0 / held.opt.timestep)):
            mujoco.mj_step(held, hdata)

        print(f"\n=== drop onto feet, root on vertical slide ({label}) ===")
        if not np.all(np.isfinite(hdata.qpos)):
            print("  UNSTABLE: non-finite state")
            continue
        print(f"  root settled {hdata.qpos[0] * 1000:+.1f} mm from spawn"
              f"   contacts {hdata.ncon}")
        for site in ("right_foot", "left_foot"):
            sid = mujoco.mj_name2id(held, mujoco.mjtObj.mjOBJ_SITE, site)
            print(f"    {site:<11} z = {hdata.site_xpos[sid][2] * 1000:+.1f} mm")
        if hdata.ncon == 0:
            print("  WARNING: no foot contact")

        worst, worst_name = 0.0, ""
        for a in range(held.nu):
            name = mujoco.mj_id2name(held, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            tau = abs(hdata.actuator_force[a])
            if tau > worst:
                worst, worst_name = tau, name
            if tau > STS3215_EFFORT:
                print(f"    {name:<34}{tau:5.2f} N.m  OVER STALL")
        print(f"  peak servo torque: {worst:.2f} / {STS3215_EFFORT} N.m "
              f"({worst / STS3215_EFFORT * 100:.0f}%) at {worst_name}")


def with_position_servos(path: Path, kp: float = 20.0, kd: float = 0.5,
                         gantry: bool = False):
    """Recompile the MJCF with a position servo on every hinge.

    Validation-only: the shipped file stays actuator-free so mjlab owns the
    actuator model. With `gantry`, the free joint is swapped for a vertical
    slide so the robot cannot topple during the contact check.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    mj = tree.getroot()

    if gantry:
        for body in mj.iter("body"):
            fj = body.find("freejoint")
            if fj is not None:
                body.remove(fj)
                ET.SubElement(body, "joint", name="gantry", type="slide",
                              axis="0 0 1", limited="false")

    act = ET.SubElement(mj, "actuator")
    for body in mj.iter("body"):
        for joint in body.findall("joint"):
            name = joint.get("name")
            if name and name != "gantry":  # the gantry rail is passive
                ET.SubElement(act, "position", name=name, joint=name,
                              kp=str(kp), kv=str(kd),
                              forcerange=f"{-STS3215_EFFORT} {STS3215_EFFORT}")
    return mujoco.MjModel.from_xml_string(
        ET.tostring(mj, encoding="unicode"),
        {p.name: p.read_bytes() for p in (ROOT / "meshes" / "K2").iterdir()
         if p.suffix.lower() in (".obj", ".stl", ".mtl")})


if __name__ == "__main__":
    main()
