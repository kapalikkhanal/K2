#!/usr/bin/env python3
"""Phase 2: convert the mass-corrected K2 URDF into a MuJoCo MJCF.

MuJoCo compiles the URDF directly, so this script fixes up the parts a URDF
cannot express and the parts Fusion got wrong:

  * package:// mesh paths -> a meshdir relative to mjcf/.
  * Y-up (Fusion) -> Z-up (MuJoCo/mjlab). The whole tree is nested under a
    world-aligned root body carrying the free joint, so the floating base's
    own frame is Z-up and gravity/heading observations come out right without
    baking a rotation into every link.
  * Feet-only collision, so the legs cannot self-snag on their own shells.
  * imu site on base_link, foot sites at the ankle soles, and the matching
    sensors.

No actuators or keyframe are emitted; mjlab adds those from its own config.

Writes mjcf/k2_physics.xml.
"""

from __future__ import annotations

import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "urdf" / "K2_phase1.urdf"
DST = ROOT / "mjcf" / "k2_physics.xml"
MESHDIR = "../meshes/K2"

# Fusion exports Y-up. This quaternion maps the robot's +Y onto world +Z and
# its +Z onto world +X, so the robot stands upright and faces world +X.
BASE_QUAT = "0.5 0.5 0.5 0.5"

# Robot_v4: the sole/collision moved off the old single "Ankle" onto the new
# ankle-roll "Feet" link (the Ankle link is now just the pitch bracket).
FOOT_LINKS = {"Feet_right": "right_foot", "Feet_left": "left_foot"}

# Ground clearance of the soles at the spawn pose, metres.
SPAWN_CLEARANCE = 0.002

# Orientation of the physical LSM6DS3 on the base. On Robot_v3 (2026-07-20) the
# user rotated the IMU 180 deg about the vertical (yaw) axis versus v2, so it
# still lies flat but its +Y now points FORWARD instead of backward. Composing
# a 180 deg yaw onto the old quat (0.707107 -0.707107 0 0) gives the value
# below. In world terms at the zero pose the chip's axes are now X -> -Y
# (right), Y -> +X (forward), Z -> +Z (up); Z is unchanged by a yaw, so resting
# accel still reads +9.81 on IMU Z. The gyro/accel report in the site frame, so
# this must match the real part or the sim2real observations come out permuted.
IMU_QUAT = "0 0 -0.707107 0.707107"

# Where the LSM6DS3 physically sits, in base_link's local (lat, up, fwd) frame.
# The user mounts it at the two holes on the horizontal TOP of base_link's lower
# block (NOT the top edge of the tall back plate, which is where auto-detecting
# the "highest flat face" wrongly put it). That block-top face is the largest
# up-facing surface on base_link, centred at lat=+0.028, up=+0.020, fwd~0; the
# board rests ~2 mm above it. This is a fixed design choice, so it is pinned
# here rather than inferred from geometry.
IMU_POS = (0.028, 0.022, -0.003)

# Reflected rotor + gearbox inertia of an STS3215, kg.m^2. Without it the
# legs have almost no inertia about their own axes and the foot contact
# chatters badly: resting IMU accel swings over [-1.8, 24.6] m/s^2 instead of
# sitting at 9.81. mjlab sets this too, but bake it in so the bare model is
# usable on its own.
ARMATURE = 0.01



# MuJoCo honours a <mujoco> extension block inside a URDF. Without
# discardvisual=false it throws away every visual-only geom, leaving just the
# collision hulls; balanceinertia guards against a CAD tensor that violates the
# triangle inequality after rescaling.
MUJOCO_URDF_HINT = """
  <mujoco>
    <compiler discardvisual="false" balanceinertia="true" fusestatic="false"/>
  </mujoco>
"""

# A URDF root link is compiled straight into the worldbody, which yields a
# fixed base. Declaring an explicit floating joint keeps base_link as a real
# body; the free joint itself is re-hung on the Z-up root below.
FLOATING_ROOT = """
  <link name="world"/>
  <joint name="floating_base" type="floating">
    <parent link="world"/>
    <child link="base_link"/>
  </joint>
"""


def fix_mesh_paths(src: Path, tmpdir: Path) -> Path:
    """Rewrite package:// URIs and make the model float, for MuJoCo's parser."""
    text = src.read_text()
    text = text.replace("package://K2/meshes/K2/", "")
    text = text.replace("</robot>", MUJOCO_URDF_HINT + FLOATING_ROOT + "</robot>")
    out = tmpdir / "k2_meshfixed.urdf"
    out.write_text(text)
    # MuJoCo resolves mesh files relative to the model file's directory.
    for mesh in (ROOT / "meshes" / "K2").iterdir():
        if mesh.suffix.lower() in (".obj", ".stl", ".mtl"):
            shutil.copy(mesh, tmpdir / mesh.name)
    return out


def compile_to_xml(urdf: Path, tmpdir: Path) -> ET.ElementTree:
    model = mujoco.MjModel.from_xml_path(str(urdf))
    raw = tmpdir / "raw.xml"
    mujoco.mj_saveLastXML(str(raw), model)
    return ET.parse(raw)


def elem(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def measure_feet(xml_path: Path):
    """Locate each sole and the stance geometry on a compiled model.

    Returns the sole centre in each ankle's own frame, plus the robot's
    lateral/forward offset and sole height in the root frame. Everything is
    read off the collision mesh, so it stays correct if the CAD changes.
    """
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    local, world = {}, {}
    for link, site in FOOT_LINKS.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                f"{site}_collision")
        mid = model.geom_dataid[gid]
        verts = model.mesh_vert[model.mesh_vertadr[mid]:
                                model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        w = verts @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]

        # The sole is the flat patch within 2 mm of the lowest vertex.
        sole = w[w[:, 2] < w[:, 2].min() + 0.002]
        centre = np.array([sole[:, 0].mean(), sole[:, 1].mean(), sole[:, 2].min()])
        world[site] = centre
        R = data.xmat[bid].reshape(3, 3)
        local[site] = R.T @ (centre - data.xpos[bid])
        # Heel and toe as explicit sites. Two points at known spots on the sole
        # turn "foot is flat on the ground" into two plain height checks, with
        # no assumption about which body axis points up -- the ankle's local Z
        # actually runs along world +X, which is easy to get wrong.
        for tag, pt in (("heel", sole[sole[:, 0].argmin()]),
                        ("toe", sole[sole[:, 0].argmax()])):
            local[f"{site}_{tag}"] = R.T @ (pt - data.xpos[bid])

    # IMU seat: centre of base_link's flat top face, where the LSM6DS3 sits.
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    pts = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid or model.geom_dataid[gid] < 0:
            continue
        mid = model.geom_dataid[gid]
        v = model.mesh_vert[model.mesh_vertadr[mid]:
                            model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        pts.append(v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid])
    pts = np.vstack(pts)
    top = pts[pts[:, 2] > pts[:, 2].max() - 0.002]
    seat = np.array([top[:, 0].mean(), top[:, 1].mean(), pts[:, 2].max()])
    Rb = data.xmat[bid].reshape(3, 3)

    centres = np.array(list(world.values()))
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "k2_root")
    return {
        "local": local,
        "imu": Rb.T @ (seat - data.xpos[bid]),
        # Offset that would put the robot's stance centre on the root axis.
        "centre_xy": centres[:, :2].mean(axis=0),
        # Root height above the soles, i.e. the leg length at this pose.
        "sole_drop": data.xpos[root][2] - centres[:, 2].min(),
        "stance_width": abs(centres[0][1] - centres[1][1]),
    }


def main():
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        tree = compile_to_xml(fix_mesh_paths(SRC, tmpdir), tmpdir)

    mj = tree.getroot()
    mj.set("model", "k2")

    compiler = mj.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        mj.insert(0, compiler)
    compiler.set("meshdir", MESHDIR)
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")
    for junk in ("strippath", "discardvisual", "fusestatic"):
        compiler.attrib.pop(junk, None)

    # 2 ms: at 5 ms the foot-plane contact injects energy and the robot
    # bounces higher than it was dropped from. mjlab decimates by 10 for the
    # usual 50 Hz control loop.
    option = mj.find("option")
    if option is None:
        option = ET.SubElement(mj, "option")
    option.set("timestep", "0.002")
    option.set("integrator", "implicitfast")

    # The default 640x480 offscreen buffer is too small for the inspection
    # renders in render_check.py.
    visual = mj.find("visual") or ET.SubElement(mj, "visual")
    elem(visual, "global", offwidth=1280, offheight=960)
    # Default site frame triads are drawn as long as the robot; shrink them so
    # they read as orientation markers rather than obscuring the model.
    elem(visual, "scale", framelength="0.06", framewidth="0.006")

    worldbody = mj.find("worldbody")
    base = worldbody.find("body")  # base_link, the URDF root

    # --- Y-up -> Z-up: nest everything under a world-aligned root ----------
    worldbody.remove(base)
    # The free joint moved in from the URDF sits on base_link, whose frame is
    # still Y-up. Drop it and re-hang it on a world-aligned root instead.
    for j in base.findall("joint") + base.findall("freejoint"):
        if j.get("type") == "free" or j.tag == "freejoint":
            base.remove(j)
    root = ET.Element("body", name="k2_root", pos="0 0 0", quat="1 0 0 0")
    ET.SubElement(root, "freejoint", name="floating_base")
    base.set("pos", "0 0 0")
    base.set("quat", BASE_QUAT)
    root.append(base)
    worldbody.append(root)

    # --- Actuator-side inertia on every hinge -------------------------------
    for body in mj.iter("body"):
        for joint in body.findall("joint"):
            if joint.get("name"):
                joint.set("armature", str(ARMATURE))

    # --- Ground plane -------------------------------------------------------
    asset = mj.find("asset")
    elem(asset, "texture", type="2d", name="groundplane", builtin="checker",
         rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3", width=512, height=512)
    elem(asset, "material", name="groundplane", texture="groundplane",
         texrepeat="5 5", reflectance="0.1")
    elem(worldbody, "geom", name="floor", type="plane", size="0 0 0.05",
         material="groundplane", contype=1, conaffinity=1)
    elem(worldbody, "light", pos="0 0 3", dir="0 0 -1", directional="true")

    # --- Collision policy: feet only ---------------------------------------
    bodies = {b.get("name"): b for b in mj.iter("body")}
    for name, body in bodies.items():
        for geom in body.findall("geom"):
            is_visual = geom.get("mesh", "").endswith("_collision") is False
            if name in FOOT_LINKS and not is_visual:
                geom.set("contype", "1")
                geom.set("conaffinity", "1")
                # Match Unitree's contact setup: foot priority must exceed the
                # terrain's default priority, otherwise MuJoCo takes the max
                # coefficient and low-friction randomization is ineffective.
                geom.set("condim", "3")
                geom.set("priority", "1")
                geom.set("friction", "0.8 0.02 0.001")
                geom.set("name", f"{FOOT_LINKS[name]}_collision")
            else:
                # Visual-only: no contacts, no mass contribution.
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.set("group", "2" if is_visual else "3")

    # --- Measure the real geometry, then place sites and the spawn pose -----
    # base_link's CAD origin is not on the robot's centreline, so compile once
    # to find where the feet actually are, then shift the body inside the root
    # and set the spawn height from the measured leg length.
    DST.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        probe = Path(td) / "probe.xml"
        compiler.set("meshdir", str((ROOT / "meshes" / "K2").resolve()))
        tree.write(probe, encoding="utf-8")
        geo = measure_feet(probe)
        compiler.set("meshdir", MESHDIR)

    dx, dy = geo["centre_xy"]
    base.set("pos", f"{-dx:.6f} {-dy:.6f} 0")
    stand = geo["sole_drop"] + SPAWN_CLEARANCE
    root.set("pos", f"0 0 {stand:.6f}")

    # --- Sites --------------------------------------------------------------
    elem(bodies["base_link"], "site", name="imu",
         pos=" ".join(f"{v:.6f}" for v in IMU_POS), quat=IMU_QUAT,
         size="0.012 0.008 0.002", type="box", rgba="1 0.5 0 1", group=5)
    for link, site in FOOT_LINKS.items():
        pos = " ".join(f"{v:.6f}" for v in geo["local"][site])
        elem(bodies[link], "site", name=site, pos=pos,
             size="0.005", rgba="0 1 0 1", group=5)
        for tag in ("heel", "toe"):
            elem(bodies[link], "site", name=f"{site}_{tag}",
                 pos=" ".join(f"{v:.6f}" for v in geo["local"][f"{site}_{tag}"]),
                 size="0.004", rgba="0 0.6 1 1", group=5)

    # --- Sensors ------------------------------------------------------------
    # Names follow mjlab's velocity-task convention (imu_ang_vel / imu_lin_vel /
    # imu_lin_acc) so the walking env's observations resolve without retargeting.
    sensor = ET.SubElement(mj, "sensor")
    elem(sensor, "gyro", name="imu_ang_vel", site="imu")
    elem(sensor, "velocimeter", name="imu_lin_vel", site="imu")
    elem(sensor, "accelerometer", name="imu_lin_acc", site="imu")
    elem(sensor, "framequat", name="imu_quat", objtype="site", objname="imu")
    # Whole-body angular momentum about base_link, used by the walking task's
    # angular_momentum reward (penalises flailing / hopping).
    elem(sensor, "subtreeangmom", name="root_angmom", body="base_link")
    for site in FOOT_LINKS.values():
        elem(sensor, "framepos", name=f"{site}_pos", objtype="site",
             objname=site)
        elem(sensor, "framelinvel", name=f"{site}_vel", objtype="site",
             objname=site)

    ET.indent(tree, space="  ")
    tree.write(DST, encoding="utf-8")

    # --- Verify it compiles and report the geometry ------------------------
    model = mujoco.MjModel.from_xml_path(str(DST))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"compiled {DST.relative_to(ROOT)}")
    print(f"  nq={model.nq} nv={model.nv} nbody={model.nbody} "
          f"ngeom={model.ngeom} nsensor={model.nsensor}")
    print(f"  total mass: {model.body_subtreemass[1]*1000:.1f} g")
    for link, site in FOOT_LINKS.items():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        print(f"  {site:<11} world {data.site_xpos[sid].round(4)}")
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
    print(f"  {'imu':<11} world {data.site_xpos[sid].round(4)}")


if __name__ == "__main__":
    main()
