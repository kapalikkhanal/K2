"""Bake the MuJoCo model into web assets for the site's joint viewer.

Outputs, all under site/assets/model/:

  meshes/*.glb  The visual meshes, vertex-merged, scale baked in and decimated
                to a face budget. The repository's ASCII OBJ exports total
                ~53 MB, which is far too much to hand a browser; these are ~2 MB.
  model.json    The body tree -- transforms, joint axes, geom placements and
                colours -- plus per-joint metadata: the servo that drives each
                joint, its operational limit, and what a positive command does.

Why the MJCF and not urdf/K2_fusion_export.urdf: the Fusion export's <visual>
origins do not assemble the robot (they were measured against a different frame
convention, and its knee limits are sign-flipped on the model-right leg). The
MJCF is the model the policy actually trains on and the one MuJoCo renders
correctly, so it is the only trustworthy source for geometry.

The generated tree is verified against MuJoCo's own forward kinematics before
anything is written -- see check_against_mujoco().

    python scripts/build_site_model.py
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hardware import k2_conventions as C  # noqa: E402

MJCF = ROOT / "mjcf" / "k2_physics.xml"
POLICY = ROOT / "policies" / "last_policy.onnx"
OUT_DIR = ROOT / "site" / "assets" / "model"
MESH_DIR = OUT_DIR / "meshes"

FACE_BUDGET = 24000
VISUAL_GROUP = "2"          # group 3 is collision geometry; the viewer skips it

GROUP = {"hip_pitch": "Hip", "hip_roll": "Hip", "hip_yaw": "Hip",
         "knee": "Knee", "ankle_pitch": "Ankle", "ankle_roll": "Ankle"}


def floats(s: str | None, default: list[float]) -> list[float]:
    return [float(v) for v in s.split()] if s else list(default)


def parse_tree() -> tuple[dict, dict]:
    """Return (assets, base_link body tree) straight from the MJCF."""
    root = ET.parse(MJCF).getroot()

    assets = {}
    for mesh in root.findall("./asset/mesh"):
        assets[mesh.get("name")] = {
            "file": mesh.get("file"),
            "scale": floats(mesh.get("scale"), [1, 1, 1]),
        }

    meshdir = root.find("compiler").get("meshdir", ".")

    def walk(body: ET.Element) -> dict:
        node = {
            "name": body.get("name"),
            "pos": floats(body.get("pos"), [0, 0, 0]),
            "quat": floats(body.get("quat"), [1, 0, 0, 0]),   # MuJoCo wxyz
            "geoms": [],
            "children": [],
        }
        joint = body.find("joint")
        if joint is not None:
            node["joint"] = {
                "name": joint.get("name"),
                "axis": floats(joint.get("axis"), [0, 0, 1]),
                "range": floats(joint.get("range"), [-3.14159, 3.14159]),
            }
        for geom in body.findall("geom"):
            if geom.get("group") != VISUAL_GROUP or geom.get("type") != "mesh":
                continue
            node["geoms"].append({
                "mesh": geom.get("mesh"),
                "pos": floats(geom.get("pos"), [0, 0, 0]),
                "quat": floats(geom.get("quat"), [1, 0, 0, 0]),
                "rgba": floats(geom.get("rgba"), [0.7, 0.7, 0.7, 1]),
            })
        for child in body.findall("body"):
            node["children"].append(walk(child))
        return node

    base = root.find(".//body[@name='base_link']")
    if base is None:
        raise SystemExit("no base_link body in the MJCF")
    return {"meshdir": meshdir, "meshes": assets}, walk(base)


def used_meshes(node: dict, out: set[str]) -> set[str]:
    for g in node["geoms"]:
        out.add(g["mesh"])
    for c in node["children"]:
        used_meshes(c, out)
    return out


def build_meshes(assets: dict, names: set[str]) -> None:
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    src_dir = (MJCF.parent / assets["meshdir"]).resolve()
    total_in = total_out = 0
    for name in sorted(names):
        a = assets["meshes"][name]
        src = src_dir / a["file"]
        mesh = trimesh.load(src, force="mesh", process=True)
        mesh.merge_vertices()
        before = len(mesh.faces)
        if before > FACE_BUDGET:
            mesh = mesh.simplify_quadric_decimation(face_count=FACE_BUDGET)
        mesh.apply_scale(a["scale"])
        mesh.visual = trimesh.visual.ColorVisuals(mesh)   # the viewer supplies colour

        out = MESH_DIR / f"{name}.glb"
        out.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
        total_in += src.stat().st_size
        total_out += out.stat().st_size
        print(f"  {name:22s} {before:6d} -> {len(mesh.faces):5d} faces  "
              f"{src.stat().st_size/1e6:5.2f} -> {out.stat().st_size/1e6:4.2f} MB")
    print(f"  total {total_in/1e6:.1f} MB -> {total_out/1e6:.1f} MB")


def policy_default_pose() -> dict[str, float]:
    """The shipped policy's default joint pose, keyed by MJCF joint name."""
    try:
        import onnx
    except ImportError:
        print("  onnx not installed; the policy-stance preset will be flat")
        return {}
    meta = {p.key: p.value for p in onnx.load(POLICY).metadata_props}
    names = meta["joint_names"].split(",")
    return dict(zip(names, (float(v) for v in meta["default_joint_pos"].split(","))))


def joint_metadata() -> dict:
    """Servo IDs, operational limits and motion descriptions, by MJCF joint name."""
    limits = C.joint_limits()
    default = policy_default_pose()
    out = {}
    for short in C.SIM_ORDER:
        mjcf_name = C.JOINT_TO_MJCF[short]
        lo, hi = limits[short]
        model_side = "right" if short.endswith("_R") else "left"
        out[mjcf_name] = {
            "short": short,
            "servo_id": C.JOINT_TO_ID[short],
            "group": GROUP[short.rsplit("_", 1)[0]],
            "model_side": model_side,
            # k2_conventions: the model's *_right chain is the robot's physical LEFT.
            "physical_side": "left" if model_side == "right" else "right",
            "lower": round(lo, 6),
            "upper": round(hi, 6),
            "positive_does": C.POS_DESC[short],
            "default": round(default.get(mjcf_name, 0.0), 6),
        }
    return out


def check_against_mujoco(tree: dict, pose: dict[str, float]) -> None:
    """Reproduce the exported tree's FK in numpy and diff it against MuJoCo."""
    import mujoco
    from scipy.spatial.transform import Rotation as R

    def quat(wxyz):
        w, x, y, z = wxyz
        return R.from_quat([x, y, z, w])

    world = {}

    def walk(node, parent_R, parent_t):
        Rl = quat(node["quat"])
        t = parent_t + parent_R.apply(node["pos"])
        Rb = parent_R * Rl
        j = node.get("joint")
        if j:
            Rb = Rb * R.from_rotvec(np.array(j["axis"]) / np.linalg.norm(j["axis"])
                                    * pose.get(j["name"], 0.0))
        world[node["name"]] = (Rb, t)
        for c in node["children"]:
            walk(c, Rb, t)

    walk(tree, R.identity(), np.zeros(3))

    m = mujoco.MjModel.from_xml_path(str(MJCF))
    d = mujoco.MjData(m)
    for name, q in pose.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        d.qpos[m.jnt_qposadr[jid]] = q
    mujoco.mj_kinematics(m, d)

    # The exported tree is rooted at base_link, whose own pos/quat are relative
    # to k2_root -- so k2_root is the frame both sides must be expressed in.
    b0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "k2_root")
    R0 = R.from_matrix(d.xmat[b0].reshape(3, 3))
    p0 = d.xpos[b0]

    worst = 0.0
    for name, (Rb, t) in world.items():
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        ref_t = R0.inv().apply(d.xpos[bid] - p0)
        ref_R = R0.inv() * R.from_matrix(d.xmat[bid].reshape(3, 3))
        dt = float(np.linalg.norm(t - ref_t))
        dr = float((Rb.inv() * ref_R).magnitude())
        worst = max(worst, dt, dr)
        if dt > 1e-6 or dr > 1e-6:
            print(f"    MISMATCH {name}: dpos={dt:.2e} drot={dr:.2e}")
    print(f"  max deviation from MuJoCo FK: {worst:.2e} (position m / rotation rad)")
    if worst > 1e-6:
        raise SystemExit("exported tree does not match MuJoCo -- refusing to write")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assets, tree = parse_tree()
    meta = joint_metadata()

    print("verifying the exported tree against MuJoCo:")
    default_pose = {n: j["default"] for n, j in meta.items()}
    check_against_mujoco(tree, {n: 0.0 for n in meta})
    check_against_mujoco(tree, default_pose)

    print("meshes:")
    build_meshes(assets, used_meshes(tree, set()))

    doc = {
        "_comment": "Generated by scripts/build_site_model.py from mjcf/k2_physics.xml -- do not edit.",
        "frame": {
            "note": "MuJoCo convention: +x forward, +y toward the model's *_right chain "
                    "(the robot's physical LEFT), +z up. The viewer rotates this to its own Y-up.",
        },
        "limits_source": "hardware/joint_limits.json (operational limits)",
        "order": [C.JOINT_TO_MJCF[s] for s in C.SIM_ORDER],
        "joints": meta,
        "presets": {
            "zero": {n: 0.0 for n in meta},
            "policy_default": default_pose,
        },
        "tree": tree,
    }
    (OUT_DIR / "model.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"  wrote {OUT_DIR / 'model.json'}")


if __name__ == "__main__":
    main()
