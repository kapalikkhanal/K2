#!/usr/bin/env python3
"""Render K2 for visual inspection: stance views and a per-joint sweep.

Produces two PNGs in images/:
  k2_stance.png  - front / side / iso views of the zero pose, sites visible.
  k2_joints.png  - every joint driven to each end of its range, one row per
                   joint, so the sign and axis of each DOF can be eyeballed.

Run headless with MUJOCO_GL=egl.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MJCF = ROOT / "mjcf" / "k2_physics.xml"
OUT = ROOT / "images"

W, H = 640, 640

# azimuth, elevation, label
STANCE_VIEWS = [
    (180, -10, "front (+X toward camera)"),
    (90, -10, "left side"),
    (135, -20, "iso"),
]


def make_renderer(model):
    r = mujoco.Renderer(model, height=H, width=W)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.sitegroup[5] = 1  # sites live in group 5 and are hidden by default
    opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    return r, opt


def shot(renderer, opt, data, model, azimuth, elevation, lookat, dist):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation = azimuth, elevation
    cam.distance, cam.lookat[:] = dist, lookat
    renderer.update_scene(data, camera=cam, scene_option=opt)
    return renderer.render()


def label(img, text):
    """Burn a text label into the top-left corner, if PIL is available."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return img
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, len(text) * 7 + 8, 18], fill=(0, 0, 0))
    d.text((4, 4), text, fill=(255, 255, 0))
    return np.asarray(im)


def save(path, rows):
    from PIL import Image
    grid = np.vstack([np.hstack(r) for r in rows])
    Image.fromarray(grid).save(path)
    print(f"wrote {path.relative_to(ROOT)}  ({grid.shape[1]}x{grid.shape[0]})")


def main():
    OUT.mkdir(exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer, opt = make_renderer(model)

    lookat = np.array([0.0, 0.0, 0.18])

    # --- stance views ------------------------------------------------------
    row = [label(shot(renderer, opt, data, model, az, el, lookat, 0.75), txt)
           for az, el, txt in STANCE_VIEWS]
    save(OUT / "k2_stance.png", [row])

    # --- per-joint sweep ---------------------------------------------------
    hinges = [j for j in range(model.njnt)
              if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    rows = {"right": [], "left": []}
    for j in hinges:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        adr = model.jnt_qposadr[j]
        lo, hi = model.jnt_range[j]
        # Face the camera along the joint's rotation axis where possible, so
        # the motion is in the image plane.
        axis = np.abs(data.xaxis[j])
        az = 90 if axis[1] > axis[0] else 180
        cells = []
        for q, tag in ((lo, "min"), (0.0, "zero"), (hi, "max")):
            mujoco.mj_resetData(model, data)
            data.qpos[adr] = q
            mujoco.mj_forward(model, data)
            img = shot(renderer, opt, data, model, az, -10, lookat, 0.75)
            cells.append(label(img, f"{name} {tag} {np.degrees(q):+.0f}"))
        rows["left" if "left" in name else "right"].append(cells)
    for side, cells in rows.items():
        save(OUT / f"k2_joints_{side}.png", cells)


if __name__ == "__main__":
    main()
