# K2

A 3D-printed 10-DOF bipedal robot: two 5-DOF legs (hip pitch / hip roll /
hip yaw / knee / ankle) driven by Feetech STS3215 bus servos, with a
MuJoCo digital twin generated from the Fusion 360 CAD export.

| | |
|---|---|
| DOF | 10 revolute (5 per leg) |
| Actuators | 10x Feetech STS3215, 30 kg·cm @ 12 V (2.94 N·m, 4.712 rad/s) |
| Mass | 926 g (243 g PLA + 600 g servos + 83 g electronics) |
| Hip height | 358 mm |
| Stance width | 156 mm |
| Controller | Raspberry Pi 5 |
| IMU | LSM6DS3 on the base top face |

![K2 stance](images/k2_stance.png)

## Layout

```
urdf/K2_fusion_export.urdf   Fusion 360 export, untouched
urdf/K2_phase1.urdf          mass- and limit-corrected (generated)
mjcf/k2_physics.xml          MuJoCo model (generated)
meshes/K2/                   visual .obj + convex collision .stl
scripts/                     the conversion pipeline
docs/transforms.md           link-to-link transformation matrices
robot_data.yaml              per-link volumes and CAD inertias
```

Both generated files are checked in so the model can be used without
running the pipeline. Re-run the scripts after any CAD change.

## Building the model

```bash
python scripts/phase1_mass.py       # -> urdf/K2_phase1.urdf
python scripts/phase2_make_mjcf.py  # -> mjcf/k2_physics.xml
python scripts/phase3_validate.py   # checks mass, limits, contact, torque
MUJOCO_GL=egl python scripts/render_check.py   # -> images/
```

Inspect it interactively:

```bash
python -m mujoco.viewer --mjcf=mjcf/k2_physics.xml
```

Sites (`imu`, `right_foot`, `left_foot`) are in group 5, which the viewer
hides by default — enable it in the rendering panel to see them.

## What the pipeline corrects

The Fusion exporter tags every link `Steel` (7850 kg/m³), giving a 5.56 kg
robot that weighs 926 g in reality. `phase1_mass.py` rescales each link to
its measured sliced print weight, then adds the servo and electronics mass
as point masses via the parallel-axis theorem.

Three things the export got wrong, fixed in the pipeline and worth knowing
if you regenerate from CAD:

- **Knee limits were sign-inverted** — as exported they allowed 90° of
  hyperextension and only 45° of flexion. Now ±90° of flexion with a hard
  stop at full extension.
- **Y-up vs Z-up** — Fusion exports Y-up. The whole tree is nested under a
  world-aligned `k2_root` body that carries the free joint, so the floating
  base's own frame is Z-up and gravity/heading observations come out right
  without baking a rotation into every link.
- **Placeholder joint dynamics** — `effort=100, velocity=1.0` replaced with
  the real STS3215 numbers.

## Conventions

- Robot faces world **+X**; **+Y** is left; **+Z** is up.
- The `imu` site matches the physical LSM6DS3 mounting: board flat, its
  **+Y pointing backwards**. Gyro and accelerometer report in this frame, so
  sim and hardware observations line up.
- Collision is **feet only**, so the legs cannot snag on their own shells.
- Timestep is **2 ms**. At 5 ms the foot–ground contact injects energy and
  the robot bounces higher than it was dropped from. Decimate by 10 for a
  50 Hz control loop.
- Every hinge carries `armature=0.01` (STS3215 reflected rotor inertia).
  Without it the resting IMU reading chatters over [-1.8, 24.6] m/s²
  instead of sitting at 9.81.

## Status

The model is validated but no policy is trained yet. The RL environment
will live in a separate `k2_rl/` package.
