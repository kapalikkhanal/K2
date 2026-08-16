# K2 — 12-DOF Biped Robot: Complete Reference

A self-contained knowledge dump for the K2 biped: the hardware, the
digital twin, the software stack, the calibration, the RL task, and how to
deploy it. Written so another engineer or LLM can pick up with full context.

Repo root: `~/K2`. Intended to be **open-sourced**, so keep it clean — only
files that are strictly needed.

---

## 1. What K2 is

K2 is a small **12-DOF bipedal robot** (two 6-DOF legs, no upper body). It was
designed in Fusion 360 (current export: **Robot_v4**, 2026-07) and is developed
**digital-twin-first**: every change is validated in a MuJoCo model before it
touches hardware, and the *same control code* drives sim or the real robot by
changing one flag.

- **Total mass:** **1280 g measured**; the current twin is 1.1786 kg and still
  needs its extra 101.4 g distributed from physical mass/CoM measurements.
- **Standing hip height:** ~364 mm; standing base-link height ~335 mm; CoM ~194 mm
- **Nominal stance width:** feet at ±55–58 mm (≈110–116 mm apart)
- **Controller:** Raspberry Pi 5 (see §9)

### Leg structure (per leg, hip → toe)
`hip_pitch → hip_roll → hip_yaw → knee → ankle_pitch → ankle_roll`, mirrored L/R.
The v4 redesign added the **ankle_roll** DOF (a second ankle servo per leg); the
ankle-roll link (`Feet`) is the foot plate that carries the sole. This DOF is
what makes **static single-foot support** possible (lateral CoM shift), which is
the mechanical prerequisite for a stable slow walk.

### Links (13)
`base_link`, and per side (`_left`/`_right`): `Hip_pitch`, `Hip_roll`,
`Hip_yaw`, `Knee`, `Ankle` (the ankle-pitch bracket), `Feet` (the ankle-roll
foot plate — **the sole/collision lives here**, not on `Ankle`).

---

## 2. Joints, axes, limits

Short name → MJCF joint name (note Fusion's odd auto-labels — keep exact spelling,
including the misspelled "anke" and the word "pitch" inside the *roll* joint):

| short | MJCF joint name | axis | limit (deg) |
|---|---|---|---|
| hip_pitch_R/L | `base_link_hip_pitch_{right,left}_joint` | pitch (world Y) | ±47 |
| hip_roll_R/L | `hip_pitch_hip_roll_{right,left}_joint` | roll (world X) | R [−45,+22], L [−22,+45] |
| hip_yaw_R/L | `hip_roll_hip_yaw_{right,left}_joint` | yaw (world Z) | ±45 |
| knee_R/L | `hip_yaw_knee_{right,left}_joint` | pitch (world Y) | **[−110, 0] both** |
| ankle_pitch_R/L | `knee_ankle_pitch_{right,left}_joint` | pitch (world Y) | ±30 |
| ankle_roll_R/L | `ankle_roll_foot_{right,left}_joint` | roll (world X) | **[−20, +40]** |

Key sign/limit facts (learned the hard way):
- **Both knees flex NEGATIVE** on v3/v4 (both ranges `[−110°, 0]`). The Fusion
  export ships the knee limit sign-inverted on *every* re-export — always fix it.
- **hip_roll is asymmetric**: ~45° abduction (leg out) vs ~22° adduction (leg in,
  tuck under body). Adduction is the tight side.
- **ankle_roll is asymmetric**: **~20° inversion (inward, sole faces in)** vs
  **~50° eversion (outward)**. The inward/inversion direction is what a lateral
  weight-shift needs, so **12–20° inward is the binding limit** for balance.
  (Started at 12° — a bracket interference — mechanically opened to 20° on
  2026-07-22. Update `ANKLE_ROLL_IN` in `scripts/phase1_mass.py` if it changes.)
- The world axes above are verified from the MJCF, not assumed. `+q` on
  `ankle_roll` drops the foot's **inner edge** (toward the other foot); `+q` on
  `ankle_pitch` lifts the toe (dorsiflex).

---

## 3. Actuators & bus

- **12× Feetech STS3215** (serial bus servos), 12 V. **2.94 N·m stall**
  (30 kg·cm), **4.712 rad/s** no-load (45 rpm), **60 g each**. Single-turn
  encoder, **0–4095 counts/rev**.
- One **serial bus**, Waveshare adapter (CH343 USB-serial, `1a86:55d3`) at
  **`/dev/ttyACM0`, 1 Mbps**. SDK: `feetech-servo-sdk` (module `scservo_sdk`),
  **PacketHandler protocol_end = 0** for STS.
- Registers: ID **5**, Baud **6**, HomingOffset **31**, TorqueEnable **40**,
  GoalPos **42**, Lock **55**, PresentPos **56**, PresentSpeed **58**
  (sign-magnitude, bit15 = direction).
- **Goal counts are hard-clamped to [40, 4055]** everywhere (never let a goal
  wrap past 0/4095 — the single-turn servo would take the long way round).
- **Port gotcha:** `/dev/ttyACM0` resets to non-writable on every replug.
  Fix each time: `sudo chmod 666 /dev/ttyACM0` (or add user to `dialout`).

### Servo ID map — chain 0..11, both ankle-rolls at the ends
Physical chain: physical-left foot (0) up to its hip (5), across to the
physical-right hip (6), down to its foot (11).

**The MJCF's `_R` joints drive physical-left servos 0..5 and `_L` drives
physical-right servos 6..11.** Fusion's `*_right` bodies sit at +root-Y, which
is the robot's physical left. This mapping was re-established from fresh
matched-phase tests on 2026-08-15: it selected the correct swing foot and made
real roll/pitch signs match MuJoCo. `calibration.json` was swapped with the map
so every physical servo retained its captured zero and sign. Change both
together or neither.

None of the held-air sign checks can catch a leg swap: every `POS_DESC` string is
mirror-symmetric ("outward", "pigeon-toed", "toe up"), so all 12 joints pass on
either leg, and a symmetric crouch is invariant under mirroring. It only shows up
once the gait has to pick a leg to shift weight onto.

**ID 0 is valid** and works through the SDK. The map lives in
`k2_conventions.JOINT_TO_ID` — the single source of truth; everything reads it.
Assign/renumber with `python -m hardware.set_servo_id --scan | --from N --to M`
(one servo at a time).

---

## 4. Calibration

Maps servo counts ↔ joint radians, per joint:

```
q = sign * (raw - home_raw) * 2π/4096
raw = home_raw + sign * q * 4096/2π
```

- `home_raw` = the encoder count at the **straight pose** (q = 0). **Sign-flips
  do NOT change the home pose** — at q=0 every servo goes to `home_raw` regardless
  of sign. So a "leaning at home" problem is always a *home* issue, never a sign.
- `sign` = ±1, the direction the joint moves for +q. Stored in
  `hardware/calibration.json`. Sim uses an identity calibration by construction,
  so miscalibration can only ever come from hardware.
- `sign` = ±1, the direction the joint moves for +q. All 12 were re-verified on
  hardware with `check_joint_signs.py` (hips, knees and all four ankles, both
  directions). After the leg swap the entries read +1 for every `_R` joint and −1
  for every `_L` joint, because each servo kept the sign measured on it.

### How to (re)calibrate — `python -m hardware.k2_calibrate` (GUI, needs DISPLAY)
1. LIMP (torque off), hand-pose the robot dead straight with **both feet flat**.
2. Capture HOME (records `home_raw`). *A leaning stand at "home" = a foot wasn't
   flat / a joint wasn't straight when captured — re-capture, don't flip signs.*
3. Nudge each joint, Flip any whose motion disagrees with its description, Save.
4. Go to CROUCH to validate.

**Debugging lesson:** single-joint direction checks can be *misjudged* (a whole
leg swinging is easy to read wrong). The reliable test is the **single-leg crouch
mirror comparison** (crouch each leg alone; they must be mirror images). And
remember: **home offsets are masked when both feet are planted** (the floor
constrains them) and only show up when a foot is **free** (swing / one-foot
balance). That is why a robot can crouch cleanly but fail during a step.

---

## 5. Digital twin (MuJoCo / MJCF)

Model file: `mjcf/k2_physics.xml`. Built by a 3-phase pipeline in `scripts/`:

1. **`phase1_mass.py`** — rebuilds link masses/inertia from *measured* print
   weights (Fusion tags every link "Steel" → ~4× too heavy) and adds STS3215
   point masses (60 g/joint) via parallel-axis; overrides the placeholder joint
   limits with the real ones. Writes `urdf/K2_phase1.urdf`.
   - Per-link PLA (g): base 30 (shell) + 70 (electronics point mass, 6 cm up);
     hip_pitch 16, hip_roll 25, hip_yaw 14, knee 27, ankle_pitch 22, Feet 14.
2. **`phase2_make_mjcf.py`** — URDF → MJCF, fixing what URDF can't express.
3. **`phase3_validate.py`** — drops it on a vertical slide with temp servos to
   check feet contact / no tunnelling / holding torque.

### MJCF conventions (non-obvious, will recur on any Fusion→MuJoCo biped)
- **Floating base:** a URDF root compiles into a *fixed* base. Inject a `world`
  link + `type="floating"` joint, then nest the tree under a world-aligned root
  body `k2_root` (carries the free joint) so the base frame is Z-up.
- **Y-up → Z-up:** Fusion is Y-up. Don't rotate every link — give `base_link`
  the `0.5 0.5 0.5 0.5` quat inside `k2_root`.
- **`discardvisual="false"`** in a `<mujoco>` block inside the URDF, or MuJoCo
  throws away visual geoms and leaves only collision hulls.
- **Feet-only collision** on the `Feet_*` bodies (so legs can't self-snag). Foot
  sole/site positions are **measured from the collision mesh**, never hardcoded.
- **Timestep 2 ms** (5 ms injects contact energy and the robot bounces higher
  than it was dropped). Decimate ×10 for a 50 Hz control loop.
- **`armature=0.01`** on every hinge (reflected rotor inertia; without it the
  foot contact chatters and resting IMU accel swings wildly).
- **IMU site + sensors** on `base_link`: `imu_ang_vel` (gyro), `imu_lin_vel`
  (velocimeter), `imu_lin_acc` (accelerometer), `imu_quat` (framequat),
  `root_angmom` (subtreeangmom). Foot sites `right_foot`/`left_foot`
  (+ heel/toe) with framepos/framelinvel sensors. **The shipped MJCF has NO
  actuators** — mjlab (training) and SimBus (direct control) add them.

---

## 6. Software stack — `~/K2/hardware/` (unified sim ↔ real)

The whole point: **the same program drives MuJoCo or the real servo chain** by
changing one flag. The abstraction sits at the **servo bus, in raw encoder
counts** — the robot's real wire language. Everything above it (calibration,
count↔rad, observation building, clamping, slew-limiting, control rate) is
byte-identical on both paths.

```
python -m hardware.k2_ctrl --bus sim --motion squat --viewer
python -m hardware.k2_ctrl --bus /dev/ttyACM0 --motion squat
```

Core modules:
- **`k2_conventions.py`** — the hub. `SIM_ORDER` (right leg hip→toe, then left),
  `JOINT_TO_ID`, `JOINT_TO_MJCF`, `POS_DESC`/`NUDGE_DIR` (calibration help),
  count↔rad math, calibration load/save. **Joint limits are read from the MJCF**
  (`write_joint_limits()` → `joint_limits.json`) — never hardcoded, so they can't
  drift from the model. The Pi loads the small JSON (no MuJoCo needed).
- **`k2_bus.py`** — `Bus` ABC + `SerialBus` (real STS3215) + `SimBus` (MuJoCo).
  Interface: `tick(dt)` / `read_pos_speed()` / `write_goals()` / `torque(on)`.
  SimBus quantises to 12-bit counts and clamps [40,4055] so the policy sees
  hardware-grade resolution; it adds position servos force-limited to 2.94 N·m.
- **`k2_ctrl.py`** — the control loop. Eases from the robot's *current* pose into
  the start pose, slew-limited, and **always releases torque on exit incl.
  Ctrl-C** (goes limp, never freezes). Modes: `squat`, `hold`.
- **`k2_calibrate.py`** — 12-servo calibration GUI (see §4).
- **`k2_motion.py`** — `SquatIK`, a flat-foot double-support squat solved against
  the MJCF (needs no IMU: both planted feet close the chain).
- **`k2_policy_run.py`** — runs a trained ONNX policy over the `Bus`, sim or
  real, one code path. Rebuilds the 45-D observation byte-for-byte the way the
  mjlab env built it at training time, reads joint order / default pose / action
  scale from the ONNX metadata, and clamps targets to the joint limits.
- **`k2_attitude.py`** — `SimAttitude` (MuJoCo ground truth) and `ImuAttitude`
  (LSM6DS3 + adaptive complementary filter), one interface. Returns
  `(gyro, projected_gravity)`. **The gyro is returned in the MJCF `imu` SITE
  frame**, not the chip frame — see §9.
- **`k2_stabilizer.py`** — `UnsafeTiltError` and bounded tilt feedback.
- **`test_default_crouch.py`** — eases into the model-derived default crouch and
  holds it, with an IMU tilt cutoff. The first thing to run after calibrating.
- **`check_joint_signs.py`** — verifies any subset of the 12 calibration signs,
  both directions, held in air. Supersedes the old per-group scripts.
- **`capture_home.py`** — snapshots the pose the robot is standing in right now
  into `calibration.json` `home_raw`, working purely in raw counts.
- **`go_home.py`**, **`set_servo_gain.py`**, **`set_servo_id.py`**,
  **`calibrate_imu_level.py`** (IMU mounting-tilt calibration).

Config files (checked in): `calibration.json`, `joint_limits.json`,
`crouch_pose.json` (the default crouch), `imu_level_calibration.json`.

---

## 7. Deploying a trained policy

`hardware/k2_policy_run.py` runs the ONNX over the same `Bus`, so the twin and
the robot execute identical code.

```bash
# twin
python -m hardware.k2_policy_run --bus sim --viewer --mode march \
  --policy policies/march_hold_v4_1999.onnx
# real
python -m hardware.k2_policy_run --bus /dev/ttyAMA0 --mode march \
  --march-after 5 --duration 20 --slew 1.0 --fall-angle 10 \
  --policy policies/march_hold_v4_1999.onnx --log run.csv
```

The 45-D observation is rebuilt exactly as the mjlab env built it:

```
[ base_ang_vel(3)      gyro, MJCF imu SITE frame
  projected_gravity(3) gravity direction, k2_root frame
  joint_pos_rel(12)    q - default_pose, SIM_ORDER
  joint_vel_rel(12)    qdot, SIM_ORDER
  last_action(12)      previous RAW policy output
  gait(3) ]            [march, march*sin(phase), march*cos(phase)]
```

Targets are `default_pose + action_scale * action`, clamped to the joint limits
and to counts [40, 4055]. Joint order, default pose, action scale and gait
frequency all come from the ONNX metadata, so a retrain cannot silently break
the deploy path.

**Safety:** torque is released on every exit path including Ctrl-C; `--slew`
caps per-tick joint motion; `--fall-angle` cuts torque above a torso tilt; the
robot is eased into the default crouch over `--approach` seconds before the
policy takes over, with the tilt cutoff already armed.

`--log` writes a 50 Hz CSV of commanded vs measured position for all 12 joints
plus gyro, gravity, tilt and raw actions. **This log is the single most useful
debugging artifact** — a mis-mapped joint tracks its command perfectly while the
IMU response is orthogonal to what that command should produce.

---

## 8. Where the real robot currently stands

- All 12 servos calibrated; all 12 signs verified on hardware with
  `check_joint_signs.py`, both directions.
- IMU mounting tilt calibrated (`calibrate_imu_level.py`); the gyro frame bug in
  §9 is fixed.
- **Permanent leg mapping validated** (§3); no deployment flag is required.
- **Hold is solid**, settling near 0.7 deg tilt with no drift.
- **March validation passed** for four continuous seconds (about six lifts) at
  50 Hz with a 10.82 deg logged peak against a 12 deg cutoff. Minimum dynamic
  servo voltage was 11.9 V and post-run temperatures were 35--39 C.
- Dynamic command tracking remains less accurate than the twin: worst final-run
  knee RMS/peak error was 3.71/8.51 deg. Actuator bandwidth identification and
  the 101.4 g twin mass correction are the next sim-to-real tasks.
- **Open defect:** the left foot's `heel`, `inner` and `outer` sole sites are not
  mirrors of the right foot's, which skews four reward terms against the left
  leg. Needs the sites corrected and a retrain.

---

## 9. Controller: Raspberry Pi 5 (not Jetson)

Benchmarked the policy MLP at **0.012 ms/inference** on a single Pi 5 CPU thread
(~0.06% of a 50 Hz budget). GPU is only for *training* or vision models. The Pi 5
is lighter/lower-power (better on `base_link`) and its I2C/SPI eases the IMU. A
Jetson would only be justified if cameras / a VLA are added later.

IMU on the Pi: **LSM6DS3**, inside the electronics tray.

**The chip frame is not the root or MJCF `imu` site frame.** Fresh held tests on
2026-08-15 found both horizontal projected-gravity axes inverted: physical
forward reported -12.5 deg pitch and physical right reported +12.8 deg roll.
The measured mounting transform is now `root_x=chip_y`, `root_y=-chip_x`,
`root_z=chip_z`; level calibration then reduced stationary tilt to 0.028 deg.

`k2_attitude.py` maps chip -> root for gravity and chip -> root -> site for the
policy gyro. `k2_imu_motion_test.py` validates the frame-invariant relationship
`gravity_dot = -omega x gravity`; the corrected hardware measured correlation
0.990 and gain 1.004 over 369 moving samples.

---

## 10. The RL task — `k2_rl/`

Built fresh on mjlab's velocity machinery (never edit mjlab site-packages).
**One policy, two behaviours:** `GaitCommand` emits `[march, sin, cos]`; a
fraction of envs march each episode and the policy observes the bit, so the same
network holds still or steps in place depending on one runtime flag.

- Actor observations are **proprio-only** — gyro, projected gravity, joint
  pos/vel, last action, gait command. No base linear velocity, because the IMU
  velocimeter is not trustworthy on hardware. This is exactly what the Pi can
  supply.
- Actuators are modelled as STS3215 position servos with **kp=20 / kd=0.5, the
  same values `hardware/k2_bus.py` uses**, so the actuator sim-to-real gap is
  zero by construction.
- Domain randomization covers the gaps chased during bring-up: pseudo-inertia
  +-15%, foot friction 0.4-1.1, PD gains x0.8-1.2, encoder bias +-0.02 rad
  (home/calibration offset), base CoM offset, control latency, random pushes.
- Default pose is a symmetric standing crouch that is self-stable under kp/kd.

**Two gotchas that will recur:**
1. **Fusion chained joint names** — `hip_pitch` is a substring of the hip_roll
   joint name. Use the parent-anchored regexes in `k2_constants.JOINT_PATTERNS`,
   never a bare `.*hip_pitch.*`.
2. **`dr.pseudo_inertia` on `.*` bodies crashes** (cholesky "not positive
   definite") because it perturbs the massless free-joint carrier. Target only
   the massive links.

Reward tip: use quadratic penalties for "keep X near a target"; exp-kernel
rewards with small std go flat and give no gradient at large error.

---

## 11. Recurring Fusion-export bugs (expect on every re-export)
1. Every link tagged `Steel` (7850 kg/m³) → ~4× too heavy. Rescale per-link to
   measured print weight, add hardware point masses.
2. **Knee limits sign-inverted.**
3. `effort=100, velocity=1.0` placeholders → real STS3215 2.94 N·m / 4.712 rad/s.
4. Fusion is **Y-up**.
Joint/link names may also change between exports (e.g. lose Fusion `_2`/`_3`
suffixes) — grep for stale references after any re-export.

---

## 12. Quick command reference

```bash
cd ~/K2
# rebuild the twin after a CAD/limit change:
python scripts/phase1_mass.py && python scripts/phase2_make_mjcf.py
python -c "from hardware import k2_conventions as C; C.write_joint_limits()"

# hardware bring-up:
python -m hardware.set_servo_id --port /dev/ttyAMA0 --scan
DISPLAY=:1 python -m hardware.k2_calibrate                  # calibrate (GUI)
python -m hardware.check_joint_signs --joints all --amp 10  # verify signs, held in air
python -m hardware.calibrate_imu_level                      # IMU mounting tilt
python -m hardware.go_home --bus /dev/ttyAMA0               # ease to straight home

# validate, then run the policy (sim first, then real):
python -m hardware.test_default_crouch --bus sim --viewer
python -m hardware.k2_policy_run --bus sim --viewer --mode march \
  --policy policies/march_hold_v4_1999.onnx
python -m hardware.k2_policy_run --bus /dev/ttyAMA0 --mode hold \
  --policy policies/march_hold_v4_1999.onnx
```

Ctrl-C always drops the robot limp. On the real robot start slow, keep a hand
near it, and always run `--mode hold` before `--mode march`.
