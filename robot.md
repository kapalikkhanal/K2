# K2 — 12-DOF Biped Robot: Complete Reference

A self-contained knowledge dump for the K2 walking robot: the hardware, the
digital twin, the software stack, the calibration, the IK walk, and the road to
RL. Written so another engineer or LLM can pick up with full context.

Repo root: `~/K2`. Intended to be **open-sourced**, so keep it clean — only
files that are strictly needed.

---

## 1. What K2 is

K2 is a small **12-DOF bipedal robot** (two 6-DOF legs, no upper body). It was
designed in Fusion 360 (current export: **Robot_v4**, 2026-07) and is developed
**digital-twin-first**: every change is validated in a MuJoCo model before it
touches hardware, and the *same control code* drives sim or the real robot by
changing one flag.

- **Total mass:** ~1056 g
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
| ankle_roll_R/L | `anke_roll_ankle_pitch_{right,left}_joint` | roll (world X) | **[−20, +50]** |

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

### Servo ID map (set 2026-07-21) — clean 0..11 chain, both ankle-rolls at the ends
Right leg: `ankle_roll 0, ankle_pitch 1, knee 2, hip_yaw 3, hip_roll 4, hip_pitch 5`.
Left leg: `hip_pitch 6, hip_roll 7, hip_yaw 8, knee 9, ankle_pitch 10, ankle_roll 11`.
R/L = the robot's **own** right/left (robot facing you → its right foot is on your
left). **ID 0 is valid** and works through the SDK. The map lives in
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
- **Verified signs (on hardware):** hip_pitch R−1/L+1, hip_roll **R+1/L+1**,
  hip_yaw R−1/L+1, knee R−1/L+1, ankle_pitch R−1/L+1, ankle_roll R−1/L+1.
  Most pairs are **mirror-opposite** (mirror-mounted servos); hip_roll is the one
  same-sign pair — confirmed correct on hardware.

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
- **`k2_walk_ik.py`** — the **IK walk** (see §7).
- **`test_swing.py`** — single-step / one-foot-balance test: eases into the
  default crouch, waits for `go`, shifts onto the stance foot + lifts the swing
  foot, returns to the crouch and holds it. Great for isolating balance.
- **`test_crouch.py`** — crouch + gentle lateral sway (both feet planted).
- **`k2_kinematics.py`** — pure-numpy forward kinematics (`k2_kinematics.npz`
  baked from the MJCF) so the Pi computes base height with no MuJoCo.
- **`go_home.py`**, **`set_servo_gain.py`**, **`set_servo_id.py`**,
  **`check_lateral_signs.py`**, **`k2_sway.py`** (one-foot reach test),
  **`k2_attitude.py`** + **`calibrate_imu_level.py`** (IMU).

Config files (checked in): `calibration.json`, `joint_limits.json`,
`crouch_pose.json` (the default crouch), `k2_kinematics.npz`,
`imu_level_calibration.json`.

---

## 7. The IK walk (quasi-static, **no RL**)

`hardware/k2_walk_ik.py`. A statically-stable test gait with bounded LSM6DS3
roll/pitch feedback — only possible because the v4 ankle roll makes single-foot
support statically stable.

- **`WalkIK`** — whole-body inverse kinematics. Given each foot's target
  (x, y, height = 0 planted / >0 lifted-and-level) and a CoM target (x, y), it
  solves the base pose + 12 joints with the feet held flat, **torso level**
  (base roll/pitch removed as unknowns). The lateral shift is taken up by
  hip_roll + ankle_roll. `base_roll` adds a fixed lateral **trim** (see below).
- **`build_walk()`** — one repeatable step cycle: shift CoM over the stance foot
  → lift & swing the other foot forward → set down → shift across → repeat. Solves
  IK per tick (warm-started; reseed-free — the current defaults converge cleanly),
  and validates the CoM stays inside the support polygon (`--check`).
- The joint path is C2-smoothed and time-scaled before playback (default limits
  0.8 rad/s and 25 rad/s²), removing the high-speed swing-leg changes that
  produced a visible jerk on hardware.
- On hardware, `TiltStabilizer` is enabled by default. It estimates torso roll
  and pitch from `ImuAttitude`, applies bounded PD feedback through IK-derived
  coordinated hip/ankle responses, and releases torque above 12° tilt.
- Played through `k2_ctrl.run` (same slew/torque-safe loop).

**Current tuned defaults** (stance widened to leverage the 20° ankle):
step 2.5 cm, lift 2.5 cm, **com_shift 39 mm**, **stance_y 55 mm**, level torso,
4 s/step before automatic time-scaling. → statically stable in sim (thin
margin), foot clearance ~62 mm during swing (no self-collision).

```
python -m hardware.k2_walk_ik --bus sim --viewer   # preview
python -m hardware.k2_walk_ik --check              # stability numbers only
python -m hardware.k2_walk_ik --bus /dev/ttyACM0   # real + IMU feedback; prompts 'go'
```

**The `base_roll` trim:** `--base-roll DEG` changes the feedback target and the
nominal IK torso roll. Its default is now zero; only add a small fixed trim after
level-calibrating the IMU and observing a consistent one-sided bias.

**Known real-vs-sim gaps / lessons on the walk:**
- Thin static margin (~1–3 mm) still means the real robot needs spotting. The
  IMU can reject torso angular error, but it cannot measure foot contact or CoM
  position and cannot recover once the CoM has left the support polygon.
- Feet self-collided during swing at a narrow (48 mm) stance because the ankle
  was limited to 12° inward; **widening to 55 mm** (with the 20° ankle) fixed it.
- Warm-start IK can "poison" later ticks if one fails; the leg-switch jerk came
  from a discontinuous CoM-x target (ease CoM onto the stance foot *during* the
  shift, not at swing start).

---

## 8. Where the real robot currently stands (as of 2026-07-22)

- All 12 servos calibrated; home re-captured from a truly-straight pose (an
  earlier home had hip_yaw_R ~7° and knee_L ~5.5° off, which caused a persistent
  left lean). Signs all verified on hardware (§4).
- `test_crouch` / `test_swing` work; right-stance weight-shift confirmed good.
- The IK walk runs on hardware but is **open-loop and marginal** — it steps but
  wobbles/tilts and needs spotting. Tuning knobs: `--base-roll` (trim),
  `--com-shift`, `--stance-y`, `--t-step` (slower), `--lift`.
- A small residual left tilt is being trimmed out with `base_roll` + a per-joint
  ankle_roll_L home offset (~+4°).

---

## 9. Controller: Raspberry Pi 5 (not Jetson)

Benchmarked the policy MLP at **0.012 ms/inference** on a single Pi 5 CPU thread
(~0.06% of a 50 Hz budget). GPU is only for *training* or vision models. The Pi 5
is lighter/lower-power (better on `base_link`) and its I2C/SPI eases the IMU. A
Jetson would only be justified if cameras / a VLA are added later.

IMU on the Pi: **LSM6DS3**, mounted flat on `base_link`. In the MJCF the IMU site
frame matches the physical chip so gyro/accel come out un-permuted; resting accel
reads +9.81 on IMU Z. `k2_attitude.py` filters gravity + de-biases the gyro and
rotates into the root frame for a walking policy's observation.

---

## 10. The road to a robust walk: RL

The open-loop IK walk proves the **hardware and kinematics can do it** (shift
weight, balance on one foot, lift, swing, step). What it fundamentally lacks is a
**feedback loop for balance** — which is exactly the drift/tilt seen on hardware.

**RL is the right next step.** A trained policy reads the IMU every tick
(angular velocity + projected gravity) and actively corrects lean/drift. Ground
already laid:
- 12 DOF with ankle roll → static one-foot support exists (the mechanical
  prerequisite).
- A validated digital twin that now matches the real robot well (masses, limits,
  and the calibration work close the sim-to-real gap).
- The MJCF already exposes the IMU sensors the policy needs.

To build it: a fresh **12-DOF training env from mjlab's velocity-locomotion
template** (the old `k2_walk_rl`/`k2_squat_rl` packages were deleted — build new,
never edit mjlab site-packages), with **domain randomization** (mass, friction,
control latency, and small calibration/home offsets — exactly the real-world
mismatches chased during bring-up) so the policy transfers. Export to ONNX, run on
the Pi over the same `Bus` abstraction. Reward tip: use quadratic penalties for
"keep X near a target"; exp-kernel rewards with small std go flat and give no
gradient at large error.

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
sudo chmod 666 /dev/ttyACM0                              # after every replug
python -m hardware.set_servo_id --port /dev/ttyACM0 --scan
DISPLAY=:1 python -m hardware.k2_calibrate               # calibrate (GUI)
python -m hardware.go_home --bus /dev/ttyACM0            # ease to straight home

# tests (sim first, then real):
python -m hardware.test_crouch --bus sim --viewer
python -m hardware.test_swing  --bus /dev/ttyACM0            # right stance
python -m hardware.test_swing  --bus /dev/ttyACM0 --stance L # left stance
python -m hardware.k2_walk_ik  --bus /dev/ttyACM0            # the IK walk
```

Ctrl-C always drops the robot limp. On the real robot, start slow, keep a hand
near it, and spot it — the IK walk is open-loop.
