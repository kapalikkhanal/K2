# k2_rl — RL locomotion for the K2 biped (mjlab)

Closed-loop RL locomotion for the 12-DOF K2. The original policy switches
between two behaviours **in place** (zero net translation):

- **hold** — stand still on both feet, stay upright, reject pushes (IMU feedback).
- **march** — step in place, alternating feet.

`Mjlab-Forward-K2` is a separate low-speed task. It retains hold/march balance
and adds a commanded forward speed in the initial 0.01--0.04 m/s range. Its
training randomizes sole height/angle, friction, actuator latency, and dynamics.

`Mjlab-Bidirectional-K2` is the conservative signed-speed successor. It trains
from scratch over -0.04--+0.04 m/s, keeps hold at the captured default crouch,
ramps both gait activation and velocity commands to prevent transition jerk,
and randomizes up to 5 mm unequal support, ±1.5° sole mounting, and light pushes.
Hold deliberately uses the previously proven learned support stance; it is not
forced into the narrower captured crouch because that variant fell in the twin.
The signed range is ±0.05 m/s, with a 0.8 s gait blend and 0.06 m/s² command
ramp for gentler hold/walk and forward/backward transitions.
Each gait begins at the v5 double-support crossing. Hold uses paired-joint
symmetry without forcing the exact default crouch, and walking hip-pitch
antisymmetry uses weight −4.

`Mjlab-Turn-K2` adds **one** thing to the bidirectional task: a commanded body
yaw rate. The command is `vx + wz` only -- there is no lateral (vy) channel.
Every term it changes is written so that with `wz == 0` it is *identical* to its
bidirectional predecessor, which is what makes warm-starting from a working
bidirectional checkpoint safe:

| bidirectional term | turning replacement | why |
|---|---|---|
| `base_heading_error` (world +X) | `heading_reference_error` | follow the *commanded* heading |
| `base_lateral_position_l2` (world +Y spawn line) | `lateral_path_deviation_l2` | corridor turns with the command |
| `base_yaw_rate_l2` (all yaw) | `yaw_rate_error_l2` | damp only yaw the command did not ask for |
| `track_angular_velocity` (twist, pinned to 0) | `track_yaw_rate_exp` | yaw target comes from the gait command |
| `heading_sin_cos` observation | `heading_error_sin_cos` | keep the feature near zero at any commanded yaw |
| `feet_lateral_vel_l2` (world Y) | `feet_lateral_vel_body_l2` | "sideways" is a body direction once you may turn |

`GaitCommand` integrates a **reference pose** (`heading_ref`, `pos_ref`) from the
commands it emits; those terms measure against it. With no yaw channel the
reference never rotates and never leaves the environment-origin line, so each
expression collapses to exactly its predecessor -- verified numerically to 0.0
max absolute difference. Deployment reproduces the same reference by integrating
the same emitted numbers, so this needs no extra sensing on the Pi.

Two further protections keep the straight gait from decaying: `rel_turning_envs`
gives 40% of resamples an exactly-zero yaw command (uniform sampling would make
straight walking measure-zero), and the hip-pitch antisymmetry penalty fades with
commanded yaw, because a turning gait is genuinely asymmetric. Hip-yaw pose
tolerance widens from 0.12 to 0.25 rad -- it is the steering joint; foot crossing
stays guarded by the unchanged inner-edge clearance and self-collision penalties.
Trained range is a deliberately conservative +/-0.25 rad/s with a 0.30 rad/s^2
command ramp.

`Mjlab-Full-K2` is isolated in `full_env_cfg.py`. It trains body-frame forward,
backward, left/right, and yaw-rate commands from scratch; it does not alter the
working forward-only policy interface.

The `march` bit is a per-episode command the policy observes (a fraction of envs
march, the rest hold), so a single trained policy does both — flip the bit at run
time. This is a closed balance-feedback loop: the policy reads the IMU every
tick and actively corrects lean and drift
(`robot.md` §10).

Built on mjlab's velocity-locomotion machinery, closely mirroring the Unitree G1
task (both are 6-DOF legs now), but written fresh — **never edit mjlab
site-packages**.

## Environment

Use the **`unitree_sim_env`** conda env (torch 2.7.0+cu128 with Blackwell/sm_120
GPU support + mjlab). The base env's torch does **not** support this GPU.

```bash
conda activate unitree_sim_env
cd ~/K2
```

## Package layout

| file | what |
|---|---|
| `k2_constants.py` | K2 `EntityCfg`: STS3215 position actuators (kp=20, kd=0.5 — same as `hardware/k2_bus.py`, so zero sim-to-real actuator gap), validated symmetric crouch keyframe, per-DOF regex patterns. |
| `mdp/gait_command.py` | `GaitCommand` — emits `[march, sin, cos]` plus optional signed forward speed and yaw rate, and integrates the reference pose those tasks measure against. |
| `mdp/rewards.py` | gait-aware rewards: `gait_contact`, `gait_swing`, `feet_planted`, `base_height_l2`. |
| `inplace_env_cfg.py` | the task: flat ground, zero-twist (stay in place), proprio-only actor obs (gyro + gravity + joints + gait — what the Pi can supply), domain randomization. |
| `forward_env_cfg.py` | 1--4 cm/s forward extension; heading feedback, foam-friction randomization, and full-sole swing clearance. |
| `bidir_env_cfg.py` | from-scratch -4--+4 cm/s forward/backward task with exact-crouch hold and smooth transitions. |
| `turn_env_cfg.py` | `vx + wz`: adds a +/-0.25 rad/s yaw-rate command to the bidirectional gait, measured against an integrated reference pose. |
| `expand_checkpoint.py` | widens a trained checkpoint by the appended command channels so it can be warm-started (zero weights, seeded normalizer, extended Adam moments). |
| `full_env_cfg.py` | robust omnidirectional task: ±4 cm/s x, ±2 cm/s y, and ±0.35 rad/s yaw. |
| `ppo_cfg.py` | PPO runner (MLP 256-128-64, 3000 iters). |
| `tasks.py` | registers the in-place hold/march pair plus the separate forward, bidirectional, turning, and full-locomotion tasks. |
| `validate_pose.py` | sanity-checks the default crouch stands (feet flat, symmetric, self-stable). |

The MJCF is the validated twin `mjcf/k2_physics.xml` (12 joints, feet-only
collision, IMU sensors). **Fusion naming gotcha:** joint names chain
`<parent>_<child>_…`, so `hip_pitch` also appears in the hip_roll joint name and
Always use the anchored patterns in `k2_constants.JOINT_PATTERNS`, never a
bare substring such as `.*hip_pitch.*`.

## Train

```bash
python -m k2_rl.train Mjlab-InPlace-K2 --env.scene.num-envs 4096 \
    --agent.max-iterations 3000 --agent.run-name inplace_v1

# Low-speed heading-aware forward walking (48-D actor observation).
python -m k2_rl.train Mjlab-Forward-K2 --env.scene.num-envs 4096 \
    --agent.max-iterations 2000 --agent.run-name forward_v1

# Forward/backward walking: train from scratch, with no --resume/load options.
conda run --no-capture-output -n unitree_sim_env \
  env MPLCONFIGDIR=/tmp/k2-mpl \
  python -m k2_rl.train Mjlab-Bidirectional-K2 \
    --env.scene.num-envs 4096 --agent.max-iterations 3000 \
    --agent.save-interval 100 --agent.run-name bidir_short_stride_v1

# Yaw turning: WARM-START from the working bidirectional checkpoint.
# Step 1 -- widen the checkpoint by the new command channel (48-D -> 49-D actor,
# 53-D -> 54-D critic). The new input column is ZERO, so the widened policy is
# numerically identical to the source until PPO learns to use it.
conda run --no-capture-output -n unitree_sim_env \
  python -m k2_rl.expand_checkpoint --task Mjlab-Turn-K2 \
    --checkpoint logs/rsl_rl/k2_bidir_v1/<run>/model_2700.pt
# -> logs/rsl_rl/k2_turn_v1/warmstart_<run>_model_2700/model_2700.pt
#
# Step 2 -- resume from it. --agent.max-iterations is ADDITIONAL iterations, and
# checkpoint numbering continues from the source (2700 -> 2700+N).
conda run --no-capture-output -n unitree_sim_env \
  env MPLCONFIGDIR=/tmp/k2-mpl \
  python -m k2_rl.train Mjlab-Turn-K2 --agent.resume True \
    --agent.load-run warmstart_<run>_model_2700 \
    --agent.load-checkpoint model_2700.pt \
    --env.scene.num-envs 4096 --agent.max-iterations 1500 \
    --agent.save-interval 100 --agent.run-name turn_v1_from_bidir_iter2700

# Full x/y/yaw locomotion (train from scratch; do not load a forward checkpoint).
python -m k2_rl.train Mjlab-Full-K2 --env.scene.num-envs 4096 \
    --agent.max-iterations 3000 --agent.run-name full_v1
```

Logs → `logs/rsl_rl/k2_inplace/<timestamp>_inplace_v1/`. An ONNX policy is
exported next to each checkpoint (`*.onnx`) for the Pi. TensorBoard:
`tensorboard --logdir logs/rsl_rl/k2_inplace`.

## Visualize a checkpoint (digital twin)

Point `--checkpoint-file` at a `model_*.pt` in the run dir. Native viewer:

```bash
RUN=logs/rsl_rl/k2_inplace/<timestamp>_inplace_v1

# HOLD (stand still, reject pushes)
python -m k2_rl.play Mjlab-InPlace-K2 --checkpoint-file $RUN/model_3000.pt

# MARCH in place (same policy, march bit forced on)
python -m k2_rl.play Mjlab-InPlace-March-K2 --checkpoint-file $RUN/model_3000.pt
```

Add `--viewer viser` to get a **Gait** dropdown (auto/hold/march) that switches
the behaviour live in the browser. `--num-envs 1` keeps it to a single robot.

## Export a checkpoint to policy.onnx

```bash
python -m k2_rl.export_onnx --task Mjlab-Forward-K2 \
    --checkpoint logs/.../model_950.pt --filename forward.onnx
```
Writes `policy.onnx` next to the checkpoint, with metadata (joint order,
stiffness/damping, default pose, `action_scale`, and gait frequency) baked in
for deployment. Forward-policy metadata also records its trained velocity range
instead of the fixed visualization command. The
`VelocityOnPolicyRunner` also auto-exports an onnx on every save, but that file
is overwritten each save — use `export_onnx` to pin a specific checkpoint.

## Run the ONNX over the Bus — twin then real (`hardware/k2_policy_run.py`)

Same code, sim or real; the runner rebuilds the policy's 45-D march, legacy
46-D forward, or heading-aware 48-D forward observation from IMU, integrated
relative yaw, joint feedback, phase clock, and known command.

```bash
P=logs/rsl_rl/k2_inplace/<ts>_inplace_v1/policy.onnx

# Digital twin (MuJoCo viewer, wall-clock):
python -m hardware.k2_policy_run --bus sim --viewer --mode hold  --policy $P
python -m hardware.k2_policy_run --bus sim --viewer --mode march --policy $P

# Forward policies only.
python -m hardware.k2_policy_run --bus sim --viewer --mode walk \
    --forward-speed 0.01 --policy policies/forward_walk_v1_iter950.onnx

# A bidirectional policy uses the same interface; negative means backward.
python -m hardware.k2_policy_run --bus sim --viewer --mode walk \
    --forward-speed -0.01 --policy policies/bidir_short_stride.onnx

# A turning policy adds --yaw-rate (rad/s, positive = left). It applies in
# mode=march (turn in place) and mode=walk (walk an arc), and is refused on a
# policy without a yaw channel.
python -m hardware.k2_policy_run --bus sim --viewer --mode walk \
    --forward-speed 0.02 --yaw-rate 0.15 --policy policies/turn_v1_iter<N>.onnx
python -m hardware.k2_policy_run --bus sim --viewer --mode march \
    --yaw-rate 0.20 --policy policies/turn_v1_iter<N>.onnx

# Real robot (Pi) — calibrate first, copy policy.onnx over, keep a hand near it:
python -m hardware.k2_policy_run --bus /dev/ttyAMA0 --mode hold  --policy $P --duration 15
python -m hardware.k2_policy_run --bus /dev/ttyAMA0 --mode march --march-after 3 --policy $P
python -m hardware.k2_policy_run --bus /dev/ttyAMA0 --mode walk \
    --forward-speed 0.01 --march-after 3 --duration 8 --fall-angle 10 \
    --policy policies/forward_walk_v1_iter950.onnx --log forward_walk_001.csv
```

The runner eases into the default crouch, then hands over to the policy; torque
is always released on exit (incl. Ctrl-C). `--slew RAD_S` caps per-tick joint
motion on hardware (default 1.0 rad/s, matching training); `--march-after S`
holds a few seconds before marching. The runner reads the training cadence from
the ONNX metadata; `--gait-freq` is only a diagnostic override. Start in
**hold**, spot the robot, then **march**.

The selected forward checkpoint measures 10.1--10.6 mm minimum clearance over
the entire sole at commands from 0.01 to 0.04 m/s in the nominal twin. Its first
hardware test must start at 0.01 m/s with the robot spotted; simulation results
are not an unsupported-hardware guarantee.
```

## Keyboard driving (`keyboard.py` twin, `real_keyboard.py` hardware)

Both run the real `hardware.k2_policy_run.run` path -- encoder quantization,
servo gains and limits, latency, ONNX Runtime -- so they exercise deployment
rather than the training viewer.

```
H  hold (also straightens the yaw command)   A  turn left    X  straighten
M  march in place                            D  turn right   Q  quit
W  walk forward     S  walk backward
```

A/D from HOLD start a turn in place. On a policy without a yaw channel A/D/X are
ignored and the turn keys are left out of the printed key list. `--yaw-rate` sets
the A/D magnitude (twin default 0.15 rad/s, hardware default 0.10 rad/s -- start
well below the trained maximum on the real robot).

## Observation sizes on the wire

| actor obs | policy | gait command |
|---|---|---|
| 45 | legacy in-place march | `[march, sin, cos]` |
| 46 | first forward | `+ vx` |
| 48 | heading-aware forward / bidirectional | `+ vx`, heading `[sin, cos]` |
| 49 | turning | `+ vx, wz`, heading is the ERROR `[sin, cos]` |

On a 49-D policy the heading channel is measured against a reference heading the
runner obtains by integrating the yaw rate it commanded itself, matching what
`GaitCommand` integrates in training. With no yaw command that reference stays at
zero and the channel is the plain relative heading, so 48-D policies are
unaffected -- verified by replaying `bidir_symmetric_swing12mm_iter2500.onnx`
through the updated runner with identical sole diagnostics.
