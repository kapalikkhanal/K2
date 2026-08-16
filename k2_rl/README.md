# k2_rl — RL locomotion for the K2 biped (mjlab)

Closed-loop RL locomotion for the 12-DOF K2. The original policy switches
between two behaviours **in place** (zero net translation):

- **hold** — stand still on both feet, stay upright, reject pushes (IMU feedback).
- **march** — step in place, alternating feet.

`Mjlab-Forward-K2` is a separate low-speed task. It retains hold/march balance
and adds a commanded forward speed in the initial 0.01--0.04 m/s range. Its
training scene mixes flat ground with 2 mm/5 mm roughness and gentle waves,
while randomizing the measured hardware actuator latency and dynamics.

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
| `mdp/gait_command.py` | `GaitCommand` — emits `[march, sin, cos]` plus optional commanded forward speed. |
| `mdp/rewards.py` | gait-aware rewards: `gait_contact`, `gait_swing`, `feet_planted`, `base_height_l2`. |
| `inplace_env_cfg.py` | the task: flat ground, zero-twist (stay in place), proprio-only actor obs (gyro + gravity + joints + gait — what the Pi can supply), domain randomization. |
| `forward_env_cfg.py` | 1--4 cm/s forward extension; heading feedback, foam-friction randomization, and full-sole swing clearance. |
| `full_env_cfg.py` | robust omnidirectional task: ±4 cm/s x, ±2 cm/s y, and ±0.35 rad/s yaw. |
| `ppo_cfg.py` | PPO runner (MLP 256-128-64, 3000 iters). |
| `tasks.py` | registers the in-place hold/march pair plus separate forward and full-locomotion tasks. |
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
