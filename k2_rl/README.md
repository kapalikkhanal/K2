# k2_rl — RL locomotion for the K2 biped (mjlab)

First closed-loop RL policy for the 12-DOF K2. One env, one policy, a runtime
switch between two behaviours **in place** (zero net translation):

- **hold** — stand still on both feet, stay upright, reject pushes (IMU feedback).
- **march** — step in place, alternating feet.

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
| `mdp/gait_command.py` | `GaitCommand` — emits `[march, sin, cos]`; the hold/march switch + phase clock. |
| `mdp/rewards.py` | gait-aware rewards: `gait_contact`, `gait_swing`, `feet_planted`, `base_height_l2`. |
| `inplace_env_cfg.py` | the task: flat ground, zero-twist (stay in place), proprio-only actor obs (gyro + gravity + joints + gait — what the Pi can supply), domain randomization. |
| `ppo_cfg.py` | PPO runner (MLP 256-128-64, 3000 iters). |
| `tasks.py` | registers `Mjlab-InPlace-K2` (play→hold) and `Mjlab-InPlace-March-K2` (play→march); both share one checkpoint. |
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
python -m k2_rl.export_onnx --checkpoint logs/.../model_800.pt   # -> policy.onnx
```
Writes `policy.onnx` next to the checkpoint, with metadata (joint order,
stiffness/damping, default pose, `action_scale`, and gait frequency) baked in
for deployment. The
`VelocityOnPolicyRunner` also auto-exports an onnx on every save, but that file
is overwritten each save — use `export_onnx` to pin a specific checkpoint.

## Run the ONNX over the Bus — twin then real (`hardware/k2_policy_run.py`)

Same code, sim or real; the 45-D observation is rebuilt byte-for-byte from the
IMU (`hardware/k2_attitude.py`, sim or LSM6DS3) + joint feedback.

```bash
P=logs/rsl_rl/k2_inplace/<ts>_inplace_v1/policy.onnx

# Digital twin (MuJoCo viewer, wall-clock):
python -m hardware.k2_policy_run --bus sim --viewer --mode hold  --policy $P
python -m hardware.k2_policy_run --bus sim --viewer --mode march --policy $P

# Real robot (Pi) — calibrate first, copy policy.onnx over, keep a hand near it:
python -m hardware.k2_policy_run --bus /dev/ttyACM0 --mode hold  --policy $P --duration 15
python -m hardware.k2_policy_run --bus /dev/ttyACM0 --mode march --march-after 3 --policy $P
```

The runner eases into the default crouch, then hands over to the policy; torque
is always released on exit (incl. Ctrl-C). `--slew RAD_S` caps per-tick joint
motion on hardware (default 1.0 rad/s, matching training); `--march-after S`
holds a few seconds before marching. The runner reads the training cadence from
the ONNX metadata; `--gait-freq` is only a diagnostic override. Start in
**hold**, spot the robot, then **march**.
```
