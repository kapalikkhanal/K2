# Optional staged training: warm up, then train together

**Current workflow:** use [turn first, then lateral](turn_lateral_training.md).
This supersedes the full-task workflow below after a stationary policy was
reported. The shared upright reward has also been corrected to the root frame.


The v6 default now samples all modes from scratch. This page describes an
explicit alternative; see [v6 design and training updates](v6_rl_update.md).

Only three tasks remain: `Mjlab-Turn-K2`, `Mjlab-Lateral-K2`, and
`Mjlab-Full-K2`. Use **Full** for the fresh run below. It trains forward,
backward, yaw turns, marching, hold, and later sideways walking. Height and
standing rotation are excluded. The previous logs directory has been removed.

The full task keeps smooth command acceleration/reversal/stopping, mirror-loss
regularization, landing and swing-velocity penalties, and 15 mm target clearance.
It uses a plane with inherited dynamics/contact randomization. These objectives
need trained-policy validation; they do not guarantee balanced or soft landings.

## 1. Turn/walk warmup: at least 1,000 PPO iterations

From the repository root, in your normal training environment:

```bash
python -m k2_rl.train Mjlab-Full-K2 \
  --env.scene.num-envs 1024 --env.commands.gait.rel-lateral-envs 0 \
  --agent.max-iterations 1000
```

This explicit override sets lateral sampling to zero. Forward/backward, both turns, march and hold
are trained. This means 1,000 **PPO iterations**, not 1,000 simulator steps;
each iteration collects 24 control steps per environment.

Check the result before adding lateral. If needed, continue the warmup:

```bash
python -m k2_rl.train Mjlab-Full-K2 \
  --env.scene.num-envs 1024 --env.commands.gait.rel-lateral-envs 0 --agent.resume True \
  --agent.load-run RUN --agent.load-checkpoint model_999.pt \
  --agent.max-iterations 500
```

Replace RUN and checkpoint filenames with actual entries under
`logs/rsl_rl/k2_full_turn_lateral_v1/`. Resume iterations are **additional**.
Use the latest accepted checkpoint, rather than assuming its suffix.

## 2. Enable lateral, preserving the warmup checkpoint

```bash
python -m k2_rl.enable_full_lateral \
  --checkpoint logs/rsl_rl/k2_full_turn_lateral_v1/RUN/model_999.pt
```

Execute the exact training command printed by this tool. It copies the checkpoint
into a new warmstart directory, refuses overwrite, initializes normalization for
the dormant lateral input, zeroes that input's weight column, and clears optimizer
momentum. Existing-command outputs are preserved at transfer. The original
checkpoint is untouched.

The printed command enables `--env.commands.gait.rel-lateral-envs 0.35`,
uses learning rate 0.0001, and runs 1,500 additional iterations. Approximately
35% of command samples practice lateral, 15% practice in-place modes, and the
rest retain the forward/backward/turn mixture. Hold sampling continues.

Both phases use 50 actor inputs, 55 critic inputs, and 12 actions, so there is
no network expansion. This is the same full task in both phases, with an explicit
checkpoint decision before lateral begins.

## 3. Continue joint training

Always retain the lateral sampling override when resuming the joint phase:

```bash
python -m k2_rl.train Mjlab-Full-K2 \
  --env.scene.num-envs 1024 \
  --env.commands.gait.rel-lateral-envs 0.35 \
  --agent.algorithm.learning-rate 0.0001 \
  --agent.resume True --agent.load-run JOINT_RUN \
  --agent.load-checkpoint model_2499.pt --agent.max-iterations 500
```

Without that override, the v6 config uses 30% lateral samples. Do not
run the initialization tool again on a joint checkpoint. Earlier motions keep
being optimized during joint training; rehearsal reduces forgetting but cannot
guarantee no regression.

## Evaluate and export

Use the same task for evaluation:

```bash
python -m k2_rl.eval_checkpoint --task Mjlab-Full-K2 \
  --checkpoint logs/rsl_rl/k2_full_turn_lateral_v1/JOINT_RUN/model_2499.pt \
  --num-envs 128 --vx 0.03 --wz 0.20 --vy 0.020 \
  --json /tmp/full-evaluation.json

python -m k2_rl.export_onnx --task Mjlab-Full-K2 \
  --checkpoint logs/rsl_rl/k2_full_turn_lateral_v1/JOINT_RUN/model_2499.pt \
  --filename accepted_policy.onnx
```

Use `--vy 0` when evaluating warmup. Compare both lateral directions, actual
displacement, falls, slip, landing loads, and all earlier motions. Inspect
release-to-stop and direction reversals in simulation before hardware use.
An iteration milestone does not establish that a gait has converged.

The export is written beside the checkpoint. Its metadata identifies the
trained lateral share, including disabling lateral controls for warmup exports.

```bash
python keyboard.py --policy logs/rsl_rl/k2_full_turn_lateral_v1/JOINT_RUN/accepted_policy.onnx
python keyboard.py --bus /dev/ttyAMA0 \
  --policy logs/rsl_rl/k2_full_turn_lateral_v1/JOINT_RUN/accepted_policy.onnx
```

Use the simulator command first; use the real command on the Pi with the robot
supported for initial checks. This change does not update ESP32 firmware.

## Implementation preflight

```bash
WARP_CACHE_PATH=/tmp/k2-warp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m k2_rl.verify_full --device cpu
```

This checks shared NumPy/Torch command smoothing, live physics, observation
layout, warmup/joint sampling, mirror transforms, a PPO update, checkpoint
transfer and warmup ONNX inference. It is not a
long training run or a hardware certification.
