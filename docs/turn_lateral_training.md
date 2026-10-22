# K2 staged training: turn, then lateral, then the ESP32

One pipeline, run top to bottom. Everything runs in the `unitree_sim_env`
conda environment from the repository root. Stage one takes about 25 minutes
on the RTX PRO 5000 (0.9 s/iteration at 4096 envs); stage two about 30.

Why two stages: the gait, the hold and every robustness decision live in
`k2_rl/turn_env_cfg.py` (`Mjlab-Turn-K2`, 49-D actor). `k2_rl/lateral_env_cfg.py`
(`Mjlab-Lateral-K2`, 50-D actor) only adds the `vy` channel and blends in the
terms a side step needs; at `vy = 0` every reward is bit-identical to stage
one. A trained turn policy is therefore widened, not retrained.

## What changed on 2026-09-15 (why the last policy crept and staggered)

* **`upright` was dead.** It read the Y-up `base_link`, so a level robot scored
  0.0068 every run (check any old `Episode_Reward/upright`). Now reads the
  Z-up root. `sole_height` randomization was sliding the soles fore/aft, not
  vertically (`axes` 2 -> 1), and `sole_tilt` was spinning them about the
  vertical instead of tilting them (`pitch_range` -> `yaw_range`). Base CoM
  randomization is now widest fore/aft and vertical, where the battery sits.
* **IMU bias randomization.** Per-episode constant offsets on projected
  gravity (~2 deg) and gyro (~1 deg/s). The policy must stand on flat soles at
  the default crouch rather than on the exact gravity reading.
* **Hold is shaped on the feet:** `hold_sole_alignment` (all sole corners on
  the floor, feet parallel), `hold_stance_stagger` (feet side by side),
  `hold_feet_still`, in addition to `hold_pair_symmetry`, `hold_stance_width`
  and `feet_planted`. All `1 - march` gated: the gait is untouched.
* **Swing apex 12 -> 15 mm** (`SWING_HEIGHT_M`, tolerance 5 mm) plus a
  march-gated `swing_shortfall` hinge that keeps a gradient alive when a foot
  drags. Smoothness: `action_acc_l2` -0.03, `action_rate_limit` tops out at
  -0.10 instead of -0.06.
* `REL_MARCH_ENVS` 0.90 -> 0.85 so the new hold terms get data.
* Firmware (`esp32-s3/src/policy_robot.cpp`): the heading reference is
  re-anchored while holding, the gyro bias is refined during hold, the boot
  bias capture averages 150 distinct samples over 1.5 s, and one late tick no
  longer latches a stop mid-walk (three in a row do).

## Stage one: Mjlab-Turn-K2 (from scratch)

```bash
conda activate unitree_sim_env
python -m k2_rl.train Mjlab-Turn-K2 --env.scene.num-envs 4096 --agent.max-iterations 1500
```

Do not warm-start from a run older than 2026-09-15: the reward structure
changed and the old critic is fitted to the old scoreboard. Checkpoints land in
`logs/rsl_rl/k2_turn_v1/<date>/model_N.pt`; `last_policy.onnx` there is always
the newest one with full metadata.

Evaluate by behaviour, never by training reward, and always evaluate the final
checkpoint explicitly (`--sweep --stride` skips non-multiples):

```bash
RUN=logs/rsl_rl/k2_turn_v1/<date>
python -m k2_rl.eval_checkpoint --task Mjlab-Turn-K2 \
  --checkpoint $RUN/model_1000.pt $RUN/model_1250.pt $RUN/model_1500.pt \
  --num-envs 128 --vx 0.03 --wz 0.20 --json /tmp/k2-turn-eval.json
```

Reference (the 2026-09-14 stage-one checkpoint, before these fixes): score
18.2, fwd/back tracking 0.97/1.04, turns 0.96, hold stagger 0.85 deg, apex
10 mm. Gate for advancing a checkpoint: every phase survives, `vx` and `wz`
ratios >= 0.9, hold `down 1.00/1.00` with `hpsplit` under 2 deg, apex >= 12 mm,
inner gap >= 8 mm, no `JITTER` note.

Twin check before anything touches hardware (hold for a full minute, then
each direction):

```bash
python keyboard.py --policy $RUN/last_policy.onnx --bus sim
```

## Stage two: Mjlab-Lateral-K2 (widen the accepted turn checkpoint)

```bash
python -m k2_rl.prepare_lateral --checkpoint $RUN/model_1500.pt
```

The converter copies the checkpoint, appends a zero weight column for `vy`,
seeds its normalizer, clears optimizer moments and iteration counter, records
the source, and refuses full-task, already-converted or mismatched-MJCF
checkpoints. It prints the exact training command; it is:

```bash
python -m k2_rl.train Mjlab-Lateral-K2 --env.scene.num-envs 4096 \
  --agent.algorithm.learning-rate 0.0001 --agent.algorithm.schedule fixed \
  --agent.resume True --agent.load-run lateral_start_<date>_model_1500 \
  --agent.load-checkpoint model_1500.pt --agent.max-iterations 2000
```

2000 iterations, not 1500: on 2026-09-14 lateral tracking was still rising at
1499 (0.46 of command) and the roll envelope was measured NOT to be the
limit. Evaluate with the sideways phases included and confirm stage one did
not regress:

```bash
LAT=logs/rsl_rl/k2_lateral_v1/<date>
python -m k2_rl.eval_checkpoint --task Mjlab-Lateral-K2 \
  --checkpoint $LAT/model_1500.pt $LAT/model_1750.pt $LAT/model_1999.pt \
  --num-envs 128 --vx 0.03 --wz 0.20 --vy 0.02 --json /tmp/k2-lateral-eval.json
python keyboard.py --policy $LAT/last_policy.onnx --bus sim
```

Gate: the stage-one gates still hold, both `lat_L`/`lat_R` progress ratios
>= 0.6 and within 25% of each other, no `LATERAL REVERSED`.

## Export and deploy to the ESP32-S3

`last_policy.onnx` in the run directory already carries the deployment
contract. To export a specific checkpoint instead:

```bash
python -m k2_rl.export_onnx --task Mjlab-Lateral-K2 --checkpoint $LAT/model_1999.pt \
  --filename policies/last_policy.onnx
```

Never deploy `<run>.onnx` (mjlab's auto-export): it lacks the gait metadata
and runs at the wrong cadence silently. Then:

```bash
policy=policies/last_policy.onnx
python esp32-s3/tools/export_policy.py "$policy" esp32-s3/src/policy_weights.h
python esp32-s3/tools/export_k2_config.py "$policy" esp32-s3/src/k2_config.h
python esp32-s3/tests/verify.py --policy "$policy"
~/.platformio/penv/bin/pio run -d esp32-s3 -e robot -t upload --upload-port /dev/ttyACM0
~/.platformio/penv/bin/pio device monitor --port /dev/ttyACM0
```

Before the first trial with a re-mounted IMU or a new battery, redo the level
calibration (`hardware/imu_level_calibration.json`) and compare the boot line
`g_chip [...]` against the value it prints as expected: a mismatch means the
firmware's tilt is wrong from the first tick. Boot with the robot on the floor
and untouched for the 1.5 s bias capture. In the 0.5 s telemetry line, `herr`
is the heading error the policy sees (should sit near 0 while holding) and
`bz` the refined yaw-gyro bias in deg/s.

Hardware limits are in the firmware config: vx 0.05 m/s, wz 0.25 rad/s,
vy 0.03 m/s, with `keyboard.py` capping lateral at 0.015 on hardware.
