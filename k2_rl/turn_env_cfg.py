"""Yaw-rate turning on top of the validated bidirectional K2 gait.

This adds ONE thing to ``Mjlab-Bidirectional-K2``: a commanded body yaw rate.
There is no lateral (vy) command -- the command is ``vx + wz`` only. Everything
that makes the bidirectional policy work on hardware is inherited untouched:
the learned two-foot hold, the 0.75 Hz alternating clock, 12 mm full-sole swing,
the ramped hold<->walk handover, sole height/tilt randomization, and the light
push schedule.

Preserving the working gait is the whole design constraint, so the task is built
so that **every changed term is bit-identical to its bidirectional predecessor
whenever the commanded yaw rate is zero**:

  * The straight-line terms that measure against world +X and the world +Y spawn
    line (``heading``, ``base_xy_position``) now measure against the gait
    command's integrated reference pose. With ``wz == 0`` that reference never
    leaves the world +X axis through the environment origin, so the expressions
    reduce to exactly what they were.
  * ``yaw_rate`` becomes a penalty on yaw rate *in excess of* the command, which
    at ``wz == 0`` is the old unconditional yaw-rate penalty.
  * ``track_angular_velocity`` takes its yaw target from the gait command rather
    than from the twist command, which the walking tasks pin to zero -- same
    kernel, same std, same roll/pitch treatment.
  * The actor's heading observation becomes heading error relative to the same
    reference, so it stays in the near-zero range the warm-start checkpoint was
    trained on instead of sweeping the full circle once the robot turns.
  * ``feet_xy_vel`` moves from world-frame to body-frame lateral foot velocity,
    which is the same quantity while facing world +X and the correct one after.

Two further protections keep the straight gait from decaying:

  * ``rel_turning_envs`` gives a fixed share of resamples an exactly-zero yaw
    command, the same way ``rel_march_envs`` protects hold. Sampling the signed
    range uniformly would make straight walking a measure-zero event.
  * The hip-pitch antisymmetry penalty fades with commanded yaw. A turning gait
    is genuinely asymmetric -- the inside leg takes the shorter step -- and the
    penalty would otherwise charge for doing the task correctly.

Warm-start from the bidirectional checkpoint with ``k2_rl.expand_checkpoint``;
the extra command channel makes the actor observation 49-D and the critic 54-D.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from k2_rl import mdp
from k2_rl.bidir_env_cfg import make_k2_bidir_env_cfg
from k2_rl.k2_constants import FOOT_SITES, JOINT_PATTERNS
from k2_rl.mdp import GaitCommandCfg


# Deliberately conservative for a first turning policy, in the same spirit as
# the forward range that was walked back from 0.08 to 0.04 m/s. At the 0.75 Hz
# cadence (0.67 s per step) 0.25 rad/s asks for 9.5 degrees of yaw per step,
# which the hip-yaw joints can supply without a lateral weight-shift strategy.
# Hardware realizes roughly half of a commanded rate, so expect ~7 deg/s.
TRAIN_YAW_RATE_RANGE = (-0.25, 0.25)
PLAY_YAW_RATE = 0.15
# Share of resamples that receive a nonzero yaw command. The remaining 40%
# reproduce the bidirectional task exactly.
REL_TURNING_ENVS = 0.6
# Full-scale reversal (0.5 rad/s) in 1.7 s, matching the time the 0.06 m/s^2
# linear ramp takes to reverse across its own +/-0.05 m/s range.
YAW_RATE_RAMP_RPS2 = 0.30
# Yaw command at which the straight-gait symmetry penalty has faded to ~13%.
SYMMETRY_YAW_FADE_STD = 0.12
# Index of the yaw channel in the 5-D gait command [march, sin, cos, vx, wz].
YAW_COMMAND_INDEX = 4


def make_k2_turn_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_k2_bidir_env_cfg(play=play)

  gait = cfg.commands["gait"]
  assert isinstance(gait, GaitCommandCfg)
  gait.yaw_rate_range = (
    (PLAY_YAW_RATE, PLAY_YAW_RATE) if play else TRAIN_YAW_RATE_RANGE
  )
  gait.rel_turning_envs = 1.0 if play else REL_TURNING_ENVS
  gait.yaw_rate_ramp_rate_rps2 = YAW_RATE_RAMP_RPS2

  # The actor's heading feature must stay near zero at every commanded yaw rate,
  # otherwise turning walks the warm-started policy straight out of the input
  # range it was trained on. Reassigning the existing key preserves the term
  # order the deployment observation layout depends on (heading, then gait).
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    assert list(terms)[-2:] == ["heading", "gait"], list(terms)
    terms["heading"] = ObservationTermCfg(
      func=mdp.heading_error_sin_cos,
      params={"command_name": "gait"},
      noise=Unoise(n_min=-0.02, n_max=0.02),
      delay_max_lag=0 if play else 1,
    )

  # Follow the commanded heading rather than world +X.
  cfg.rewards["heading"] = RewardTermCfg(
    func=mdp.heading_reference_error,
    weight=-5.0,
    params={"command_name": "gait"},
  )
  # Stay in the corridor around the commanded path rather than the spawn line.
  cfg.rewards["base_xy_position"] = RewardTermCfg(
    func=mdp.lateral_path_deviation_l2,
    weight=-20.0,
    params={"command_name": "gait"},
  )
  # Damp only the yaw rate the command did not ask for.
  cfg.rewards["yaw_rate"] = RewardTermCfg(
    func=mdp.yaw_rate_error_l2,
    weight=-1.0,
    params={"command_name": "gait", "yaw_command_index": YAW_COMMAND_INDEX},
  )
  # Same kernel and std as the inherited term; the yaw target now comes from the
  # gait command instead of the permanently-zero twist command.
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=mdp.track_yaw_rate_exp,
    weight=1.5,
    params={
      "command_name": "gait",
      "std": math.sqrt(0.5),
      "yaw_command_index": YAW_COMMAND_INDEX,
    },
  )
  # "Sideways" is a body-frame direction once the robot may face anywhere.
  cfg.rewards["feet_xy_vel"] = RewardTermCfg(
    func=mdp.feet_lateral_vel_body_l2,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", site_names=list(FOOT_SITES))},
  )
  # Do not charge a turning gait for the stride asymmetry that turning requires.
  cfg.rewards["gait_symmetry"].params["yaw_command_index"] = YAW_COMMAND_INDEX
  cfg.rewards["gait_symmetry"].params["yaw_fade_std"] = SYMMETRY_YAW_FADE_STD

  # NOTE (2026-08-21): a +/-0.06 rad encoder_bias widening was tried here and
  # REVERTED. It rested on the belief that the robot stood ~5 deg more
  # asymmetric than the twin -- but comparing the SAME policy's hold pose in sim
  # vs hardware showed the twin's own hip-yaw split is +6.15 deg against the
  # robot's +4.99, i.e. the real robot is slightly MORE symmetric, and every
  # joint agrees within 1.5 deg. There was no asymmetry to be robust to.
  # Worse, telling the policy to distrust its encoders raised loop gain: on
  # hardware, action-rate RMS +51%, jerk +53..87%, and gyro energy in the known
  # 2.5-8 Hz closed-loop resonance band went from 54% to 72%. The twin cannot
  # see that failure -- it has no resonance -- so the 128-robot score IMPROVED
  # (5.9 -> 11.8) while the real robot got visibly worse. Do not widen this
  # without a hardware jitter metric in the loop.

  # Hardware root cause (2026-08-21): marching in place with ZERO yaw
  # commanded, the real robot yaws left at +0.61 deg/s while the twin yaws
  # +0.06 deg/s -- a 10x gap that matches the +0.54 deg/s additive offset
  # measured during commanded turns, and fully explains "left turns fast, right
  # turns slow" (it adds to +wz and subtracts from -wz). It is NOT sensor bias:
  # the static residual after ImuAttitude debiasing is +0.024 deg/s, 5% of it
  # (`hardware/test_imu_yaw_bias.py`).
  #
  # The cause is the gait's own left/right asymmetry. Hip yaw is by far the
  # worst joint -- swing amplitude R:L is 0.41 in sim and 0.58 on hardware, so
  # the left leg twists roughly twice as far as the right. Equal and opposite
  # leg twists cancel their reaction torque on the body; unequal ones ratchet it
  # around. The twin hides this because its feet do not slip or comply.
  #
  # `gait_antisymmetry_l2` already penalizes exactly this (u_R + u_L)^2 form and
  # is only applied to hip pitch today. Hip yaw wants the same treatment, faded
  # with commanded yaw so a real turn -- which is asymmetric on purpose -- is
  # not charged for it.
  cfg.rewards["gait_symmetry_yaw"] = RewardTermCfg(
    func=mdp.gait_antisymmetry_l2,
    weight=-4.0,
    params={
      "command_name": "gait",
      "right_cfg": SceneEntityCfg(
        "robot", joint_names=[r"hip_roll_hip_yaw_right_joint"]
      ),
      "left_cfg": SceneEntityCfg(
        "robot", joint_names=[r"hip_roll_hip_yaw_left_joint"]
      ),
      "yaw_command_index": YAW_COMMAND_INDEX,
      "yaw_fade_std": SYMMETRY_YAW_FADE_STD,
    },
  )

  # STRIDE LENGTH. At a fixed cadence, stride is not a free parameter:
  #   stride = commanded_speed / (2 * gait_freq)
  # so `feet_under_base` (below) only PERMITS a longer step, it cannot demand
  # one -- easing it from -0.25 to -0.15 measurably changed nothing. Raising the
  # commanded speed is the other arithmetic option and is the wrong lever (it
  # was walked back once already). That leaves cadence.
  #
  # 0.75 -> 0.58 Hz is a 30% longer step at the same speed, which takes the
  # measured forward hip-pitch swing from ~6.7 deg toward ~8.7 deg (+2 deg).
  # TRADE-OFF: a slower cycle means longer single support, and lateral rocking
  # at gait frequency is already what caps this robot's speed (it fell at
  # 0.025 m/s, and touched 13.8 of 14 deg at 0.018). Slower may give the policy
  # more time to correct, or may give the rocking more time to build. Watch
  # rollp2p and tilt in evaluation, not just the stride number.
  # 0.75 -> 0.58 was validated on hardware (backward roll p2p 19.8 -> 14.3 deg,
  # tilt max 13.4 -> 8.0, airborne-leg velocity RMS down 20-40%), so the worry
  # that longer single support would worsen the lateral rocking was wrong -- it
  # damped it. Operator asked for a further 25% step: 0.58 / 1.25 = 0.46 Hz.
  # That is a 1.09 s swing per step, 63% longer than the original 0.67 s.
  # 0.75 -> 0.58 Hz, validated on hardware: backward roll p2p 19.8 -> 14.3 deg,
  # tilt max 13.4 -> 8.0, airborne-leg velocity RMS down 20-40%, and it survived
  # 0.025 m/s -- the speed at which the 0.75 Hz gait fell. Longer single support
  # DAMPED the lateral rocking rather than feeding it.
  #
  # This is the exact cadence that produced turn_v5_iter2900. Two later
  # experiments were tried and REVERTED at the operator's request:
  #   0.46 Hz (+25% stride) -- worked, but visibly too slow.
  #   0.70 Hz with vx +/-0.07 -- faster legs, untested on hardware.
  cfg.commands["gait"].gait_freq = 0.58

  # `feet_under_base` is the one term whose PURPOSE is to keep
  # the feet under the body, so it is what caps step length -- the forward task
  # already halved it from -0.5 to -0.25 when the speed range was raised. Ease
  # it a little further for a slightly longer step. Deliberately a small move:
  # this term is also what stops the swing foot landing far ahead of the body,
  # and the robot is already close to a lateral-rocking limit at 0.018 m/s.
  cfg.rewards["feet_under_base"].weight = -0.15

  # DOMAIN RANDOMIZATION -- inherited from the bidirectional task, unchanged.
  # This is the turn_v5_iter2900 recipe. Widenings tried and reverted:
  #   encoder_bias +/-0.06 BACKFIRED on hardware (action rate +51%, jerk +87%,
  #     gyro energy in the 2.5-8 Hz resonance band 54% -> 72%) while the twin
  #     scored it two points HIGHER, because the twin has no resonance.
  #   friction 0.2-1.5 / inertia +/-20% / sole 6 mm / vx +/-0.07 (run v8) --
  #     scored well (14.58 vs 13.85) but the operator judged the resulting gait
  #     a limp on hardware. Reverted in favour of fixing the limp directly.

  # LEG SYMMETRY. Operator observation on hardware: the robot limps -- and the
  # logs agree, hip-pitch swing measured 8.3 deg right against 9.7 left in
  # forward walking (~15% mismatch), with the same split backward.
  #
  # `gait_symmetry` above already penalizes (u_R + u_L)^2 on hip pitch, which
  # does see amplitude mismatch, but weakly: a 1.4 deg mismatch is 0.024 rad, so
  # the whole term is worth 6e-4 before weighting and the policy can afford to
  # ignore it. And that form is simply WRONG for knee and ankle pitch, whose
  # antiphase sum is a nonzero constant (see the gait_antisymmetry_l2 docstring)
  # -- penalizing it there fights their normal shape.
  #
  # This instead samples each leg at ITS OWN swing apex and compares like with
  # like, so it reads amplitude mismatch directly on every sagittal joint.
  cfg.rewards["leg_symmetry"] = RewardTermCfg(
    func=mdp.swing_amplitude_symmetry_l2,
    weight=-25.0,
    params={
      "command_name": "gait",
      "right_cfg": SceneEntityCfg(
        "robot",
        joint_names=[
          "base_link_hip_pitch_right_joint",
          "hip_yaw_knee_right_joint",
          "knee_ankle_pitch_right_joint",
        ],
      ),
      "left_cfg": SceneEntityCfg(
        "robot",
        joint_names=[
          "base_link_hip_pitch_left_joint",
          "hip_yaw_knee_left_joint",
          "knee_ankle_pitch_left_joint",
        ],
      ),
    },
  )

  # Hip yaw is the steering joint. The bidirectional 0.12 rad tolerance charges
  # nearly the whole per-joint pose reward for the ~0.17 rad excursion a step of
  # the commanded turn needs. Widen it to the value the omnidirectional task
  # already uses; foot crossing stays guarded by the (unchanged) inner-edge
  # clearance and self-collision penalties, not by this pose term.
  cfg.rewards["pose"].params["std_walking"][JOINT_PATTERNS["hip_yaw"]] = 0.25

  return cfg
