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

  # Hip yaw is the steering joint. The bidirectional 0.12 rad tolerance charges
  # nearly the whole per-joint pose reward for the ~0.17 rad excursion a step of
  # the commanded turn needs. Widen it to the value the omnidirectional task
  # already uses; foot crossing stays guarded by the (unchanged) inner-edge
  # clearance and self-collision penalties, not by this pose term.
  cfg.rewards["pose"].params["std_walking"][JOINT_PATTERNS["hip_yaw"]] = 0.25

  return cfg
