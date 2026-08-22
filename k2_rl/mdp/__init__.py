"""MDP functions for K2 in-place tasks.

Re-exports mjlab's velocity-task MDP (rewards, observations, events, terms) and
adds the K2-specific gait command + gait-aware rewards.
"""

# Reuse everything mjlab already provides for legged locomotion.
from mjlab.tasks.velocity.mdp import *  # noqa: F401,F403
from mjlab.tasks.velocity import mdp as _vel_mdp  # noqa: F401

# K2 custom terms.
from k2_rl.mdp.actions import (  # noqa: F401
  DelayedJointPositionAction,
  DelayedJointPositionActionCfg,
)
from k2_rl.mdp.gait_command import GaitCommand, GaitCommandCfg  # noqa: F401
from k2_rl.mdp.observations import (  # noqa: F401
  heading_error_sin_cos,
  heading_sin_cos,
)
from k2_rl.mdp.rewards import (  # noqa: F401
  action_rate_soft_limit_l2,
  base_ang_vel_xy_l2,
  base_height_l2,
  base_heading_error,
  base_lateral_position_l2,
  base_lateral_velocity_l2,
  base_yaw_rate_l2,
  base_lin_vel_xy_l2,
  base_tilt_l2,
  base_xy_position_l2,
  feet_planted,
  feet_lateral_clearance_l2,
  feet_lateral_vel_body_l2,
  feet_lateral_vel_l2,
  feet_under_base_l2,
  feet_xy_vel_l2,
  gait_antisymmetry_l2,
  gait_contact,
  gait_swing,
  heading_reference_error,
  hold_joint_pair_symmetry_l2,
  lateral_path_deviation_l2,
  swing_amplitude_symmetry_l2,
  swing_sole_height_spread_l2,
  track_forward_velocity_exp,
  track_yaw_rate_exp,
  yaw_rate_error_l2,
)
