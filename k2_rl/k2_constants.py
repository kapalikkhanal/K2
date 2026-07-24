"""K2 robot constants for mjlab RL.

Builds the mjlab ``EntityCfg`` for the 12-DOF K2 biped from the validated
digital twin (``mjcf/k2_physics.xml``). This is the K2 analogue of mjlab's
``unitree_g1/g1_constants.py``.

Key facts baked in here (see ``robot.md`` / the k2-twin memory):
  * 12 joints, 6 per leg: hip_pitch -> hip_roll -> hip_yaw -> knee ->
    ankle_pitch -> ankle_roll, mirrored L/R.
  * Actuators are Feetech STS3215 serial-bus *position* servos. The real
    control law is a joint-space PD: tau = kp*(q_des - q) - kd*qdot, saturated
    at the 2.94 N.m stall torque. We train with the SAME kp/kd that the
    deployment SimBus/real bus uses (``hardware/k2_bus.py`` DEFAULT_KP/KD =
    20 / 0.5) so the sim-to-real actuator gap is zero by construction.
  * The MJCF already encodes feet-only collision (foot geoms contype/conaffinity
    = 1, every other geom = 0), so no CollisionCfg override is needed.

FUSION NAMING GOTCHA (critical for every regex in the task):
  Fusion names each joint ``<parent>_<child>_<side>_joint``, so substrings
  collide -- e.g. ``hip_pitch`` appears in BOTH ``base_link_hip_pitch_..._joint``
  and ``hip_pitch_hip_roll_..._joint``; ``ankle_pitch`` appears in both the
  pitch joint and the (misspelled) roll joint ``anke_roll_ankle_pitch_...``.
  Always anchor on the parent token. The unique per-DOF patterns live in
  ``JOINT_PATTERNS`` below -- use them everywhere.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg, DelayedActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF and assets.
##

# Repo root is the parent of this package (~/K2).
K2_ROOT: Path = Path(__file__).resolve().parents[1]
K2_XML: Path = K2_ROOT / "mjcf" / "k2_physics.xml"
K2_MESHDIR: Path = K2_ROOT / "meshes" / "K2"
assert K2_XML.exists(), f"Missing MJCF: {K2_XML}"


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(K2_XML))
  assets: dict[str, bytes] = {}
  # spec.meshdir is the compiler meshdir ("../meshes/K2"); mirror mjlab's G1.
  update_assets(assets, K2_MESHDIR, spec.meshdir)
  spec.assets = assets
  return spec


##
# Per-DOF regex patterns (parent-anchored -- see module docstring).
##

JOINT_PATTERNS: dict[str, str] = {
  "hip_pitch": r"base_link_hip_pitch_(left|right)_joint",
  "hip_roll": r"hip_pitch_hip_roll_(left|right)_joint",
  "hip_yaw": r"hip_roll_hip_yaw_(left|right)_joint",
  "knee": r"hip_yaw_knee_(left|right)_joint",
  "ankle_pitch": r"knee_ankle_pitch_(left|right)_joint",
  "ankle_roll": r"anke_roll_ankle_pitch_(left|right)_joint",
}

# Body names of the two feet (carry the sole collision + foot sites).
FOOT_BODIES: tuple[str, str] = ("Feet_left", "Feet_right")
FOOT_SITES: tuple[str, str] = ("left_foot", "right_foot")
FOOT_HEEL_SITES: tuple[str, str] = ("left_foot_heel", "right_foot_heel")
FOOT_TOE_SITES: tuple[str, str] = ("left_foot_toe", "right_foot_toe")
FOOT_GEOMS: tuple[str, str] = ("left_foot_collision", "right_foot_collision")

##
# Actuator config (STS3215 position servo).
##

STS3215_STALL_TORQUE = 2.94  # N.m
STS3215_KP = 20.0  # matches hardware/k2_bus.py DEFAULT_KP
STS3215_KD = 0.5  # matches hardware/k2_bus.py DEFAULT_KD
STS3215_ARMATURE = 0.01  # matches the per-joint armature already in the MJCF

K2_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*",),  # all 12 joints share one servo type
  stiffness=STS3215_KP,
  damping=STS3215_KD,
  effort_limit=STS3215_STALL_TORQUE,
  armature=STS3215_ARMATURE,
)

##
# Default crouch keyframe.
##

# Symmetric standing crouch (MJCF joint coordinates; sim uses identity
# calibration so these equal the hardware `q` convention). Derived from the
# validated hardware crouch_pose.json, symmetrized L/R and pulled a few degrees
# off the ankle_pitch limit. Base z is solved so the soles rest on the floor
# (see validate_pose.py; recomputed there and pinned here).
_CROUCH_JOINTS: dict[str, float] = {
  JOINT_PATTERNS["hip_pitch"]: -0.40,
  JOINT_PATTERNS["hip_roll"]: 0.0,
  JOINT_PATTERNS["hip_yaw"]: 0.0,
  JOINT_PATTERNS["knee"]: -0.85,
  JOINT_PATTERNS["ankle_pitch"]: 0.45,
  JOINT_PATTERNS["ankle_roll"]: 0.0,
}

# Base height (m) of k2_root at the crouch. Pinned from validate_pose.py so the
# feet start flat on the ground with a hair of clearance.
CROUCH_BASE_HEIGHT = 0.318

CROUCH_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, CROUCH_BASE_HEIGHT),
  joint_pos=_CROUCH_JOINTS,
  joint_vel={".*": 0.0},
)

##
# Assemble.
##

K2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(K2_ACTUATOR,),
  soft_joint_pos_limit_factor=0.9,
)


def get_k2_robot_cfg(actuator_delay: bool = False) -> EntityCfg:
  """Fresh K2 EntityCfg instance.

  Training can randomize 0--10 ms of servo/bus latency (0--5 physics steps).
  Play/export remains deterministic and delay-free.
  """
  actuator = (
    DelayedActuatorCfg(
      base_cfg=K2_ACTUATOR,
      delay_target="position",
      delay_min_lag=0,
      delay_max_lag=5,
      delay_hold_prob=0.9,
      delay_update_period=10,
    )
    if actuator_delay
    else K2_ACTUATOR
  )
  articulation = EntityArticulationInfoCfg(
    actuators=(actuator,),
    soft_joint_pos_limit_factor=0.9,
  )
  return EntityCfg(
    init_state=CROUCH_KEYFRAME,
    collisions=(),  # MJCF already encodes feet-only collision.
    spec_fn=get_spec,
    articulation=articulation,
  )


# Action scale (rad) applied to the policy output before adding the default
# pose. STS3215 kp is low, so mjlab's effort/stiffness formula would give a tiny
# ~2 deg scale; we use a fixed 0.25 rad (~14 deg typical) so the policy has room
# to lift a foot for marching while staying well inside joint limits.
K2_ACTION_SCALE = 0.20


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_k2_robot_cfg())
  viewer.launch(robot.spec.compile())
