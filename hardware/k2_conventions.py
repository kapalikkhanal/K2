"""Shared conventions for K2: joint order, servo IDs, and count <-> radian math.

Everything above the bus layer uses these, so the sim and the real robot are
driven by identical code. Joint limits are read from the MJCF rather than
hardcoded, so they cannot drift out of sync with the model.

STS3215: 0..4095 counts per revolution. Registers 42 GoalPos, 56 PresentPos,
58 PresentSpeed (sign-magnitude, bit15 = direction).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MJCF = ROOT / "mjcf" / "k2_physics.xml"
HERE = Path(__file__).resolve().parent
CALIB_PATH = HERE / "calibration.json"
LIMITS_PATH = HERE / "joint_limits.json"

COUNTS_PER_REV = 4096
RAD_PER_COUNT = 2.0 * math.pi / COUNTS_PER_REV
COUNTS_PER_RAD = COUNTS_PER_REV / (2.0 * math.pi)

# Never command a goal outside this. The STS3215 is single-turn, so a goal
# that wraps past 0/4095 makes the servo take the long way round (~300 deg).
# Saturate instead of wrapping. See the re-homing note in the hardware memory.
COUNT_MIN, COUNT_MAX = 40, 4055

# Policy/observation joint order: right leg hip->toe, then left leg hip->toe.
# Robot_v4 (2026-07-21): a 6th DOF per leg. The old single "ankle" is now
# ankle_pitch (the sagittal ankle, still driven by the same servo) plus a new
# ankle_roll below it (the added servo). 12 joints total.
SIM_ORDER = (
    "hip_pitch_R", "hip_roll_R", "hip_yaw_R", "knee_R", "ankle_pitch_R", "ankle_roll_R",
    "hip_pitch_L", "hip_roll_L", "hip_yaw_L", "knee_L", "ankle_pitch_L", "ankle_roll_L",
)

# Physical servo IDs -- CONFIRMED CORRECT on hardware 2026-07-28.
#
# Two transpositions were tried on the real robot that day and BOTH made it
# worse, so do not "fix" this map again without a held-air check first:
#   * hips  4/7 <-> 5/6 (pitch/roll)  -> worse
#   * ankles 0/1 and 10/11 (pitch/roll) -> worse
# Both had already been contradicted by check_joint_signs, where every joint
# matched POS_DESC at +-10..17 deg.
#
# The map is a clean daisy-chain: right foot (0) up to the right hip (5),
# across to the left hip (6), down to the left foot (11).
#
# CAUTION: these checks are all MIRROR-SYMMETRIC in wording ("outward",
# "pigeon-toed", "toe up"), so they confirm each joint's AXIS and SIGN but are
# blind to whether the "_R" servos are on the leg the MJCF calls right. See the
# note on JOINT_TO_MJCF below -- the MJCF's `right_*` bodies sit at +y, which is
# the robot's own LEFT.
# LEGS SWAPPED 2026-07-28, proven on hardware. The MJCF's `_R` joints drive the
# servos on the OTHER physical leg (IDs 6..11), and `_L` drives 0..5. Measured
# back-to-back on the real robot with the same policy and the same 10 deg tilt
# cutoff: unswapped safety-stopped at 11.4 deg after 10.5 s of marching, swapped
# ran the full 15 s at 3.9 deg. The decisive number is not survival but gait
# coherence -- the fraction of base roll locked to the 0.75 Hz gait clock went
# 60% -> ~100% (amplitude 1.45 -> 2.57 deg at identical total roll), i.e. the
# lateral weight shift now loads the leg that is actually planting.
#
# calibration.json's R/L entries were swapped by the SAME permutation, so every
# servo keeps its own measured home_raw/sign (backup: calibration.json
# .prelegswap.bak). Signs stay valid because every POS_DESC is mirror-symmetric
# ("outward", "pigeon-toed", "toe up"), which is also precisely why none of the
# held-air sign checks could ever have caught this.
#
# PREVIOUS (unswapped) MAP, if this ever needs reverting -- revert BOTH this and
# calibration.json together:
#   "ankle_roll_R": 0, "ankle_pitch_R": 1, "knee_R": 2, "hip_yaw_R": 3,
#   "hip_roll_R": 4, "hip_pitch_R": 5, "hip_pitch_L": 6, "hip_roll_L": 7,
#   "hip_yaw_L": 8, "knee_L": 9, "ankle_pitch_L": 10, "ankle_roll_L": 11
JOINT_TO_ID = {
    "ankle_roll_R": 11, "ankle_pitch_R": 10, "knee_R": 9, "hip_yaw_R": 8, "hip_roll_R": 7, "hip_pitch_R": 6,
    "hip_pitch_L": 5, "hip_roll_L": 4, "hip_yaw_L": 3, "knee_L": 2, "ankle_pitch_L": 1, "ankle_roll_L": 0,
}
IDS = [JOINT_TO_ID[j] for j in SIM_ORDER]

# Short joint name -> exact Robot_v4 MJCF joint name.
JOINT_TO_MJCF = {
    "hip_pitch_R": "base_link_hip_pitch_right_joint",
    "hip_roll_R": "hip_pitch_hip_roll_right_joint",
    "hip_yaw_R": "hip_roll_hip_yaw_right_joint",
    "knee_R": "hip_yaw_knee_right_joint",
    "ankle_pitch_R": "knee_ankle_pitch_right_joint",
    "ankle_roll_R": "ankle_roll_foot_right_joint",
    "hip_pitch_L": "base_link_hip_pitch_left_joint",
    "hip_roll_L": "hip_pitch_hip_roll_left_joint",
    "hip_yaw_L": "hip_roll_hip_yaw_left_joint",
    "knee_L": "hip_yaw_knee_left_joint",
    "ankle_pitch_L": "knee_ankle_pitch_left_joint",
    "ankle_roll_L": "ankle_roll_foot_left_joint",
}
MJCF_TO_JOINT = {v: k for k, v in JOINT_TO_MJCF.items()}

# What a nudge in NUDGE_DIR does, measured off the MJCF (not guessed) so the
# operator can confirm each sign by eye. Each description names a motion that
# is unambiguous from one viewing angle -- "leg rotates (yaw)" is not good
# enough, because it reads the same in both directions.
# Descriptions below were re-measured from the current Robot_v4 MJCF. They
# describe the direction used by NUDGE_DIR, not the raw encoder direction.
POS_DESC = {
    "hip_pitch_R": "right foot swings BACKWARD",
    "hip_roll_R": "right foot swings OUTWARD, away from the left leg",
    "hip_yaw_R": "right TOE rotates INWARD (pigeon-toed)",
    "knee_R": "right knee BENDS (heel rises relative to toe)",
    "ankle_pitch_R": "right TOE tilts UP (dorsiflex), heel drops",
    "ankle_roll_R": "right foot's OUTER edge LIFTS, inner edge (toward the left foot) DROPS",
    "hip_pitch_L": "left foot swings BACKWARD",
    "hip_roll_L": "left foot swings OUTWARD, away from the right leg",
    "hip_yaw_L": "left TOE rotates INWARD (pigeon-toed)",
    "knee_L": "left knee BENDS (heel rises relative to toe)",
    "ankle_pitch_L": "left TOE tilts UP (dorsiflex), heel drops",
    "ankle_roll_L": "left foot's OUTER edge LIFTS, inner edge (toward the right foot) DROPS",
}

# Which way to nudge when checking a sign. BOTH knees are one-sided on v3/v4
# ([-110, 0] each), so a positive nudge cannot move either knee at all -- nudge
# them negative. (This used to special-case only knee_R, a leftover from v2
# when the left knee ran [0, +90]; on v4 that left knee_L stuck at 0.) The
# ankles are symmetric, so +1 is fine. Signs here are in MJCF convention.
NUDGE_DIR = {j: (-1 if j in ("knee_R", "knee_L") else 1) for j in SIM_ORDER}


def joint_limits() -> dict[str, tuple[float, float]]:
    """Per-joint (lower, upper) in radians.

    Read from joint_limits.json, which `write_joint_limits()` bakes from the
    MJCF. Loading a small JSON keeps the deploy path (Pi) free of MuJoCo; the
    JSON is the single source shared by PC and Pi so they cannot disagree.
    """
    with open(LIMITS_PATH) as f:
        return {j: tuple(v) for j, v in json.load(f).items()}


def write_joint_limits(path: Path = LIMITS_PATH) -> None:
    """Regenerate operational limits from the MJCF (needs MuJoCo).

    The v4 knee mechanism physically permits 15 degrees past straight, but
    locomotion must not command that hyperextended branch because it shifts the
    torso forward. Preserve the CAD range in MJCF and cap the hardware/runtime
    limit at 0 here.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(MJCF))
    out = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name in MJCF_TO_JOINT:
            lo, hi = model.jnt_range[j]
            short = MJCF_TO_JOINT[name]
            if short in ("knee_R", "knee_L"):
                hi = min(float(hi), 0.0)
            out[short] = [float(lo), float(hi)]
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


LIMITS = joint_limits()


def clamp_q(joint: str, q: float) -> float:
    lo, hi = LIMITS[joint]
    return max(lo, min(hi, q))


# --- calibration -----------------------------------------------------------
# q_sim = sign * (raw - home_raw) * RAD_PER_COUNT
# The sim bus uses the identity calibration below, so a miscalibration can
# only ever come from the real robot -- which is exactly where it lives.

def default_calibration() -> dict:
    return {"home_raw": {j: 2048 for j in SIM_ORDER},
            "sign": {j: 1 for j in SIM_ORDER}}


def load_calibration(path: Path = CALIB_PATH) -> dict | None:
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_calibration(calib: dict, path: Path = CALIB_PATH) -> None:
    with open(path, "w") as f:
        json.dump(calib, f, indent=2)


def raw_to_q(joint: str, raw: int, calib: dict) -> float:
    return calib["sign"][joint] * (raw - calib["home_raw"][joint]) * RAD_PER_COUNT


def q_to_raw(joint: str, q: float, calib: dict) -> int:
    raw = calib["home_raw"][joint] + calib["sign"][joint] * q * COUNTS_PER_RAD
    return int(round(max(COUNT_MIN, min(COUNT_MAX, raw))))


def decode_speed(u16: int) -> int:
    """STS present-speed is sign-magnitude (bit15 = direction)."""
    return -(u16 & 0x7FFF) if (u16 & 0x8000) else (u16 & 0x7FFF)


def q_vector(pos_counts: dict[int, int], calib: dict) -> np.ndarray:
    """Servo counts keyed by ID -> joint angles in SIM_ORDER."""
    missing = [JOINT_TO_ID[j] for j in SIM_ORDER
               if JOINT_TO_ID[j] not in pos_counts]
    if missing:
        raise RuntimeError(f"position read is missing servo IDs {missing}")
    q = np.zeros(len(SIM_ORDER), np.float32)
    for k, j in enumerate(SIM_ORDER):
        sid = JOINT_TO_ID[j]
        q[k] = raw_to_q(j, pos_counts[sid], calib)
    return q


def qd_vector(speed_counts: dict[int, int], calib: dict) -> np.ndarray:
    missing = [JOINT_TO_ID[j] for j in SIM_ORDER
               if JOINT_TO_ID[j] not in speed_counts]
    if missing:
        raise RuntimeError(f"speed read is missing servo IDs {missing}")
    qd = np.zeros(len(SIM_ORDER), np.float32)
    for k, j in enumerate(SIM_ORDER):
        sid = JOINT_TO_ID[j]
        qd[k] = calib["sign"][j] * speed_counts[sid] * RAD_PER_COUNT
    return qd


def goal_counts(q_target: np.ndarray, calib: dict) -> dict[int, int]:
    """Joint angles in SIM_ORDER -> goal counts keyed by servo ID."""
    return {JOINT_TO_ID[j]: q_to_raw(j, clamp_q(j, float(q_target[k])), calib)
            for k, j in enumerate(SIM_ORDER)}


# --- STS3215 register map --------------------------------------------------
ADDR_ID = 5
ADDR_BAUD = 6
ADDR_HOMING_OFFSET = 31
ADDR_TORQUE = 40
ADDR_GOAL_POS = 42
ADDR_LOCK = 55
ADDR_PRESENT_POS = 56
ADDR_PRESENT_SPEED = 58
