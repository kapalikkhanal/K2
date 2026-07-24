#!/usr/bin/env python3
"""Phase 1: rebuild K2 link masses and joint dynamics from measured hardware.

The Fusion 360 export tags every link "Steel" (7850 kg/m^3), which produces
inertials that are ~4x too heavy. This script replaces them with:

  1. Measured sliced print weights (Premium PLA) per link.
  2. STS3215 servo hardware, 60 g per joint, added at the joint location via
     the parallel-axis theorem.
  3. Base electronics (Pi 5, Waveshare bus driver, IMU, print mount) added at
     their mounting points on base_link.

Link inertia tensors keep their CAD shape and are scaled by the mass ratio,
which is exact for a uniform-density part whose geometry is unchanged.

Writes urdf/K2_phase1.urdf; the Fusion export is left untouched.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "urdf" / "K2_fusion_export.urdf"
DST = ROOT / "urdf" / "K2_phase1.urdf"

# --- Measured masses -------------------------------------------------------

# Sliced print weight (Premium PLA, RaiseCloud), kg. Left/right are identical.
# Robot_v3 (2026-07-20): links lost their Fusion _2/_3 suffixes; base shell is
# now 30 g (shrunk 3 cm) with electronics lumped separately below; knee is 27 g.
# Robot_v4 (2026-07-21): a 6th DOF per leg was added -- the old single "Ankle"
# (a foot, 16.5 g) is now an ankle-PITCH bracket ("Ankle") carrying a new
# ankle-ROLL foot plate ("Feet"). One extra STS3215 per leg is added
# automatically by the per-joint servo loop below. The two new plastic masses
# are CAD-volume estimates (volume x the ~265 kg/m^3 effective print density
# implied by the old 16.5 g ankle): bracket ~22 g, foot ~14 g. Verify by
# weighing the printed parts.
PRINT_MASS = {
    "base_link": 0.030,
    "Hip_pitch_right": 0.016,
    "Hip_pitch_left": 0.016,
    "Hip_roll_right": 0.025,
    "Hip_roll_left": 0.025,
    "Hip_yaw_right": 0.014,
    "Hip_yaw_left": 0.014,
    "Knee_right": 0.027,
    "Knee_left": 0.027,
    "Ankle_right": 0.022,   # v4 ankle-pitch bracket (CAD-volume est.)
    "Ankle_left": 0.022,
    "Feet_right": 0.014,    # v4 ankle-roll foot plate (CAD-volume est.)
    "Feet_left": 0.014,
}

# Feetech STS3215 (30 kg.cm, 12 V): 55 g servo + 5 g horn and screws.
SERVO_MASS = 0.060

# Electronics mounted on base_link. Positions are base_link-local, metres, in
# the Fusion Y-up frame (+Y up, +X forward).
# Robot_v3 (2026-07-20): the user reports the base assembly weighs 100 g total,
# of which the printed shell is 30 g (above) and the remaining 70 g of
# electronics (Pi 5 + Waveshare driver + IMU + bracket + cabling) sits ~6 cm
# UP from the base, mounted on the holder. Lump it as one point mass at
# +Y 0.060; x/z centred over the base. Raising it 6 cm lifts the whole-body
# CoM, which matters for the lateral-balance / single-foot-support budget.
BASE_PAYLOAD = [
    ("electronics_stack", 0.070, (0.028, 0.060, 0.000)),
]

# --- STS3215 joint dynamics ------------------------------------------------

EFFORT = 2.94  # N.m, 30 kg.cm at 12 V
VELOCITY = 4.712  # rad/s, 45 rpm no-load

# The Robot_v3 Fusion export ships pure placeholder limits again -- every one
# is an exact fraction of pi (pi/3, pi/2, pi/12) which no real mechanism
# produces -- so every joint below is overridden. Signs: a negative right-knee
# angle swings the foot backward (anatomical flexion) and the left leg mirrors
# it; hip_roll adduction (leg toward centreline) is +right / -left.
#
#   knee     : the user deepened the knee bend on v3 and the CAD now clears
#              110 deg of flexion (was 98). Use the full 110 deg with a hard
#              stop at full extension (no hyperextension). *Verify on the
#              physical build -- 110 is CAD-derived, not yet hand-measured.*
#   hip_pitch: keep the hand-measured 47 deg (v3 export's 60 is a placeholder;
#              the pitch mechanism was not changed).
#   hip_roll : the user widened adduction from 12 -> 22 deg on one side to help
#              shift the CoM over the stance foot. Keep abduction at the
#              measured 45 deg (v3 export's 90 is a placeholder).
#   hip_yaw  : keep +-45 deg (v3 export's +-90 is a placeholder).
KNEE_MAX = 1.919862        # 110 deg flexion (CAD clearance, verify on hardware)
HIP_PITCH_MAX = 0.820305   # 47 deg (hand-measured)
HIP_ROLL_ABD = 0.785398    # 45 deg abduction
HIP_ROLL_ADD = 0.383972    # 22 deg adduction (widened from 12 on v3)
HIP_YAW_MAX = 0.785398     # 45 deg
# Robot_v4 ankle joints. The v4 export ships plausible CAD limits
# (pitch [-25,+50], roll [-20,+30]) but for early RL the user chose symmetric,
# conservative ranges: pitch +-30 deg, roll +-20 deg. The ankle-pitch axis is
# mirrored L/R and the ankle-roll axis too, but symmetric limits need no
# per-side sign handling. NOTE: the flexion-direction sign of each ankle axis
# has NOT been verified numerically the way the knee was -- do that in
# phase3_validate before trusting these for a directional gait.
ANKLE_PITCH_MAX = 0.523599  # 30 deg
# Real mechanical ankle-roll range, measured on the build (2026-07-21): only
# 12 deg INVERSION (foot rolls inward / sole faces inward) but 50 deg EVERSION
# (outward). In the MJCF joint sign +q = eversion (outward), so the range is
# [-12, +50] deg. The axes are mirrored L/R, so the same signed tuple is right
# for both ankles. The inversion side is the one a lateral weight-shift needs,
# so 12 deg is the binding limit -- one-foot support still just clears
# (CoM reach 35 mm vs 30 mm needed) but the static margin is thin (~4-5 mm).
ANKLE_ROLL_IN = 0.349066    # 20 deg inversion (inward) -- mechanically opened up
                            # from 12 deg on 2026-07-22 (was a bracket limit)
ANKLE_ROLL_OUT = 0.872665   # 50 deg eversion (outward)
JOINT_LIMITS = {
    # Robot_v3: BOTH knee axes point world -Y (v2 had them mirrored), so both
    # knees flex NEGATIVE now -- the left range is [-110, 0] like the right, not
    # [0, +110]. Using the old left range gave a wrong-way left knee / split
    # stance. hip_pitch is symmetric (+-47) so its axis flip needs no limit
    # change, and hip_roll/hip_yaw axes stayed mirrored on v3.
    "hip_yaw_knee_right_joint": (-KNEE_MAX, 0.0),
    "hip_yaw_knee_left_joint": (-KNEE_MAX, 0.0),
    "base_link_hip_pitch_right_joint": (-HIP_PITCH_MAX, HIP_PITCH_MAX),
    "base_link_hip_pitch_left_joint": (-HIP_PITCH_MAX, HIP_PITCH_MAX),
    "hip_pitch_hip_roll_right_joint": (-HIP_ROLL_ABD, HIP_ROLL_ADD),
    "hip_pitch_hip_roll_left_joint": (-HIP_ROLL_ADD, HIP_ROLL_ABD),
    "hip_roll_hip_yaw_right_joint": (-HIP_YAW_MAX, HIP_YAW_MAX),
    "hip_roll_hip_yaw_left_joint": (-HIP_YAW_MAX, HIP_YAW_MAX),
    # v4 ankle pitch (knee -> Ankle) and ankle roll (Ankle -> Feet). The roll
    # joint name carries Fusion's odd "anke_roll_ankle_pitch" auto-label; it is
    # the ROLL DOF despite the name. Keep the exact spelling (note "anke").
    "knee_ankle_pitch_right_joint": (-ANKLE_PITCH_MAX, ANKLE_PITCH_MAX),
    "knee_ankle_pitch_left_joint": (-ANKLE_PITCH_MAX, ANKLE_PITCH_MAX),
    "anke_roll_ankle_pitch_right_joint": (-ANKLE_ROLL_IN, ANKLE_ROLL_OUT),
    "anke_roll_ankle_pitch_left_joint": (-ANKLE_ROLL_IN, ANKLE_ROLL_OUT),
}


def point_mass_inertia(m, r):
    """Parallel-axis contribution of a point mass m at offset r from the CoM."""
    x, y, z = r
    return {
        "ixx": m * (y * y + z * z),
        "iyy": m * (x * x + z * z),
        "izz": m * (x * x + y * y),
        "ixy": -m * x * y,
        "ixz": -m * x * z,
        "iyz": -m * y * z,
    }


def main():
    tree = ET.parse(SRC)
    robot = tree.getroot()
    robot.set("name", "K2")

    # Servo bodies sit at the joint they drive; attribute each to the child
    # link so the mass rides with the moving segment.
    servo_on_child = {}
    for joint in robot.findall("joint"):
        child = joint.find("child").get("link")
        origin = joint.find("origin")
        # The joint origin is expressed in the PARENT frame, but the child
        # frame is coincident with it, so the servo sits at the child origin.
        servo_on_child.setdefault(child, []).append((0.0, 0.0, 0.0))

    print(f"{'link':<20} {'CAD (kg)':>9} {'PLA':>8} {'+hw':>8} {'total':>8}")
    print("-" * 58)
    total = 0.0

    for link in robot.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        if inertial is None:
            continue

        mass_el = inertial.find("mass")
        inertia_el = inertial.find("inertia")
        origin_el = inertial.find("origin")
        cad_mass = float(mass_el.get("value"))
        com = [float(v) for v in origin_el.get("xyz").split()]

        # 1. Rescale the printed shell to its measured weight.
        pla = PRINT_MASS[name]
        scale = pla / cad_mass
        inertia = {k: float(inertia_el.get(k)) * scale for k in
                   ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}

        # 2. Collect rigid point masses attached to this link.
        attachments = [(SERVO_MASS, r) for r in servo_on_child.get(name, [])]
        if name == "base_link":
            attachments += [(m, r) for _, m, r in BASE_PAYLOAD]

        hw = sum(m for m, _ in attachments)
        new_mass = pla + hw

        # 3. Combined centre of mass.
        cx = pla * com[0] + sum(m * r[0] for m, r in attachments)
        cy = pla * com[1] + sum(m * r[1] for m, r in attachments)
        cz = pla * com[2] + sum(m * r[2] for m, r in attachments)
        new_com = (cx / new_mass, cy / new_mass, cz / new_mass)

        # 4. Shift the shell inertia to the new CoM, then add each point mass
        #    about that same CoM (parallel-axis both ways).
        d = [com[i] - new_com[i] for i in range(3)]
        for k, v in point_mass_inertia(pla, d).items():
            inertia[k] += v
        for m, r in attachments:
            dr = [r[i] - new_com[i] for i in range(3)]
            for k, v in point_mass_inertia(m, dr).items():
                inertia[k] += v

        mass_el.set("value", f"{new_mass:.6f}")
        origin_el.set("xyz", " ".join(f"{v:.6f}" for v in new_com))
        for k, v in inertia.items():
            inertia_el.set(k, f"{v:.6e}")

        total += new_mass
        print(f"{name:<20} {cad_mass:>9.3f} {pla*1000:>7.1f}g {hw*1000:>7.1f}g "
              f"{new_mass*1000:>7.1f}g")

    print("-" * 58)
    print(f"{'TOTAL':<20} {'':>9} {'':>8} {'':>8} {total*1000:>7.1f}g")

    # 5. Real actuator limits in place of the Fusion placeholders.
    for joint in robot.findall("joint"):
        limit = joint.find("limit")
        if limit is not None:
            limit.set("effort", str(EFFORT))
            limit.set("velocity", str(VELOCITY))
            override = JOINT_LIMITS.get(joint.get("name"))
            if override is not None:
                limit.set("lower", f"{override[0]:.6f}")
                limit.set("upper", f"{override[1]:.6f}")
                print(f"  limit override: {joint.get('name')} -> "
                      f"[{np.degrees(override[0]):.0f}, "
                      f"{np.degrees(override[1]):.0f}] deg")

    # 6. Drop the ros2_control block: MuJoCo cannot parse it and the RL stack
    #    does not use it.
    for tag in ("ros2_control", "gazebo"):
        for el in robot.findall(tag):
            robot.remove(el)

    ET.indent(tree, space="  ")
    tree.write(DST, xml_declaration=True, encoding="utf-8")
    print(f"\nwrote {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
