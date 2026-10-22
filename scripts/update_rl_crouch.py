#!/usr/bin/env python3
"""Regenerate the RL neutral pose after changing MJCF geometry or inertias."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hardware.k2_conventions import SIM_ORDER, JOINT_TO_MJCF
from hardware.k2_motion import SquatIK


def main():
    ik = SquatIK()
    # Recenter for the current mass distribution; the old -10 mm trim was
    # measured on an earlier robot and is not a v6 calibration.
    q, height = ik.solve(35.0, base_shift_m=0.0)
    result = {
        'description': '35 degree knees, flat TPU soles, fore/aft CoM centered; no hardware trim',
        'mjcf_sha256': hashlib.sha256((ROOT / 'mjcf/k2_physics.xml').read_bytes()).hexdigest(),
        'total_mass_kg': float(ik.model.body_mass.sum()),
        'base_height_m': height,
        'joint_pos': {JOINT_TO_MJCF[name]: float(value) for name, value in zip(SIM_ORDER, q)},
    }
    path = ROOT / 'k2_rl/crouch.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(path)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
