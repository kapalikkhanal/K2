#!/usr/bin/env python3
"""Check the RL neutral pose and report gravity-only supported joint loads.

This is a static calculation, not a motor rating or a walking certification.
Single-foot cases hold the double-support pose and can require an infeasible
centre of pressure; use them only to examine load transfer sensitivity.
"""
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def audit():
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'mjcf/k2_physics.xml'))
    data = mujoco.MjData(model)
    crouch = json.loads((ROOT / 'k2_rl/crouch.json').read_text())
    data.qpos[2] = crouch['base_height_m']
    for name, value in crouch['joint_pos'].items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    feet = [model.site(f'{side}_foot').id for side in ('left', 'right')]
    jacobians = []
    for site in feet:
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, site)
        jacobians.append(np.vstack((jp, jr)))
    cases = {}
    for name, shares in [('double_support', (.5, .5)), ('left_support', (1., 0.)), ('right_support', (0., 1.))]:
        generalized = np.zeros(model.nv)
        for share, jac in zip(shares, jacobians):
            wrench = np.linalg.solve(jac[:, :6].T, share * data.qfrc_bias[:6])
            generalized += jac.T @ wrench
        tau = data.qfrc_bias[6:] - generalized[6:]
        cases[name] = {
            'peak_abs_nm': float(abs(tau).max()),
            'fraction_of_2_94_nm': float(abs(tau).max() / 2.94),
            'joint_nm': {model.joint(j).name: float(tau[model.jnt_dofadr[j]-6]) for j in range(1, model.njnt)},
        }
    return {
        'mass_kg': float(model.body_mass.sum()),
        'base_mass_kg': float(model.body('base_link').mass[0]),
        'com_world_m': data.subtree_com[0].tolist(),
        'foot_sites_world_m': data.site_xpos[feet].tolist(),
        'static_loads': cases,
        'battery_increment_nm_at_50mm': .150 * 9.81 * .050,
        'limitations': 'Gravity only; no acceleration, impacts, speed/voltage derating or thermal limit. Single-support wrench feasibility is not enforced.',
    }


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
