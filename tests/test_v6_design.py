"""Physical model and training regressions for the battery/TPU revision."""
import hashlib
import json
from pathlib import Path
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class V6DesignTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / 'mjcf/k2_physics.xml'))
        self.data = mujoco.MjData(self.model)
        self.crouch = json.loads((ROOT / 'k2_rl/crouch.json').read_text())
        self.data.qpos[2] = self.crouch['base_height_m']
        for name, q in self.crouch['joint_pos'].items():
            self.data.qpos[self.model.joint(name).qposadr[0]] = q
        mujoco.mj_forward(self.model, self.data)

    def test_generated_pose_matches_model(self):
        self.assertEqual(self.crouch['mjcf_sha256'], hashlib.sha256(
            (ROOT / 'mjcf/k2_physics.xml').read_bytes()).hexdigest(),
            'Run scripts/update_rl_crouch.py after model changes')
        feet = self.data.site_xpos[[self.model.site(f'{s}_foot').id for s in ('left', 'right')]]
        np.testing.assert_allclose(feet[:, 2], 0, atol=1e-5)
        self.assertAlmostEqual(self.data.subtree_com[0, 0], feet[:, 0].mean(), places=5)
        for side in ('left', 'right'):
            gid = self.model.geom(f'{side}_foot_collision').id
            mid = self.model.geom_dataid[gid]
            start, count = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
            vertices = self.model.mesh_vert[start:start+count]
            world = vertices @ self.data.geom_xmat[gid].reshape(3, 3).T + self.data.geom_xpos[gid]
            self.assertLess(abs(world[:, 2].min()), 1e-5)

    def test_mass_accounting(self):
        self.assertAlmostEqual(float(self.model.body('base_link').mass[0]), .375)
        self.assertAlmostEqual(self.model.body_mass.sum(), 1.4219540134)
        pads = json.loads((ROOT / 'urdf/sole_pads.json').read_text())['pads']
        for side, pad in pads.items():
            self.assertLessEqual(pad['mass_kg'], .010)
            expected = pad['mjcf_inertial']['mass'] + pad['mass_kg']
            self.assertAlmostEqual(float(self.model.body(f'Feet_{side}').mass[0]), expected)

    def test_imu_frame_preserved(self):
        from hardware.k2_attitude import R_SITE_TO_ROOT
        sid = self.model.site('imu').id
        np.testing.assert_allclose(self.data.site_xmat[sid].reshape(3, 3), R_SITE_TO_ROOT, atol=1e-6)

    def test_training_covers_lateral_and_vertical_sole_variation(self):
        from k2_rl.full_env_cfg import make_k2_full_env_cfg
        cfg = make_k2_full_env_cfg()
        self.assertGreater(cfg.commands['gait'].rel_lateral_envs, 0)
        self.assertEqual(cfg.events['sole_height'].params['axes'], [1])
        for side in ('left', 'right'):
            bid = self.model.body(f'Feet_{side}').id
            axis = self.data.xmat[bid].reshape(3, 3)[:, 1]
            self.assertGreater(abs(axis[2]), .999)
        self.assertIn('yaw_range', cfg.events['sole_tilt'].params)
        self.assertNotIn('pitch_range', cfg.events['sole_tilt'].params)


if __name__ == '__main__':
    unittest.main()
