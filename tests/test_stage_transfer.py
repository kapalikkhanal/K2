"""Ensure stage transfer preserves weights and refuses incompatible designs."""
import hashlib
import unittest
import torch
from k2_rl.k2_constants import K2_XML, _CROUCH_JOINTS, CROUCH_BASE_HEIGHT
from k2_rl.prepare_lateral import prepare


class StageTransferTests(unittest.TestCase):
  def checkpoint(self):
    result = {'iter': 999, 'infos': {'k2_design': {
      'mjcf_sha256': hashlib.sha256(K2_XML.read_bytes()).hexdigest(),
      'joint_pos': _CROUCH_JOINTS, 'base_height_m': CROUCH_BASE_HEIGHT,
      'gait_command_dim': 5, 'command_profile': None}},
      'optimizer_state_dict': {'state': {0: {'step': 10}}, 'param_groups': [{'lr': .001}]}}
    for key, n in (('actor_state_dict', 49), ('critic_state_dict', 54)):
      result[key] = {'mlp.0.weight': torch.randn(8, n),
                     'obs_normalizer._mean': torch.randn(1, n),
                     'obs_normalizer._var': torch.ones(1, n),
                     'obs_normalizer._std': torch.ones(1, n)}
    return result

  def test_expansion_and_source_preservation(self):
    original = self.checkpoint()
    result = prepare(original)
    for key in ('actor_state_dict', 'critic_state_dict'):
      torch.testing.assert_close(result[key]['mlp.0.weight'][:, :-1], original[key]['mlp.0.weight'])
      self.assertTrue((result[key]['mlp.0.weight'][:, -1] == 0).all())
      torch.testing.assert_close(result[key]['obs_normalizer._mean'][:, :-1], original[key]['obs_normalizer._mean'])
    self.assertEqual(original['iter'], 999)
    self.assertTrue(original['optimizer_state_dict']['state'])
    self.assertEqual(result['iter'], 0)
    self.assertEqual(result['optimizer_state_dict']['state'], {})
    self.assertEqual(result['optimizer_state_dict']['param_groups'][0]['lr'], .0001)

  def test_reject_full_stale_or_already_transferred(self):
    for field, value in (('gait_command_dim', 6), ('mjcf_sha256', 'stale'),
                         ('command_profile', 'smooth_full_v1')):
      data = self.checkpoint()
      data['infos']['k2_design'][field] = value
      with self.assertRaises(ValueError): prepare(data)
    with self.assertRaises(ValueError): prepare(prepare(self.checkpoint()))


if __name__ == '__main__':
  unittest.main()
