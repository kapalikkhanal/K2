"""Geometric reward checks for stance and swing, independent of a trained policy."""
import unittest
from types import SimpleNamespace as NS
import torch
from k2_rl.mdp.lateral import hold_sole_alignment_l2, lateral_swing_shortfall_l2
from k2_rl.lateral_env_cfg import make_k2_lateral_env_cfg
from k2_rl.turn_env_cfg import make_k2_turn_env_cfg


class LateralStanceTests(unittest.TestCase):
  def setUp(self):
    self.command = torch.zeros(1, 6)
    self.pos = torch.tensor([[[-.04,.035,0],[-.04,-.035,0],
                              [.04,.035,0],[.04,-.035,0],
                              [0,.02,0],[0,-.02,0],
                              [0,.05,0],[0,-.05,0]]])
    self.env = NS(scene={'robot': NS(data=NS(site_pos_w=self.pos))},
                  command_manager=NS(get_command=lambda _: self.command))
    self.params = {name: NS(name='robot',site_ids=[2*i,2*i+1])
                   for i,name in enumerate(('heel_cfg','toe_cfg','inner_cfg','outer_cfg'))}

  def stance(self):
    return hold_sole_alignment_l2(self.env,'gait',**self.params)

  def test_grounded_parallel_and_mode_gating(self):
    self.assertEqual(self.stance().item(),0)
    self.pos[0,3,1] += .02  # yaw one toe out
    self.assertGreater(self.stance().item(),0)
    self.command[0,0] = 1
    self.assertEqual(self.stance().item(),0)

  def test_edge_lift_and_ground_tolerance(self):
    self.pos[:,:,2] = .0015
    self.assertEqual(self.stance().item(),0)
    self.pos[0,2,2] = .012
    self.assertGreater(self.stance().item(),0)

  def test_drag_gradient_and_both_lateral_signs(self):
    for direction in (-1,1):
      for phase in (-1,1):
        self.pos[:,:,2] = 0
        self.command[0] = torch.tensor([1,phase,0,0,0,.02*direction])
        self.pos.requires_grad_(True)
        cost = lateral_swing_shortfall_l2(self.env,'gait',.012,**self.params)
        self.assertGreater(cost.item(),0)
        cost.sum().backward()
        self.assertLess(self.pos.grad[:,:,2].sum().item(),0)
        self.pos=self.pos.detach();self.env.scene['robot'].data.site_pos_w=self.pos
        self.pos[:,0 if phase>0 else 1::2,2] = .012
        self.assertEqual(lateral_swing_shortfall_l2(self.env,'gait',.012,**self.params).item(),0)
    self.command[0,5]=0
    self.pos[:,:,2]=0
    self.assertEqual(lateral_swing_shortfall_l2(self.env,'gait',.012,**self.params).item(),0)

  def test_turn_task_unchanged_and_height_retained(self):
    turn=make_k2_turn_env_cfg();lateral=make_k2_lateral_env_cfg()
    for name in ('hold_sole_alignment','hold_feet_still','lateral_swing_shortfall'):
      self.assertNotIn(name,turn.rewards)
      self.assertIn(name,lateral.rewards)
    self.assertEqual(lateral.rewards['gait_swing'].params['target_height'],.012)


if __name__=='__main__': unittest.main()
