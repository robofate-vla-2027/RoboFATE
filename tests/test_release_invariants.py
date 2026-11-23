"""Full-size metadata and explicitly synthetic restored-state comparisons."""
import copy
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from opendm.model.dm05.dm05_arch import DM05Config, DM05ForConditionalGeneration
from robofate.collection import compare_restored_state
from robofate.config import PAPER
from robofate.model import RoboFATE


class ArchitectureTests(unittest.TestCase):
    def test_full_sized_meta_parameter_counts(self):
        metadata = json.loads((Path(__file__).parents[1]/'configs/dm05_architecture.json').read_text())
        with torch.device('meta'):
            model = RoboFATE(DM05ForConditionalGeneration(DM05Config(**metadata)), PAPER['model'])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 7947323057)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 7533995969)
        self.assertEqual(sum(p.numel() for p in model.dm.model.visual.parameters()), 413327088)
        self.assertTrue(all(p.device.type == 'meta' for p in model.parameters()))


class RestorationTests(unittest.TestCase):
    def state(self):
        return {
            'robot_joint_positions':[0.]*14, 'robot_joint_velocities':[0.]*14,
            'object_positions':[[0., 0., 1.]], 'object_linear_velocities':[[0.]*3],
            'object_angular_velocities':[[0.]*3], 'object_quaternions':[[1.,0.,0.,0.]],
            'articulation_positions':[], 'articulation_velocities':[],
            'controller_targets':[0.]*14, 'interpolation_state':[0.], 'gripper_state':[0.,0.],
            'attachments':[], 'task_flags':{'synthetic':True}, 'task_progress':0,
            'termination_flags':{'done':False}, 'rng_states':{'synthetic_rng':[1,2,3]},
            'simulator_time':7., 'action_counter':175, 'remaining_actions':325, 'remaining_time':13.,
            'contact_solver_state':{'synthetic_contacts':[]}, 'task_dynamic_state':{'synthetic':{}},
            'rgb':{camera:np.zeros((4,4,3),dtype=np.uint8) for camera in ('head','left_wrist','right_wrist')},
        }

    def test_known_empty_articulation_fields_are_valid(self):
        state = self.state()
        self.assertTrue(compare_restored_state(state, copy.deepcopy(state))['qualified'])

    def test_missing_dynamic_state_is_not_known_empty(self):
        state = self.state()
        restored = copy.deepcopy(state)
        del restored['task_dynamic_state']
        result = compare_restored_state(state, restored)
        self.assertFalse(result['qualified'])
        self.assertEqual(result['missing_fields'], ['task_dynamic_state'])

    def test_quaternion_sign_is_not_a_rotation_difference(self):
        state = self.state()
        restored = copy.deepcopy(state)
        restored['object_quaternions'] = [[-1.,0.,0.,0.]]
        self.assertTrue(compare_restored_state(state, restored)['qualified'])

    def test_joint_tolerance_and_exact_rgb(self):
        state = self.state()
        restored = copy.deepcopy(state)
        restored['robot_joint_positions'][0] += 0.5e-6
        self.assertTrue(compare_restored_state(state, restored)['qualified'])
        restored['robot_joint_positions'][0] += 2e-6
        self.assertFalse(compare_restored_state(state, restored)['qualified'])
        restored = copy.deepcopy(state)
        restored['rgb']['head'][0,0,0] = 1
        self.assertFalse(compare_restored_state(state, restored)['qualified'])

    def test_empty_camera_arrays_are_not_observations(self):
        state = self.state()
        state['rgb'] = {camera:np.zeros((0,0,3),dtype=np.uint8) for camera in state['rgb']}
        self.assertFalse(compare_restored_state(state, copy.deepcopy(state))['qualified'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
