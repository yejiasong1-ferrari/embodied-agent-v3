"""Physical regressions for releasing a round payload from tilted fingers."""
import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import mujoco
import numpy as np

from config import AgentConfig
from robot.mujoco_robot import MuJoCoRobot
from simulation.environment import Simulation


class ReleaseContactTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = json.loads(
            (Path(__file__).parent / 'fixtures/v3_tilted_release.json').read_text()
        )

    def restore(self, sim, robot, support_offset=(0., 0.)):
        mujoco.mj_resetData(sim.model, sim.data)
        for name in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart'):
            getattr(sim.data, name)[:] = self.state[name]
        support_joint = int(sim.model.joint('pink_bottle_joint').qposadr[0])
        sim.data.qpos[support_joint:support_joint + 2] += support_offset
        mujoco.mj_forward(sim.model, sim.data)
        robot.resume()
        robot.confirm_grasp()

    def assert_stable_on_support(self, sim):
        # Ground truth is confined to the test's oracle. The controller only
        # receives jaw proprioception and contact sensors through its interface.
        source = sim.model.body('cyan_mug').id
        target = sim.model.body('pink_bottle').id
        tilt = np.arccos(np.clip(sim.data.xmat[source, 8], -1., 1.))
        self.assertLess(tilt, np.deg2rad(3.))
        self.assertLess(np.linalg.norm(sim.data.xpos[source, :2] - sim.data.xpos[target, :2]), .022)
        self.assertAlmostEqual(sim.data.xpos[source, 2] - sim.data.xpos[target, 2], .08, delta=.004)

    def test_loaded_release_settles_across_small_support_offsets(self):
        cfg = AgentConfig(version=3)
        with patch.object(mujoco, 'Renderer'), Simulation(cfg) as sim:
            robot = MuJoCoRobot(sim)
            original_friction = sim.model.geom_friction.copy()
            original_solver = int(sim.model.opt.noslip_iterations)
            for offset in ((0., 0.), (-.003, 0.), (.003, 0.), (0., -.003), (0., .003)):
                with self.subTest(offset=offset):
                    self.restore(sim, robot, offset)
                    self.assertTrue(robot.get_contact_state().bilateral)
                    self.assertTrue(robot.open_gripper())
                    self.assertTrue(robot.hold(1.5))
                    self.assert_stable_on_support(sim)
                    self.assertGreater(robot.get_gripper_state()['width'], .079)
                    np.testing.assert_array_equal(sim.model.geom_friction, original_friction)
                    self.assertEqual(sim.model.opt.noslip_iterations, original_solver)

    def test_release_respects_stop_and_keeps_physical_friction_when_interrupted(self):
        cfg = AgentConfig(version=3)
        with patch.object(mujoco, 'Renderer'), Simulation(cfg) as sim:
            robot = MuJoCoRobot(sim)
            self.restore(sim, robot)
            self.assertFalse(robot.open_gripper(lambda: 'USER_STOP'))
            self.assertFalse(robot.last_release_evidence['completed'])
            self.assertFalse(robot._control.one_sided_friction)
            self.assertLess(robot.get_gripper_state()['width'], .065)
            self.assertEqual(robot.last_error, 'USER_STOP')

    def test_fixture_reproduces_tip_with_the_previous_slow_release(self):
        cfg = AgentConfig(version=3)
        with patch.object(mujoco, 'Renderer'), Simulation(cfg) as sim:
            robot = MuJoCoRobot(sim)
            self.restore(sim, robot)
            robot._bilateral_grasp_verified = False
            robot._control.one_sided_friction = True
            self.assertTrue(robot._control.grasp_to(255, duration=1.))
            self.assertTrue(robot.hold(1.5))
            source = sim.model.body('cyan_mug').id
            self.assertLess(sim.data.xmat[source, 8], .5)

    def test_v2_keeps_its_existing_opening_behavior(self):
        cfg = AgentConfig(version=2)
        with patch.object(mujoco, 'Renderer'), Simulation(cfg) as sim:
            robot = MuJoCoRobot(sim)
            self.restore(sim, robot)
            with patch.object(robot._control, 'grasp_to', return_value=True) as opening:
                self.assertTrue(robot.open_gripper())
            self.assertEqual(opening.call_args.kwargs['duration'], 1.)

