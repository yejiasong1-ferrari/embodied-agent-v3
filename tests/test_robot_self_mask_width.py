import unittest
import numpy as np
from config import AgentConfig
from simulation.environment import Simulation
from robot.mujoco_robot import MuJoCoRobot


class SelfMaskWidthTests(unittest.TestCase):
    def test_mask_follows_the_measured_finger_width(self):
        with Simulation(AgentConfig()) as sim:
            robot=MuJoCoRobot(sim);points=np.zeros((1,3))
            sim.data.qpos[int(sim.model.joint('finger_joint1').qposadr[0])]=.013
            sim.data.qpos[int(sim.model.joint('finger_joint2').qposadr[0])]=.013
            robot._planning.gripper_width=.08
            robot.self_mask(points)
            self.assertAlmostEqual(robot._planning.gripper_width,robot.get_gripper_state()['width'])
            self.assertAlmostEqual(robot._planning.gripper_width,.026)


if __name__=='__main__':unittest.main()
