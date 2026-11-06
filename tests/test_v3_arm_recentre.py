import unittest
import numpy as np
import mujoco
from config import AgentConfig
from simulation.environment import Simulation
from robot.mujoco_robot import MuJoCoRobot
from manipulation.motion_planner import MotionPlanner

CROWDED=np.array([2.72,.66,2.72,-2.44,-.26,1.82,.09])


def margin(robot,q):
    lower,upper=robot.joint_limits
    return float(np.min(np.minimum(q-lower,upper-q)))


class ArmRecentreTests(unittest.TestCase):
    def setUp(self):
        self.config=AgentConfig();self.sim=Simulation(self.config);self.sim.__enter__()
        self.robot=MuJoCoRobot(self.sim)
        for i,v in enumerate(CROWDED):
            self.sim.data.qpos[int(self.sim.model.joint(f'joint{i+1}').qposadr[0])]=v
        mujoco.mj_forward(self.sim.model,self.sim.data)
    def tearDown(self):self.sim.__exit__(None,None,None)

    def test_ik_offers_a_roomy_posture_when_near_seeds_hug_the_joint_stops(self):
        solutions=self.robot.solve_ik(self.robot.get_ee_pose(),self.robot.get_joint_positions())
        self.assertTrue(any(margin(self.robot,q)>=.4 for q in solutions))

    def test_roomy_preference_reaches_a_posture_away_from_the_stops(self):
        planner=MotionPlanner(self.robot,self.config)
        pose=self.robot.get_ee_pose()
        path=planner.plan_to_pose(pose,prefer_roomy=True)
        self.assertIsNotNone(path)
        self.assertGreaterEqual(margin(self.robot,np.asarray(path[-1])),.4)

    def test_default_planning_keeps_the_nearest_posture(self):
        planner=MotionPlanner(self.robot,self.config)
        path=planner.plan_to_pose(self.robot.get_ee_pose())
        self.assertIsNotNone(path)
        self.assertLess(np.linalg.norm(np.asarray(path[-1])-CROWDED),.5)


if __name__=='__main__':unittest.main()
