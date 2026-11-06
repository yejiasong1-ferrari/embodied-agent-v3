import unittest
import threading
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
from config import AgentConfig
from simulation.mujoco_camera import MujocoCamera
from robot.robot_interface import Pose


class DetailSelfMaskTests(unittest.TestCase):
    def test_detail_and_routine_captures_both_mask_robot_pixels(self):
        cfg=AgentConfig()
        renderer=Mock();rgb=np.ones((48,64,3),np.uint8)*120;depth=np.ones((48,64))*.95
        renderer.render.side_effect=[rgb,depth,rgb,depth]
        pose=np.eye(4);pose[:3,:3]=np.diag([1.,-1.,-1.]);pose[2,3]=1.
        robot=Mock();robot.get_ee_pose.return_value=Pose(pose[:3,3],pose[:3,:3])
        robot.self_mask.side_effect=lambda p:p[:,0]<0
        camera=MujocoCamera.__new__(MujocoCamera)
        camera._robot=robot;camera.config=NS(mode='wrist');camera.index=0;camera.ee_to_camera=np.eye(4)
        camera._sim=NS(config=cfg,lock=threading.RLock(),renderer=renderer,get_detail_renderer=lambda:renderer,
            model=NS(camera=lambda name:NS(id=0),cam_fovy=[65]),
            data=NS(cam_xmat=[np.eye(3)],cam_xpos=[np.zeros(3)]))
        for frame in [camera.capture(),camera.capture_detail()]:
            self.assertIsNotNone(frame.robot_mask)
            self.assertTrue(frame.robot_mask[:,:32].all())
            self.assertFalse(frame.robot_mask[:,32:].any())


if __name__=='__main__':unittest.main()
