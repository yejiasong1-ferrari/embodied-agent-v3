import unittest
from types import SimpleNamespace as NS
import numpy as np
from config import AgentConfig
from manipulation.viewpoint_planner import ViewpointPlanner


def camera(offset):
    ee=np.eye(4);ee[:3,3]=offset;ee[:3,:3]=np.diag([1,-1,-1])
    return NS(ee_to_camera=ee)


class SideCameraViewpointTests(unittest.TestCase):
    def hand_for(self,position,offset):
        target=NS(position=np.array(position,float))
        return next(iter(ViewpointPlanner(AgentConfig()).candidates(target,camera(offset))))

    def test_camera_looks_at_the_target_whatever_way_the_hand_is_turned(self):
        for position in [(.16,.16,.04),(.5,-.3,.04),(-.4,.2,.04)]:
            offset=np.array([.065,0,.04]);pose=self.hand_for(position,offset)
            transform=np.eye(4);transform[:3,:3]=pose.rotation;transform[:3,3]=pose.position
            lens=transform@np.array([*offset,1.])
            np.testing.assert_allclose(lens[:2],np.array(position)[:2],atol=1e-6)

    def test_hand_is_not_folded_onto_the_base_for_a_target_beside_it(self):
        pose=self.hand_for((.16,.16,.04),np.array([.065,0,.04]))
        naive=np.array([.16-.065,.16])
        self.assertGreater(np.linalg.norm(pose.position[:2]),np.linalg.norm(naive))

    def test_centred_camera_is_unchanged(self):
        pose=self.hand_for((.3,.1,.04),np.array([0,0,.02]))
        np.testing.assert_allclose(pose.position[:2],[.3,.1],atol=1e-9)


if __name__=='__main__':unittest.main()
