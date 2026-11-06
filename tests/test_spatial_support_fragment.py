import unittest
from types import SimpleNamespace as NS
import numpy as np
from config import AgentConfig
from spatial.spatial_planner import SpatialPlanner


def body(id,category,position,size,points):
    position=np.array(position,float);size=np.array(size,float)
    return NS(id=id,category=category,points=points,position=position,size=size,manipulation_geometry={},
              high=position+size/2,low=position-size/2)


def disc(center,radius,z):
    a=np.linspace(0,2*np.pi,40,endpoint=False)
    return np.column_stack((np.array(center)+np.column_stack((np.cos(a),np.sin(a)))*radius,np.full(40,z)))


class SupportFragmentTests(unittest.TestCase):
    def setUp(self):
        self.cup=body('cup','cup',(.38,-.18,.05),(.07,.07,.10),disc((.38,-.18),.035,.05))
        self.pink=body('pink','bottle',(.38,-.18,.14),(.056,.056,.08),disc((.38,-.18),.028,.14))
        self.spot=np.array([.38,-.18,.14])

    def check(self,world_objects,support_id=None):
        world=NS(objects={o.id:o for o in world_objects},memory=None,support_points=np.zeros((0,3)))
        return SpatialPlanner(AgentConfig()).feasibility(self.pink,self.cup,world,self.spot,'on',support_id=support_id)

    def test_handle_fragment_touching_the_cup_is_part_of_the_cup(self):
        handle=body('handle','bottle',(.443,-.18,.05),(.069,.023,.10),np.array([[.41,-.19,.05],[.477,-.17,.05],[.45,-.18,.05]]))
        ok,reason,_=self.check([self.cup,self.pink,handle])
        self.assertTrue(ok,reason)

    def test_taller_touching_object_still_counts(self):
        tall=body('tall','bottle',(.443,-.18,.15),(.069,.023,.30),np.array([[.41,-.19,.05],[.477,-.17,.05],[.45,-.18,.05]]))
        ok,reason,_=self.check([self.cup,self.pink,tall])
        self.assertFalse(ok)


if __name__=='__main__':unittest.main()
