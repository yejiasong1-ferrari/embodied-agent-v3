import unittest
from types import SimpleNamespace as NS
import numpy as np
from spatial.geometry import footprint,supported_by


def circle(center,radius):
    angles=np.linspace(0,2*np.pi,40,endpoint=False)
    xy=np.array(center)+np.column_stack((np.cos(angles),np.sin(angles)))*radius
    return np.column_stack((xy,np.full(len(xy),.1)))


def round_body(category,points,position,size):
    return NS(category=category,points=points,position=np.array(position,float),size=np.array(size,float),
              manipulation_geometry={},high=np.array([position[0],position[1],position[2]+size[2]/2]))


class PartialRoundFootprintTests(unittest.TestCase):
    def side_view(self):
        # Only the camera-facing half of a 7 cm round outline was measured.
        angles=np.linspace(0,np.pi,40)
        xy=np.column_stack((.38+np.cos(angles)*.035,-.18-np.sin(angles)*.035*.6))
        return np.column_stack((xy,np.full(len(xy),.05)))

    def test_support_footprint_covers_the_measured_diameter(self):
        cup=round_body('cup',self.side_view(),(.38,-.18,.05),(.07,.07,.10))
        poly=footprint(cup)
        self.assertGreaterEqual(np.ptp(poly[:,1]),.069)
        small=round_body('cup',circle((.38,-.18),.028),(.38,-.18,.14),(.056,.056,.08))
        self.assertTrue(supported_by(small,cup,np.array([.38,-.18,.14])))

    def test_a_larger_payload_still_overhangs(self):
        cup=round_body('cup',self.side_view(),(.38,-.18,.05),(.07,.07,.10))
        wide=round_body('cup',circle((.38,-.18),.045),(.38,-.18,.14),(.09,.09,.08))
        self.assertFalse(supported_by(wide,cup,np.array([.38,-.18,.14])))

    def test_complete_outline_is_left_unchanged(self):
        angles=np.linspace(0,2*np.pi,60,endpoint=False)
        xy=np.column_stack((.38+np.cos(angles)*.035,-.18+np.sin(angles)*.035))
        cup=round_body('cup',np.column_stack((xy,np.full(len(xy),.05))),(.38,-.18,.05),(.07,.07,.10))
        self.assertAlmostEqual(float(np.ptp(footprint(cup)[:,0])),.07,places=3)


if __name__=='__main__':unittest.main()
