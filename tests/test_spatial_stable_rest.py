import unittest
from types import SimpleNamespace as NS
import numpy as np
from spatial.geometry import supported_by,footprint
from perception.geometry_analyzer import measured_round_base


def block(position,size,yaw=0.):
    c,s=np.cos(yaw),np.sin(yaw);r=np.array([[c,-s],[s,c]])
    corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*np.asarray(size[:2])/2
    xy=position[:2]+corners@r.T
    points=np.column_stack((xy,np.full(4,position[2]+size[2]/2)));points=np.repeat(points,3,axis=0)
    return NS(category='box',points=points,position=np.array(position,float),size=np.array(size,float),
              manipulation_geometry={},high=np.array([position[0],position[1],position[2]+size[2]/2]))


class StableRestTests(unittest.TestCase):
    def test_measured_base_excludes_high_protrusion_only_for_support(self):
        obj=block((.3,.2,.04),(.064,.064,.08));obj.low=obj.position-obj.size/2
        angles=np.linspace(-1.25,1.25,40)
        xy=np.column_stack([np.cos(angles),np.sin(angles)])*.032+obj.position[:2]
        obj.points=np.vstack([np.c_[xy,np.full(len(xy),z)] for z in (.008,.025,.04)])
        high_protrusion=obj.position+np.array([.085,0.,.035])
        obj.points=np.vstack([obj.points,high_protrusion]);obj.manipulation_geometry={}
        radius=measured_round_base(obj,obj.points)
        self.assertAlmostEqual(radius,.033,places=4)
        # A pad too small for the measured base is still rejected.
        obj.manipulation_geometry['measured_base_radius_m']=radius
        for width,expected in ((.06,True),(.035,False)):
            pad=block((.3,.2,.04),(width,width,.08))
            self.assertEqual(supported_by(obj,pad,np.array([.3,.2,.12])),expected)
        self.assertGreater(footprint(obj)[:,0].max(),.38)
        # Do not invent a base when the lower rim is missing or the low
        # silhouette actually includes a protrusion too.
        self.assertIsNone(measured_round_base(obj,obj.points[obj.points[:,2]>.02]))
        self.assertIsNone(measured_round_base(obj,np.vstack([obj.points,[.38,.2,.010]])))

    def test_identical_blocks_with_small_yaw_offset_rest_stably(self):
        base=block((.3,.2,.04),(.07,.06,.08));top=block((.3,.2,.12),(.07,.06,.08),np.deg2rad(20))
        self.assertTrue(supported_by(top,base,top.position))

    def test_a_centimetre_of_settling_offset_still_counts_as_resting_on(self):
        base=block((.3,.2,.04),(.07,.06,.08),np.deg2rad(5));top=block((.309,.203,.12),(.07,.06,.08))
        self.assertTrue(supported_by(top,base,top.position))

    def test_centre_must_stay_over_the_support(self):
        base=block((.3,.2,.04),(.07,.06,.08));top=block((.3,.2,.12),(.07,.06,.08))
        self.assertTrue(supported_by(top,base,np.array([.3,.2,.12])))
        self.assertFalse(supported_by(top,base,np.array([.3+.04,.2,.12])))

    def test_height_must_match_the_support_surface(self):
        base=block((.3,.2,.04),(.07,.06,.08));top=block((.3,.2,.2),(.07,.06,.08))
        self.assertFalse(supported_by(top,base,top.position))

    def test_positive_margin_still_requires_every_corner(self):
        base=block((.3,.2,.04),(.07,.06,.08));top=block((.3,.2,.12),(.07,.06,.08),np.deg2rad(20))
        self.assertFalse(supported_by(top,base,top.position,.005))


class PartialHullTests(unittest.TestCase):
    def test_a_support_hull_missing_a_corner_is_completed_from_its_measured_size(self):
        from spatial.geometry import footprint
        pts=np.array([[-.285,-.22],[-.355,-.22],[-.355,-.272],[-.35,-.279]]);pts=np.column_stack((pts,np.full(4,.08)))
        support=NS(category='bottle',points=np.repeat(pts,3,axis=0),position=np.array([-.324,-.2486,.04]),
                   size=np.array([.0693,.0593,.08]),manipulation_geometry={})
        top=block((-.313,-.2413,.12),(.07,.06,.08))
        self.assertTrue(supported_by(top,NS(**{**support.__dict__,'high':np.array([-.324,-.2486,.08])}),top.position))


if __name__=='__main__':unittest.main()


class HandleWidenedHullTests(unittest.TestCase):
    def test_cup_seen_only_by_its_handle_side_still_gets_its_whole_body_as_support(self):
        centre=np.array([.38,-.18]);angles=np.linspace(np.pi*.6,np.pi*1.4,12)
        body=centre+np.column_stack((np.cos(angles),np.sin(angles)))*.035 # far half-body only
        handle=np.array([[.485,-.145],[.456,-.191],[.46,-.19]])
        points=np.column_stack((np.concatenate([body,handle]),np.full(15,.1)))
        cup=NS(category='cup',points=points,position=np.array([.38,-.18,.05]),size=np.array([.08,.08,.1]),
               manipulation_geometry={},high=np.array([.38,-.18,.1]))
        pink=block((.38,-.18,.141),(.056,.056,.081))
        pink.category='bottle';pink.points=np.empty((0,3))
        self.assertTrue(supported_by(pink,cup,np.array([.38,-.18,.141])))


class OverlapAreaTests(unittest.TestCase):
    def test_overlap_of_square_with_shifted_square_is_exact_even_with_duplicate_vertices(self):
        from spatial.geometry import convex_overlap_area
        a=np.array([[0,0],[1,0],[1,1],[1,1],[0,1]],float);b=a+np.array([.5,.25])
        self.assertAlmostEqual(convex_overlap_area(a,b),.5*.75,places=6)
        self.assertAlmostEqual(convex_overlap_area(a,b[::-1]),.5*.75,places=6)
        self.assertEqual(convex_overlap_area(a,a+5),0.)
