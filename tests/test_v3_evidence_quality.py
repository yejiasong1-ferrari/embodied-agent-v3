import unittest
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
from perception.camera import CameraFrame
from perception.candidate_comparison import CandidateComparison
from perception.candidate_evidence import candidate_view_quality


class EvidenceQualityTests(unittest.TestCase):
    def setUp(self):
        pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=1.
        rgb=np.full((80,80,3),[220,0,0],np.uint8)
        depth=np.ones((80,80));depth[28:52,28:52]=.9
        self.frame=CameraFrame(rgb,depth,np.array([[100,0,40],[0,100,40],[0,0,1]]),pose,1.,'scan:0')
        self.obj=NS(id='anonymous',low=np.array([-.1,-.1,0]),high=np.array([.1,.1,.1]),
                    position=np.array([0.,0.,.05]),bbox=(28,28,52,52),observation_clipped=False,
                    points=np.empty((0,3)))
        self.comparison=CandidateComparison(NS(p=NS(log=Mock())))

    def quality(self,frame):
        return candidate_view_quality(self.obj,frame,(20,20,60,60))

    def test_clear_body_is_usable_without_classifying_its_shape(self):
        quality=self.quality(self.frame)
        self.assertTrue(quality['usable'])
        self.assertEqual(quality['foreground_occlusion_fraction'],0)

    def test_foreground_link_does_not_replace_clear_candidate_photo(self):
        self.comparison.remember([self.obj],self.frame)
        original=self.comparison.gallery['anonymous']
        depth=self.frame.depth.copy();depth[25:55,25:47]=.6
        obstructed=replace(self.frame,depth=depth,timestamp=2.,view_id='identity_side:1')
        self.assertFalse(self.quality(obstructed)['usable'])
        self.comparison.remember([self.obj],obstructed)
        self.assertIs(self.comparison.gallery['anonymous'],original)
        self.assertEqual(original['evidence_views'],[])

    def test_missing_depth_cannot_validate_an_image(self):
        missing=replace(self.frame,depth=np.full_like(self.frame.depth,np.nan))
        self.assertFalse(self.quality(missing)['usable'])

    def test_background_and_repeated_bad_views_stay_in_raw_log_only(self):
        p=self.comparison.resolver.p
        ref=NS(description='指定物体',color='red',category='box')
        p.references=[ref];p.world=NS(reference_bindings={('指定物体','red','box'):'selected'})
        bad=replace(self.frame,depth=np.full_like(self.frame.depth,np.nan))
        def rejection(id,t):
            self.obj.id=id;self.obj.color='red'
            self.comparison.remember([self.obj],replace(bad,timestamp=t))
            return p.log.event.call_args.args[1]
        self.assertFalse(rejection('background',1)['show_in_studio'])
        self.assertTrue(rejection('selected',2)['show_in_studio'])
        self.assertFalse(rejection('selected',3)['show_in_studio'])
        self.assertTrue(rejection('selected',19)['show_in_studio'])
        self.assertEqual(p.log.event.call_count,4)

    def test_robot_mask_blocks_body_even_without_valid_depth(self):
        robot=np.zeros((80,80),bool);robot[25:55,25:55]=True
        frame=replace(self.frame,robot_mask=robot)
        self.assertFalse(self.quality(frame)['usable'])

    def test_background_outside_body_is_not_occlusion(self):
        depth=self.frame.depth.copy();depth[:15,:]=.2
        self.assertTrue(self.quality(replace(self.frame,depth=depth))['usable'])

    def test_better_view_recovers_after_rejected_view(self):
        depth=self.frame.depth.copy();depth[25:55,25:55]=.6
        self.comparison.remember([self.obj],replace(self.frame,depth=depth))
        self.assertNotIn('anonymous',self.comparison.gallery)
        self.comparison.remember([self.obj],replace(self.frame,timestamp=2.))
        self.assertEqual(self.comparison.gallery['anonymous']['timestamp'],2.)


if __name__=='__main__':unittest.main()
