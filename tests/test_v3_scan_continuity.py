"""Identity continuity regressions; fixtures are not task success evidence."""
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from test_v3_identity import IdentityTests
from perception.visual_identity import reference_key
from recovery.failure_classifier import SkillFailure
from recovery.recovery_manager import RecoveryManager
from simulation.studio_presenter import failure_text


class ScanContinuityTests(IdentityTests):
    def test_distant_same_colour_neighbour_does_not_cancel_stable_identity(self):
        self.confirm();other=deepcopy(self.obj);other.id='neighbour'
        other.position=self.obj.position+np.array([.092,.016,.004])
        self.resolver.check_continuity([self.obj,other],self.frame(480,640))
        self.assertTrue(self.resolver.valid(self.ref))

    def test_same_spot_no_rival_survives_a_view_dependent_colour_shift(self):
        self.confirm();anchor=self.resolver.anchors[reference_key(self.ref)]
        anchor['appearance']=np.roll(anchor['appearance'],7)  # histogram of another viewing angle
        with patch('perception.visual_identity.cv2.compareHist',return_value=.30):
            self.resolver.check_continuity([self.obj],self.frame(480,640))
        self.assertTrue(self.resolver.valid(self.ref))
        with patch('perception.visual_identity.cv2.compareHist',return_value=-.5):
            self.resolver.check_continuity([self.obj],self.frame(480,640))
        self.assertFalse(self.resolver.valid(self.ref))

    def test_competing_instance_blocks_motion_even_with_stable_colour(self):
        self.confirm();other=deepcopy(self.obj);other.id='contender'
        other.position=self.obj.position+np.array([.02,0,0])
        self.resolver.check_continuity([self.obj,other],self.frame(480,640))
        with self.assertRaises(SkillFailure) as failure:self.resolver.require_references([self.ref],fresh=True)
        self.assertEqual(failure.exception.evidence['reason_code'],'IDENTITY_REVIEW_REQUIRED')

    def test_post_scan_review_uses_current_camera_and_preserves_instance(self):
        self.confirm();self.resolver.anchors[reference_key(self.ref)]['review_required']=True
        context=SimpleNamespace(goal=SimpleNamespace(object=self.ref),source_id=self.obj.id,log=self.p.log,refresh=Mock())
        with patch('perception.visual_identity.prompted_depth_mask',side_effect=lambda frame,*_:np.ones(frame.depth.shape,bool)):
            self.resolver.reconcile_after_scan(context)
        self.assertTrue(self.resolver.valid(self.ref));context.refresh.assert_called_once()
        self.camera.move_viewpoint.assert_not_called()
        self.assertEqual(self.client.complete.call_args.args[2]['continuity_review']['previous_tracking_id'],self.obj.id)

    def test_model_reviewed_stable_neighbour_does_not_trigger_endless_review(self):
        self.confirm();other=deepcopy(self.obj);other.id='other'
        other.position=self.obj.position+np.array([.02,0,0]);other.query=''
        self.world.objects[other.id]=other
        self.resolver.check_continuity([self.obj,other],self.frame(480,640))
        self.p.observe.return_value=[self.obj,other]
        self.confirm()
        self.resolver.check_continuity([self.obj,other],self.frame(480,640))
        self.assertTrue(self.resolver.valid(self.ref))
        other.position+=np.array([.02,0,0])
        self.resolver.check_continuity([self.obj,other],self.frame(480,640))
        self.assertFalse(self.resolver.valid(self.ref))

    def test_post_scan_ambiguous_cannot_authorize(self):
        self.confirm();self.resolver.anchors[reference_key(self.ref)]['review_required']=True
        self.client.complete.return_value=self.reply('ambiguous')
        context=SimpleNamespace(goal=SimpleNamespace(object=self.ref),source_id=self.obj.id,log=self.p.log,refresh=Mock())
        with self.assertRaises(SkillFailure):self.resolver.reconcile_after_scan(context)
        self.assertFalse(self.resolver.valid(self.ref));context.refresh.assert_not_called()

    def test_repeated_identity_conflict_does_not_spend_grasp_recovery(self):
        context=SimpleNamespace(candidate=None,current_skill='APPROACH_OBJECT',source=self.obj,robot=Mock(),held=False)
        manager=RecoveryManager(self.cfg,Mock(),Mock())
        evidence={'reason_code':'IDENTITY_REVIEW_REQUIRED','reference':self.ref.description,'review':{'rival_ids':['other'],'similarity':.99}}
        self.assertEqual(manager.decide(SkillFailure('TARGET_LOST',evidence),context),'REOBSERVE')
        changed={**evidence,'review':{'rival_ids':['other'],'similarity':.98}}
        self.assertEqual(manager.decide(SkillFailure('TARGET_LOST',changed),context),'REOBSERVE')
        self.assertEqual(manager.failed_candidates,set())

    def test_new_conflict_after_review_is_not_suppressed(self):
        self.confirm();self.resolver.anchors[reference_key(self.ref)]['review_required']=True
        def conflict_again():self.resolver.anchors[reference_key(self.ref)]['review_required']=True
        context=SimpleNamespace(goal=SimpleNamespace(object=self.ref),source_id=self.obj.id,log=self.p.log,refresh=Mock(side_effect=conflict_again))
        with patch('perception.visual_identity.prompted_depth_mask',side_effect=lambda frame,*_:np.ones(frame.depth.shape,bool)):
            with self.assertRaises(SkillFailure):self.resolver.reconcile_after_scan(context)

    def test_pregrasp_rechecks_identity_before_contact_motion(self):
        from skills.manipulation import ApproachSkill
        resolver=Mock();resolver.reconcile_after_scan.side_effect=SkillFailure('TARGET_LOST')
        context=SimpleNamespace(log=Mock(),candidate=SimpleNamespace(id='top_grasp:0'),
            config=self.cfg,perception=SimpleNamespace(observe=Mock(),identity_resolver=resolver),refresh=Mock(),move=Mock())
        with self.assertRaises(SkillFailure):ApproachSkill().refine(context)
        context.move.assert_not_called()

    def test_prior_volume_recovers_occluded_surface_without_using_colour(self):
        from perception.camera import CameraFrame
        from perception.visual_identity import prompted_depth_mask
        rgb=np.zeros((64,64,3),np.uint8);depth=np.ones((64,64))
        depth[10:54,10:54]=.96;depth[10:54,27:37]=.8
        pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=1.
        frame=CameraFrame(rgb,depth,np.array([[100,0,32],[0,100,32],[0,0,1]]),pose,1.,'camera')
        prior=SimpleNamespace(low=np.array([-.23,-.23,0]),high=np.array([.23,.23,.04]),uncertainty_m=.003)
        mask=prompted_depth_mask(frame,(10,10,54,54),30,prior)
        self.assertIsNotNone(mask)
        self.assertTrue(mask[30,15]);self.assertTrue(mask[30,45])
        self.assertFalse(mask[30,32]);self.assertFalse(mask[0,0])

    def test_visible_identity_conflict_has_accurate_ui_description(self):
        text=failure_text({'failure':'TARGET_LOST','evidence':{'identity_review_required':True}})
        self.assertIn('身份待复核',text);self.assertNotIn('看不到',text)


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(ScanContinuityTests(name) for name in ScanContinuityTests.__dict__ if name.startswith('test_'))
