"""Phase policy regression: image identity persists; permissions do not."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from test_v3_identity import IdentityTests
from perception.visual_identity import reference_key,appearance
from llm.schemas import Goal,Reference
from skills.skill_base import SkillContext
from recovery.failure_classifier import SkillFailure
from world.world_model import WorldObject


class PhaseTests(IdentityTests):
    # Reuse only fixture helpers, not the inherited test suite (see load_tests).
    def setup_pair(self):
        self.confirm();ref=Reference(description='绿色积木')
        obj=WorldObject('reference','green box','green','box',np.array([.5,.1,.05]),np.array([0,0,0,1]),np.array([.06,.06,.1]),.95,1.)
        obj.bbox=(300,100,350,150);self.world.objects[obj.id]=obj
        self.world.bind_reference(ref,obj.id,{'source':'vlm_identity','image_sha256':'a'*64})
        self.resolver.anchors[reference_key(ref)]={'id':obj.id,'position':obj.position.copy(),'appearance':appearance(self.frame(480,640).rgb,obj.bbox),'last_observed':1.,'uncertainty_m':.003,'visibility':'visible','review_required':False}
        self.goal=Goal(goal='place_right',object=self.ref,target=ref)
        self.context=SimpleNamespace(goal=self.goal,current_skill='APPROACH_OBJECT',held=False)
        return ref,obj

    def test_reference_missing_during_approach_preserves_model_identity(self):
        ref,obj=self.setup_pair();obj.visible=False
        self.resolver.check_continuity([self.obj],self.frame(480,640))
        self.resolver.require_phase(self.context)
        self.assertIn(reference_key(ref),self.world.reference_bindings)
        self.assertFalse(self.resolver.valid(ref))
        self.assertTrue(self.resolver.valid(ref,fresh=False))

    def test_source_missing_still_blocks_grasp(self):
        ref,obj=self.setup_pair();self.obj.visible=False
        self.context.current_skill='GRASP_OBJECT'
        with self.assertRaises(SkillFailure):self.resolver.require_phase(self.context)

    def test_observation_can_recover_without_identity_but_grasp_cannot(self):
        self.setup_pair();self.world.reference_bindings.clear()
        self.context.observation_motion=True;self.resolver.require_phase(self.context)
        self.context.observation_motion=False
        with self.assertRaises(SkillFailure):self.resolver.require_phase(self.context)

    def test_expired_reference_blocks_manipulation(self):
        ref,obj=self.setup_pair();obj.visible=False;self.p.last_frame=SimpleNamespace(timestamp=200.)
        with self.assertRaises(SkillFailure):self.resolver.require_phase(self.context)

    def test_uncertain_reference_memory_is_not_permission(self):
        ref,obj=self.setup_pair();anchor=self.resolver.anchors[reference_key(ref)]
        anchor['uncertainty_m']=.04
        with self.assertRaises(SkillFailure):self.resolver.require_phase(self.context)

    def test_reference_displacement_conflict_needs_model_review(self):
        ref,obj=self.setup_pair();obj.position+=.2
        self.resolver.check_continuity([self.obj,obj],self.frame(480,640))
        self.assertTrue(self.resolver.anchors[reference_key(ref)]['review_required'])
        with self.assertRaises(SkillFailure):self.resolver.require_phase(self.context)
        # Later local sightings cannot erase the model review requirement.
        self.resolver.check_continuity([self.obj,obj],self.frame(480,640))
        self.assertFalse(self.resolver.valid(ref,fresh=False))

    def test_same_colour_ambiguity_is_not_renewed_from_local_pixels(self):
        self.setup_pair()
        rival=WorldObject('rival','red cup','red','cup',self.obj.position+.02,np.array([0,0,0,1]),self.obj.size,.95,1.)
        self.resolver.check_continuity([self.obj,rival],self.frame(480,640))
        self.assertFalse(self.resolver.valid(self.ref,fresh=False))

    def test_final_acceptance_requires_all_references_fresh(self):
        ref,obj=self.setup_pair();obj.visible=False
        with self.assertRaises(SkillFailure):self.resolver.require_all(self.goal)

    def test_carrying_uses_bounded_reference_memory(self):
        ref,obj=self.setup_pair();obj.visible=False
        self.context.current_skill='MOVE_TO_TARGET';self.context.held=True
        self.resolver.require_phase(self.context)
        self.p.last_frame=SimpleNamespace(timestamp=1000.)
        with self.assertRaises(SkillFailure):self.resolver.require_phase(self.context)

    def test_observation_scope_restores_flags_after_collision_failure(self):
        context=SimpleNamespace(observation_motion=False,cartesian_speed_limit=.1,held=True,log=Mock(),move=Mock(side_effect=SkillFailure('COLLISION_RISK')))
        with self.assertRaises(SkillFailure):SkillContext.observe_move(context,np.zeros(3))
        self.assertFalse(context.observation_motion);self.assertEqual(context.cartesian_speed_limit,.1)
        context.move.assert_called_once_with(unittest.mock.ANY,None,allow_contact=True)

    def test_multiple_placement_references_are_all_checked(self):
        from copy import deepcopy
        ref,obj=self.setup_pair();third=Reference(description='黄色积木')
        other=deepcopy(obj);other.id='third';other.color='yellow';self.world.objects[other.id]=other
        self.world.bind_reference(third,other.id,{'source':'vlm_identity','image_sha256':'b'*64})
        anchor=deepcopy(self.resolver.anchors[reference_key(ref)]);anchor['id']=other.id
        anchor['review_required']=True;self.resolver.anchors[reference_key(third)]=anchor
        self.context.goal=SimpleNamespace(object=self.ref,references=[ref,third])
        self.context.current_skill='MOVE_TO_TARGET';self.context.held=True
        with self.assertRaises(SkillFailure):self.resolver.require_phase(self.context)

    def test_descent_requires_fresh_reference_after_placement_check(self):
        ref,obj=self.setup_pair();obj.visible=False
        self.context.current_skill='PLACE_OBJECT';self.context.held=True
        self.context.placement_reference_checked=False
        self.resolver.require_phase(self.context)
        self.context.placement_reference_checked=True
        self.p.last_frame=SimpleNamespace(timestamp=70.)
        with self.assertRaises(SkillFailure):self.resolver.require_phase(self.context)

    def test_masked_appearance_ignores_changed_background(self):
        import cv2
        a=np.full((100,100,3),240,np.uint8);b=np.full((100,100,3),20,np.uint8)
        mask=np.zeros((100,100),bool);mask[30:70,30:70]=True
        a[mask]=[0,0,220];b[mask]=[0,0,220]
        first=appearance(a,(0,0,100,100),mask);second=appearance(b,(0,0,100,100),mask)
        self.assertGreater(cv2.compareHist(first,second,cv2.HISTCMP_CORREL),.99)


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(PhaseTests(name) for name in PhaseTests.__dict__ if name.startswith('test_'))
