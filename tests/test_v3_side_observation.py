import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import numpy as np
from config import AgentConfig
from perception.camera import CameraFrame
from perception.candidate_comparison import CandidateNeedsViews
from recovery.failure_classifier import SkillFailure
from skills.identity_observation import identify_with_side_review,move_identity_camera,observe_candidate_sides


class SideObservationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=AgentConfig(version=3,strict_visual_identity=True)
        self.ref=NS(description='object',color=None,category='object')
        self.resolver=NS(calls=2,comparison=NS(gallery={}))
        self.p=NS(identity_resolver=self.resolver,identify=Mock(),observation_count=1)
        self.c=NS(perception=self.p,config=self.cfg,held=False,world=NS(memory=NS(mission_active=False),objects={}),
            planner=Mock(),executor=Mock(),robot=Mock(),budget=Mock(),log=Mock(),cartesian_speed_limit=.08,observation_motion=False)
        self.c.planner.last_diagnostics=[];self.c.planner.last_failure='IK_FAILED'

    def test_uncertain_gets_new_observation_before_second_model_decision(self):
        target=object();self.p.identify.side_effect=[CandidateNeedsViews(['a','b']),target]
        with patch('skills.identity_observation.observe_candidate_sides',return_value=2) as observe:
            self.assertIs(identify_with_side_review(self.c,self.ref),target)
        observe.assert_called_once_with(self.c,['a','b'])
        self.assertEqual(self.p.identify.call_count,2)
        self.assertIsNone(self.resolver.observation_context)

    def test_repeated_uncertainty_has_only_one_side_round(self):
        self.p.identify.side_effect=CandidateNeedsViews(['a','b'])
        with patch('skills.identity_observation.observe_candidate_sides',return_value=2) as observe:
            for _ in range(2):
                with self.assertRaises(CandidateNeedsViews):identify_with_side_review(self.c,self.ref)
        self.assertEqual(observe.call_count,1)

    def test_budget_exhaustion_does_not_move_camera(self):
        self.resolver.calls=9;self.p.identify.side_effect=[CandidateNeedsViews(['a','b']),'guess']
        with patch('skills.identity_observation.observe_candidate_sides') as observe:
            self.assertEqual(identify_with_side_review(self.c,self.ref),'guess')
        observe.assert_not_called()

    def test_budget_reserves_all_reviews_joint_and_fresh_confirmation(self):
        self.resolver.calls=5
        self.p.identify.side_effect=[CandidateNeedsViews(['a'],required_calls=6),'guess']
        with patch('skills.identity_observation.observe_candidate_sides') as observe:
            self.assertEqual(identify_with_side_review(self.c,self.ref),'guess')
        observe.assert_not_called()

    def test_request_error_does_not_trigger_side_search(self):
        self.p.identify.side_effect=SkillFailure('LOW_CONFIDENCE',{'terminal':True})
        with patch('skills.identity_observation.observe_candidate_sides') as observe:
            with self.assertRaises(SkillFailure):identify_with_side_review(self.c,self.ref)
        observe.assert_not_called()

    def test_carrying_never_searches_unconfirmed_candidate(self):
        self.c.held=True;self.p.identify.side_effect=CandidateNeedsViews(['a'])
        with patch('skills.identity_observation.observe_candidate_sides') as observe:
            with self.assertRaises(CandidateNeedsViews):identify_with_side_review(self.c,self.ref)
        observe.assert_not_called()

    def test_unreachable_pose_is_not_executed(self):
        self.c.planner.plan_to_pose.return_value=None
        self.assertFalse(move_identity_camera(self.c,object()))
        self.c.executor.execute.assert_not_called()

    def test_motion_failure_restores_observation_permissions(self):
        self.c.executor.execute.return_value=(False,'COLLISION_RISK')
        with self.assertRaises(SkillFailure):move_identity_camera(self.c,object())
        self.assertFalse(self.c.observation_motion);self.assertEqual(self.c.cartesian_speed_limit,.08)
        self.assertEqual(self.c.executor.execute.call_args.kwargs['cartesian_speed_limit'],.05)

    def test_side_capture_uses_detail_and_requires_same_tracked_instance(self):
        frame=CameraFrame(np.zeros((960,1280,3),np.uint8),np.ones((960,1280)),np.eye(3),np.eye(4),2.,'camera:0')
        self.p.camera=Mock(config=NS(mode='wrist'));self.p.camera.capture_detail.return_value=frame
        self.c.world.objects={'a':NS(id='a')}
        def observed(**kw):
            self.resolver.comparison.gallery['a']={'timestamp':kw['frame'].timestamp}
            self.assertTrue(kw['frame'].view_id.startswith('identity_side:'))
            return [NS(id='a')]
        self.p.observe=Mock(side_effect=observed)
        with patch('skills.identity_observation.ViewpointPlanner') as planner,patch('skills.identity_observation.move_identity_camera',return_value=True):
            planner.return_value.candidates.return_value=[object()]
            self.assertEqual(observe_candidate_sides(self.c,['a']),1)
        self.p.camera.capture_detail.assert_called_once()

    def test_unreachable_low_views_try_higher_oblique_fallback(self):
        frame=CameraFrame(np.zeros((960,1280,3),np.uint8),np.ones((960,1280)),np.eye(3),np.eye(4),2.,'camera:0')
        self.p.camera=Mock(config=NS(mode='wrist'));self.p.camera.capture_detail.return_value=frame
        self.c.world.objects={'a':NS(id='a')}
        def observed(**kw):
            self.resolver.comparison.gallery['a']={'timestamp':kw['frame'].timestamp}
            return [NS(id='a')]
        self.p.observe=Mock(side_effect=observed)
        with patch('skills.identity_observation.ViewpointPlanner') as planner,patch(
                'skills.identity_observation.move_identity_camera',side_effect=[False]*4+[True]) as move:
            planner.return_value.candidates.return_value=[object() for _ in range(8)]
            self.assertEqual(observe_candidate_sides(self.c,['a']),1)
            self.assertEqual(move.call_count,5)

    def test_rejected_new_image_is_not_reported_as_successful_side_capture(self):
        frame=CameraFrame(np.zeros((960,1280,3),np.uint8),np.ones((960,1280)),np.eye(3),np.eye(4),2.,'camera:0')
        self.p.camera=Mock(config=NS(mode='wrist'));self.p.camera.capture_detail.return_value=frame
        self.c.world.objects={'a':NS(id='a')}
        self.resolver.comparison.gallery['a']={'timestamp':1.}
        self.p.observe=Mock(return_value=[NS(id='a')])
        with patch('skills.identity_observation.ViewpointPlanner') as planner,patch(
                'skills.identity_observation.move_identity_camera',return_value=True):
            planner.return_value.candidates.return_value=[object()]
            self.assertEqual(observe_candidate_sides(self.c,['a']),0)
        self.assertTrue(any(call.args[1].get('status')=='unavailable' for call in self.c.log.event.call_args_list))

if __name__=='__main__':unittest.main()
