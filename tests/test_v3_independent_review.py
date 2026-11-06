import unittest
import numpy as np
from test_v3_candidate_comparison import ComparisonTests
from llm.schemas import VisualDisambiguation
from recovery.failure_classifier import SkillFailure
from world.world_model import AmbiguousReferenceError

class IndependentTests(ComparisonTests):
    def setUp(self):
        super().setUp();self.cfg.localization.independent_candidate_review=True

    def test_each_candidate_gets_its_own_image_and_id(self):
        other=self.pair()
        self.client.complete.side_effect=[self.answer({'A':'no'}),self.answer({'B':'yes'})]
        selected,evidence=self.resolver.comparison.choose(self.ref,None,True)
        self.assertEqual(selected['id'],other.id)
        calls=self.client.complete.call_args_list
        self.assertEqual([c.args[2]['candidate_ids'] for c in calls],[['A'],['B']])
        self.assertTrue(all(c.kwargs['image'].shape==(640,640,3) for c in calls))
        self.assertEqual(evidence['review_mode'],'independent_objects')
        self.assertEqual(set(evidence['individual_images']),{'A','B'})
        self.assertEqual(self.resolver.calls,2)

    def test_budget_is_checked_before_partial_reviews(self):
        self.pair();self.resolver.calls=self.cfg.localization.max_vlm_calls-1
        with self.assertRaises(SkillFailure):self.resolver.comparison.choose(self.ref,None,True)
        self.client.complete.assert_not_called()

    def test_wrong_label_cannot_be_mapped_to_another_object(self):
        self.pair();self.client.complete.return_value=self.answer({'B':'yes'})
        with self.assertRaises(AmbiguousReferenceError):self.resolver.comparison.choose(self.ref,None,True)
        self.assertFalse(self.world.reference_bindings)

    def test_uncertainty_does_not_become_forced_choice(self):
        self.pair();self.client.complete.side_effect=[self.answer({'A':'uncertain'}),self.answer({'B':'yes'})]
        with self.assertRaises(AmbiguousReferenceError):self.resolver.comparison.choose(self.ref,None,True)

    def test_long_explanation_does_not_reject_valid_decision(self):
        result=VisualDisambiguation(candidates=[{'candidate_id':'A','verdict':'no','confidence':.95,'evidence':'x'*308}])
        self.assertEqual(result.candidates[0].verdict,'no')
        with self.assertRaises(ValueError):VisualDisambiguation(candidates=[{'candidate_id':'A','verdict':'no','confidence':.95,'evidence':'x'*2001}])

    def test_visible_geometry_is_measured_and_missing_samples_stay_unknown(self):
        self.assertIsNone(self.resolver.comparison.visible_profile(self.obj))
        rng=np.random.default_rng(0)
        self.obj.points=rng.uniform([0,0,0],[.04,.04,.10],(500,3))
        profile=self.resolver.comparison.visible_profile(self.obj)
        self.assertEqual(len(profile['sections']),4)
        self.assertNotIn('category',profile)
        self.assertIn('unknown',profile['limitation'])

    def test_measured_shape_does_not_authorize_identity_without_model(self):
        other=self.pair()
        for row in self.resolver.comparison.gallery.values():row['visible_geometry']={'observed_height_m':.08}
        self.client.complete.side_effect=[self.answer({'A':'uncertain'}),self.answer({'B':'uncertain'})]
        with self.assertRaises(AmbiguousReferenceError):self.resolver.comparison.choose(self.ref,None,True)
        self.assertFalse(self.world.reference_bindings)
        self.assertIn('visible_rgbd_geometry',self.client.complete.call_args.args[2])

    def test_single_remaining_candidate_keeps_both_camera_views(self):
        other=self.pair();gallery=self.resolver.comparison.gallery;del gallery[other.id]
        row=gallery[self.obj.id]
        row['evidence_views']=[{'patch':row['patch'],'view':'side:1','timestamp':2.}]
        self.client.complete.return_value=self.answer({'A':'yes'})
        self.resolver.comparison.choose(self.ref,None,True)
        call=self.client.complete.call_args
        self.assertEqual(call.kwargs['image'].shape,(640,1280,3))
        self.assertEqual(len(call.args[2]['views']),2)

    def test_affirmative_verdict_cannot_override_model_reported_attribute_conflict(self):
        self.pair()
        for field in ('shape_match','color_match'):
            for verdict in ('no','uncertain'):
                answer=self.answer({'A':'yes'})
                setattr(answer.candidates[0],field,verdict)
                self.client.complete.side_effect=[answer,self.answer({'B':'no'})]
                self.resolver.calls=0
                with self.assertRaises(AmbiguousReferenceError):self.resolver.comparison.choose(self.ref,None,True)
        self.assertFalse(self.world.reference_bindings)
        self.assertTrue(any(c.args[0]=='identity_review_inconsistent' for c in self.p.log.event.call_args_list))


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(IndependentTests(n) for n in IndependentTests.__dict__ if n.startswith('test_'))
