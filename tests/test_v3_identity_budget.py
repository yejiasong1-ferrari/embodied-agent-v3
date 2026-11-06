"""Identity budget is allotted per referent; wording follows the kind of reference."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
from test_v3_independent_review import IndependentTests
from llm.schemas import Reference
from perception.candidate_comparison import CandidateNeedsViews
from perception.identity_guidance import reference_guidance
from recovery.failure_classifier import SkillFailure
from simulation.studio_presenter import failure_text
from skills.identity_observation import identify_with_side_review
from world.world_model import AmbiguousReferenceError


class IdentityBudgetTests(IndependentTests):
    def setUp(self):
        super().setUp();self.cap=self.cfg.localization.max_vlm_calls
        self.other=Reference(description='粉色圆柱',color='pink',category='bottle')

    def test_one_referent_cannot_starve_another(self):
        self.resolver.charge(self.other,self.cap)
        self.assertFalse(self.resolver.can_spend(self.other))
        self.assertTrue(self.resolver.can_spend(self.ref,2))
        peer=self.pair();self.client.complete.side_effect=[self.answer({'A':'no'}),self.answer({'B':'yes'})]
        selected,_=self.resolver.comparison.choose(self.ref,None,True)
        self.assertEqual(selected['id'],peer.id)
        self.assertEqual(self.resolver.spent(self.ref),2)

    def test_own_allowance_still_stops_the_referent(self):
        self.pair();self.resolver.charge(self.ref,self.cap-1)
        with self.assertRaises(SkillFailure) as caught:self.resolver.comparison.choose(self.ref,None,True)
        self.client.complete.assert_not_called()
        self.assertIsNotNone(caught.exception)

    def test_calls_without_a_referent_count_against_everyone(self):
        self.resolver.calls=self.cap
        self.assertFalse(self.resolver.can_spend(self.ref));self.assertFalse(self.resolver.can_spend(self.other))

    def test_side_review_reservation_uses_the_referents_own_spend(self):
        resolver=NS(calls=self.cap,spent=lambda ref:3,comparison=NS(gallery={}))
        p=NS(identity_resolver=resolver,identify=Mock(side_effect=[CandidateNeedsViews(['a','b']),object()]),observation_count=1)
        c=NS(perception=p,config=self.cfg,held=False,world=NS(memory=NS(mission_active=False),objects={}),log=Mock())
        with patch('skills.identity_observation.observe_candidate_sides',return_value=2) as observe:
            identify_with_side_review(c,self.ref)
        observe.assert_called_once()

    def test_side_review_budget_shortfall_commits_best_guess_instead_of_failing(self):
        resolver=NS(calls=self.cap,spent=lambda ref:self.cap-1,comparison=NS(gallery={}))
        p=NS(identity_resolver=resolver,identify=Mock(side_effect=[CandidateNeedsViews(['a','b']),'guess']),observation_count=1)
        c=NS(perception=p,config=self.cfg,held=False,world=NS(memory=NS(mission_active=False),objects={}),log=Mock())
        with patch('skills.identity_observation.observe_candidate_sides') as observe:
            self.assertEqual(identify_with_side_review(c,self.ref),'guess')
        observe.assert_not_called()
        self.assertIn((self.ref.description,self.ref.color,self.ref.category),resolver.decide_now)

    def test_confident_exclusion_is_reused_while_evidence_is_unchanged(self):
        other=self.pair()
        self.client.complete.side_effect=[self.answer({'A':'no'}),self.answer({'B':'uncertain'})]
        with self.assertRaises(CandidateNeedsViews) as caught:self.resolver.comparison.choose(self.ref,None,True)
        self.assertEqual(caught.exception.candidate_ids,[other.id])
        self.assertEqual(caught.exception.required_calls,3)
        self.assertEqual(self.resolver.calls,2)
        self.client.complete.side_effect=[self.answer({'B':'yes'})]
        selected,_=self.resolver.comparison.choose(self.ref,None,True)
        self.assertEqual(selected['id'],other.id)
        self.assertEqual(self.resolver.calls,3)
        self.assertEqual(self.client.complete.call_count,3)

    def test_changed_evidence_is_reviewed_again(self):
        self.pair();self.client.complete.side_effect=[self.answer({'A':'no'}),self.answer({'B':'uncertain'})]
        with self.assertRaises(AmbiguousReferenceError):self.resolver.comparison.choose(self.ref,None,True)
        for row in self.resolver.comparison.gallery.values():row['timestamp']=2.
        self.client.complete.side_effect=[self.answer({'A':'no'}),self.answer({'B':'uncertain'})]
        with self.assertRaises(AmbiguousReferenceError):self.resolver.comparison.choose(self.ref,None,True)
        self.assertEqual(self.client.complete.call_count,4)

    def test_everyday_noun_is_not_held_to_plain_cylinder_rules(self):
        self.pair();self.client.complete.side_effect=[self.answer({'A':'no'}),self.answer({'B':'yes'})]
        self.resolver.comparison.choose(self.ref,None,True)
        payload=self.client.complete.call_args_list[0].args[2]
        self.assertEqual(payload['reference_kind'],'everyday_category')
        self.assertIsNone(payload['required_whole_form'])
        self.assertIn('handle',payload['guidance']);self.assertNotIn('plain cylinder',payload['instruction'])

    def test_explicit_geometry_keeps_the_literal_whole_form_check(self):
        guide=reference_guidance('粉色圆柱')
        self.assertEqual((guide['reference_kind'],guide['required_whole_form']),('explicit_geometry','plain_cylinder'))
        self.assertIn('entire outer silhouette',guide['guidance'])

    def test_failure_text_separates_budget_from_model_uncertainty(self):
        budget=failure_text({'failure':'LOW_CONFIDENCE','evidence':{'reason_code':'IDENTITY_BUDGET_EXHAUSTED'}})
        plain=failure_text({'failure':'LOW_CONFIDENCE','evidence':{}})
        self.assertIn('次数',budget);self.assertNotEqual(budget,plain)


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(IdentityBudgetTests(n) for n in IdentityBudgetTests.__dict__ if n.startswith('test_'))
