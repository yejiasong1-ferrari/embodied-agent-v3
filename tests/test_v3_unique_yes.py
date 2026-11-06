import unittest
from llm.schemas import VisualDisambiguation,SpatialConstraint
from test_v3_candidate_comparison import ComparisonTests


class UniqueYesTests(ComparisonTests):
    def mixed(self,other_verdict,other_confidence,yes_confidence=.95):
        rows=[{'candidate_id':'A','verdict':other_verdict,'confidence':other_confidence,'evidence':'only weak partial evidence'},
              {'candidate_id':'B','verdict':'yes','confidence':yes_confidence,'evidence':'clear match'}]
        return VisualDisambiguation(candidates=rows)

    def test_unique_yes_beats_a_candidate_the_model_could_not_lean_towards(self):
        other=self.pair();self.client.complete.return_value=self.mixed('uncertain',.5)
        self.resolver.comparison.gallery[other.id]['view']='identity_side:9' # it has had its own close look
        chosen,_=self.resolver.comparison.choose(self.ref,None,True)
        self.assertEqual(chosen['id'],other.id)
        self.assertFalse(self.world.reference_bindings) # fresh confirmation is still required

    def test_leaning_uncertain_still_needs_more_evidence(self):
        self.pair();self.client.complete.return_value=self.mixed('uncertain',.8)
        with self.assertRaises(Exception):self.resolver.comparison.choose(self.ref,None,True)

    def test_unique_yes_must_itself_be_confident(self):
        self.pair();self.client.complete.return_value=self.mixed('uncertain',.5,yes_confidence=.7)
        with self.assertRaises(Exception):self.resolver.comparison.choose(self.ref,None,True)


class RelationNormalisationTests(unittest.TestCase):
    def test_above_without_gap_means_resting_on_the_reference(self):
        self.assertEqual(SpatialConstraint(relation='above').relation,'on')

    def test_explicit_gap_keeps_the_floating_meaning(self):
        self.assertEqual(SpatialConstraint(relation='above',distance_m=.08).relation,'above')

    def test_below_is_never_rewritten(self):
        self.assertEqual(SpatialConstraint(relation='below').relation,'below')


def load_tests(loader,tests,pattern):
    return unittest.TestSuite([UniqueYesTests(n) for n in UniqueYesTests.__dict__ if n.startswith('test_')]
        +[RelationNormalisationTests(n) for n in RelationNormalisationTests.__dict__ if n.startswith('test_')]
        +[UniqueYesNeedsOwnSideViewTests(n) for n in UniqueYesNeedsOwnSideViewTests.__dict__ if n.startswith('test_')]
        +[ColourPrefilterTests(n) for n in ColourPrefilterTests.__dict__ if n.startswith('test_')])


class ColourPrefilterTests(ComparisonTests):
    def test_candidate_of_another_colour_is_not_reviewed(self):
        other=self.pair()
        self.resolver.comparison.gallery[self.obj.id]['colors']={'red':.9}
        self.resolver.comparison.gallery[other.id]['colors']={'red':.3}
        rows=self.resolver.comparison.candidates(self.ref,None)
        self.assertEqual([r['id'] for r in rows],[self.obj.id])

    def test_two_comparable_candidates_both_stay(self):
        other=self.pair()
        self.resolver.comparison.gallery[self.obj.id]['colors']={'red':.9}
        self.resolver.comparison.gallery[other.id]['colors']={'red':.6}
        self.assertEqual(len(self.resolver.comparison.candidates(self.ref,None)),2)


class UniqueYesNeedsOwnSideViewTests(ComparisonTests):
    def test_selection_through_the_shortcut_first_asks_for_a_view_of_the_selected_only(self):
        from perception.candidate_comparison import CandidateNeedsViews
        other=self.pair()
        rows=[{'candidate_id':'A','verdict':'uncertain','confidence':.5,'evidence':'weak'},
              {'candidate_id':'B','verdict':'yes','confidence':.95,'evidence':'clear match'}]
        self.client.complete.return_value=VisualDisambiguation(candidates=rows)
        with self.assertRaises(CandidateNeedsViews) as caught:self.resolver.comparison.choose(self.ref,None,True)
        self.assertEqual(caught.exception.candidate_ids,[other.id])
