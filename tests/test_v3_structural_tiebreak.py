import unittest
from llm.schemas import Reference
from test_v3_joint_comparison import JointTests
from world.world_model import AmbiguousReferenceError


class StructuralTiebreakTests(JointTests):
    def unresolved(self,b_verdict='uncertain',a_confidence=1.,b_form='plain_cylinder'):
        answer=self.decision('B');answer.status='ambiguous';answer.selected_candidate_id=None
        answer.candidates[0].confidence=a_confidence
        answer.candidates[1].verdict=b_verdict;answer.candidates[1].confidence=.7;answer.candidates[1].color_match='uncertain'
        answer.structures[1].whole_form=b_form
        if b_form!='plain_cylinder':answer.structures[1].neck='present'
        return answer

    def test_only_structural_survivor_with_independent_yes_goes_on_to_confirmation(self):
        other,_=self.rows()
        self.client.complete.side_effect=[self.answer({'A':'yes'}),self.answer({'B':'yes'}),self.unresolved()]
        chosen,evidence=self.resolver.comparison.choose(self.ref,None,True)
        self.assertEqual(chosen['id'],other.id)
        self.assertEqual(evidence['joint_review']['structural_tiebreak'],other.id)
        self.assertFalse(self.world.reference_bindings)

    def test_survivor_needs_its_own_independent_yes(self):
        self.rows()
        self.client.complete.side_effect=[self.answer({'A':'yes'}),self.answer({'B':'uncertain'}),self.unresolved()]
        with self.assertRaises(AmbiguousReferenceError):self.resolver.comparison.choose(self.ref,None,True)

    def test_exclusion_needs_high_confidence(self):
        self.rows()
        self.client.complete.side_effect=[self.answer({'A':'yes'}),self.answer({'B':'yes'}),self.unresolved(a_confidence=.7)]
        with self.assertRaises(AmbiguousReferenceError):self.resolver.comparison.choose(self.ref,None,True)

    def test_survivor_must_itself_fit_the_explicit_form(self):
        self.rows()
        self.client.complete.side_effect=[self.answer({'A':'yes'}),self.answer({'B':'yes'}),self.unresolved(b_form='stepped_cylindrical')]
        with self.assertRaises(AmbiguousReferenceError):self.resolver.comparison.choose(self.ref,None,True)

    def test_everyday_category_never_uses_the_structural_path(self):
        self.rows();self.ref=Reference(description='粉色杯子')
        self.client.complete.side_effect=[self.answer({'A':'yes'}),self.answer({'B':'yes'}),self.unresolved()]
        with self.assertRaises(AmbiguousReferenceError):self.resolver.comparison.choose(self.ref,None,True)


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(StructuralTiebreakTests(n) for n in StructuralTiebreakTests.__dict__ if n.startswith('test_'))
