import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from test_v3_candidate_comparison import ComparisonTests
from skills.skill_registry import SkillRegistry
from skills.manipulation import VerifyTaskSkill
from recovery.failure_classifier import SkillFailure
from world.world_model import AmbiguousReferenceError
from llm.llm_client import ModelRequestError
from simulation.studio_presenter import failure_text


class FreshComparisonTests(ComparisonTests):
    def test_final_review_excludes_historical_crop(self):
        other=self.pair();gallery=self.resolver.comparison.gallery
        gallery[self.obj.id]['timestamp']=101.
        self.p.last_frame=SimpleNamespace(timestamp=101.,view_id='camera:0')
        self.client.complete.return_value=self.answer({'A':'yes'})
        chosen,_=self.resolver.comparison.choose(self.ref,self.obj.id,False)
        self.assertEqual(chosen['id'],self.obj.id)
        self.assertEqual(self.client.complete.call_args.args[2]['candidate_ids'],['A'])

    def test_current_competitor_still_prevents_confirmation(self):
        other=self.pair()
        for row in self.resolver.comparison.gallery.values():row.update(timestamp=101.,view='camera:0')
        self.p.last_frame=SimpleNamespace(timestamp=101.,view_id='camera:0')
        self.client.complete.return_value=self.answer({'A':'yes','B':'uncertain'})
        with self.assertRaises(AmbiguousReferenceError):self.resolver.comparison.choose(self.ref,self.obj.id,False)


class FinalReviewTests(unittest.TestCase):
    def setup_context(self):
        ref=SimpleNamespace(description='box',color='blue',category='box');key=('box','blue','box')
        resolver=Mock();resolver.anchors={key:{'review_required':True}}
        resolver.identify.return_value=SimpleNamespace(id='tracked')
        c=SimpleNamespace(config=SimpleNamespace(strict_visual_identity=True),
            perception=SimpleNamespace(last_frame=None,identity_resolver=resolver),
            goal=SimpleNamespace(object=ref,references=[]),world=SimpleNamespace(reference_bindings={key:'tracked'}),
            log=Mock(),final_geometry_evidence={'accepted':True})
        registry=SkillRegistry();skill=Mock();skill.execute.return_value=True;skill.validate.return_value=True
        registry.skills['VERIFY_TASK']=skill
        return c,registry,skill,resolver

    def test_unresolved_identity_becomes_recoverable_failure_without_success(self):
        c,r,skill,resolver=self.setup_context();resolver.identify.side_effect=AmbiguousReferenceError('unresolved')
        with self.assertRaises(SkillFailure) as raised:r.execute('VERIFY_TASK',c)
        self.assertEqual(raised.exception.evidence['reason_code'],'IDENTITY_REVIEW_REQUIRED')
        c.log.event.assert_not_called();skill.validate.assert_not_called()

    def test_success_is_emitted_only_after_all_checks(self):
        c,r,skill,resolver=self.setup_context()
        self.assertTrue(r.execute('VERIFY_TASK',c))
        resolver.identify.assert_called_once_with(c.goal.object,allow_search=False)
        self.assertTrue(c.log.event.call_args.args[1]['identity_verified'])

    def test_monitor_failure_cannot_emit_success(self):
        c,r,skill,resolver=self.setup_context();skill.monitor.side_effect=SkillFailure('TARGET_LOST')
        with self.assertRaises(SkillFailure):r.execute('VERIFY_TASK',c)
        c.log.event.assert_not_called()

    def test_geometry_measurement_is_not_final_success(self):
        c,_,_,_=self.setup_context()
        VerifyTaskSkill.record_measurement(c,{'accepted':True})
        self.assertEqual(c.log.event.call_args.args[0],'geometric_validation')

    def test_content_rejection_is_not_generic_http_error(self):
        exc=ModelRequestError(400,{'code':'data_inspection_failed'})
        self.assertFalse(exc.retryable)
        self.assertIn('审核',failure_text({'failure':'LOW_CONFIDENCE','evidence':{'reason_code':exc.reason_code}}))


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(loader.loadTestsFromName(cls.__name__+'.'+name,__import__(__name__))
        for cls in (FreshComparisonTests,FinalReviewTests)
        for name in cls.__dict__ if name.startswith('test_'))

