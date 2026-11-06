import unittest
from unittest.mock import patch
from llm.schemas import Reference,VisualComparisonDecision
from test_v3_candidate_comparison import ComparisonTests
from perception.joint_comparison import joint_review,requested_form
from recovery.failure_classifier import SkillFailure
from world.world_model import AmbiguousReferenceError


class JointTests(ComparisonTests):
    def setUp(self):
        super().setUp();self.cfg.localization.independent_candidate_review=True
        self.ref=Reference(description='粉色圆柱')
        self.p.task_description='把粉色圆柱放到红色杯子上'

    def rows(self):
        other=self.pair()
        for row in self.resolver.comparison.gallery.values():row['colors']={'pink':.8}
        rows=list(self.resolver.comparison.gallery.values());rows[0]['view']='identity_side:14'
        return other,{'A':rows[0],'B':rows[1]}

    def decision(self,selected='B',form='plain_cylinder',neck='absent',shoulder='absent'):
        structures=[{'candidate_id':k,'whole_form':form if k==selected else 'stepped_cylindrical',
                     'neck':neck if k==selected else 'present','shoulder':shoulder if k==selected else 'present',
                     'evidence':'visible whole silhouette'} for k in ['A','B']]
        return VisualComparisonDecision(status='selected',selected_candidate_id=selected,confidence=.95,
            structures=structures,candidates=[{'candidate_id':k,'verdict':'yes' if k==selected else 'no',
                'confidence':.95,'evidence':'compared whole form','shape_match':'yes' if k==selected else 'no',
                'color_match':'yes'} for k in ['A','B']],evidence='literal whole shape distinguishes alternatives')

    def test_two_independent_yes_get_joint_comparison_before_stopping(self):
        other,labels=self.rows()
        self.client.complete.side_effect=[self.answer({'A':'yes'}),self.answer({'B':'yes'}),self.decision()]
        chosen,evidence=self.resolver.comparison.choose(self.ref,None,True)
        self.assertEqual(chosen['id'],other.id)
        self.assertEqual(self.client.complete.call_args.args[1],'visual_comparison')
        self.assertEqual(self.client.complete.call_args.kwargs['image'].shape,(960,1280,3))
        self.assertEqual(evidence['joint_review']['selected'],other.id)
        self.assertFalse(self.world.reference_bindings) # still needs fresh VLM confirmation

    def test_stepped_winner_cannot_override_explicit_plain_cylinder(self):
        _,labels=self.rows();self.client.complete.return_value=self.decision('A','stepped_cylindrical','present','present')
        selected,evidence=joint_review(self.resolver.comparison,self.ref,labels)
        self.assertIsNone(selected)
        self.assertEqual(evidence['rejection_reason'],'whole_form_contradicts_explicit_request')

    def test_neck_or_unknown_shoulder_cannot_pass_as_plain_cylinder(self):
        _,labels=self.rows()
        for neck,shoulder in [('present','absent'),('absent','unknown')]:
            self.client.complete.return_value=self.decision(neck=neck,shoulder=shoulder)
            self.assertIsNone(joint_review(self.resolver.comparison,self.ref,labels)[0])

    def test_no_unoffered_label_or_missing_competitor(self):
        _,labels=self.rows();answer=self.decision();answer.candidates.pop();answer.structures.pop()
        self.client.complete.return_value=answer
        with self.assertRaises(AmbiguousReferenceError):joint_review(self.resolver.comparison,self.ref,labels)

    def test_ambiguity_never_forces_winner(self):
        _,labels=self.rows();answer=self.decision();answer.status='ambiguous';answer.selected_candidate_id=None
        self.client.complete.return_value=answer
        self.assertIsNone(joint_review(self.resolver.comparison,self.ref,labels)[0])

    def test_no_budget_means_no_extra_request(self):
        _,labels=self.rows();self.resolver.calls=self.cfg.localization.max_vlm_calls
        with self.assertRaises(SkillFailure):joint_review(self.resolver.comparison,self.ref,labels)
        self.client.complete.assert_not_called()

    def test_geometry_words_do_not_override_everyday_categories(self):
        self.assertIsNone(requested_form('粉色杯子'));self.assertIsNone(requested_form('红色瓶子'))
        self.assertEqual(requested_form('白色球体'),'sphere')
        self.assertEqual(requested_form('blue cube'),'cuboid')

    def test_labels_do_not_define_the_valid_shape(self):
        _,labels=self.rows();labels={'A':labels['B'],'B':labels['A']}
        self.client.complete.return_value=self.decision('A')
        selected,_=joint_review(self.resolver.comparison,self.ref,labels)
        self.assertEqual(selected,'A')

    def test_explicit_color_conflict_cannot_be_ignored_by_joint_winner(self):
        _,labels=self.rows();answer=self.decision();answer.candidates[1].color_match='no'
        self.client.complete.return_value=answer
        self.assertIsNone(joint_review(self.resolver.comparison,self.ref,labels)[0])

    def test_joint_choice_cannot_switch_a_held_identity(self):
        self.rows();self.world.memory.held_id=self.obj.id
        self.client.complete.side_effect=[self.answer({'A':'yes'}),self.answer({'B':'yes'}),self.decision()]
        with self.assertRaises(AmbiguousReferenceError):self.resolver.comparison.choose(self.ref,self.obj.id,True)

    def test_joint_client_upload_preserves_1280_by_960(self):
        import base64,io,json
        from types import SimpleNamespace
        from PIL import Image
        from llm.llm_client import LLMClient
        client=LLMClient(self.cfg.llm);client.api_key='test-placeholder';self.cfg.llm.mode='auto'
        reply={'choices':[{'message':{'content':self.decision().model_dump_json()}}]}
        with patch('llm.llm_client.urlopen') as open_:
            open_.return_value.__enter__.return_value.read.return_value=json.dumps(reply).encode()
            client.complete(VisualComparisonDecision,'visual_comparison',{'candidate_ids':['A','B']},
                image=self.frame(960,1280).rgb)
        body=json.loads(open_.call_args.args[0].data)
        url=body['messages'][1]['content'][1]['image_url']['url']
        self.assertEqual(Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1]))).size,(1280,960))


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(JointTests(n) for n in JointTests.__dict__ if n.startswith('test_'))
