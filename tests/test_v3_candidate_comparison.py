import unittest
from copy import deepcopy
from unittest.mock import Mock,patch
import numpy as np
from test_v3_identity import IdentityTests
from llm.schemas import VisualDisambiguation
from perception.visual_identity import reference_key
from world.world_model import AmbiguousReferenceError


class ComparisonTests(IdentityTests):
    def pair(self):
        other=deepcopy(self.obj);other.id='alternative';other.position+=np.array([.3,0,0]);other.query=''
        self.world.objects[other.id]=other
        for i,obj in enumerate([self.obj,other]):
            self.resolver.comparison.gallery[obj.id]={'id':obj.id,'position':obj.position.copy(),
                'patch':self.frame(40,40).rgb,'scene':self.frame(480,640).rgb,'view':f'camera:{i}',
                'timestamp':1.,'colors':{'red':.8},'area':1600}
        return other

    def answer(self,verdicts):
        return VisualDisambiguation(candidates=[{'candidate_id':k,'verdict':v,'confidence':.95,'evidence':'visible distinguishing body shape'} for k,v in verdicts.items()])

    def test_compares_full_instruction_with_anonymous_images(self):
        other=self.pair();self.client.complete.return_value=self.answer({'A':'no','B':'yes'})
        chosen,evidence=self.resolver.comparison.choose(self.ref,None,True)
        self.assertEqual(chosen['id'],other.id)
        call=self.client.complete.call_args
        self.assertEqual(call.args[2]['user_command'],self.p.task_description)
        self.assertNotIn('bottle',str(call.args[2]));self.assertEqual(call.kwargs['image'].shape,(960,1280,3))
        self.assertTrue((self.p.log.path/evidence['review_image']).exists())

    def test_all_no_or_uncertain_never_selects(self):
        self.pair()
        for verdicts in [{'A':'no','B':'no'},{'A':'uncertain','B':'yes'},{'A':'yes','B':'yes'},{'A':'yes'}]:
            self.client.complete.return_value=self.answer(verdicts)
            with self.assertRaises(AmbiguousReferenceError):self.resolver.comparison.choose(self.ref,None,True)

    def test_correction_requires_old_candidate_rejection(self):
        self.confirm();other=self.pair();self.client.complete.return_value=self.answer({'A':'no','B':'yes'})
        chosen,_=self.resolver.comparison.choose(self.ref,self.obj.id,True)
        self.assertEqual(chosen['id'],other.id)
        self.assertEqual(self.world.reference_bindings[reference_key(self.ref)],self.obj.id)

    def test_no_identity_switch_after_manipulation_started_or_while_held(self):
        other=self.pair();self.client.complete.return_value=self.answer({'A':'no','B':'yes'})
        for active,held in [(True,None),(False,self.obj.id)]:
            self.world.memory.mission_active=active;self.world.memory.held_id=held
            with self.assertRaises(AmbiguousReferenceError):self.resolver.comparison.choose(self.ref,self.obj.id,True)

    def test_compared_correction_commits_only_after_fresh_confirmation(self):
        self.confirm();other=self.pair();self.resolver.anchors[reference_key(self.ref)]['review_required']=True
        self.p.observe.return_value=[other];other.query=self.ref.description
        self.client.complete.side_effect=[self.answer({'A':'no','B':'yes'}),self.reply()]
        with patch('perception.visual_identity.prompted_depth_mask',side_effect=lambda f,*_:np.ones(f.depth.shape,bool)):
            self.resolver.identify(self.ref)
        self.assertEqual(self.world.reference_bindings[reference_key(self.ref)],other.id)
        self.assertTrue(any(c.args[0]=='identity_corrected' for c in self.p.log.event.call_args_list))

    def test_no_correction_when_old_candidate_is_missing_from_comparison(self):
        self.pair();self.client.complete.return_value=self.answer({'A':'no','B':'yes'})
        with self.assertRaises(AmbiguousReferenceError):self.resolver.comparison.choose(self.ref,'missing-old',True)

    def test_comparison_consumes_shared_identity_budget(self):
        from recovery.failure_classifier import SkillFailure
        self.pair();self.resolver.calls=self.cfg.localization.max_vlm_calls
        with self.assertRaises(SkillFailure) as exc:self.resolver.comparison.choose(self.ref,None,True)
        self.assertTrue(exc.exception.evidence['terminal']);self.client.complete.assert_not_called()

    def test_single_candidate_still_requires_visual_review(self):
        other=self.pair();del self.resolver.comparison.gallery[other.id]
        self.client.complete.return_value=self.answer({'A':'no'})
        with self.assertRaises(AmbiguousReferenceError):self.resolver.comparison.choose(self.ref,None,True)
        self.client.complete.assert_called_once()
        self.assertFalse(self.world.reference_bindings)

    def test_fresh_full_image_cannot_override_single_compared_candidate(self):
        other=self.pair();del self.resolver.comparison.gallery[other.id]
        other.query=self.ref.description;self.p.observe.return_value=[other]
        self.client.complete.side_effect=[self.answer({'A':'yes'}),self.reply(),self.reply()]
        with patch('perception.visual_identity.prompted_depth_mask',side_effect=lambda f,*_:np.ones(f.depth.shape,bool)):
            with self.assertRaises(AmbiguousReferenceError):self.resolver.identify(self.ref)
        self.assertNotIn(reference_key(self.ref),self.world.reference_bindings)

    def test_six_candidates_screen_all_before_final_choice(self):
        self.pair();gallery=self.resolver.comparison.gallery
        template=next(iter(gallery.values()))
        for i in range(4):
            row=dict(template);row['id']=f'extra{i}';row['position']=np.array([2+i,0,0.])
            gallery[row['id']]=row;self.world.objects[row['id']]=deepcopy(self.obj)
        self.client.complete.side_effect=[self.answer({'A':'no','B':'yes','C':'no','D':'no'}),
            self.answer({'A':'no','B':'no'}),self.answer({'A':'yes'})]
        chosen,evidence=self.resolver.comparison.choose(self.ref,None,True)
        self.assertEqual(chosen['id'],'alternative');self.assertEqual(self.client.complete.call_count,3)
        self.assertEqual(len(evidence['screening_batches']),2)

    def test_uncertain_screening_candidate_is_not_discarded(self):
        self.pair();gallery=self.resolver.comparison.gallery;template=next(iter(gallery.values()))
        for i in range(3):
            row=dict(template);row['id']=f'extra{i}';row['position']=np.array([2+i,0,0.])
            gallery[row['id']]=row;self.world.objects[row['id']]=deepcopy(self.obj)
        self.client.complete.side_effect=[self.answer({'A':'yes','B':'no','C':'no','D':'no'}),
            self.answer({'A':'uncertain'}),self.answer({'A':'yes','B':'uncertain'})]
        with self.assertRaises(AmbiguousReferenceError):self.resolver.comparison.choose(self.ref,None,True)
        self.assertEqual(len(self.client.complete.call_args.args[2]['candidate_ids']),2)

    def test_background_colour_is_excluded_from_candidate_recall(self):
        from perception.camera import CameraFrame
        from llm.schemas import Reference
        rgb=np.full((64,64,3),[220,0,0],np.uint8);rgb[27:37,27:37]=[0,220,0]
        depth=np.ones((64,64));depth[27:37,27:37]=.94
        pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=1.
        frame=CameraFrame(rgb,depth,np.array([[100,0,32],[0,100,32],[0,0,1]]),pose,1.,'camera:0')
        obj=deepcopy(self.obj);obj.position=np.array([0.,0.,.03]);obj.size=np.array([.12,.12,.06]);obj.bbox=(27,27,37,37)
        self.world.objects[obj.id]=obj;self.resolver.comparison.remember([obj],frame)
        self.assertEqual(self.resolver.comparison.candidates(Reference(description='红色物体'),None),[])
        self.assertEqual(len(self.resolver.comparison.candidates(Reference(description='绿色物体'),None)),1)

    def test_offline_preparation_does_not_request_model(self):
        self.pair();prepared=self.resolver.comparison.prepare(self.ref,None)
        self.assertEqual(set(prepared[0]),{'A','B'});self.client.complete.assert_not_called()
        self.assertEqual(self.resolver.calls,0)

    def test_failed_fresh_confirmation_keeps_original_binding(self):
        self.confirm();self.pair();self.resolver.anchors[reference_key(self.ref)]['review_required']=True
        self.client.complete.side_effect=[self.answer({'A':'no','B':'yes'}),self.reply('ambiguous'),self.reply('ambiguous')]
        with self.assertRaises(AmbiguousReferenceError):self.resolver.identify(self.ref)
        self.assertEqual(self.world.reference_bindings[reference_key(self.ref)],self.obj.id)
        self.assertFalse(self.resolver.valid(self.ref))


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(ComparisonTests(n) for n in ComparisonTests.__dict__ if n.startswith('test_'))
