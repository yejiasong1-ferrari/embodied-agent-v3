"""V3 authorization tests use synthetic images/model replies, not demo evidence."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from config import AgentConfig
from llm.schemas import VisualIdentity,Reference,Goal
from perception.visual_identity import VisualIdentityResolver,reference_key
from perception.camera import CameraFrame
from world.world_model import WorldModel,WorldObject,AmbiguousReferenceError,ReferenceNotFoundError
from recovery.failure_classifier import SkillFailure
from skills.skill_registry import SkillRegistry
from perception.active_perception import ActivePerception


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.cfg=AgentConfig(version=3,strict_visual_identity=True)
        self.cfg.localization.max_search_views=1
        self.world=WorldModel(self.cfg.perception)
        self.ref=Reference(description='红色杯子')
        self.obj=WorldObject('measured','red bottle','red','bottle',np.array([.3,.1,.05]),np.array([0,0,0,1]),np.array([.06,.06,.1]),.95,1)
        self.obj.query=self.ref.description
        self.obj.bbox=(100,100,200,200);self.world.objects[self.obj.id]=self.obj
        self.frame=lambda h,w:CameraFrame(np.broadcast_to(np.array([200,0,0],np.uint8),(h,w,3)).copy(),np.ones((h,w)),np.array([[500,0,w/2],[0,500,h/2],[0,0,1]]),np.eye(4),1.,'camera:0')
        self.camera=Mock(viewpoint_count=1)
        self.camera.capture.side_effect=lambda:self.frame(480,640)
        self.camera.capture_detail.side_effect=lambda:self.frame(960,1280)
        self.client=Mock(online=True)
        path=Path(self.tmp.name);(path/'images').mkdir()
        self.p=SimpleNamespace(world=self.world,config=self.cfg,client=self.client,camera=self.camera,task_description='把红色杯子放到黄色盘子上',log=SimpleNamespace(path=path,event=Mock()),observe=Mock(return_value=[self.obj]))
        self.resolver=VisualIdentityResolver(self.p)

    def reply(self,status='confirmed',confidence=.95):
        return VisualIdentity(status=status,bbox=(.15,.2,.4,.6) if status=='confirmed' else None,label='red vessel',color='red',category='bottle',confidence=confidence,evidence='visible cylindrical drinking vessel; no alternative fits')

    def confirm(self):
        self.client.complete.return_value=self.reply()
        with patch('perception.visual_identity.prompted_depth_mask',side_effect=lambda frame,*args:np.ones(frame.depth.shape,bool)):
            return self.resolver.identify(self.ref)

    def test_unique_local_candidate_cannot_authorize(self):
        self.client.online=False
        with self.assertRaises(SkillFailure):self.resolver.identify(self.ref)
        self.assertEqual(self.world.reference_bindings,{})

    def test_schema_requires_valid_confirmed_box(self):
        with self.assertRaises(ValueError):VisualIdentity(status='confirmed',confidence=.99,evidence='yes')
        with self.assertRaises(ValueError):VisualIdentity(status='confirmed',bbox=(0,0,100,100),confidence=.99,evidence='yes')

    def test_full_command_and_pixels_are_sent_and_evidence_persisted(self):
        self.assertEqual(self.confirm().id,'measured')
        call=self.client.complete.call_args
        self.assertEqual(call.args[2]['user_command'],self.p.task_description)
        self.assertEqual(call.kwargs['image'].shape,(480,640,3))
        evidence=self.world.reference_binding_evidence[reference_key(self.ref)]
        self.assertEqual(evidence['source'],'vlm_identity')
        self.assertTrue((self.p.log.path/evidence['image']).exists())
        self.assertEqual(len(evidence['image_sha256']),64)
        self.assertTrue(self.resolver.valid(self.ref))
        # Model may confirm ordinary naming without changing geometric category.
        self.assertEqual(self.obj.category,'bottle')

    def test_detail_needed_upgrades_actual_resolution(self):
        self.client.complete.side_effect=[self.reply('detail_needed'),self.reply()]
        with patch('perception.visual_identity.prompted_depth_mask',return_value=np.ones((960,1280),bool)):
            self.resolver.identify(self.ref)
        self.assertEqual([c.kwargs['image'].shape for c in self.client.complete.call_args_list],[(480,640,3),(960,1280,3)])

    def test_absence_does_not_spend_detail_call(self):
        self.client.complete.return_value=self.reply('not_visible')
        with self.assertRaises(ReferenceNotFoundError):self.resolver.identify(self.ref)
        self.camera.capture_detail.assert_not_called()
        self.assertEqual(self.world.reference_bindings,{})

    def test_ambiguous_never_autobinds_single_local_object(self):
        self.client.complete.return_value=self.reply('ambiguous')
        with self.assertRaises(AmbiguousReferenceError):self.resolver.identify(self.ref)
        self.assertEqual(self.world.reference_bindings,{})

    def test_model_failure_has_bounded_retry_no_fallback(self):
        self.client.complete.side_effect=RuntimeError('network')
        with self.assertRaises(SkillFailure):self.resolver.identify(self.ref)
        self.assertEqual(self.client.complete.call_count,2)
        self.assertEqual(self.world.reference_bindings,{})

    def test_budget_cannot_authorize_local_object(self):
        self.cfg.localization.max_vlm_calls=0
        with self.assertRaises(SkillFailure):self.resolver.identify(self.ref)
        self.client.complete.assert_not_called()

    def test_invalid_depth_does_not_authorize(self):
        self.client.complete.return_value=self.reply()
        with patch('perception.visual_identity.prompted_depth_mask',return_value=None):
            with self.assertRaises(AmbiguousReferenceError):self.resolver.identify(self.ref)
        self.assertEqual(self.world.reference_bindings,{})

    def test_geometry_must_resolve_one_instance(self):
        self.p.observe.return_value=[self.obj,self.obj]
        with self.assertRaises(AmbiguousReferenceError):self.confirm()
        self.assertEqual(self.world.reference_bindings,{})

    def test_occlusion_preserves_evidence_but_requires_observation(self):
        self.confirm();self.resolver.check_continuity([],self.frame(480,640))
        self.assertFalse(self.resolver.valid(self.ref))
        self.assertEqual(self.world.reference_bindings[reference_key(self.ref)],self.obj.id)
        self.assertTrue(self.resolver.valid(self.ref,fresh=False))

    def test_same_colour_nearby_instance_revokes(self):
        self.confirm()
        rival=WorldObject('other','red cup','red','cup',self.obj.position+.02,np.array([0,0,0,1]),self.obj.size,.95,1)
        self.resolver.check_continuity([self.obj,rival],self.frame(480,640))
        self.assertFalse(self.resolver.valid(self.ref))

    def test_all_references_required_before_motion(self):
        self.confirm()
        goal=Goal(goal='place_right',object=self.ref,target=Reference(description='绿色积木'))
        with self.assertRaises(SkillFailure):self.resolver.require_all(goal)

    def test_skill_dispatch_cannot_bypass_identity(self):
        registry=SkillRegistry();registry.skills['GRASP_OBJECT']=Mock()
        goal=Goal(goal='place_right',object=self.ref,target=Reference(description='绿色积木'))
        context=SimpleNamespace(config=self.cfg,goal=goal,held=False,perception=SimpleNamespace(identity_resolver=self.resolver))
        with self.assertRaises(SkillFailure):registry.execute('GRASP_OBJECT',context)
        registry.skills['GRASP_OBJECT'].execute.assert_not_called()

    def test_model_confirmed_wrong_colour_is_vetoed(self):
        self.client.complete.return_value=self.reply()
        self.camera.capture.side_effect=lambda:self.frame(480,640).__class__(np.broadcast_to(np.array([0,0,200],np.uint8),(480,640,3)).copy(),np.ones((480,640)),np.eye(3),np.eye(4),1.,'camera:0')
        self.camera.capture_detail.side_effect=lambda:self.frame(960,1280).__class__(np.broadcast_to(np.array([0,0,200],np.uint8),(960,1280,3)).copy(),np.ones((960,1280)),np.eye(3),np.eye(4),1.,'camera:0')
        with patch('perception.visual_identity.prompted_depth_mask',side_effect=lambda frame,*args:np.ones(frame.depth.shape,bool)):
            with self.assertRaises(AmbiguousReferenceError):self.resolver.identify(self.ref)
        self.assertEqual(self.world.reference_bindings,{})

    def test_two_references_cannot_bind_one_instance(self):
        self.confirm()
        second=Reference(description='红色瓶子')
        self.client.complete.return_value=self.reply()
        with patch('perception.visual_identity.prompted_depth_mask',side_effect=lambda frame,*args:np.ones(frame.depth.shape,bool)):
            with self.assertRaises(AmbiguousReferenceError):self.resolver.identify(second)
        self.assertNotIn(reference_key(second),self.world.reference_bindings)

    def test_offline_task_stops_before_robot_resume(self):
        from agent.embodied_agent import EmbodiedAgent
        self.cfg.llm.mode='offline';self.cfg.log_root=self.tmp.name
        robot=Mock();agent=EmbodiedAgent(robot,self.camera,self.cfg)
        result=agent.execute('把红色杯子放到绿色积木右边')
        self.assertFalse(result.success)
        self.assertEqual(result.metrics['grasp_attempts'],0)
        robot.resume.assert_not_called()

    def test_real_pixel_pipeline_preserves_both_reference_observations(self):
        from agent.task_log import TaskLog
        rgb=np.full((480,640,3),25,np.uint8);depth=np.ones((480,640))
        rgb[200:250,200:250]=[220,0,0];depth[200:250,200:250]=.9
        rgb[200:250,400:450]=[0,220,0];depth[200:250,400:450]=.9
        pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=1.
        frame=CameraFrame(rgb,depth,np.array([[500,0,320],[0,500,240],[0,0,1]]),pose,1.,'camera:0')
        log=TaskLog(self.tmp.name,'two actual pixel objects',self.cfg)
        world=WorldModel(self.cfg.perception)
        p=ActivePerception(self.camera,world,self.cfg,self.client,log)
        p.observe(frame=frame)
        self.assertEqual(len(world.objects),2)
        from perception.object_detector import Detection
        mask=np.zeros((480,640),bool);mask[200:250,200:250]=True
        detection=Detection((200,200,250,250),'box','red',.95,mask);detection.query='红色杯子'
        objects=p.observe(frame=frame,identity_detection=detection)
        self.assertEqual(len(objects),2)
        self.assertTrue(all(o.visible for o in world.objects.values()))
        self.assertEqual(len([o for o in objects if o.query=='红色杯子']),1)

    def test_review_cannot_silently_switch_to_a_different_instance(self):
        self.confirm()
        self.resolver.anchors[reference_key(self.ref)]['review_required']=True
        from copy import deepcopy
        other=deepcopy(self.obj);other.id='other';self.world.objects[other.id]=other
        self.p.observe.return_value=[other]
        with patch('perception.visual_identity.prompted_depth_mask',side_effect=lambda frame,*args:np.ones(frame.depth.shape,bool)):
            with self.assertRaises(AmbiguousReferenceError):self.resolver.identify(self.ref)
        self.assertEqual(self.world.reference_bindings[reference_key(self.ref)],self.obj.id)

    def test_default_entry_config_is_v3(self):
        cfg=AgentConfig.load();self.assertTrue(cfg.strict_visual_identity);self.assertEqual(cfg.version,3)

    def test_active_perception_uses_resolver_before_coarse_match(self):
        p=ActivePerception.__new__(ActivePerception);p.identity_resolver=Mock()
        p.identity_resolver.identify.return_value=self.obj
        self.assertIs(p.identify(self.ref),self.obj)
        p.identity_resolver.identify.assert_called_once_with(self.ref)

if __name__=='__main__':unittest.main()
