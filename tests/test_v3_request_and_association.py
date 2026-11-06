"""Identity geometry and provider failures; entirely offline, no model requests."""
import io,json,unittest
from copy import deepcopy
from unittest.mock import patch
from urllib.error import HTTPError
import numpy as np
from llm.llm_client import LLMClient,ModelRequestError
from llm.schemas import VisualIdentity
from config import AgentConfig
from world.object_memory import ObjectMemory
from world.world_model import WorldObject
from simulation.studio_presenter import failure_text
from test_v3_identity import IdentityTests
from recovery.failure_classifier import SkillFailure


class AssociationTests(unittest.TestCase):
    def setUp(self):
        self.memory=ObjectMemory(AgentConfig().perception)
        self.old=WorldObject('selected','purple body','purple','bottle',np.array([-.414,.011,.0739]),
            np.array([0,0,0,1]),np.array([.0398,.0342,.1479]),.95,1.)
        self.memory.objects[self.old.id]=self.old
        self.fresh=deepcopy(self.old);self.fresh.id='';self.fresh.color='pink'
        self.fresh.position=np.array([-.420,.020,.0658]);self.fresh.size=np.array([.0518,.0518,.1316])
        self.fresh.tracking_hint=self.old.id

    def test_recorded_geometry_retains_selected_id_across_colour_change(self):
        # Geometry from task_20261102T132448_39742c64 observations 32 and 47.
        self.memory.update([self.fresh],2.)
        self.assertEqual(self.fresh.id,'selected')

    def test_nearby_distinct_body_cannot_take_selected_id(self):
        self.fresh.position[0]+=.045
        self.memory.update([self.fresh],2.)
        self.assertNotEqual(self.fresh.id,'selected')

    def test_stacked_body_cannot_take_selected_id(self):
        self.fresh.position[2]+=.10
        self.memory.update([self.fresh],2.)
        self.assertNotEqual(self.fresh.id,'selected')

    def test_no_hint_does_not_authorize_cross_colour_reassociation(self):
        del self.fresh.tracking_hint
        self.memory.update([self.fresh],2.)
        self.assertNotEqual(self.fresh.id,'selected')

    def test_partial_views_do_not_steal_confirmed_stationary_body(self):
        # Replay the measured centres/volumes either side of the November 5
        # track switch. The second body-sized observation returns after a gap.
        old=self.old;old.position=np.array([.160,.160,.040]);old.size=np.array([.064,.064,.080])
        def detection(center,size):
            obj=deepcopy(old);obj.id='';obj.position=np.array(center);obj.size=np.array(size)
            return obj
        far=detection([.230,.156,.022],[.018,.016,.044])
        near=detection([.184,.142,.013],[.014,.018,.026])
        self.memory.update([far,near],2.,{old.id})
        self.assertFalse(old.visible)
        self.assertNotIn(old.id,{far.id,near.id})
        self.memory.update([],3.,{old.id})
        returned=detection([.160,.160,.040],[.064,.064,.080])
        neighbour=detection([.230,.156,.022],[.018,.016,.044])
        self.memory.update([neighbour,returned],4.,{old.id})
        self.assertEqual(returned.id,old.id)
        self.assertNotEqual(neighbour.id,old.id)


class RequestTests(unittest.TestCase):
    def test_http_error_exposes_safe_diagnostics_and_retryability(self):
        client=LLMClient(AgentConfig().llm);client.api_key='test-private-key'
        events=[];client.event_sink=lambda name,data:events.append((name,data))
        for status,retryable in [(400,False),(401,False),(429,True),(500,True)]:
            body={'error':{'code':'InvalidParameter','message':'bad image test-private-key data:image/jpeg;base64,'+'A'*120},'request':'must not be logged'}
            error=HTTPError('https://example.invalid',status,'rejected',{},io.BytesIO(json.dumps(body).encode()))
            with patch('llm.llm_client.urlopen',side_effect=error):
                with self.assertRaises(ModelRequestError) as raised:
                    client.complete(VisualIdentity,'visual_identity',{})
            self.assertEqual(raised.exception.retryable,retryable)
            self.assertEqual(raised.exception.http_status,status)
            log=json.dumps(events[-1]);self.assertNotIn('test-private-key',log)
            self.assertNotIn('A'*80,log);self.assertNotIn('must not be logged',log)
            self.assertEqual(events[-1][1]['provider_error']['code'],'InvalidParameter')

    def test_interface_request_error_has_distinct_ui_text(self):
        text=failure_text({'failure':'LOW_CONFIDENCE','evidence':{'reason_code':'MODEL_REQUEST_FAILED','http_status':400}})
        self.assertIn('400',text);self.assertNotIn('识别把握不足',text)


class IdentityRequestTests(IdentityTests):
    def test_nonretryable_request_stops_without_detail_capture_or_binding(self):
        self.client.complete.side_effect=ModelRequestError(400,{'code':'InvalidParameter'})
        with self.assertRaises(SkillFailure) as raised:self.resolver.identify(self.ref)
        self.assertTrue(raised.exception.evidence['terminal'])
        self.assertEqual(raised.exception.evidence['reason_code'],'MODEL_REQUEST_FAILED')
        self.client.complete.assert_called_once();self.camera.capture_detail.assert_not_called()
        self.assertFalse(self.world.reference_bindings)

    def test_temporary_request_failure_keeps_bounded_retry(self):
        self.client.complete.side_effect=[ModelRequestError(503,{}),self.reply()]
        with patch('perception.visual_identity.prompted_depth_mask',side_effect=lambda f,*_:np.ones(f.depth.shape,bool)):
            self.resolver.identify(self.ref)
        self.assertEqual(self.client.complete.call_count,2)


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(loader.loadTestsFromName(cls.__name__+'.'+name,__import__(__name__))
        for cls in (AssociationTests,RequestTests,IdentityRequestTests)
        for name in cls.__dict__ if name.startswith('test_'))
