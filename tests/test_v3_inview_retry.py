"""A selected candidate that projects into the frame is in view: absence at low resolution earns a detail look."""
import unittest
from unittest.mock import patch
import numpy as np
from test_v3_identity import IdentityTests


class InViewRetryTests(IdentityTests):
    def selected(self):
        self.obj.position=np.array([0.,0.,.5])
        evidence={'reviews':[],'label_to_object_id':{'A':self.obj.id}}
        return patch.object(self.resolver.comparison,'choose',return_value=({'id':self.obj.id,'view':'gripper_camera:0'},evidence))

    def test_not_visible_on_a_selected_candidate_retries_in_detail(self):
        self.client.complete.side_effect=[self.reply('not_visible'),self.reply()]
        with self.selected(),patch('perception.visual_identity.prompted_depth_mask',return_value=np.ones((960,1280),bool)):
            self.assertEqual(self.resolver.identify(self.ref).id,'measured')
        self.assertEqual([c.kwargs['image'].shape for c in self.client.complete.call_args_list],[(480,640,3),(960,1280,3)])

    def test_two_absences_still_stop_at_the_detail_frame(self):
        self.client.complete.return_value=self.reply('not_visible')
        with self.selected():
            with self.assertRaises(Exception):self.resolver.identify(self.ref)
        self.assertEqual(self.client.complete.call_count,2)
        self.assertEqual(self.world.reference_bindings,{})


del IdentityTests
if __name__=='__main__':unittest.main()
