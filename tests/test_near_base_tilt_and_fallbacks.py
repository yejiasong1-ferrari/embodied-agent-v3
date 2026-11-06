import unittest
from types import SimpleNamespace
import numpy as np
from config import AgentConfig
from world.world_model import WorldModel,WorldObject
from world.object_memory import ObjectMemory
from perception.geometry_analyzer import GeometryAnalyzer
from manipulation.grasp_planner import GraspPlanner,tilted_rotation
from spatial.geometry import footprint
from perception.candidate_comparison import CandidateComparison


def cylinder(x,y):
    theta=np.linspace(0,2*np.pi,100,endpoint=False);points=[]
    for z in np.linspace(.015,.15,35):points.extend(np.column_stack([x+.025*np.cos(theta),y+.025*np.sin(theta),np.full_like(theta,z)]))
    for radius in np.linspace(.002,.025,20):points.extend(np.column_stack([x+radius*np.cos(theta),y+radius*np.sin(theta),np.full_like(theta,.15)]))
    return np.asarray(points)


class TiltedGraspTests(unittest.TestCase):
    def setUp(self):self.config=AgentConfig();self.world=WorldModel(self.config.perception)

    def candidates(self,x,y):
        obj=WorldObject('target','blue object','blue','object',np.array([x,y,.075]),np.array([0,0,0,1]),np.array([.05,.05,.15]),.95,1,
                        points=cylinder(x,y),uncertainty_m=.004)
        self.world.objects[obj.id]=obj;GeometryAnalyzer(self.config).analyze(obj,self.world)
        return GraspPlanner(self.config).candidates(obj,self.world)

    def test_object_near_the_base_gets_tilted_alternatives_pointing_outward(self):
        found=[c for c in self.candidates(.16,.16) if ':t' in c.id]
        self.assertTrue(found)
        outward=np.array([.16,.16])/np.linalg.norm([.16,.16])
        for c in found:
            np.testing.assert_allclose(c.orientation.T@c.orientation,np.eye(3),atol=1e-7)
            self.assertLess(float(c.orientation[:2,2]@outward),-.3)  # the tool leans so the wrist moves away from the base
            self.assertLess(float(c.orientation[2,2]),-.7)           # while the fingers still come from above

    def test_distant_object_keeps_only_straight_grasps(self):
        self.assertFalse([c for c in self.candidates(.45,.1) if ':t' in c.id])

    def test_tilt_keeps_the_closing_axis_horizontal(self):
        rotation,approach=tilted_rotation(.7,np.deg2rad(30),np.array([.16,.16]))
        self.assertAlmostEqual(float(rotation[2,1]),0.,places=9)
        np.testing.assert_allclose(rotation[:,2],approach,atol=1e-9)


class FragmentTests(unittest.TestCase):
    def memory(self):
        memory=ObjectMemory(AgentConfig().perception);memory.held_id='cup'
        cup=WorldObject('cup','cyan cup','cyan','cup',np.array([.3,.3,.35]),np.array([0,0,0,1]),np.array([.064,.064,.08]),.9,1)
        sliver=WorldObject('sliver','cyan cup','cyan','cup',np.array([.34,.3,.37]),np.array([0,0,0,1]),np.array([.021,.014,.008]),.9,1)
        memory.objects.update({'cup':cup,'sliver':sliver});return memory

    def test_thin_handle_fragment_touching_the_held_body_is_adopted(self):
        memory=self.memory()
        self.assertEqual(memory.adopt_adjacent_fragments('cup'),{'sliver'})


class GeometryTests(unittest.TestCase):
    def test_partial_triangle_of_a_box_top_is_completed_to_its_measured_rectangle(self):
        corner=np.array([[.0,.0],[.07,.0],[.0,.11]]);xy=[]
        for a,b in ((corner[0],corner[1]),(corner[1],corner[2]),(corner[2],corner[0])):xy.extend(a+(b-a)*t for t in np.linspace(0,1,10))
        points=np.column_stack([np.asarray(xy)+[.5,-.2],np.full(len(xy),.16)])
        box=WorldObject('box','yellow box','yellow','box',np.array([.535,-.145,.08]),np.array([0,0,0,1]),np.array([.07,.11,.16]),.9,1,points=points)
        area=float(np.prod(np.ptp(footprint(box),axis=0)))
        self.assertGreater(area,.9*.07*.11)


class DecisionPolicyTests(unittest.TestCase):
    def test_comparison_must_decide_once_more_evidence_is_unavailable(self):
        resolver=SimpleNamespace(side_reviewed=set(),decide_now=set(),p=SimpleNamespace(world=SimpleNamespace(memory=SimpleNamespace(held_id=None))))
        comparison=CandidateComparison.__new__(CandidateComparison);comparison.resolver=resolver
        ref=SimpleNamespace(description='yellow box',color='yellow',category='box');key=('yellow box','yellow','box')
        self.assertFalse(comparison.must_decide(ref))
        resolver.decide_now.add(key);self.assertTrue(comparison.must_decide(ref))
        resolver.decide_now.clear();resolver.side_reviewed.add(key);self.assertTrue(comparison.must_decide(ref))
        resolver.side_reviewed.clear();resolver.p.world.memory.held_id='x';self.assertTrue(comparison.must_decide(ref))

    def test_plate_beats_a_block_when_the_model_flips_between_views(self):
        plate=WorldObject('plate','yellow plate','yellow','plate',np.array([.4,0,.01]),np.array([0,0,0,1]),np.array([.24,.24,.02]),.9,1)
        box=WorldObject('box','yellow box','yellow','box',np.array([.5,0,.08]),np.array([0,0,0,1]),np.array([.07,.11,.16]),.9,1)
        world=WorldModel(AgentConfig().perception);world.objects.update(plate=plate,box=box)
        comparison=CandidateComparison.__new__(CandidateComparison)
        comparison.resolver=SimpleNamespace(p=SimpleNamespace(world=world,log=SimpleNamespace(event=lambda *a,**k:None)))
        ref=SimpleNamespace(description='黄色盘子',color='yellow',category='plate')
        labels={'A':{'id':'plate','colors':{'yellow':.9}},'B':{'id':'box','colors':{'yellow':.9}}}
        reviews={'A':SimpleNamespace(verdict='uncertain',confidence=.5,shape_match='yes',color_match='yes'),
                 'B':SimpleNamespace(verdict='yes',confidence=.95,shape_match='yes',color_match='yes')}
        self.assertEqual(comparison.best_guess(ref,labels,reviews,{}), 'A')

    def recovery_context(self,**extra):
        from unittest.mock import Mock
        from robot.robot_interface import ContactState
        context=SimpleNamespace(candidate=SimpleNamespace(id='top_grasp:4',grasp_type='top_grasp',
            contact_region='top',approach_direction='top'),current_skill='MOVE_TO_TARGET',
            source=SimpleNamespace(position=np.array([.16,.16,.08])),robot=Mock(),held=True,
            config=AgentConfig(),goal=SimpleNamespace(),world=SimpleNamespace(objects={}),
            perception=SimpleNamespace(last_frame=None),safety=SimpleNamespace(last_failure=None),
            target_distance=lambda:0.)
        context.robot.get_contact_state.return_value=ContactState()
        for key,value in extra.items():setattr(context,key,value)
        return context

    def test_path_and_placement_failures_try_another_grasp_instead_of_stopping(self):
        from recovery.failure_classifier import SkillFailure
        from recovery.recovery_manager import RecoveryManager
        from unittest.mock import Mock
        from robot.robot_interface import ContactState
        replanner=Mock();replanner.client.online=False
        manager=RecoveryManager(AgentConfig(),replanner,Mock())
        context=self.recovery_context(candidate=SimpleNamespace(id='top_grasp:0',grasp_type='top_grasp',
            contact_region='top',approach_direction='top'))
        self.assertEqual(manager.decide(SkillFailure('NO_VALID_PLACEMENT'),context),'CHANGE_GRASP')
        manager2=RecoveryManager(AgentConfig(),replanner,Mock())
        context.held=False
        self.assertEqual(manager2.decide(SkillFailure('AMBIGUOUS_TARGET',{'reference':'黄色盘子'}),context),'REOBSERVE')
        manager3=RecoveryManager(AgentConfig(),replanner,Mock())
        context.current_skill='FIND_OBJECT';context.held=False;context.candidate=None
        self.assertEqual(manager3.decide(SkillFailure('IK_FAILED'),context),'REOBSERVE')

    def test_blocked_carry_replans_a_route_instead_of_releasing(self):
        from recovery.failure_classifier import SkillFailure
        from recovery.recovery_manager import RecoveryManager
        from unittest.mock import Mock
        from robot.robot_interface import ContactState
        replanner=Mock();replanner.client.online=False
        manager=RecoveryManager(AgentConfig(),replanner,Mock())
        context=self.recovery_context(source=SimpleNamespace(position=np.array([-.03,-.38,.36])))
        self.assertEqual(manager.decide(SkillFailure('PATH_BLOCKED'),context),'REPLAN')
        replanner.client.online=True
        replanner.reason.return_value=SimpleNamespace(action='CHANGE_APPROACH')
        self.assertEqual(manager.decide(SkillFailure('PATH_BLOCKED'),context),'REPLAN')

    def test_llm_cannot_force_a_regrasp_on_the_first_blocked_carry(self):
        from recovery.failure_classifier import SkillFailure
        from recovery.recovery_manager import RecoveryManager
        from unittest.mock import Mock
        from robot.robot_interface import ContactState
        replanner=Mock();replanner.client.online=True
        replanner.reason.return_value=SimpleNamespace(action='CHANGE_GRASP')
        manager=RecoveryManager(AgentConfig(),replanner,Mock())
        context=self.recovery_context(current_skill='GRASP_OBJECT',held=False)
        manager.decide(SkillFailure('GRASP_FAILED'),context)
        context.held=True;context.current_skill='MOVE_TO_TARGET'
        self.assertEqual(manager.decide(SkillFailure('PATH_BLOCKED'),context),'REPLAN')

    def test_park_keeps_the_object_when_every_landing_is_blocked(self):
        from skills.manipulation import park_held_object
        from recovery.failure_classifier import SkillFailure
        from unittest.mock import Mock
        from robot.robot_interface import Pose
        context=SimpleNamespace(held=True,payload={'offset':np.array([0.,0.,.15]),'size':np.array([.06,.06,.08])},
            source=SimpleNamespace(id='blue',position=np.array([.3,.2,.04]),size=np.array([.06,.06,.08])),
            prelift_position=np.array([.3,.2,.04]),robot=Mock(),
            world=SimpleNamespace(memory=SimpleNamespace(held_id='blue',last_released_id=None)),log=Mock())
        context.robot.get_ee_pose.return_value=Pose(np.array([.05,-.38,.50]),np.eye(3))
        context.move=Mock(side_effect=SkillFailure('PATH_BLOCKED'))
        context.refresh=Mock()
        self.assertFalse(park_held_object(context,'blocked'))
        self.assertTrue(context.held)
        context.robot.open_gripper.assert_not_called()


if __name__=='__main__':unittest.main()
