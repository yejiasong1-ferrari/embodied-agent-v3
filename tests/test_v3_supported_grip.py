"""Regression for the transition from supported closure to a loaded trajectory."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import TestCase
from unittest.mock import Mock, patch
import mujoco
import numpy as np

from config import AgentConfig
from manipulation.trajectory_executor import retime
from robot.mujoco_robot import MuJoCoRobot
from robot.robot_interface import ContactState, Pose
from simulation.environment import Simulation
from skills.skill_base import SkillContext
from validation.safety_monitor import SafetyMonitor


class SupportedGripTests(TestCase):
    def test_measured_probe_does_not_unload_pad_during_transition(self):
        cfg=AgentConfig.load(Path(__file__).resolve().parents[1]/'config/v3.json')
        # Measured arm posture from the first (failed) lift. Only the unused
        # renderer is stubbed; dynamics, control, contact and safety are real.
        with patch.object(mujoco,'Renderer'), Simulation(cfg) as sim:
            robot=MuJoCoRobot(sim);safety=SafetyMonitor(robot,cfg)
            self.assertLess(robot._planning.payload_tilt_limit_deg,12.)
            q=[2.354903,.113135,.198586,-2.705888,-.069670,2.816081,.262074]
            sim.data.qpos[robot._addresses]=q;sim.data.ctrl[:7]=q;sim.data.ctrl[-1]=255
            for name in ('finger_joint1','finger_joint2'):
                sim.data.qpos[int(sim.model.joint(name).qposadr[0])]=.04
            mujoco.mj_forward(sim.model,sim.data)
            self.assertTrue(robot.hold(.35))
            safety.allow_finger_contact=True
            self.assertTrue(robot.close_gripper(.05804, safety.check))
            self.assertTrue(robot.hold(.24,safety.check))
            self.assertTrue(robot.get_contact_state().bilateral)
            robot.confirm_grasp()
            ee=robot.get_ee_pose();safety.carry_reference_rotation=ee.rotation
            start=robot.get_joint_positions()
            goal=min(robot.solve_ik(Pose(ee.position+[0,0,.007],ee.rotation),start),
                     key=lambda x:np.linalg.norm(x-start))
            path,ts=retime([start,goal],cfg.safety.carrying_velocity)
            self.assertTrue(robot.execute_trajectory(path,ts,lambda:safety.check(carrying=True)))
            self.assertTrue(robot.get_contact_state().bilateral)
            delta=robot.get_ee_pose().position-ee.position
            self.assertGreater(delta[2],.004)
            self.assertLess(np.linalg.norm(delta[:2]),.002)

    def test_one_sided_adjustment_stops_on_bilateral_contact_or_excess_force(self):
        cfg=AgentConfig();r=Mock(config=cfg)
        r.get_gripper_state.return_value={'command':150.};r._control.grasp_to.return_value=True
        r.get_contact_state.side_effect=[ContactState(1,0), ContactState(1,0),ContactState(1,1)]
        self.assertTrue(MuJoCoRobot.snug_gripper(r,None))
        commands=[call.args[0] for call in r._control.grasp_to.call_args_list]
        self.assertTrue(150>commands[0]>commands[1]>=150-.002/.08*255)
        r._control.grasp_to.reset_mock()
        r.get_contact_state.side_effect=[ContactState(5,0)]
        self.assertTrue(MuJoCoRobot.snug_gripper(r,None))
        r._control.grasp_to.assert_not_called()

    def test_only_high_empty_transits_get_extra_speed(self):
        cfg=AgentConfig(version=3)
        ctx=NS(config=cfg,world=NS(objects={'obstacle':NS(high=[0,0,.20])}),
               robot=Mock(),log=Mock(),held=False,cartesian_speed_limit=None)
        limits=[]
        ctx.move=lambda *args,**kw:limits.append(ctx.cartesian_speed_limit)
        ctx.robot.get_ee_pose.return_value=NS(position=[0,0,.50])
        for held,height in ((False,.5),(False,.27),(True,.5)):
            ctx.held=held;SkillContext.observe_move(ctx,[0,0,height])
        self.assertEqual(limits,[.08,.05,.05])
        self.assertIsNone(ctx.cartesian_speed_limit)

    def test_oblique_closure_stops_before_chasing_unilateral_pressure(self):
        cfg=AgentConfig(version=3);r=Mock(config=cfg)
        r._monitor.return_value=None
        r.get_gripper_state.return_value={'command':220.}
        r._control.grasp_to.return_value=True
        r.get_contact_state.side_effect=[ContactState(),ContactState(1,0),ContactState(4.1,0)]
        self.assertTrue(MuJoCoRobot.close_obliquely(r,.064))
        self.assertEqual(r._control.grasp_to.call_count,2)
        commands=[c.args[0] for c in r._control.grasp_to.call_args_list]
        self.assertTrue(220>commands[0]>commands[1])
        r._control.grasp_to.reset_mock();r._monitor.return_value='USER_STOP'
        self.assertFalse(MuJoCoRobot.close_obliquely(r,.064))
        r._control.grasp_to.assert_not_called()
