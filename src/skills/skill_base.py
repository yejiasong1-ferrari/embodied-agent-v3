from dataclasses import dataclass,field
import time
import numpy as np
from robot.robot_interface import Pose
from recovery.failure_classifier import SkillFailure,FailureType,FailureClassifier


class Skill:
    preconditions=()
    success_conditions=()
    failure_conditions=()
    def check_preconditions(self,context):
        for field in self.preconditions:
            if not getattr(context,field,None): raise SkillFailure(FailureType.TARGET_NOT_FOUND,{"missing":field})
    def execute(self,context): raise NotImplementedError
    def monitor(self,context):
        reason=context.safety.check(carrying=context.held)
        if reason: raise SkillFailure(FailureClassifier.classify(reason),{"reason":reason})
    def validate(self,context,result): return result is not False


def drop_source_slivers(c,include_visible=False):
    """Same-colour slivers beside the body (rim ring, handle) are parts of it, not obstacles for the gripper."""
    src=c.source
    for stale in [o for o in c.world.memory.objects.values() if o.id!=src.id and o.movable and (include_visible or not o.visible)
                  and o.color==src.color and float(np.prod(o.size))<.35*float(np.prod(src.size))
                  and np.linalg.norm(o.position[:2]-src.position[:2])<.08]:
        c.world.memory.objects.pop(stale.id,None)


@dataclass
class SkillContext:
    robot: object
    perception: object
    world: object
    goal: object
    config: object
    planner: object
    executor: object
    grasp_planner: object
    safety: object
    log: object
    recovery: object
    source: object=None
    target: object=None
    candidate: object=None
    held: bool=False
    payload: dict|None=None
    current_skill: str=""
    source_id: str|None=None
    target_id: str|None=None
    prelift_position: object=None
    placement_position: object=None
    grasp_verified: bool=False
    lift_verified: bool=False
    final_validated: bool=False
    any_grasp_verified: bool=False
    any_lift_verified: bool=False
    grasp_attempts: int=0
    first_grasp_verified: bool=False
    first_lift_success: bool=False
    placement_plan: object=None
    placement_support_id: str|None=None
    push_attempts: int=0
    regrasp_attempts: int=0
    cartesian_speed_limit: float|None=None
    observation_motion: bool=False
    correction_history: list=field(default_factory=list)
    blocked_placements: list=field(default_factory=list)
    logged_held_fragments: set=field(default_factory=set)
    steps: int=0
    started: float=field(default_factory=time.monotonic)

    def budget(self):
        if self.steps>self.config.recovery.max_steps:raise TimeoutError("skill step budget exhausted")
        if self.safety.cancelled(): raise SkillFailure("USER_STOP")
        if time.monotonic()-self.started>self.config.recovery.timeout_s: raise TimeoutError("task wall-clock budget exhausted")

    def target_distance(self):
        return None if self.source is None else float(np.linalg.norm(self.source.position-self.robot.get_ee_pose().position))

    def attachment_visible(self,predicted):
        """Test measured surface points against the attached shape hypothesis.

        A cropped face has a biased centroid. It can still provide evidence
        for attachment, but contact alone can never satisfy this check.
        """
        if len(self.source.points)<12 or not self.robot.get_contact_state().bilateral: return False
        pose=self.robot.get_ee_pose()
        if 'surface_local' in self.payload:
            from perception.held_registration import register_translation
            reference=predicted+np.asarray(self.payload['surface_local'])@pose.rotation.T
            shift,evidence=register_translation(self.source.points,reference)
            self.log.event('visual_attachment',evidence)
            return shift is not None and np.linalg.norm(shift)<=self.config.safety.slip_distance_m
        rotation=pose.rotation@np.asarray(self.payload["relative_rotation"])
        local=(self.source.points-predicted)@rotation
        half=np.asarray(self.payload["size"])/2
        if self.source.category in {"cup","bottle"}:
            distances=np.column_stack((np.linalg.norm(local[:,:2],axis=1)-np.mean(half[:2]),np.abs(local[:,2])-half[2]))
        else: distances=np.abs(local)-half
        signed=np.linalg.norm(np.maximum(distances,0),axis=1)+np.minimum(np.max(distances,axis=1),0)
        residual=float(np.median(np.abs(signed)))
        fraction=float(np.mean(signed<.008))
        ok=residual<.006 and fraction>.85
        self.log.event("visual_attachment",{"surface_residual_m":residual,"inlier_fraction":fraction,"consistent":ok})
        return ok

    def gripper_holding(self):
        state=self.robot.get_contact_state()
        return min(state.left_force,state.right_force)>self.config.safety.tactile_hold_force_n

    def refresh(self,scan=False):
        self.budget()
        if self.held and self.payload is not None:
            pose=self.robot.get_ee_pose()
            # Used only to associate actual segmented pixels after wrist
            # rotation; it never creates an observation or a visible flag.
            self.world.memory.held_expected_position=pose.position+pose.rotation@np.asarray(self.payload['offset'])
        else:self.world.memory.held_expected_position=None
        if scan and self.held:
            observed=self.perception.observe(reground=True)
        elif scan:
            self.perception.scan([self.goal.object]+self.goal.references,force_all=False)
            observed=[]
        else: observed=self.perception.observe([self.goal.object.description])
        # A tilted or rotated hand exposes the handle as a separate same-colour piece later than the
        # lift did; a piece touching the held body belongs to it, not to the obstacle set.
        if self.held and self.source_id is not None:self.world.memory.adopt_adjacent_fragments(self.source_id)
        newly_attached=self.world.memory.held_fragments-self.logged_held_fragments
        for fragment_id in sorted(newly_attached):
            from manipulation.payload_geometry import merge_observed_fragment
            self.payload=merge_observed_fragment(self.payload,self.robot.get_ee_pose(),self.world.objects[fragment_id])
            self.log.event('held_fragment_confirmed',{'source_id':self.source_id,'fragment_id':fragment_id,
                'basis':'repeated adjacent co-motion in RGB-D observations',
                'collision_size_m':np.asarray(self.payload['collision_size']).tolist()})
        self.logged_held_fragments.update(newly_attached)
        if self.source_id in self.world.objects: self.source=self.world.objects[self.source_id]
        if self.held and self.source is not None and self.source.visible and self.payload is not None and 'surface_local' in self.payload:
            from perception.held_registration import register_translation
            pose=self.robot.get_ee_pose();predicted=pose.position+pose.rotation@np.asarray(self.payload['offset'])
            reference=predicted+np.asarray(self.payload['surface_local'])@pose.rotation.T
            shift,evidence=register_translation(self.source.points,reference)
            self.log.event('held_visual_registration',evidence)
            if shift is not None:
                self.source.position=predicted+shift
                self.source.uncertainty_m=max(self.source.uncertainty_m,evidence['surface_residual_m'])
        if self.source is not None:
            if self.held and self.payload is not None and 'footprint_local' in self.payload:
                offsets=np.asarray(self.payload['footprint_local'])@self.robot.get_ee_pose().rotation.T
                self.source.manipulation_geometry['carried_footprint_offsets']=offsets[:,:2].tolist()
            else:self.source.manipulation_geometry.pop('carried_footprint_offsets',None)
        if self.target_id in self.world.objects: self.target=self.world.objects[self.target_id]
        elif self.target_id in self.world.regions: self.target=self.world.regions[self.target_id]
        return observed

    def observe_move(self,position,rotation=None):
        """Bounded sensor repositioning through the normal collision controller."""
        previous=getattr(self,'observation_motion',False);speed=self.cartesian_speed_limit
        self.observation_motion=True
        speed_limit = .05
        if not self.held and self.config.version >= 3:
            clearance_z = max((o.high[2] for o in self.world.objects.values()), default=.15)
            clearance_z += self.config.planner.palm_to_contact_m + .045
            if min(position[2], self.robot.get_ee_pose().position[2]) >= clearance_z:
                speed_limit = .08
        self.cartesian_speed_limit=min(speed or speed_limit,speed_limit)
        try:
            self.log.event('observation_motion',{'carrying':self.held,'speed_limit_m_s':self.cartesian_speed_limit})
            return self.move(position,rotation,allow_contact=self.held)
        finally:self.observation_motion=previous;self.cartesian_speed_limit=speed

    def move(self,position,rotation=None,allow_contact=False,support_id=None):
        if self.config.strict_visual_identity:
            self.perception.identity_resolver.require_phase(self)
        """Short Cartesian segment -> validated trajectory -> sensor feedback."""
        self.safety.allow_finger_contact=allow_contact or self.held
        start=self.robot.get_ee_pose()
        rotation=start.rotation if rotation is None else rotation
        goal=np.asarray(position,dtype=float)
        step = self.config.planner.cartesian_step_m
        if self.observation_motion and not self.held and (self.cartesian_speed_limit or 0) > .05:
            step *= 2  # fewer stop/settle points on high empty observation transits
        count=max(1,int(np.ceil(np.linalg.norm(goal-start.position)/step)))
        for index in range(1,count+1):
            if self.config.strict_visual_identity:
                self.perception.identity_resolver.require_phase(self)
            self.budget()
            point=start.position+(goal-start.position)*(index/count)
            support="observed_table" if support_id in self.world.regions else support_id
            if self.current_skill=='RETREAT' and not self.held and self.source is not None:drop_source_slivers(self,True)
            self.robot.configure_collision_world(self.world,self.source_id if allow_contact else None,self.payload,support)
            start_collision=self.robot.check_collision(self.robot.get_joint_positions())
            corridor=(.10 if self.held and self.candidate is not None and getattr(self.candidate,'grasp_type',None)=='side_grasp' else None)
            path=self.planner.plan_to_pose(Pose(point,rotation),carrying=self.held,cartesian_corridor_m=corridor)
            if path is None:
                solutions=self.robot.solve_ik(Pose(point,rotation),self.robot.get_joint_positions())
                collisions=[self.robot.check_collision(q) for q in solutions]
                raise SkillFailure(self.planner.last_failure or "PATH_BLOCKED",{
                    "requested_position":point.tolist(),
                    "ik_solutions":len(solutions),
                    "goal_collision_reasons":collisions,
                    "start_collision_reason":start_collision,
                    "route_diagnostics":self.planner.last_diagnostics[-6:],
                })
            self.perception.annotations={"target":None if self.target is None else self.target.position.tolist(),
                "grasp":None if self.candidate is None else self.candidate.position.tolist(),
                "trajectory":[self.robot.forward_kinematics(q).position.tolist() for q in path]}
            ok,reason=self.executor.execute(path,carrying=self.held,cartesian_speed_limit=self.cartesian_speed_limit)
            if not ok: raise SkillFailure(FailureClassifier.classify(reason),{"reason":reason})
            self.refresh()
            if self.held and not self.source.visible:
                self.refresh(scan=True)
                if not self.source.visible:
                    # A wrist camera a few centimetres from the fingers often sees only a sliver of the
                    # object it holds. Two fingers still pressing on it are proof enough that it is in hand.
                    if not self.gripper_holding():raise SkillFailure("TARGET_LOST",{"reason":"held object occluded in all allowed views"})
                    self.log.event("held_visual_lost_tactile_hold",{"source_id":self.source_id})
            if self.held and self.source.visible:
                predicted=self.robot.get_ee_pose().position+self.robot.get_ee_pose().rotation@np.asarray(self.payload["offset"])
                slip=float(np.linalg.norm(self.source.position-predicted))
                if slip>self.config.safety.slip_distance_m and not self.attachment_visible(predicted):
                    self.log.event("uncertain_tracking",{"single_view_slip_m":slip,"action":"hold_and_reobserve"})
                    self.refresh(scan=True)
                    slip=float(np.linalg.norm(self.source.position-predicted))
                    if self.source.visible and slip>self.config.safety.slip_distance_m and not self.attachment_visible(predicted) and not self.gripper_holding():
                        raise SkillFailure("OBJECT_SLIPPED",{"visual_slip_m":slip,"observed":self.source.position.tolist(),"expected":predicted.tolist()})
