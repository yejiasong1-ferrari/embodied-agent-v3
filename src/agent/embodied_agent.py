"""Observe -> plan -> execute a bounded skill -> validate -> recover."""
from dataclasses import dataclass
from enum import Enum
import time
from config import AgentConfig
from world.world_model import WorldModel
from llm.llm_client import LLMClient
from llm.schemas import Goal,Reference
from perception.active_perception import ActivePerception
from perception.debug_visualizer import write_report
from manipulation.motion_planner import MotionPlanner
from manipulation.trajectory_executor import TrajectoryExecutor
from manipulation.grasp_planner import GraspPlanner
from validation.safety_monitor import SafetyMonitor
from recovery.failure_classifier import SkillFailure
from recovery.recovery_manager import RecoveryManager
from skills.skill_base import SkillContext
from skills.skill_registry import SkillRegistry
from .task_log import TaskLog
from .task_interpreter import TaskInterpreter
from .task_planner import TaskPlanner
from .replanner import Replanner


class State(str,Enum):
    IDLE="IDLE"
    UNDERSTAND="UNDERSTAND"
    OBSERVE="OBSERVE"
    PLAN="PLAN"
    EXECUTE="EXECUTE"
    MONITOR="MONITOR"
    VERIFY="VERIFY"
    RECOVER="RECOVER"
    REPLAN="REPLAN"
    SUCCESS="SUCCESS"
    FAILED="FAILED"


@dataclass(frozen=True)
class AgentResult:
    success: bool
    state: str
    message: str
    task_id: str
    log_path: str
    metrics: dict


class EmbodiedAgent:
    def __init__(self,robot,camera,config:AgentConfig|None=None,event_sink=None,image_sink=None,log_sink=None):
        self.robot,self.camera,self.config=robot,camera,config or AgentConfig.load()
        self.event_sink=event_sink or (lambda state,data:None)
        self.image_sink=image_sink
        self.log_sink=log_sink
        self.world=WorldModel(self.config.perception)
        self.state=State.IDLE
        self.registry=SkillRegistry()
        self.context=None
        self._cancelled=False

    def cancel(self):
        self._cancelled=True
        self.robot.stop()

    def transition(self,state,**data):
        self.state=state
        self.log.event("state",{"state":state.value,**data})
        self.event_sink(state.value,data)

    def execute(self,task:str)->AgentResult:
        self.context=None
        self._cancelled=False
        self.log=TaskLog(self.config.log_root,task,self.config,event_sink=self.log_sink)
        client=LLMClient(self.config.llm,self.log.event)
        started=time.monotonic(); message=""; success=False; failure=None
        try:
            if self.world.memory.held_id is not None:
                raise SkillFailure("OBJECT_SLIPPED",{"reason":"previous task stopped while holding an object; inspect and reset before a new task"})
            if self.config.strict_visual_identity and not client.online:
                raise SkillFailure("LOW_CONFIDENCE", {"reason":"V3 requires an online vision model; local identity fallback is disabled"})
            self.robot.resume()
            self.world.memory.mission_active=False
            self.world.memory.retained_ids.clear()
            self.world.reference_bindings.clear()
            self.world.reference_binding_evidence.clear()
            for obj in self.world.objects.values():
                obj.query=""
                obj.semantic_confidence=0.
            self.camera.set_manipulation_active(False)
            self.transition(State.UNDERSTAND)
            goal=TaskInterpreter(client,self.log).understand(task)
            self.log.event("interpretation",goal.model_dump())
            perception=ActivePerception(self.camera,self.world,self.config,client,self.log)
            perception.task_description=task
            perception.image_sink=self.image_sink
            perception.set_references([goal.object]+goal.references)
            safety=SafetyMonitor(self.robot,self.config,lambda:self._cancelled)
            recovery=RecoveryManager(self.config,Replanner(client),self.log)
            planner=TaskPlanner(client,self.log)
            c=SkillContext(self.robot,perception,self.world,goal,self.config,MotionPlanner(self.robot,self.config),
                TrajectoryExecutor(self.robot,safety,self.config,self.log),GraspPlanner(self.config),safety,self.log,recovery)
            self.context=c
            self.transition(State.OBSERVE)
            c.refresh(scan=True)
            self.event_sink("OBSERVE",{"objects":[o.semantic_label for o in self.world.objects.values() if o.visible],
                                       "world_revision":self.world.revision})
            self.transition(State.PLAN)
            # Clearing an obstruction expands to ordinary, independently
            # verified manipulation subgoals, using only observed geometry.
            jobs=[]
            if goal.goal in {"extract","clear_then_place"}:
                source=perception.identify(goal.object)
                for blocker in planner.blockers(source,self.world):
                    staging=planner.free_region(blocker,self.world,self.config.safety.workspace)
                    staging.id=f"reserved_staging_{len(jobs)}"
                    self.world.regions[staging.id]=staging
                    jobs.append((Goal(goal="place_on",object=Reference(description=blocker.semantic_label,color=blocker.color,category=blocker.category),
                                      target=Reference(description="free table region",category="region")),staging))
                if goal.goal=="extract":
                    staging=planner.free_region(source,self.world,self.config.safety.workspace)
                    jobs.append((Goal(goal="place_on",object=goal.object,target=Reference(description="free table region",category="region")),staging))
                else: jobs.append((goal,None))
            else: jobs=[(goal,None)]
            for subgoal,region in jobs:
                c.goal=subgoal
                c.held=False; c.payload=None; c.grasp_verified=False; c.lift_verified=False; c.final_validated=False
                c.source=c.target=c.candidate=None; c.source_id=c.target_id=None
                recovery.failed_candidates.clear()
                if region:
                    region.id=f"staging_region_{len(self.world.objects)}"
                    self.world.regions.clear()
                    self.world.regions[region.id]=region
                plan=planner.plan(subgoal,self.world)
                names=[step.skill for step in plan.steps]
                self.log.event("plan",plan.model_dump()); self.event_sink("PLAN",{"skills":names})
                index=0
                while index<len(names):
                    c.budget(); c.steps+=1
                    if c.steps>self.config.recovery.max_steps: raise TimeoutError("skill step budget exhausted")
                    name=names[index]
                    self.transition(State.VERIFY if name.startswith("VERIFY") else State.EXECUTE,skill=name)
                    try:
                        if not self.registry.execute(name,c): raise SkillFailure("GRASP_FAILED",{"skill":name})
                        self.log.event("skill",{"name":name,"success":True})
                        self.transition(State.MONITOR,skill=name)
                        index+=1
                    except SkillFailure as exc:
                        if self._cancelled: raise SkillFailure("USER_STOP")
                        self.log.event("failure",{"skill":name,"failure":exc.failure.value,"evidence":exc.evidence})
                        if (name in {"MOVE_TO_TARGET","PLACE_OBJECT"} and c.held and c.placement_position is not None
                                and exc.failure.value in {"PATH_BLOCKED","IK_FAILED"}):
                            if c.goal.relation not in {'on','centered_with'}:
                                c.blocked_placements.append(c.placement_position.copy())
                            self.log.event("transport_route_rejected",{
                                "placement_position":c.placement_position.tolist(),
                                "world_revision":c.world.revision,
                                "reason":exc.failure.value,
                            })
                        self.transition(State.RECOVER,failure=exc.failure.value)
                        action=recovery.decide(exc,c)
                        if action=="ABORT": raise
                        self.robot.resume(); safety.last_failure=None
                        released_during_placement=(name=="PLACE_OBJECT" and exc.failure.value=="OBJECT_SLIPPED")
                        if action in {"CHANGE_GRASP","CHANGE_APPROACH"} and c.held and exc.failure.value!='OBJECT_SLIPPED':
                            from skills.manipulation import park_held_object
                            if not park_held_object(c,'carry or placement unreachable; try another grasp'):
                                # Keep the object; a blocked park must not become a mid-air drop.
                                action='REPLAN'
                        if action in {"CHANGE_GRASP","CHANGE_APPROACH"}:
                            if c.held:
                                self.world.memory.last_released_id=c.source_id
                                c.held=False; c.payload=None; self.world.memory.held_id=None
                            self.robot.open_gripper(); c.grasp_verified=False; c.lift_verified=False
                            # Exit the target contact region before allowing
                            # free-space motions for another grasp orientation.
                            pose=self.robot.get_ee_pose()
                            import numpy as np
                            try:
                                c.move(pose.position+np.array([0,0,.08]),pose.rotation,allow_contact=True)
                                self.camera.set_manipulation_active(False)
                                c.refresh(scan=True)
                            except SkillFailure as cleanup:
                                self.log.event("failure",{"skill":name,"failure":cleanup.failure.value,
                                    "evidence":cleanup.evidence,"phase":"recovery_cleanup"})
                                if cleanup.evidence.get("reason_code")!="IDENTITY_REVIEW_REQUIRED":raise
                                resolver=getattr(perception,'identity_resolver',None)
                                if resolver is not None:
                                    key=(c.goal.object.description,c.goal.object.color,c.goal.object.category)
                                    resolver.decide_now=getattr(resolver,'decide_now',set())|{key}
                                index=names.index("FIND_OBJECT")
                                self.transition(State.REPLAN,action="REOBSERVE")
                                continue
                            if released_during_placement:
                                # A lost grip on descent may still leave the
                                # object stable at the requested destination.
                                # Only fresh visual task validation can decide.
                                self.log.event("released_object_recheck",{"reason":"grip lost during placement",
                                    "next_skill":"VERIFY_TASK"})
                                index=names.index("VERIFY_TASK")
                            else:index=names.index("APPROACH_OBJECT")
                        elif action=="REOBSERVE":
                            if exc.failure.value in {'AMBIGUOUS_TARGET'} or exc.evidence.get('reason_code')=='IDENTITY_REVIEW_REQUIRED':
                                resolver=getattr(perception,'identity_resolver',None)
                                requested=exc.evidence.get('reference')
                                references=[c.goal.object,*c.goal.references]
                                if c.goal.target is not None:references.append(c.goal.target)
                                match=next((r for r in references if r.description==requested),c.goal.object)
                                if resolver is not None:
                                    key=(match.description,match.color,match.category)
                                    resolver.decide_now=getattr(resolver,'decide_now',set())|{key}
                                    self.log.event('identity_decide_now',{'reference':match.description,
                                        'reason':'recovery: pick the likeliest candidate rather than stop'})
                            # A failed approach must release the camera scan lock
                            # before searching again. Never do this while carrying.
                            if (not c.held and self.config.strict_visual_identity
                                    and exc.evidence.get('reason_code')=='IDENTITY_REVIEW_REQUIRED'):
                                from skills.localization import localize_target
                                c.observation_view_offset=getattr(c,'observation_view_offset',0)+1
                                localize_target(c,c.source_id,move=True)
                            elif not c.held:
                                self.camera.set_manipulation_active(False)
                                c.refresh(scan=True)
                            elif self.config.strict_visual_identity and name=='PLACE_OBJECT':
                                c.placement_reference_checked=False
                                c.refresh()
                            elif self.config.strict_visual_identity:
                                # Reposition the calibrated wrist through payload-safe
                                # observation motion, never the free-camera scan API.
                                from skills.localization import localize_target
                                references=[c.goal.object,*c.goal.references]
                                requested=exc.evidence.get('reference')
                                reference=next((r for r in references if r.description==requested),c.goal.target or c.goal.object)
                                obj=c.world.resolve(reference)
                                localize_target(c,obj.id,move=True)
                                if not perception.identity_resolver.valid(reference,fresh=False):
                                    checked=perception.identify(reference)
                                    if checked.id!=obj.id:raise SkillFailure('AMBIGUOUS_TARGET',{'reason':'reference instance changed during carried recovery'})
                            else:c.refresh(scan=True)
                            index=names.index('PLACE_OBJECT') if name=='PLACE_OBJECT' and c.held else names.index('VERIFY_TASK') if name in {'RETREAT','VERIFY_TASK'} else names.index("FIND_OBJECT") if not c.held else names.index("MOVE_TO_TARGET")
                        else:
                            if not c.held:
                                pose=self.robot.get_ee_pose()
                                import numpy as np
                                c.move(pose.position+np.array([0,0,.08]),pose.rotation,allow_contact=True)
                                self.camera.set_manipulation_active(False)
                            c.refresh(scan=True)
                            index=names.index("MOVE_TO_TARGET") if c.held else names.index("APPROACH_OBJECT")
                        self.transition(State.REPLAN,action=action)
                if not c.final_validated: raise SkillFailure("RELATION_NOT_SATISFIED",{"reason":"missing final visual acceptance"})
            success=True; message="TASK SUCCESS: fresh visual relation and stability verified"
            self.transition(State.SUCCESS)
        except Exception as exc:
            self.robot.stop()
            failure=exc.failure.value if isinstance(exc,SkillFailure) else type(exc).__name__
            message=str(exc)
            self.transition(State.FAILED,failure=failure,message=message,evidence=exc.evidence if isinstance(exc,SkillFailure) else {})
        c=self.context
        metrics={"execution_time_s":time.monotonic()-started,"llm_calls":client.calls,
                 "recovery_attempts":len(c.recovery.history) if c else 0,
                 "grasp_success":bool(c and c.any_grasp_verified),"lift_success":bool(c and c.any_lift_verified),
                 "grasp_attempts":c.grasp_attempts if c else 0,
                 "first_grasp_verified":bool(c and c.first_grasp_verified),
                 "first_lift_success":bool(c and c.first_lift_success),
                 "placement_success":bool(c and c.final_validated),"failure":failure,
                 "unexpected_contact_events":c.safety.collision_count if c else 0,
                 "observations":c.perception.observation_count if c else 0}
        strategies=[e['data'] for e in self.log.events if e['type']=='adaptive_grasp']
        scans=[e['data'] for e in self.log.events if e['type']=='object_multiview_complete']
        validations=[e['data'] for e in self.log.events if e['type']=='relation_validation']
        metrics.update(visual_identity_calls=c.perception.identity_resolver.calls if c and c.perception.identity_resolver else 0,
            identity_confirmations=sum(e['type']=='target_binding' and e['data'].get('source')=='vlm_identity' for e in self.log.events),
            visual_grounding_calls=c.perception.grounder.calls if c else 0,
            fast_path_used=any(s['fast_path_used'] for s in strategies),
            multiview_used=any(s['scan_views']>0 for s in scans),
            scan_views=sum(s['scan_views'] for s in scans),
            grasp_reasoner_calls=sum(e['type']=='llm_request' and e['data'].get('purpose')=='grasp_strategy' for e in self.log.events),
            grasp_strategies=[s['strategy'] for s in strategies],
            push_attempts=c.push_attempts if c else 0,regrasp_attempts=c.regrasp_attempts if c else 0,
            final_spatial_constraints=validations[-1].get('constraints',[]) if validations else [])
        result=AgentResult(success,self.state.value,message,self.log.task_id,str(self.log.path),metrics)
        self.log.event("result",result.__dict__); self.log.write("result.json",result.__dict__); write_report(self.log)
        return result
