"""Coarse targets authorize observation motion only; closing requires a fresh lock."""
import numpy as np
import time
from robot.robot_interface import Pose
from manipulation.viewpoint_planner import ViewpointPlanner
from recovery.failure_classifier import SkillFailure


def travel_to_observation(c,pose):
    current=c.robot.get_ee_pose();highest=max((o.high[2] for o in c.world.objects.values()),default=.15)
    height=max(current.position[2],pose.position[2],highest+c.config.planner.palm_to_contact_m+.08)
    c.observe_move(np.array([*current.position[:2],height]),current.rotation)
    delta=pose.position[:2]-current.position[:2];t=np.clip(-current.position[:2]@delta/max(delta@delta,1e-9),0,1)
    if np.linalg.norm(current.position[:2]+t*delta)<c.config.planner.base_clearance_radius_m:
        a0=np.arctan2(current.position[1],current.position[0]);a1=np.arctan2(pose.position[1],pose.position[0]);radius=max(np.linalg.norm(current.position[:2]),np.linalg.norm(pose.position[:2]),c.config.planner.transfer_radius_m)
        for angle in np.linspace(a0,a1,max(2,int(abs(a1-a0)/.20)+1)):
            c.observe_move(np.array([radius*np.cos(angle),radius*np.sin(angle),height]),current.rotation)
    c.observe_move(np.array([*pose.position[:2],height]),pose.rotation)
    c.observe_move(pose.position,pose.rotation)


def localize_target(c,identity,move=True):
    cfg=c.config.localization
    obj=c.world.objects.get(identity)
    if obj is None:
        raise SkillFailure('TARGET_LOST',{'reason':'tracked instance left memory during search','object_id':identity})
    if not cfg.enabled:return obj
    camera=c.perception.camera;c.perception.phase='local'
    c.log.event('stage',{'stage':'APPROACHING_FOR_OBSERVATION' if move else 'TARGET_REOBSERVATION','object_id':identity})
    poses=list(ViewpointPlanner(c.config).candidates(obj,camera)) if camera.config.mode=='wrist' and move else [None]
    last=None
    if poses and move:
        offset=getattr(c,'observation_view_offset',0)%len(poses)
        poses=poses[offset:]+poses[:offset]
    observed_views=0
    for attempt,pose in enumerate(poses):
        # Only views actually taken count: an unreachable viewpoint is skipped in favour of the next one.
        if observed_views>=cfg.max_local_observations:break
        c.budget()
        if pose is not None:
            try:travel_to_observation(c,pose)
            except SkillFailure as exc:
                if exc.failure.value in {'USER_STOP','COLLISION_RISK'}:raise
                last=exc;c.observation_view_offset=getattr(c,'observation_view_offset',0)+1;c.log.event('viewpoint_rejected',{'reason':exc.failure.value});continue
        elif move and camera.config.mode=='gantry':
            camera.move_observation(obj.position+np.array([0,0,cfg.observation_distance_m]),obj.position)
        c.log.event('stage',{'stage':'LOCAL_REOBSERVATION','object_id':identity,'attempt':attempt+1})
        observed_views+=1
        c.perception.observe(reground=True);c.refresh();obj=c.world.objects.get(identity)
        if obj and c.perception.localizer.precise(obj,time.monotonic()):
            if c.config.strict_visual_identity:
                resolver=c.perception.identity_resolver
                refs=[c.goal.object,*c.goal.references]
                for reference in refs:
                    key=(reference.description,reference.color,reference.category)
                    if c.world.reference_bindings.get(key)==identity and resolver.anchors.get(key,{}).get('review_required'):
                        checked=resolver.identify(reference,allow_search=False)
                        if checked.id!=identity:raise SkillFailure('AMBIGUOUS_TARGET',{'reference':reference.description,'reason':'instance changed during close observation'})
            c.log.event('stage',{'stage':'PRECISE_LOCKED' ,'object_id':identity});return obj
        last=SkillFailure('TARGET_LOST' if obj is None or not obj.visible else 'LOW_CONFIDENCE',{'reason':'fresh close-range precision gate','object_id':identity})
    raise last or SkillFailure('LOW_CONFIDENCE',{'reason':'no safe local observation pose'})
