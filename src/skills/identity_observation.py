"""Bounded observation-only side views of measured, unconfirmed candidates."""
from dataclasses import replace
import numpy as np
from manipulation.viewpoint_planner import ViewpointPlanner
from robot.robot_interface import Pose
from recovery.failure_classifier import SkillFailure,FailureClassifier


def identify_with_side_review(c,reference):
    from perception.candidate_comparison import CandidateNeedsViews
    resolver=getattr(c.perception,'identity_resolver',None)
    if resolver is None:return c.perception.identify(reference)
    previous=getattr(resolver,'observation_context',None)
    resolver.observation_context=c
    try:
        try:return c.perception.identify(reference)
        except CandidateNeedsViews as exc:
            key=(reference.description,reference.color,reference.category)
            reviewed=getattr(resolver,'side_reviewed',set())

            def decide(reason):
                # Never stop on "cannot tell": take the likeliest candidate and let the fresh
                # confirmation and the measured colour check keep the final word.
                resolver.decide_now=getattr(resolver,'decide_now',set())|{key}
                c.log.event('identity_decide_now',{'reference':reference.description,'reason':reason})
                return c.perception.identify(reference)
            if key in reviewed or c.held or c.world.memory.mission_active:return decide('further side views unavailable')
            # Reserve a comparison and fresh identity confirmation before
            # moving. New photos cannot create additional model-call budget.
            spent=getattr(resolver,'spent',None)
            used=spent(reference) if callable(spent) else resolver.calls
            if used+exc.required_calls>c.config.localization.max_vlm_calls:return decide('identity budget too small for side views')
            resolver.side_reviewed=reviewed|{key}
            try:observed=observe_candidate_sides(c,exc.candidate_ids)
            except SkillFailure as failure:
                if failure.failure.value=='USER_STOP':raise
                c.log.event('identity_side_view_failed',{'reference':reference.description,'failure':failure.failure.value})
                c.robot.resume()
                return decide('side view could not be reached safely')
            if not observed:return decide('no side view produced usable evidence')
            return c.perception.identify(reference)
    finally:resolver.observation_context=previous


def move_identity_camera(c,pose):
    if c.held:raise SkillFailure('AMBIGUOUS_TARGET',{'reason':'side identity search cannot move a held object'})
    c.budget();c.robot.configure_collision_world(c.world,None,None,None)
    path=c.planner.plan_to_pose(pose)
    if path is None:return False
    old=getattr(c,'observation_motion',False);speed=c.cartesian_speed_limit
    c.observation_motion=True;c.cartesian_speed_limit=min(speed or .05,.05)
    try:
        ok,reason=c.executor.execute(path,carrying=False,cartesian_speed_limit=c.cartesian_speed_limit)
        if not ok:raise SkillFailure(FailureClassifier.classify(reason),{'reason':reason})
    finally:c.observation_motion=old;c.cartesian_speed_limit=speed
    return True


def restore_identity_camera(c,camera_pose):
    transform=camera_pose@np.linalg.inv(c.perception.camera.ee_to_camera)
    return move_identity_camera(c,Pose(transform[:3,3],transform[:3,:3]))


def observe_candidate_sides(c,candidate_ids):
    if c.held:raise SkillFailure('AMBIGUOUS_TARGET',{'reason':'cannot perform identity side search while holding'})
    camera=c.perception.camera
    if getattr(camera.config,'mode','')!='wrist':return 0
    count=0
    for identity in candidate_ids[:4]:
        obj=c.world.objects.get(identity)
        if obj is None:continue
        c.log.event('identity_side_observation',{'object_id':identity,'status':'planning'})
        captured=False
        # The planner also offers higher oblique poses after the four low
        # sides. Keep those collision-checked alternatives when sides fail IK.
        for index,pose in enumerate(ViewpointPlanner(c.config).candidates(obj,camera,side=True)):
            try:
                if not move_identity_camera(c,pose):
                    c.log.event('viewpoint_rejected',{'object_id':identity,'viewpoint_index':index,
                        'reason':'no_safe_path','planner_failure':getattr(c.planner,'last_failure',None),
                        'diagnostics':getattr(c.planner,'last_diagnostics',[])[-3:]})
                    continue
            except SkillFailure as exc:
                if exc.failure.value in {'COLLISION_RISK','USER_STOP'}:raise
                c.log.event('viewpoint_rejected',{'object_id':identity,'viewpoint_index':index,
                    'reason':exc.failure.value});continue
            frame=camera.capture_detail()
            expected=(c.config.camera.detail_height,c.config.camera.detail_width)
            if frame.rgb.shape[:2]!=expected:raise ValueError('side identity image calibration mismatch')
            # A dedicated view ID prevents treating this arbitrary pose as a
            # preset scan pose when the selected candidate is revisited later.
            frame=replace(frame,view_id=f'identity_side:{c.perception.observation_count+1}')
            objects=c.perception.observe(frame=frame)
            row=c.perception.identity_resolver.comparison.gallery.get(identity)
            if (any(o.id==identity for o in objects) and row is not None
                    and row['timestamp']==frame.timestamp):
                count+=1;c.log.event('identity_side_observation',{'object_id':identity,'status':'captured',
                    'view':frame.view_id,'resolution':[frame.rgb.shape[1],frame.rgb.shape[0]]})
                captured=True
                break
            c.log.event('viewpoint_rejected',{'object_id':identity,'viewpoint_index':index,
                'reason':'no_usable_tracked_candidate_image'})
        if not captured:
            c.log.event('identity_side_observation',{'object_id':identity,'status':'unavailable',
                'reason':'no_safe_view_with_usable_evidence'})
    return count
