import time
import numpy as np
from dataclasses import replace
from llm.schemas import GraspStrategyDecision
from perception.geometry_analyzer import GeometryAnalyzer
from perception.multiview_scanner import MultiViewObjectScanner
from recovery.failure_classifier import SkillFailure
from robot.robot_interface import Pose


ROOMY_MARGIN_RAD=.15


class AdaptiveGraspPlanner:
    def __init__(self,config):self.config=config;self.analyzer=GeometryAnalyzer(config);self.scanner=MultiViewObjectScanner(config);self.reasoner_calls=0
    def choose(self,c):
        start=time.monotonic();c.log.event('stage',{'stage':'ANALYZE_GEOMETRY'})
        force_scan=bool(getattr(c,'force_multiview_grasp',False))
        model=self.analyzer.analyze(c.source,c.world);constraints=c.goal.hard_constraints
        fast=(model['top_grasp_score']>=self.config.grasp.fast_top_score_threshold and model.get('gripper_fit_top')
              and c.source.stability_displacement_m is not None and c.source.stability_displacement_m<=.004
              and c.source.uncertainty_m<=self.config.localization.precise_uncertainty_m
              and not constraints&{'avoid_top_contact','prefer_handle','avoid_squeezing'} and c.goal.manipulation.grasp_specific_region in {None,'top'}
              and not force_scan)
        views=0
        if not fast and (force_scan or model['complexity_score']>=self.config.grasp.complexity_threshold or not model['graspable_regions']):
            c.log.event('stage',{'stage':'NEED_MORE_GEOMETRY'});model,points,views=self.scanner.scan(c);c.source.points=points
        if c.config.strict_visual_identity:
            measured_position=c.source.position.copy()
            c.perception.identity_resolver.reconcile_after_scan(c)
            if np.linalg.norm(c.source.position-measured_position)>.012:
                raise SkillFailure('LOW_CONFIDENCE',{'reason':'target moved after geometry scan; reconstruction must be renewed'})
            if views:
                c.source.points=points
                model=self.analyzer.analyze(c.source,c.world,points)
                if len(self.scanner.fusion.cache.get(c.source.id,{}).get('views',()))>=2:
                    from perception.geometry_analyzer import measured_round_base
                    radius=measured_round_base(c.source,points)
                    if radius is not None:
                        model['measured_base_radius_m']=radius
        c.force_multiview_grasp=False
        c.log.event('manipulation_geometry',{'object_id':c.source.id,'model':model})
        candidates=c.grasp_planner.candidates(c.source,c.world)
        if fast:candidates=[x for x in candidates if x.grasp_type=='top_grasp']
        if 'avoid_top_contact' in constraints:candidates=[x for x in candidates if x.contact_region!='top']
        if 'prefer_handle' in constraints:candidates=[x for x in candidates if x.contact_region=='handle']
        if c.goal.manipulation.grasp_specific_region:candidates=[x for x in candidates if x.contact_region==c.goal.manipulation.grasp_specific_region]
        candidates=[x for x in candidates if x.id not in c.recovery.failed_candidates and x.clearance>=c.config.safety.collision_margin_m]
        heading=np.arctan2(c.source.position[1],c.source.position[0])
        def cost(candidate):
            yaw=np.arctan2(candidate.orientation[1,0],candidate.orientation[0,0]);angle=abs(np.arctan2(np.sin(yaw-heading),np.cos(yaw-heading)))
            return ((candidate.grasp_type!='top_grasp',-candidate.score,-candidate.clearance,candidate.grasp_width,angle)
                    if fast else (-candidate.score,angle))
        candidates.sort(key=cost)
        feasible=[];pregrasp_paths={};lift_margins={};route_margins={};clearances={}
        c.robot.configure_collision_world(c.world,c.source_id,None,None)
        for candidate in candidates:
            c.budget();distance=c.config.planner.side_approach_clearance_m if candidate.grasp_type=='side_grasp' else c.config.planner.approach_clearance_m
            hover=candidate.position-candidate.approach_vector*distance
            contact_solutions=[q for q in c.robot.solve_ik(Pose(candidate.position,candidate.orientation),c.robot.get_joint_positions())
                if c.robot.check_collision(q) is None]
            contact_ok=bool(contact_solutions)
            hover_solutions=c.robot.solve_ik(Pose(hover,candidate.orientation),c.robot.get_joint_positions()) if contact_ok else []
            hover_ok=any(c.robot.check_collision(q) is None for q in hover_solutions)
            clearance=distance
            if contact_ok and not hover_ok and candidate.grasp_type=='top_grasp':
                # Beside the base the full pregrasp height can be out of reach while
                # a lower one is not. A lower pregrasp is still clear of the object.
                for lower in (.09,.07,.05,.03):
                    if lower>=distance:continue
                    lowered=candidate.position-candidate.approach_vector*lower
                    if any(c.robot.check_collision(q) is None for q in c.robot.solve_ik(Pose(lowered,candidate.orientation),c.robot.get_joint_positions())):
                        hover_ok=True;clearance=lower;break
            clearances[candidate.id]=clearance
            if contact_ok and hover_ok and candidate.grasp_type=='side_grasp':
                c.robot.configure_collision_world(c.world,None,None,None)
                path=c.planner.plan_to_pose(Pose(hover,candidate.orientation))
                hover_ok=path is not None
                if hover_ok:pregrasp_paths[candidate.id]=path
                c.robot.configure_collision_world(c.world,c.source_id,None,None)
            lift_solutions=[]
            if contact_ok and hover_ok:
                lift_pose=Pose(candidate.position+np.array([0,0,.045]),candidate.orientation)
                lift_solutions=[q for q in c.robot.solve_ik(lift_pose,contact_solutions[0])
                    if c.robot.check_collision(q) is None]
            if contact_ok and hover_ok and lift_solutions:
                lower,upper=c.robot.joint_limits
                margin_of=lambda q:float(np.min(np.minimum(q-lower,upper-q)))
                lift_margins[candidate.id]=max(margin_of(q) for q in lift_solutions)
                # The carry leaves the pick spot outward and upward. A pose that is fine to
                # grasp but has no joint range left a hand's width further on cannot be carried.
                outward=candidate.position[:2]/max(float(np.linalg.norm(candidate.position[:2])),1e-9)
                probe=Pose(lift_pose.position+np.array([*(outward*.10),.06]),candidate.orientation)
                probe_solutions=[q for q in c.robot.solve_ik(probe,lift_solutions[0]) if c.robot.check_collision(q) is None]
                route_margins[candidate.id]=min(lift_margins[candidate.id],max((margin_of(q) for q in probe_solutions),default=0.))
                feasible.append(replace(candidate,reachability=True,collision_free=True))
            if len(feasible)>=6 and any(route_margins[x.id]>=ROOMY_MARGIN_RAD for x in feasible):break
        feasible.sort(key=lambda candidate:(route_margins[candidate.id]<ROOMY_MARGIN_RAD,-candidate.score,-lift_margins[candidate.id],-candidate.clearance,candidate.grasp_width))
        c.log.event('grasp_candidates',[{'id':x.id,'type':x.grasp_type,'position':x.position.tolist(),'width':x.grasp_width,'score':x.score,
            'clearance':x.clearance,'lift_joint_margin_rad':lift_margins[x.id],'route_joint_margin_rad':route_margins[x.id],
            'tilt_deg':float(np.rad2deg(np.arccos(np.clip(-x.approach_vector[2],-1,1)))),
            'region':x.contact_region,'approach':x.approach_direction,'reachability':x.reachability,'collision_free':x.collision_free} for x in feasible])
        if not feasible:
            if views==0 and not force_scan:
                c.force_multiview_grasp=True
                c.log.event('grasp_retry_multiview',{'reason':'single-view candidate width or route unsafe'})
                return self.choose(c)
            raise SkillFailure('GRASP_FAILED',{'reason':'no feasible grasp candidate for measured geometry/constraints'})
        roomy=[x for x in feasible if route_margins[x.id]>=ROOMY_MARGIN_RAD]
        if roomy and len(roomy)<len(feasible):
            c.log.event('grasp_candidates_narrowed',{'kept':[x.id for x in roomy],'dropped':[x.id for x in feasible if x not in roomy],
                'reason':'carry would leave no joint range'})
            feasible=roomy
        if fast and c.config.version >= 3:
            # On a well-measured simple top grasp, a narrow closing axis
            # leaves space for both pads. Treat sub-3 mm differences as noise
            # and keep the existing clearance/kinematics order within that set.
            narrowest = min(x.grasp_width for x in feasible)
            feasible.sort(key=lambda x: x.grasp_width > narrowest + .003)
        chosen=feasible[0];client=c.recovery.replanner.client;reasoned=False;reasoner_called=False
        margin=abs(feasible[0].score-feasible[1].score) if len(feasible)>1 else 1.
        semantic=bool(constraints-{'avoid_collision'}) or c.source.category in {'cup','bottle'}
        if (not fast and self.config.grasp.llm_reasoner_enabled and client.online and self.reasoner_calls<self.config.grasp.max_reasoner_calls
            and (semantic or margin<self.config.grasp.reasoner_candidate_margin or len(c.recovery.history)>1)):
            self.reasoner_calls+=1
            reasoner_called=True
            try:
                image_kwargs={}
                if c.config.strict_visual_identity and c.perception.last_frame is not None:
                    import cv2
                    from perception.pointcloud import world_to_pixel
                    frame=c.perception.last_frame;image=frame.rgb.copy()
                    for item in feasible:
                        pixels,z=world_to_pixel([item.position],frame)
                        if z[0]>0:
                            uv=tuple(np.rint(pixels[0]).astype(int))
                            cv2.circle(image,uv,5,(255,220,0),2)
                            cv2.putText(image,item.id,uv,cv2.FONT_HERSHEY_SIMPLEX,.4,(255,220,0),1)
                    from PIL import Image
                    Image.fromarray(image).save(c.log.path/'images'/f'grasp_choices_{self.reasoner_calls}.png')
                    image_kwargs={'image':image}
                decision=client.complete(GraspStrategyDecision,'grasp_strategy',{'goal':c.goal.model_dump(),'geometry':model,
                    'candidates':[{'id':x.id,'strategy':x.grasp_type,'region':x.contact_region,'approach':x.approach_direction,'score':x.score} for x in feasible]},**image_kwargs)
                selected=next((x for x in feasible if x.id==decision.selected_candidate_id and x.grasp_type==decision.strategy and x.contact_region==decision.preferred_region),None)
                if selected is None:raise ValueError('reasoner selected unavailable candidate')
                chosen=selected;reasoned=True;c.log.event('grasp_strategy',decision.model_dump())
            except (RuntimeError,ValueError) as exc:c.log.event('grasp_reasoner_fallback',{'reason':str(exc),'backend':'local_geometry'})
        c.log.event('adaptive_grasp',{'fast_path_used':fast,'multiview_used':views>0,'llm_grasp_reasoner_called':reasoner_called,'llm_selection_used':reasoned,
            'scan_views':views,'planning_time_s':time.monotonic()-start,'candidate':chosen.id,'strategy':chosen.grasp_type,
            'region':chosen.contact_region,'approach':chosen.approach_direction})
        c.pregrasp_path=pregrasp_paths.get(chosen.id)
        c.pregrasp_clearance=clearances.get(chosen.id) if chosen.grasp_type=='top_grasp' else None
        c.grasp_surface_offsets=c.source.points-c.source.position
        return chosen
