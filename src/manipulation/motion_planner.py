import numpy as np
from robot.robot_interface import Pose
from .ik_solver import IKSolver


class MotionPlanner:
    def __init__(self,robot,config):
        self.robot,self.config=robot,config
        self.ik=IKSolver(robot)
        self.last_failure=None
        self.last_diagnostics=[]

    def plan_joint_path(self,start,goal,allow_rrt=True,allow_direct=True):
        count=max(2,int(np.ceil(np.max(np.abs(goal-start))/self.config.planner.joint_step_rad))+1)
        points=[start+a*(goal-start) for a in np.linspace(0,1,count)]
        if allow_direct and all(self.robot.check_collision(p) is None for p in points): return [start,goal]
        if allow_rrt:
            # Reuse the existing independent OMPL RRT-Connect wrapper.
            from motion_planning import plan_rrt_connect
            lower,upper=self.robot.joint_limits
            diagnostics=[]
            plan=plan_rrt_connect(start,[goal],lower=lower,upper=upper,
                 is_state_valid=lambda q:self.robot.check_collision(q) is None,
                 timeout_s=self.config.planner.rrt_timeout_s,diagnostics=diagnostics)
            self.last_diagnostics.append({'route':'rrt','queries':diagnostics})
            if plan is not None: return plan.as_arrays()
        self.last_failure="PATH_BLOCKED"; return None

    def plan_to_pose(self,pose,carrying=False,cartesian_corridor_m=None,start_joints=None,prefer_roomy=False):
        self.last_failure=None;self.last_diagnostics=[]
        if not np.all((pose.position>=np.array(self.config.safety.workspace)[:,0]) & (pose.position<=np.array(self.config.safety.workspace)[:,1])):
            self.last_failure="COLLISION_RISK"; return None
        start=self.robot.get_joint_positions() if start_joints is None else np.asarray(start_joints)
        start_collision=self.robot.check_collision(start)
        if start_collision is not None:
            self.last_failure="PATH_BLOCKED"
            self.last_diagnostics.append({'route':'start','reason':start_collision})
            return None
        solutions=self.ik.solve(pose,start)
        if not solutions:
            # IK seeded from a cramped current posture can come back empty although the pose is reachable
            # from the roomy middle of the joint range. Re-seed there; the path search below handles the rest.
            solutions=self.reseeded_solutions(pose)
            if solutions:self.last_diagnostics.append({'route':'reseeded_ik','solutions':len(solutions)})
        if not solutions: self.last_failure="IK_FAILED"; return None
        start_position=(self.robot.get_ee_pose().position if start_joints is None
                        else self.robot.forward_kinematics(start).position)
        def accepted(path):
            from .trajectory_executor import retime
            velocity=self.config.safety.carrying_velocity if carrying else self.config.safety.max_velocity
            positions,_=retime(path,velocity)
            if any(self.robot.check_collision(p) is not None for p in positions):return False
            if cartesian_corridor_m is None:return True
            # A short Cartesian command must not become a multi-metre RRT
            # excursion that shakes a friction-held object loose.
            segment=pose.position-start_position
            denom=max(float(segment@segment),1e-12)
            stride=max(1,len(positions)//100)
            for q in [*positions[::stride],positions[-1]]:
                point=self.robot.forward_kinematics(q).position
                alpha=np.clip(float((point-start_position)@segment)/denom,0,1)
                if np.linalg.norm(point-(start_position+alpha*segment))>cartesian_corridor_m:return False
            return True
        lower,upper=self.robot.joint_limits
        def crowded(q):return float(np.min(np.minimum(q-lower,upper-q)))<.4
        if prefer_roomy and not carrying and cartesian_corridor_m is None:
            # An explicit re-centring move may change arm branch. After a side
            # observation the arm can sit near its joint stops, where the later
            # Cartesian descent and transport have no room. Search the roomy
            # postures completely (direct and RRT) before settling for others.
            ranked=sorted(solutions,key=lambda q:np.linalg.norm(q-start))
            groups=[[q for q in ranked if not crowded(q)],[q for q in ranked if crowded(q)]]
        else:
            groups=[sorted(solutions,key=lambda q:np.linalg.norm(q-start))]
        # A direct path rejected by the dense execution check must not be
        # returned again in the RRT pass. Otherwise no detour is searched.
        for ordered in groups:
            for q in ordered:
                path=self.plan_joint_path(start,q,allow_rrt=False)
                if path and accepted(path):return path
                self.last_diagnostics.append({'route':'direct','reason':'dense_rejected' if path else 'coarse_blocked'})
            for q in ordered:
                path=self.plan_joint_path(start,q,allow_direct=False)
                if path and accepted(path):return path
                if path:self.last_diagnostics.append({'route':'rrt','reason':'execution_rejected'})
        via=self.route_through_posture(start,pose,solutions,accepted)
        if via is not None:return via
        self.last_failure='PATH_BLOCKED'
        return None

    def reseeded_solutions(self,pose):
        postures=getattr(self.robot,'via_postures',None)
        if postures is None:return []
        found=[]
        for via in postures(pose.position[:2]):
            for q in self.ik.solve(pose,via):
                if not any(np.linalg.norm(q-other)<1e-3 for other in found):found.append(q)
        return found

    def route_through_posture(self,start,pose,solutions,accepted):
        """Last resort for a blocked move: go through a calibrated roomy posture. The
        arm leaves a crowded branch for the open middle of its range and then goes on
        to the goal, instead of giving up on the one direct way."""
        postures=getattr(self.robot,'via_postures',None)
        if postures is None:return None
        for via in postures(pose.position[:2]):
            if self.robot.check_collision(via) is not None:continue
            first=self.plan_joint_path(start,via)
            if first is None:continue
            for q in sorted(solutions,key=lambda q:np.linalg.norm(q-via))[:4]:
                second=self.plan_joint_path(via,q)
                if second is None:continue
                path=[*list(first),*list(second)[1:]]
                if accepted(path):
                    self.last_diagnostics.append({'route':'via_posture','posture':[round(float(v),3) for v in via]})
                    return path
        return None

    def preview_route(self,waypoints,carrying=False,cartesian_corridor_m=None,step_m=None,budget=None):
        """Check the same short Cartesian commands without moving the robot."""
        joints=self.robot.get_joint_positions().copy()
        step=self.config.planner.cartesian_step_m if step_m is None else step_m
        lower,upper=self.robot.joint_limits
        minimum_margin=float('inf');near_limit_segments=0
        for waypoint_index,waypoint in enumerate(waypoints):
            origin=self.robot.forward_kinematics(joints).position
            count=max(1,int(np.ceil(np.linalg.norm(waypoint.position-origin)/step)))
            for segment_index in range(1,count+1):
                if budget is not None:budget()
                point=origin+(waypoint.position-origin)*(segment_index/count)
                path=self.plan_to_pose(Pose(point,waypoint.rotation),carrying=carrying,
                    cartesian_corridor_m=cartesian_corridor_m,start_joints=joints)
                if path is None:
                    return False,{'waypoint':waypoint_index,'segment':segment_index,
                        'requested_position':point.tolist(),'failure':self.last_failure,
                        'diagnostics':self.last_diagnostics[-6:]}
                joints=np.asarray(path[-1]).copy()
                if waypoint_index>0:
                    margin=float(np.min(np.minimum(joints-lower,upper-joints)))
                    minimum_margin=min(minimum_margin,margin)
                    near_limit_segments+=margin<.025
        return True,{'waypoints':len(waypoints),'final_joints':joints.tolist(),
            'minimum_joint_margin_rad':minimum_margin,
            'near_joint_limit_segments':near_limit_segments}

    def check_collision(self,joints): return self.robot.check_collision(joints)
    def validate_trajectory(self,path):
        return all(self.robot.check_collision(q) is None for q in path)
