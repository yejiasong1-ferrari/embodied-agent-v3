"""Safe sensor poses generated locally from measured targets and robot calibration."""
import numpy as np
from robot.robot_interface import Pose


class ViewpointPlanner:
    def __init__(self,config):self.config=config
    def candidates(self,target,camera,side=False):
        distance=self.config.localization.observation_distance_m
        # Top-oblique followed by opposing sides. The object geometry, never
        # a named simulator fixture, is the pivot for every view.
        offsets=[np.array([0,0,distance]),np.array([.28,0,.22]),np.array([-.28,0,.22]),np.array([0,-.28,.22]),np.array([0,.28,.22])]
        if side:
            # Try higher obliques before low side poses: a low wrist approach
            # can bring the fingers near adjacent objects unnecessarily.
            offsets=offsets[1:]+[np.array([.28,0,.10]),np.array([-.28,0,.10]),np.array([0,-.28,.10]),np.array([0,.28,.10])]
        for offset in offsets:
            location=target.position+offset;forward=target.position-location;forward/=np.linalg.norm(forward)
            up=np.array([0.,1.,0.]) if abs(forward[2])>.98 else np.array([0.,0.,1.])
            right=np.cross(forward,up);right/=np.linalg.norm(right);down=np.cross(forward,right)
            transform=np.eye(4);transform[:3,:3]=np.column_stack([right,down,forward]);transform[:3,3]=location
            inverse=np.linalg.inv(camera.ee_to_camera)
            hand=transform@inverse
            if np.linalg.norm(camera.ee_to_camera[:2,3])>.01:
                # A camera mounted beside the hand leaves the hand displaced from
                # the camera. Turn the view about its own optical axis so the hand
                # sits where the arm reaches comfortably, not folded onto the base.
                best=None
                for turn in (0,np.pi/2,np.pi,3*np.pi/2):
                    c,s=np.cos(turn),np.sin(turn);spin=np.array([[c,-s,0],[s,c,0],[0,0,1]])
                    rotated=transform.copy();rotated[:3,:3]=transform[:3,:3]@spin
                    candidate=rotated@inverse
                    score=abs(float(np.linalg.norm(candidate[:2,3]))-self.config.planner.transfer_radius_m)
                    if best is None or score<best[0]-1e-6:best=(score,candidate)
                hand=best[1]
            yield Pose(hand[:3,3],hand[:3,:3])
