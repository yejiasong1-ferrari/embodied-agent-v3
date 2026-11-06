"""Footprints are reconstructed surface hulls; no simulator inputs."""
import cv2
import numpy as np


def round_body_uncovered(obj,poly):
    """A side attachment (handle) can widen a hull whose body half was never seen.
    Check the measured body circle itself: much of it outside the hull means missing surface."""
    angles=np.linspace(0,2*np.pi,24,endpoint=False)
    ring=obj.position[:2]+np.column_stack((np.cos(angles),np.sin(angles)))*np.asarray(obj.size[:2])/2*.9
    contour=poly.astype(np.float32).reshape(-1,1,2)
    outside=sum(cv2.pointPolygonTest(contour,(float(x),float(y)),False)<0 for x,y in ring)
    return outside/len(ring)>.2


def footprint(obj,position=None):
    points=getattr(obj,'points',np.empty((0,3)))
    verified_world=getattr(obj,'manipulation_geometry',{}).get('verified_footprint_world')
    verified=getattr(obj,'manipulation_geometry',{}).get('verified_footprint_offsets')
    carried=getattr(obj,'manipulation_geometry',{}).get('carried_footprint_offsets')
    if verified_world is not None:
        poly=np.asarray(verified_world,dtype=float)
        if position is not None:
            anchor=np.asarray(obj.manipulation_geometry['verified_footprint_anchor'],dtype=float)
            poly=poly+np.asarray(position[:2])-anchor
        return poly
    if verified is not None:
        poly=np.asarray(verified)+obj.position[:2]
    elif carried is not None:
        poly=np.asarray(carried)+obj.position[:2]
    elif len(points)>=8:
        poly=cv2.convexHull(np.asarray(points[:,:2],np.float32)).reshape(-1,2).astype(float)
        # A partial face is not the object's whole footprint. Box dimensions
        # retained from a complete RGB-D view bound its unseen opposite face.
        if obj.category=='box' and (np.any(np.ptp(poly,axis=0)<np.asarray(obj.size[:2])*.85)
                or abs(float(cv2.contourArea(poly.astype(np.float32))))<.85*float(np.prod(obj.size[:2]))):
            corners=obj.position[:2]+np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*obj.size[:2]/2
            poly=cv2.convexHull(np.concatenate([poly,corners]).astype(np.float32)).reshape(-1,2).astype(float)
        elif obj.category in ('cup','bottle') and (np.any(np.ptp(poly,axis=0)<np.asarray(obj.size[:2])*.85)
                or abs(float(cv2.contourArea(poly.astype(np.float32))))<.8*np.pi/4*float(np.prod(obj.size[:2]))
                or round_body_uncovered(obj,poly)):
            # A side view sees only part of a round body's outline. Add the
            # circle of the diameter already measured over earlier views, so
            # the unseen far side is not mistaken for missing support.
            angles=np.linspace(0,2*np.pi,48,endpoint=False)
            ring=obj.position[:2]+np.column_stack((np.cos(angles),np.sin(angles)))*np.asarray(obj.size[:2])/2
            poly=cv2.convexHull(np.concatenate([poly,ring]).astype(np.float32)).reshape(-1,2).astype(float)
        elif obj.category=='plate' and np.any(np.ptp(poly,axis=0)<np.asarray(obj.size[:2])*.85):
            # Keep the complete plate extent measured from an earlier view
            # when the carried object or fingers hide most of its surface.
            angles=np.linspace(0,2*np.pi,48,endpoint=False)
            poly=obj.position[:2]+np.column_stack((np.cos(angles),np.sin(angles)))*np.asarray(obj.size[:2])/2
    else:
        # WorldModel dimensions are already RGB-D estimates.
        half=np.asarray(obj.size[:2])/2
        poly=np.asarray(obj.position[:2])+np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*half
    if position is not None:poly=poly+np.asarray(position[:2])-obj.position[:2]
    return poly


def overlap(a,b):
    for polygon in (a,b):
        edges=np.roll(polygon,-1,axis=0)-polygon
        for edge in edges:
            axis=np.array([-edge[1],edge[0]])
            if np.linalg.norm(axis)<1e-8:continue
            av,bv=a@axis,b@axis
            if av.max()<bv.min() or bv.max()<av.min():return False
    return True


def ray_exit_distance(polygon,center,outward):
    """Measured footprint boundary on a ray through the object's centre."""
    center=np.asarray(center,float);outward=np.asarray(outward,float)
    hits=[]
    for start,end in zip(polygon,np.roll(polygon,-1,axis=0)):
        edge=end-start;matrix=np.column_stack([outward,-edge])
        if abs(np.linalg.det(matrix))<1e-10:continue
        distance,segment=np.linalg.solve(matrix,start-center)
        if distance>=0 and -.0001<=segment<=1.0001:hits.append(distance)
    return min(hits) if hits else None


def polygon_gap(a,b):
    if overlap(a,b):return 0.
    def distance(points,polygon):
        start=polygon;delta=np.roll(polygon,-1,axis=0)-polygon
        offset=points[:,None,:]-start[None,:,:]
        t=np.clip(np.sum(offset*delta,axis=2)/np.maximum(np.sum(delta**2,axis=1),1e-12),0,1)
        return np.linalg.norm(offset-t[...,None]*delta,axis=2).min()
    return float(min(distance(a,b),distance(b,a)))


def projected_edges(obj,axis,position=None):
    measured_position=obj.position if position is None else np.asarray(position)
    if abs(axis[2])>.9:
        values=(measured_position[2]+np.array([-.5,.5])*obj.size[2])*axis[2]
    else:values=footprint(obj,position)@axis[:2]
    return float(values.min()),float(values.max())


def support_covered(world,obj,position,margin=.025):
    """All footprint corners and edge midpoints need nearby observed support."""
    points=world.support_points
    if len(points)<9:return False
    half=np.max(np.abs(footprint(obj,position)-np.asarray(position[:2])),axis=0)+margin
    sample=np.asarray(position[:2])+np.array([(x,y) for x in (-1,0,1) for y in (-1,0,1)])*half
    distances=np.sqrt(np.min(np.sum((sample[:,None,:]-points[None,:,:2])**2,axis=2),axis=1))
    return bool(np.all(distances<.025))


def convex_overlap_area(subject,clipper):
    """Sutherland-Hodgman clip of one convex hull by another. cv2.intersectConvexConvex
    returns a small fraction of the true overlap for hulls with many near-duplicate points."""
    def counter_clockwise(hull):
        hull=np.asarray(hull,float)
        return hull if cv2.contourArea(hull.astype(np.float32),True)>=0 else hull[::-1]
    out=[tuple(q) for q in counter_clockwise(subject)];clipper=counter_clockwise(clipper)
    for a,b in zip(clipper,np.roll(clipper,-1,axis=0)):
        points=out;out=[]
        if not points:break
        edge=b-a
        side=lambda q:edge[0]*(q[1]-a[1])-edge[1]*(q[0]-a[0])
        for q,r in zip(points,points[1:]+points[:1]):
            sq,sr=side(q),side(r)
            if sq>=0:out.append(q)
            if (sq>=0)!=(sr>=0):
                t=sq/(sq-sr);out.append((q[0]+t*(r[0]-q[0]),q[1]+t*(r[1]-q[1])))
    return abs(float(cv2.contourArea(np.array(out,np.float32)))) if len(out)>2 else 0.


def rests_stably(base,polygon,centre=None):
    """A rest is stable without every corner inside: same-size blocks stacked with
    a small yaw offset overhang at the corners, yet stay put because most of the
    base is supported and the centre of mass sits well inside the support.
    Footprints come from partial point clouds that can be a centimetre off, so the
    measured centre of the payload, when known, is the centre of mass."""
    mine=cv2.convexHull(np.asarray(base,np.float32)).reshape(-1,2);polygon=cv2.convexHull(np.asarray(polygon,np.float32)).reshape(-1,2)
    area=abs(float(cv2.contourArea(mine)))
    if area<1e-9:return False
    shared=convex_overlap_area(mine,polygon)
    if centre is None:
        moments=cv2.moments(mine);centre=(moments['m10']/moments['m00'],moments['m01']/moments['m00'])
    depth=cv2.pointPolygonTest(polygon,(float(centre[0]),float(centre[1])),True)
    larger_support=abs(float(cv2.contourArea(polygon)))>=.7*area
    return bool(larger_support and shared/area>=.5 and depth>=.25*np.sqrt(area)/2)


def support_footprint(obj,position):
    radius=getattr(obj,'manipulation_geometry',{}).get('measured_base_radius_m')
    if radius is None:return footprint(obj,position)
    angles=np.linspace(0,2*np.pi,48,endpoint=False)
    return np.asarray(position[:2])+radius*np.column_stack((np.cos(angles),np.sin(angles)))


def supported_by(obj,support,position,margin=0.):
    """Check the measured base against the support; clearance still uses the whole object."""
    polygon=footprint(support).astype(np.float32)
    if abs(position[2]-obj.size[2]/2-support.high[2])>.01:return False
    base=support_footprint(obj,position)
    if all(cv2.pointPolygonTest(polygon,tuple(map(float,p)),True)>=margin for p in base):return True
    return margin<=0 and rests_stably(base,polygon,position[:2])
