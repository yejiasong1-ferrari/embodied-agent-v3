"""V3: only image-backed model evidence may authorize a physical referent.

Local pixels/depth propose and measure geometry; they never certify identity.
All model inputs/outputs and the actual input RGB-D calibration are persisted.
"""
import hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from llm.schemas import VisualIdentity,Reference
from llm.llm_client import ModelRequestError
from world.world_model import AmbiguousReferenceError,ReferenceNotFoundError
from recovery.failure_classifier import SkillFailure
from .object_detector import Detection
from .depth_components import split_mask_3d
from .identity_guidance import reference_guidance
from .pointcloud import pixel_to_world


def reference_key(ref):return (ref.description,ref.color,ref.category)


def prompted_depth_mask(frame,bbox,min_pixels=30,prior=None):
    """Box-prompted depth components, independent of colour vocabulary.

    This is a conservative baseline, not a claimed SAM installation. If a box
    contains competing similarly sized depth components, refuse geometry.
    """
    h,w=frame.depth.shape;x0,y0,x1,y1=bbox
    gate=np.zeros((h,w),bool);gate[y0:y1,x0:x1]=True
    gate &= np.isfinite(frame.depth)&(frame.depth>.025)&(frame.depth<3.)
    if frame.robot_mask is not None:gate &= ~frame.robot_mask
    v,u=np.nonzero(gate)
    if len(u)<min_pixels:return None
    pts=pixel_to_world(np.column_stack((u,v)),frame.depth[v,u],frame.intrinsics,frame.camera_to_world)
    gate[v,u] &= pts[:,2]>.009
    if prior is not None:
        # The model confirms identity; existing measured volume separates the
        # referent from a different object occluding it (e.g. cup over plate).
        # No missing surface or depth is synthesized.
        padding=max(.012,min(.025,float(prior.uncertainty_m)))
        gate[v,u] &= np.all((pts>=prior.low-padding)&(pts<=prior.high+padding),axis=1)
    parts=split_mask_3d(frame,gate,min_pixels)
    parts.sort(key=np.count_nonzero,reverse=True)
    if prior is not None and parts:
        # Occlusion may split a previously measured surface into crescents.
        # Union only observed pieces inside that object's measured volume.
        return np.logical_or.reduce(parts)
    if not parts or (len(parts)>1 and parts[1].sum()>.45*parts[0].sum()):return None
    return parts[0]


def appearance(rgb,bbox,mask=None):
    x0,y0,x1,y1=map(int,bbox);patch=rgb[y0:y1,x0:x1]
    if not patch.size:return None
    hsv=cv2.cvtColor(patch,cv2.COLOR_RGB2HSV)
    selection=None if mask is None else mask[y0:y1,x0:x1].astype(np.uint8)
    hist=cv2.calcHist([hsv],[0],selection,[18],[0,180])
    return cv2.normalize(hist,hist).flatten()


class VisualIdentityResolver:
    def __init__(self,perception):
        self.p=perception;self.calls=0;self.anchors={};self.ledger={}
        from .candidate_comparison import CandidateComparison
        self.comparison=CandidateComparison(self)

    def charge(self,ref,count=1):
        """Count one model request against the referent that caused it."""
        self.calls+=count
        key=reference_key(ref);self.ledger[key]=self.ledger.get(key,0)+count

    def spent(self,ref):
        """Requests this referent has used. Each referent has its own allowance,
        so a long search for the object cannot starve confirming the target.
        Calls recorded without a referent count against every referent."""
        attributed=sum(self.ledger.values())
        return min(self.ledger.get(reference_key(ref),0),self.calls)+max(0,self.calls-attributed)

    @staticmethod
    def prior_review(evidence,object_id):
        """The comparison's own verdict on the selected object, as context only."""
        label=next((k for k,v in evidence.get('label_to_object_id',{}).items() if v==object_id),None)
        row=next((r for r in evidence.get('reviews',[]) if r.get('candidate_id')==label),None)
        return None if row is None else {k:row.get(k) for k in ('verdict','confidence','evidence')}

    def can_spend(self,ref,count=1):
        return self.spent(ref)+count<=self.p.config.localization.max_vlm_calls

    def event(self,status,ref,**extra):
        self.p.log.event('visual_identity',{'status':status,'query':ref.description,**extra})

    def revoke(self,key,reason):
        self.p.world.reference_bindings.pop(key,None)
        self.p.world.reference_binding_evidence.pop(key,None)
        self.anchors.pop(key,None)
        self.p.log.event('identity_revoked',{'query':key[0],'reason':reason})

    def publish(self,key,anchor,frame):
        age=max(0.,float(frame.timestamp-anchor.get('last_observed',frame.timestamp)))
        uncertainty=anchor.get('uncertainty_m',.01)+age*self.p.config.localization.identity_uncertainty_growth_m_s
        anchor['age_s']=age;anchor['position_uncertainty_m']=uncertainty
        self.p.log.event('identity_state',{'query':key[0],'object_id':anchor['id'],
            'identity':'review_required' if anchor.get('review_required') else 'vlm_confirmed',
            'visibility':anchor.get('visibility','visible'),'age_s':age,
            'position_uncertainty_m':uncertainty})

    def check_continuity(self,objects,frame):
        self.comparison.remember(objects,frame)
        from itertools import product
        from .pointcloud import world_to_pixel
        for key,anchor in list(self.anchors.items()):
            if getattr(self,'confirming',False):continue
            current=next((o for o in objects if o.id==anchor['id']),None)
            if current is None:
                old=self.p.world.objects.get(anchor['id']);visibility='unexpected_missing'
                if old is not None:
                    with np.errstate(divide='ignore',invalid='ignore'):
                        pixels,z=world_to_pixel(np.array(list(product(*zip(old.low,old.high)))),frame)
                    h,w=frame.rgb.shape[:2]
                    if np.all(z<=0):visibility='out_of_view'
                    elif np.all(z>0) and np.all(np.isfinite(pixels)):
                        lo=pixels.min(0);hi=pixels.max(0)
                        inside=np.prod(np.maximum(0,np.minimum(hi,[w,h])-np.maximum(lo,0)))
                        fraction=inside/max(1.,float(np.prod(hi-lo)))
                        if fraction==0:visibility='out_of_view'
                        elif fraction<.85:visibility='partially_out_of_view'
                        else:
                            x0,y0=np.maximum(lo,0).astype(int);x1,y1=np.minimum(hi,[w,h]).astype(int)
                            depth=frame.depth[y0:y1,x0:x1]
                            valid=np.isfinite(depth)&(depth>0)
                            if np.any(valid) and np.mean(depth[valid]<np.min(z)-.02)>.5:visibility='occluded'
                anchor['visibility']=visibility
                # Visibility loss preserves historical identity evidence, but
                # never manufactures fresh position measurements.
                self.publish(key,anchor,frame);continue
            distance=float(np.linalg.norm(current.position-anchor['position']))
            hist=getattr(current,'identity_appearance',None)
            if hist is None and current.bbox:hist=appearance(frame.rgb,current.bbox)
            similarity=0. if hist is None else float(cv2.compareHist(anchor['appearance'],hist,cv2.HISTCMP_CORREL))
            held=self.p.world.memory.held_id==current.id
            fragments=set(self.p.world.memory.held_fragments) if held else set()
            rivals=[o for o in objects if o.id!=current.id and o.color==current.color and o.id not in fragments and
                    np.linalg.norm(o.position-current.position)<.10]
            # A neighbouring colour component is not an alternative identity.
            # Only contenders inside the predicted body's association envelope
            # compete. Handles outside it remain collision geometry, never get
            # merged or silently promoted to semantic identities here.
            radius=float(np.clip(.5*np.linalg.norm(current.size[:2]),.025,.05))
            reviewed=anchor.get('reviewed_neighbours',{})
            competitors=[o for o in rivals if np.linalg.norm(o.position-anchor['position'])<=radius
                and not (o.id in reviewed and np.linalg.norm(o.position-reviewed[o.id])<.012
                         and distance<.012 and similarity>=.35)]
            # The same tracked body at the same spot with no rival nearby changed only its
            # look: a new viewing angle turns a yellow top into a shaded brown side. Colour
            # histograms are view dependent, so only a gross mismatch counts there.
            if held:competitors=[]  # the body in the gripper has no rival; a handle seen separately is its own part
            just_released=self.p.world.memory.last_released_id==current.id
            stationary=distance<.015 and not competitors
            # We just put this same tracked body down. The centimetre-scale
            # displacement gate would otherwise demand a new identity review.
            conflict=(distance>.07 and not held and not just_released) or similarity<(-.3 if stationary else .35) or bool(competitors)
            if conflict and not anchor.get('review_required'):
                anchor['review_required']=True
                anchor['review_reason']={'similarity':similarity,'displacement_m':distance,'rival_ids':[o.id for o in competitors]}
                self.p.log.event('identity_review_required',{'query':key[0],'object_id':current.id,
                    'reason':'appearance, displacement or instance association conflict',
                    'similarity':similarity,'displacement_m':distance,'rival_ids':[o.id for o in competitors]})
            anchor['visibility']='visible'
            if not anchor.get('review_required'):
                anchor['position']=current.position.copy();anchor['last_observed']=frame.timestamp
                anchor['uncertainty_m']=current.uncertainty_m
                if just_released:self.p.world.memory.last_released_id=None
            self.publish(key,anchor,frame)

    def valid(self,ref,fresh=True):
        key=reference_key(ref);world=self.p.world
        obj=world.objects.get(world.reference_bindings.get(key));anchor=self.anchors.get(key,{})
        evidence=world.reference_binding_evidence.get(key,{})
        certified=(obj is not None and bool(anchor) and not anchor.get('review_required') and
                   evidence.get('source')=='vlm_identity' and bool(evidence.get('image_sha256')))
        if not certified:return False
        if fresh:return obj.visible and anchor.get('visibility','visible')=='visible'
        # The body in the closed gripper cannot be anyone else; the wrist camera often loses sight of it.
        if world.memory.held_id==obj.id:return True
        cfg=self.p.config.localization
        now=self.p.last_frame.timestamp if getattr(self.p,'last_frame',None) is not None else world.timestamp
        age=max(0.,float(now-anchor.get('last_observed',now)))
        uncertainty=anchor.get('uncertainty_m',obj.uncertainty_m)+age*cfg.identity_uncertainty_growth_m_s
        return age<=cfg.identity_memory_s and uncertainty<=cfg.identity_memory_uncertainty_m

    def require_all(self,goal):
        self.require_references([goal.object,*goal.references],fresh=True)

    def require_references(self,references,fresh=False):
        seen={}
        for ref in references:
            if ref.category=='region':
                region=self.p.world.resolve(ref,require_visible=True)
                if region.evidence_source!='observed_support_region':
                    raise SkillFailure('LOW_CONFIDENCE',{'reason':'unobserved placement region'})
                continue
            key=reference_key(ref);identity=self.p.world.reference_bindings.get(key)
            if identity is not None and identity in seen and seen[identity]!=key:
                raise SkillFailure('AMBIGUOUS_TARGET',{'reason':'distinct references resolved to the same instance'})
            seen[identity]=key
            if not self.valid(ref,fresh=fresh):
                anchor=self.anchors.get(key,{})
                raise SkillFailure('TARGET_LOST',{'reason':'identity review or fresh observation required',
                    'reference':ref.description,'visibility':anchor.get('visibility','unconfirmed'),
                    'identity_review_required':anchor.get('review_required',False),'fresh_required':fresh,
                    'reason_code':'IDENTITY_REVIEW_REQUIRED' if anchor.get('review_required') else 'FRESH_OBSERVATION_REQUIRED',
                    'review':anchor.get('review_reason',{})})

    def require_phase(self,c):
        if getattr(c,'observation_motion',False):
            # Observation motion authorizes no grasp/release and still passes
            # the same local trajectory, payload and collision validation.
            return
        if c.current_skill in {'RETREAT','VERIFY_TASK'} and not c.held:
            # After release, retreat/final inspection are observation actions.
            # Final acceptance is still gated by fresh model-backed identities.
            return
        source_only={'APPROACH_OBJECT','GRASP_OBJECT','VERIFY_GRASP','LIFT_OBJECT'}
        if c.current_skill in source_only:
            self.require_references([c.goal.object],fresh=not c.held)
            # Other referents must have model evidence and bounded memory.
            self.require_references(c.goal.references,fresh=False)
        elif c.current_skill=='MOVE_TO_TARGET':
            self.require_references([c.goal.object,*c.goal.references],fresh=False)
        elif c.current_skill=='PLACE_OBJECT':
            self.require_references([c.goal.object],fresh=False)
            if getattr(c,'placement_reference_checked',False):
                self.require_references(c.goal.references,fresh=False)
                now=self.p.last_frame.timestamp
                for ref in c.goal.references:
                    if ref.category=='region':continue
                    anchor=self.anchors[reference_key(ref)]
                    if now-anchor['last_observed']>self.p.config.localization.placement_reference_max_age_s:
                        raise SkillFailure('TARGET_LOST',{'reference':ref.description,'reason':'placement observation lease expired'})
        elif c.current_skill in {'RETREAT','VERIFY_TASK'}:
            self.require_references([c.goal.object,*c.goal.references],fresh=False)
        else:self.require_all(c.goal)

    def review_context(self,ref,frame):
        """Current-view geometry is context, not semantic confirmation."""
        key=reference_key(ref);anchor=self.anchors.get(key)
        if not anchor:return None
        from .pointcloud import world_to_pixel
        candidates=[]
        for obj in self.p.world.objects.values():
            if not obj.visible or np.linalg.norm(obj.position-anchor['position'])>.15:continue
            uv,z=world_to_pixel([obj.position],frame)
            if z[0]>0:
                candidates.append({'tracking_id':obj.id,'center_pixel':uv[0].tolist(),
                    'measured_size_m':obj.size.tolist()})
        return {'previous_tracking_id':anchor['id'],
            'previous_model_evidence':self.p.world.reference_binding_evidence.get(key,{}).get('reason'),
            'conflict':anchor.get('review_reason',{}),'current_candidates':candidates,
            'instruction':'Reidentify the previously confirmed instance in the current image. Nearby regions may be object parts or independent objects. Tracking IDs and measurements are proposals, not identity facts. If multiple independent objects fit and continuity cannot distinguish them, return ambiguous.'}

    def reconcile_after_scan(self,c):
        """Resolve newly raised semantic conflicts before spending on grasp plans."""
        key=reference_key(c.goal.object)
        if self.anchors.get(key,{}).get('review_required'):
            c.log.event('stage',{'stage':'POST_SCAN_IDENTITY_REVIEW','object_id':c.source_id})
            try:checked=self.identify(c.goal.object,allow_search=False)
            except (AmbiguousReferenceError,ReferenceNotFoundError) as exc:
                raise SkillFailure('TARGET_LOST',{'reason_code':'IDENTITY_REVIEW_REQUIRED',
                    'reason':'post-scan identity unresolved','reference':c.goal.object.description,
                    'identity_review_required':True,'review':self.anchors.get(key,{}).get('review_reason',{})}) from exc
            if checked.id!=c.source_id:raise SkillFailure('AMBIGUOUS_TARGET',{'reason':'instance changed after scan'})
            c.refresh()
        self.require_references([c.goal.object],fresh=True)

    def in_frame(self,frame,object_id):
        """True when most of the tracked object's projected box lies inside the frame."""
        from itertools import product
        from .pointcloud import world_to_pixel
        selected=self.p.world.objects.get(object_id)
        if selected is None:return False
        h,w=frame.rgb.shape[:2]
        with np.errstate(divide='ignore',invalid='ignore'):
            uv,z=world_to_pixel(np.array(list(product(*zip(selected.low,selected.high)))),frame)
        if not (np.all(z>0) and np.all(np.isfinite(uv))):return False
        raw=np.r_[uv.min(0),uv.max(0)];area=max((raw[2]-raw[0])*(raw[3]-raw[1]),1e-9)
        inside=max(min(raw[2],w)-max(raw[0],0),0)*max(min(raw[3],h)-max(raw[1],0),0)
        return inside/area>=.5

    def identify(self,ref,*,allow_search=True):
        p=self.p
        if ref.category=='region':return p.world.resolve(ref,require_visible=True)
        if self.valid(ref):return p.world.resolve(ref)
        if not p.client.online:
            self.event('model_unavailable',ref)
            raise SkillFailure('LOW_CONFIDENCE',{'reason':'V3 vision model unavailable; identity fallback forbidden'})
        # New evidence is never suppressed by the old size/view signature cache.
        previous_id=p.world.reference_bindings.get(reference_key(ref))
        original_id=previous_id;comparison_evidence=None
        try:choice=self.comparison.choose(ref,previous_id,allow_search)
        except SkillFailure:raise
        except ModelRequestError as exc:
            raise SkillFailure('LOW_CONFIDENCE',{'reason_code':exc.reason_code,
                'http_status':exc.http_status,'terminal':not exc.retryable,
                'reason':'candidate comparison request rejected'}) from exc
        except RuntimeError as exc:
            exhausted='budget exhausted' in str(exc)
            raise SkillFailure('LOW_CONFIDENCE',{'reason':'candidate comparison model request failed','terminal':True,'cause':str(exc)[:300],
                **({'reason_code':'IDENTITY_BUDGET_EXHAUSTED','reference':ref.description} if exhausted else {})}) from exc
        if choice is not None:
            chosen,comparison_evidence=choice;previous_id=chosen['id']
            if allow_search:
                if chosen['view'].startswith('identity_side:'):
                    context=getattr(self,'observation_context',None)
                    if context is None:raise AmbiguousReferenceError('selected side view needs safe observation context')
                    from skills.identity_observation import restore_identity_camera
                    if not restore_identity_camera(context,chosen['camera_pose']):
                        raise SkillFailure('PATH_BLOCKED',{'reason':'cannot safely revisit selected side view'})
                else:
                    try:view_index=int(chosen['view'].rsplit(':',1)[1])
                    except (ValueError,IndexError):raise AmbiguousReferenceError('candidate observation view unavailable')
                    p.camera.update_collision_world(p.world)
                    if not p.camera.move_viewpoint(view_index):raise SkillFailure('PATH_BLOCKED',{'reason':'cannot revisit selected candidate'})
                    # A scan label only names the last preset view requested; the frame may
                    # have been taken after an approach move. If the preset pose lacks the
                    # selected body, return to the exact pose the evidence came from.
                    context=getattr(self,'observation_context',None)
                    if context is not None and chosen.get('camera_pose') is not None and not self.in_frame(p.camera.capture(),previous_id):
                        from skills.identity_observation import restore_identity_camera
                        try:restore_identity_camera(context,chosen['camera_pose'])
                        except SkillFailure as exc:
                            if exc.failure.value in {'COLLISION_RISK','USER_STOP'}:raise
            self.event('candidate_selected',ref,object_id=previous_id)
        last='not_visible';detail=False
        for index in range(min(p.camera.viewpoint_count,p.config.localization.max_search_views) if allow_search else 1):
            p.camera.update_collision_world(p.world)
            if index and not p.camera.move_viewpoint(index):continue
            for resolution in range(2):
                if not self.can_spend(ref):
                    self.event('budget_exhausted',ref)
                    raise SkillFailure('LOW_CONFIDENCE',{'reason':'V3 identity call budget exhausted','reason_code':'IDENTITY_BUDGET_EXHAUSTED',
                        'reference':ref.description,'terminal':True})
                detail=resolution==1
                frame=p.camera.capture_detail() if detail else p.camera.capture()
                h,w=frame.rgb.shape[:2]
                expected=(p.config.camera.detail_height,p.config.camera.detail_width) if detail else (p.config.camera.height,p.config.camera.width)
                if (h,w)!=expected:raise ValueError('identity input resolution does not match camera calibration')
                name=f'identity_{self.calls:03d}_{"detail" if detail else "scan"}'
                path=Path(p.log.path)/'images'/f'{name}.png';Image.fromarray(frame.rgb).save(path)
                np.savez_compressed(path.with_suffix('.npz'),depth=frame.depth,intrinsics=frame.intrinsics,camera_to_world=frame.camera_to_world,timestamp=frame.timestamp)
                request_path=path.with_suffix('.jpg')
                Image.fromarray(frame.rgb).save(request_path,format='JPEG',quality=88)
                digest=hashlib.sha256(request_path.read_bytes()).hexdigest()
                selected_context=None;in_view=True
                if comparison_evidence is not None and previous_id in p.world.objects:
                    from itertools import product
                    from .pointcloud import world_to_pixel
                    selected=p.world.objects[previous_id]
                    with np.errstate(divide='ignore',invalid='ignore'):
                        uv,z=world_to_pixel(np.array(list(product(*zip(selected.low,selected.high)))),frame)
                    if np.all(z>0) and np.all(np.isfinite(uv)):
                        raw=np.r_[uv.min(0),uv.max(0)];area=max((raw[2]-raw[0])*(raw[3]-raw[1]),1e-9)
                        inside=max(min(raw[2],w)-max(raw[0],0),0)*max(min(raw[3],h)-max(raw[1],0),0)
                        in_view=inside/area>=.5
                    if in_view and np.all(z>0) and np.all(np.isfinite(uv)):
                        selected_context={'projected_bbox_normalized':np.clip(np.r_[uv.min(0),uv.max(0)]/np.array([w,h,w,h]),0,1).tolist(),
                            'prior_review':self.prior_review(comparison_evidence,previous_id),
                            'instruction':'This is the measured projection of the candidate selected in the comparison, not proof of identity. Verify its whole shape and colour against the user description in this fresh image. prior_review is the earlier close-up judgement of this same region, given as context: judge by the whole body together with its attachments, because a side attachment seen from a new angle changes the silhouette. Return ambiguous if the fresh image contradicts the description or another object fits better; do not silently substitute a different instance.'}
                if not in_view:
                    # Asking the model about a frame that mostly lacks the selected body only
                    # burns the identity budget; look from another pose instead.
                    self.event('candidate_out_of_view',ref,view=frame.view_id,object_id=previous_id)
                    last='not_visible'
                    break
                self.charge(ref)
                try:
                    answer=p.client.complete(VisualIdentity,'visual_identity',{
                        'user_command':p.task_description,'reference':ref.model_dump(),
                        'image_width':w,'image_height':h,'view_id':frame.view_id,
                        'continuity_review':self.review_context(ref,frame),
                        'selected_candidate':selected_context,
                        'reference_guidance':reference_guidance(ref.description),
                        'coordinate_format':'Prefer coordinate_space=normalized, bbox decimals 0..1. Example [0.10,0.20,0.35,0.60]. If using 0..1000 declare grid1000; actual image pixels declare pixel. Never label pixel coordinates normalized.',
                        'instruction':'Find the physical referent in this full image. Return normalized bbox only for a uniquely confirmed instance. Everyday naming may differ; explicit user attributes must still hold. If two objects fit, return ambiguous. Never choose merely because only one colour component exists.'},image=frame.rgb)
                except ModelRequestError as exc:
                    self.event('model_error',ref,image=str(path.relative_to(p.log.path)),http_status=exc.http_status)
                    if not exc.retryable:
                        raise SkillFailure('LOW_CONFIDENCE',{'reason_code':exc.reason_code,
                            'http_status':exc.http_status,'terminal':True,
                            'reason':'vision service rejected request; repeated observation cannot repair it'}) from exc
                    if resolution==0:continue
                    raise SkillFailure('LOW_CONFIDENCE',{'reason_code':exc.reason_code,
                        'http_status':exc.http_status,'reason':'vision service temporarily unavailable'}) from exc
                except RuntimeError:
                    self.event('model_error',ref,image=str(path.relative_to(p.log.path)))
                    if resolution==0:continue
                    raise SkillFailure('LOW_CONFIDENCE',{'reason':'V3 vision request failed after bounded retry'})
                last=answer.status
                self.event(last,ref,review=answer.model_dump(),image=str(path.relative_to(p.log.path)),image_sha256=digest,timestamp=frame.timestamp,resolution=[w,h])
                if last=='confirmed' and answer.confidence>=.85:
                    values=np.array(answer.bbox,float)
                    if answer.coordinate_space=='grid1000':values/=1000
                    if answer.coordinate_space!='pixel':values*=np.array([w,h,w,h])
                    box=np.rint(values).astype(int)
                    if box[2]>w or box[3]>h or box[0]<0 or box[1]<0 or box[2]<=box[0] or box[3]<=box[1]:
                        last='detail_needed';self.event('invalid_box',ref);continue
                    mask=prompted_depth_mask(frame,tuple(box),p.config.perception.min_pixels,
                        p.world.objects.get(previous_id) if previous_id is not None else None)
                    if mask is None:
                        independent=prompted_depth_mask(frame,tuple(box),p.config.perception.min_pixels) if previous_id else None
                        if independent is not None:
                            last='ambiguous';self.event('instance_conflict',ref,reason='new image proposal conflicts with selected candidate geometry')
                        else:
                            last='depth_unreliable';self.event(last,ref)
                        continue
                    # Local measurements may veto a contradiction, never authorize identity.
                    # A model hallucinating 'green' over blue pixels must not bind it.
                    if ref.color:
                        from .object_detector import color_masks
                        palettes=color_masks(frame.rgb)
                        expected_mask=palettes.get(ref.color)
                        if expected_mask is not None and np.count_nonzero(expected_mask & mask)<.15*np.count_nonzero(mask):
                            total=max(np.count_nonzero(mask),1)
                            last='detail_needed';self.event('attribute_conflict',ref,measured_fraction={k:round(float(np.count_nonzero(v&mask)/total),3) for k,v in palettes.items()});continue
                    facets=Reference(description=answer.label,color=answer.color,category=answer.category)
                    detection=Detection(tuple(box),facets.category or 'object',facets.color or 'unknown',answer.confidence,mask)
                    detection.query=ref.description
                    self.confirming=True
                    try:objects=p.observe(frame=frame,identity_detection=detection,identity_tracking_id=previous_id)
                    finally:self.confirming=False
                    observed_objects=objects
                    objects=[o for o in objects if getattr(o,'identity_prompt_match',o.query==ref.description)]
                    if len(objects)!=1:
                        last='depth_unreliable';self.event(last,ref);continue
                    obj=objects[0]
                    if comparison_evidence is None:
                        # A full-scene proposal must join the candidate route
                        # once its real pixels have produced a measured instance.
                        checked_choice=self.comparison.choose(ref,original_id,allow_search)
                        if checked_choice is not None:
                            selected,comparison_evidence=checked_choice
                            if selected['id']!=obj.id:
                                return self.identify(ref,allow_search=allow_search)
                    if previous_id is not None and obj.id!=previous_id:
                        last='ambiguous';self.event('instance_conflict',ref,previous_id=previous_id,proposed_id=obj.id);continue
                    other_bindings=[key for key,identity in p.world.reference_bindings.items()
                                    if identity==obj.id and key!=reference_key(ref)]
                    if other_bindings:
                        last='ambiguous';self.event('instance_conflict',ref);continue
                    # Binding is to a measured, tracked instance, not model coordinates.
                    evidence={'source':'vlm_identity','image':str(request_path.relative_to(p.log.path)),
                              'raw_image':str(path.relative_to(p.log.path)),
                              'image_sha256':digest,'timestamp':frame.timestamp,'view_id':frame.view_id,
                              'bbox':box.tolist(),'confidence':answer.confidence,'reason':answer.evidence,
                              'user_command':p.task_description,'candidate_comparison':comparison_evidence}
                    if original_id is not None and original_id!=obj.id:
                        p.log.event('identity_corrected',{'query':ref.description,'previous_id':original_id,'object_id':obj.id,
                            'comparison':comparison_evidence,'reason':'model rejected previous candidate; new instance freshly confirmed'})
                    p.world.bind_reference(ref,obj.id,evidence)
                    self.anchors[reference_key(ref)]={'id':obj.id,'position':obj.position.copy(),'appearance':getattr(obj,'identity_appearance',appearance(frame.rgb,obj.bbox or box)),
                        'last_observed':frame.timestamp,'uncertainty_m':obj.uncertainty_m,'visibility':'visible','review_required':False,
                        'reviewed_neighbours':{other.id:other.position.copy() for other in observed_objects
                            if other.id!=obj.id and np.linalg.norm(other.position-obj.position)<.10}}
                    p.log.event('target_binding',{'query':ref.description,'object_id':obj.id,**evidence})
                    return obj
                # Resolution cannot reveal an object outside the view or behind an occluder.
                # The exception is a candidate whose measured box projects inside this
                # frame: the object is in view, so a higher-resolution look can still
                # settle a shape that the wide frame rendered too small to judge.
                if last in {'not_visible','occluded'} and not (last=='not_visible' and selected_context is not None and resolution==0):break
        committed=self.commit_best_guess(ref,previous_id,comparison_evidence,last)
        if committed is not None:return committed
        if last=='not_visible':raise ReferenceNotFoundError(f'{ref.description}: not seen in searched views')
        raise AmbiguousReferenceError(f'{ref.description}: VLM identity unresolved ({last})')

    def commit_best_guess(self,ref,object_id,comparison_evidence,status):
        """Bind the candidate the comparison judged most probable when no frame could add a fresh
        confirmation. Colour measured from the pixels still vetoes it; nothing is guessed against
        the evidence, but a clean "cannot confirm" is not a reason to stop the task."""
        p=self.p;obj=p.world.objects.get(object_id) if object_id is not None else None
        if obj is None or comparison_evidence is None:return None
        row=self.comparison.gallery.get(object_id,{})
        if ref.color and float(row.get('colors',{}).get(ref.color,0))<.30:return None
        key=reference_key(ref)
        if any(identity==object_id and other!=key for other,identity in p.world.reference_bindings.items()):return None
        frame=getattr(p,'last_frame',None)
        histogram=getattr(obj,'identity_appearance',None)
        if histogram is None and frame is not None and obj.bbox:histogram=appearance(frame.rgb,obj.bbox)
        if histogram is None:return None
        evidence={'source':'vlm_identity','image':comparison_evidence['review_image'],'raw_image':comparison_evidence['review_image'],
                  'image_sha256':comparison_evidence['comparison_sha256'],'timestamp':float(obj.last_seen),'view_id':'candidate_comparison',
                  'bbox':[int(v) for v in obj.bbox] if obj.bbox else [0,0,0,0],'confidence':float(obj.semantic_confidence),
                  'reason':f'most probable candidate from comparison; fresh confirmation ended as {status}',
                  'user_command':p.task_description,'candidate_comparison':comparison_evidence,'committed_best_guess':True}
        p.world.bind_reference(ref,object_id,evidence)
        self.anchors[key]={'id':object_id,'position':obj.position.copy(),'appearance':histogram,
            'last_observed':float(obj.last_seen),'uncertainty_m':obj.uncertainty_m,'visibility':'visible','review_required':False,
            'reviewed_neighbours':{}}
        p.log.event('target_binding',{'query':ref.description,'object_id':object_id,**evidence})
        self.event('committed_best_guess',ref,object_id=object_id,final_status=status)
        return obj
