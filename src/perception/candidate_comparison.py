"""Anonymous, image-backed alternatives for V3 semantic selection."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from llm.schemas import VisualDisambiguation
from .object_detector import color_masks
from .candidate_evidence import candidate_view_quality
from .identity_guidance import reference_guidance
from world.world_model import AmbiguousReferenceError

SINGLE_CANDIDATE_RULE=('This image contains ONLY ONE candidate. If VIEW 1 and VIEW 2 are shown, they are observations of the SAME '
    'tracked candidate at different camera poses, not alternatives; compare their visible structure and report uncertain if they '
    'conflict. Judge it independently; no other candidate is shown and its label gives no hint. Follow the guidance field for how '
    'to read this kind of reference. Return one candidate entry, yes only if the requested attributes fit, no for visible '
    'contradiction, uncertain if insufficient. Keep evidence under 220 characters. Use measured surface spans as supplementary '
    'evidence, acknowledging occlusion. Return actual verdict values, not a JSON schema.')


class CandidateNeedsViews(AmbiguousReferenceError):
    def __init__(self, candidate_ids, required_calls=None):
        self.candidate_ids=candidate_ids
        self.required_calls=len(candidate_ids)+2 if required_calls is None else required_calls
        super().__init__('candidate appearance unresolved; a new side observation is required')


class CandidateComparison:
    def __init__(self,resolver):
        self.resolver=resolver;self.gallery={};self.serial=0
        # Exclusions (no, confident) stay valid while a candidate's evidence is unchanged.
        self.settled={}
        self.last_quality_notice={}

    def quality_notice(self,obj,now):
        """Scope Studio notices to task referents; keep all raw measurements."""
        p=self.resolver.p
        bindings=getattr(getattr(p,'world',None),'reference_bindings',{})
        refs=[]
        for ref in getattr(p,'references',()):
            bound=bindings.get((ref.description,ref.color,ref.category))
            if (bound==obj.id or (bound is None and
                    (ref.color==getattr(obj,'color',None) if ref.color else
                     ref.category and ref.category==getattr(obj,'category',None)))):
                if ref.description not in refs:refs.append(ref.description)
        show=any(now-self.last_quality_notice.get(key,-float('inf'))>=15 for key in refs)
        if show:self.last_quality_notice.update({key:now for key in refs})
        return {'task_references':refs,'show_in_studio':show}

    @staticmethod
    def settled_key(ref,row):
        return (ref.description,ref.color,ref.category,row['id'],row['timestamp'],
                tuple(v['timestamp'] for v in row.get('evidence_views',[])[:1]))

    @staticmethod
    def visible_profile(obj):
        points=obj.points
        if len(points)<40:return None
        low,high=np.percentile(points[:,2],[2,98]);height=float(high-low)
        if height<.015:return None
        sections=[]
        for a,b in ((0,.25),(.25,.5),(.5,.75),(.75,1.01)):
            part=points[(points[:,2]>=low+a*height)&(points[:,2]<low+b*height)]
            if len(part)<10:continue
            widths=np.percentile(part[:,:2],95,axis=0)-np.percentile(part[:,:2],5,axis=0)
            sections.append({'relative_height_interval':[a,min(b,1.)],
                'visible_xy_span_m':widths.tolist(),'surface_samples':len(part)})
        return {'observed_height_m':height,'sections':sections,
            'limitation':'Measured visible RGB-D surface spans only. Occlusion and viewpoint can shrink a span. Missing sections are unknown, not absent; spans do not prove a category or identity.'}

    def remember(self,objects,frame):
        # Every crop comes from its own calibrated observation, never from a
        # remembered bbox applied to a different camera pose.
        scene=frame.rgb.copy()
        from .pointcloud import pixel_to_world
        yy,xx=np.indices(frame.depth.shape)
        valid=np.isfinite(frame.depth)&(frame.depth>.025)&(frame.depth<3)
        if frame.robot_mask is not None:valid &= ~frame.robot_mask
        xyz=np.full((*frame.depth.shape,3),np.nan)
        xyz[valid]=pixel_to_world(np.column_stack((xx[valid],yy[valid])),frame.depth[valid],frame.intrinsics,frame.camera_to_world)
        for obj in objects:
            if obj.bbox is None or obj.observation_clipped:continue
            x0,y0,x1,y1=map(int,obj.bbox);h,w=frame.rgb.shape[:2]
            # Include the previously measured body, not just a segmented cap.
            from itertools import product
            from .pointcloud import world_to_pixel
            with np.errstate(divide='ignore',invalid='ignore'):
                uv,z=world_to_pixel(np.array(list(product(*zip(obj.low,obj.high)))),frame)
            if np.all(z>0) and np.all(np.isfinite(uv)):
                lo=np.floor(uv.min(0)).astype(int);hi=np.ceil(uv.max(0)).astype(int)
                x0=min(x0,int(lo[0]));y0=min(y0,int(lo[1]));x1=max(x1,int(hi[0]));y1=max(y1,int(hi[1]))
            x0=max(0,x0-8);y0=max(0,y0-8);x1=min(w,x1+8);y1=min(h,y1+8)
            if x1<=x0 or y1<=y0:continue
            quality=candidate_view_quality(obj,frame,(x0,y0,x1,y1))
            if not quality['usable']:
                self.resolver.p.log.event('identity_evidence_rejected',{
                    'object_id':obj.id,'view':frame.view_id,
                    'reason':'occluded_or_insufficient_depth','quality':quality,
                    'previous_image_retained':obj.id in self.gallery,**self.quality_notice(obj,float(frame.timestamp))})
                continue
            patch=frame.rgb[y0:y1,x0:x1].copy()
            # Context crop can include neighbours; only this measured body's
            # depth-supported pixels contribute to candidate colour recall.
            points=xyz[y0:y1,x0:x1]
            body=np.all((points>=obj.low-.004)&(points<=obj.high+.004),axis=2)&(points[:,:,2]>.009)
            if body.sum()<30:continue
            fractions={k:float((v&body).sum()/body.sum()) for k,v in color_masks(patch).items()}
            old=self.gallery.get(obj.id)
            if old and np.linalg.norm(old['position']-obj.position)<.04:
                fractions={k:max(value,old['colors'].get(k,0)) for k,value in fractions.items()}
            self.gallery[obj.id]={'id':obj.id,'position':obj.position.copy(),'patch':patch,
                'scene':scene,'view':frame.view_id,'timestamp':float(frame.timestamp),
                'colors':fractions,'area':(x1-x0)*(y1-y0),'visible_geometry':self.visible_profile(obj),
                'camera_pose':frame.camera_to_world.copy(),'evidence_views':[],'quality':quality}
            if old and np.linalg.norm(old['position']-obj.position)<.025:
                history=old.get('evidence_views',[])
                old_pose=old.get('camera_pose')
                if old_pose is not None and (np.linalg.norm(old_pose[:3,3]-frame.camera_to_world[:3,3])>.04
                        or float(old_pose[:3,2]@frame.camera_to_world[:3,2])<.94):
                    history=[{k:old[k] for k in ('patch','view','timestamp','camera_pose')},*history]
                self.gallery[obj.id]['evidence_views']=[v for v in history if frame.timestamp-v['timestamp']<=120][:1]
        now=float(frame.timestamp)
        self.gallery={k:v for k,v in self.gallery.items() if now-v['timestamp']<=120}

    def candidates(self,ref,previous):
        world=self.resolver.p.world
        rows=[r for r in self.gallery.values() if r['id'] in world.objects and
              (not ref.color or r['colors'].get(ref.color,0)>=.08 or r['id']==previous)]
        if ref.color and len(rows)>1:
            # Measured colour is physical evidence. A candidate showing under half of
            # the best candidate's share of the requested colour, and under half of
            # its own surface, belongs to another colour and needs no model call.
            best=max(r['colors'].get(ref.color,0) for r in rows)
            rows=[r for r in rows if r['id']==previous or r['colors'].get(ref.color,0)>=.5
                  or r['colors'].get(ref.color,0)>=.5*best]
        # Several colour fragments can describe one body. Group only proposals
        # with nearly identical physical centres, without merging world objects.
        groups=[]
        for row in sorted(rows,key=lambda r:(r['id']==previous,r['area']),reverse=True):
            if any(np.linalg.norm(row['position']-r['position'])<.035 for r in groups):continue
            groups.append(row)
        return groups

    def prepare(self,ref,previous,rows=None):
        r=self.resolver;p=r.p;rows=self.candidates(ref,previous) if rows is None else rows
        if not rows:return None
        if len(rows)>4:raise ValueError('prepare one batch of at most four; use choose for all candidates')
        labels={chr(65+i):row for i,row in enumerate(rows)}
        sheet=Image.new('RGB',(1280,960),(20,24,29));draw=ImageDraw.Draw(sheet)
        for i,(label,row) in enumerate(labels.items()):
            x=i*1280//len(rows);width=1280//len(rows)
            draw.text((x+10,8),f'{label}: scene / view {row["view"]}',fill='white')
            for y,source,height in [(30,row['scene'],320),(390,row['patch'],510)]:
                tile=ImageOps.contain(Image.fromarray(source),(width-16,height))
                sheet.paste(tile,(x+(width-tile.width)//2,y))
            draw.text((x+10,365),f'{label}: object detail',fill='white')
        self.serial+=1;path=Path(p.log.path)/'images'/f'identity_comparison_{self.serial:03d}.jpg'
        sheet.save(path,quality=88)
        payload={'user_command':p.task_description,'query':ref.description,
            'candidate_ids':list(labels),'observations':[{'candidate_id':k,'view':v['view'],'timestamp':v['timestamp']} for k,v in labels.items()],
            **reference_guidance(ref.description),
            'instruction':'First compare every lettered candidate by whole-object shape and structure against the requested form/category, disregarding hue and brightness for this first comparison. Then verify explicit colour under illumination among shape-compatible candidates. If no shape is specified, use the stated attributes without inventing a shape. Scene above, corresponding object detail below; crops are independently scaled. Do not reject a shaded body merely for being darker than its lit top, or accept a coloured cap alone as the whole object colour. Apply the same criteria to all candidates. Return yes only for a uniquely justified referent satisfying the explicit attributes; when shape and colour conflict or views cannot decide, use uncertain. All no and multiple uncertain are valid. In each evidence, state visible shape first, colour check second, and any missing view; keep within 300 characters. No internal object labels are provided.'}
        return labels,sheet,payload,path

    def review_batch(self,ref,previous,rows):
        r=self.resolver;p=r.p
        independent=p.config.localization.independent_candidate_review and (len(rows)>1 or any(row.get('evidence_views') for row in rows))
        # Candidates already excluded on unchanged evidence need no new request.
        needed=sum(1 for row in rows if self.settled_key(ref,row) not in self.settled) if independent else 1
        if not r.can_spend(ref,needed):
            from recovery.failure_classifier import SkillFailure
            raise SkillFailure('LOW_CONFIDENCE',{'reason':'V3 identity call budget exhausted','terminal':True})
        labels,sheet,payload,path=self.prepare(ref,previous,rows)
        individual_images={}
        if independent:
            judgements=[]
            for label,row in labels.items():
                cached=self.settled.get(self.settled_key(ref,row))
                if cached is not None:
                    judgements.append(cached.model_copy(update={'candidate_id':label}))
                    p.log.event('identity_candidate_review',{'query':ref.description,'candidate_id':label,
                        'review':cached.model_dump(),'reused':True})
                    continue
                # One object per request prevents scene/crop and A/B feature
                # attribution errors. No internal category or expected answer
                # is supplied. Padding preserves the actual crop proportions.
                views=[row,*row.get('evidence_views',[])][:2]
                image=Image.new('RGB',(640*len(views),640),(20,24,29))
                draw=ImageDraw.Draw(image)
                for i,view in enumerate(views):
                    tile=ImageOps.contain(Image.fromarray(view['patch']),(624,600))
                    image.paste(tile,(i*640+(640-tile.width)//2,32+(600-tile.height)//2))
                    draw.text((i*640+12,8),f'VIEW {i+1} - SAME CANDIDATE',fill='white')
                single_path=path.with_name(path.stem+f'_{label}.jpg')
                image.save(single_path,quality=88)
                single_payload={'user_command':p.task_description,'query':ref.description,
                    'candidate_ids':[label],
                    'visible_rgbd_geometry':row.get('visible_geometry'),
                    'views':[{'view':v['view'],'timestamp':v['timestamp']} for v in views],
                    **reference_guidance(ref.description),
                    'instruction':SINGLE_CANDIDATE_RULE}
                r.charge(ref)
                part=p.client.complete(VisualDisambiguation,'visual_disambiguation',single_payload,image=np.asarray(image))
                if len(part.candidates)!=1 or part.candidates[0].candidate_id!=label:
                    raise AmbiguousReferenceError('single-object review must evaluate exactly the offered candidate')
                judgements.extend(part.candidates)
                if part.candidates[0].verdict=='no' and part.candidates[0].confidence>=.65:
                    self.settled[self.settled_key(ref,row)]=part.candidates[0]
                individual_images[label]={'image':str(single_path.relative_to(p.log.path)),
                    'sha256':hashlib.sha256(single_path.read_bytes()).hexdigest()}
                p.log.event('identity_candidate_review',{'query':ref.description,'candidate_id':label,
                    'review':part.candidates[0].model_dump(),**individual_images[label]})
            answer=VisualDisambiguation(candidates=judgements)
        else:
            r.charge(ref)
            answer=p.client.complete(VisualDisambiguation,'visual_disambiguation',payload,image=np.asarray(sheet))
        # Reject internally contradictory affirmative replies without assigning
        # a semantic identity ourselves. The original response stays in logs.
        for index,review in enumerate(answer.candidates):
            if review.verdict=='yes' and any(v in ('no','uncertain')
                    for v in (review.shape_match,review.color_match)):
                p.log.event('identity_review_inconsistent',{'query':ref.description,
                    'review':review.model_dump(),'action':'require_new_evidence'})
                answer.candidates[index]=review.model_copy(update={'verdict':'uncertain',
                    'confidence':min(review.confidence,.5)})
        reviews={v.candidate_id:v for v in answer.candidates}
        if set(reviews)!=set(labels) or len(reviews)!=len(answer.candidates):
            raise AmbiguousReferenceError('model must review every offered candidate exactly once')
        evidence={'query':ref.description,'selected':None,
            'label_to_object_id':{k:v['id'] for k,v in labels.items()},'reviews':[v.model_dump() for v in answer.candidates],
            'review_image':str(path.relative_to(p.log.path)),'comparison_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        if independent:evidence.update(review_mode='independent_objects',individual_images=individual_images)
        p.log.event('identity_comparison_batch',evidence)
        return labels,reviews,evidence

    @staticmethod
    def excluded(review):
        # A confident "no" excludes a candidate. A candidate the model could
        # not even lean towards ("uncertain", low confidence) cannot outweigh
        # a unique confident "yes"; the selected one still has to pass the
        # separate fresh identity confirmation before any manipulation.
        return (review.verdict=='no' and review.confidence>=.65) or (review.verdict=='uncertain' and review.confidence<=.6)

    def must_decide(self,ref):
        """More evidence is no longer available: side views were already taken, are unsafe or
        impossible, or the body is in the gripper. Then the likeliest candidate is the answer."""
        r=self.resolver;key=(ref.description,ref.color,ref.category)
        return (key in getattr(r,'side_reviewed',set()) or key in getattr(r,'decide_now',set())
                or r.p.world.memory.held_id is not None)

    def everyday_shape_score(self,ref,row):
        """Measured proportions decide plate vs block when the model flips between views."""
        import re
        obj=self.resolver.p.world.objects.get(row['id'])
        size=np.asarray(obj.size if obj is not None else (row.get('visible_geometry') or {}).get('size') or (1,1,1),float)
        height,span=float(size[2]),max(float(np.max(size[:2])),1e-6)
        ratio=height/span;text=ref.description
        if re.search(r'盘|碟|plate|dish',text,re.I):return 1. if ratio<.45 else 0.
        if re.search(r'方块|积木|block|cuboid|cube',text,re.I):return 1. if ratio>=.5 else 0.
        if re.search(r'杯|mug|\bcup\b',text,re.I):return 1. if ratio>=.6 else .2
        if re.search(r'瓶|bottle',text,re.I):return 1. if ratio>=.8 else .2
        return .5

    def best_guess(self,ref,labels,reviews,evidence):
        """Pick the most probable candidate from all the evidence instead of giving up.

        Ranking: measured colour and body proportion first (they do not flip with viewpoint),
        then the model's latest verdict. A later contradictory view cannot bury a body whose
        measured shape is the one the user named. Always returns a candidate when any remain."""
        def fit(label):
            verdict=reviews[label];row=labels[label]
            colour=float(row['colors'].get(ref.color,0)) if ref.color else 1.
            rank={'yes':2,'uncertain':1,'no':0}[verdict.verdict]
            confidence=verdict.confidence if verdict.verdict!='no' else -verdict.confidence
            shape=self.everyday_shape_score(ref,row)
            return (5*shape+2*colour+rank+confidence+(.5 if verdict.shape_match=='yes' else 0)
                    +(.3 if verdict.color_match=='yes' else 0),colour,shape)
        ranked=sorted(labels,key=lambda k:fit(k),reverse=True)
        table={labels[k]['id']:round(fit(k)[0],3) for k in ranked}
        chosen=None
        for label in ranked:
            score,colour,shape=fit(label);verdict=reviews[label]
            distinctive=shape>=.8 and any(fit(k)[2]<.5 for k in labels)
            if ref.color and colour<.30 and not distinctive:continue
            if verdict.verdict=='no' and verdict.confidence>=.85 and not distinctive:continue
            chosen=label;break
        if chosen is None:return None
        verdict=reviews[chosen]
        self.resolver.p.log.event('identity_best_guess',{'query':ref.description,'selected':labels[chosen]['id'],
            'verdict':verdict.verdict,'confidence':verdict.confidence,'scores':table,
            'reason':'evidence cannot be improved; the most probable candidate is taken'})
        evidence['best_guess']=True
        return chosen

    def choose(self,ref,previous,allow_search):
        p=self.resolver.p;rows=self.candidates(ref,previous)
        if previous is not None and not allow_search:
            # Revalidate a tracked instance in the current observation, not
            # against historical crops from before it was carried elsewhere.
            frame=getattr(p,'last_frame',None)
            if frame is not None:
                rows=[r for r in rows if r['view']==frame.view_id
                      and abs(r['timestamp']-frame.timestamp)<.001]
        if not rows:return None
        p.log.event('identity_candidates',{'query':ref.description,'candidate_ids':[v['id'] for v in rows],
            'measured_colour_fraction':{v['id']:round(float(v['colors'].get(ref.color,0)),3) for v in rows} if ref.color else {}})
        all_ids={v['id'] for v in rows};screenings=[]
        if len(rows)>4:
            survivors=[]
            for offset in range(0,len(rows),4):
                labels,reviews,evidence=self.review_batch(ref,previous,rows[offset:offset+4])
                screenings.append(evidence)
                survivors.extend(row for key,row in labels.items()
                    if not (reviews[key].verdict=='no' and reviews[key].confidence>=.65))
            rows=survivors
            if not rows or len(rows)>4:
                raise AmbiguousReferenceError('screening could not narrow candidates; additional observation required')
        labels,reviews,evidence=self.review_batch(ref,previous,rows)
        yes=[v for v in reviews.values() if v.verdict=='yes' and v.confidence>=.85]
        selected=(yes[0].candidate_id if (not self.must_decide(ref) and len(yes)==1 and all(self.excluded(v)
            for v in reviews.values() if v.candidate_id!=yes[0].candidate_id)) else None)
        if (selected is not None and allow_search and not self.must_decide(ref)
                and any(v.verdict!='no' for v in reviews.values() if v.candidate_id!=selected)):
            # The others were set aside only because the model could not lean towards
            # them. The one selected still needs a close oblique look of its own for the
            # fresh confirmation, but those unclear alternatives need no risky side visit.
            row=labels[selected]
            if not str(row.get('view','')).startswith('identity_side:'):
                p.log.event('identity_comparison',{**evidence,'selected':None,'deferred_for_side_view':row['id']})
                raise CandidateNeedsViews([row['id']],required_calls=2)
        # After actual side evidence, resolve competing affirmative or unclear
        # independent judgements by looking at the alternatives TOGETHER.
        if (selected is None and len(labels)>1 and evidence.get('review_mode')=='independent_objects'
                and any(str(v.get('view','')).startswith('identity_side:') for v in labels.values())
                and any(v.verdict!='no' for v in reviews.values())):
            from .joint_comparison import joint_review
            selected,joint=joint_review(self,ref,labels)
            if selected is None:
                from .joint_comparison import structural_survivor
                selected=structural_survivor(ref,reviews,joint['response'])
                if selected is not None:joint['structural_tiebreak']=labels[selected]['id']
            evidence['joint_review']=joint
        evidence['selected']=None if selected is None else labels[selected]['id']
        evidence['screening_batches']=screenings
        p.log.event('identity_comparison',evidence)
        if selected is None:
            pending=[labels[k]['id'] for k,v in reviews.items() if v.verdict!='no' or v.confidence<.65]
            if pending and not self.must_decide(ref):
                independent=p.config.localization.independent_candidate_review and len(rows)>1
                raise CandidateNeedsViews(pending,required_calls=(len(pending)+2 if independent else 2))
            selected=self.best_guess(ref,labels,reviews,evidence)
            if selected is None:raise AmbiguousReferenceError('no offered candidate matches the reference')
        chosen=labels[selected]
        if previous and chosen['id']!=previous:
            if p.world.memory.held_id is not None or p.world.memory.mission_active or not allow_search:
                raise AmbiguousReferenceError('identity correction requires a new task plan before manipulation')
            if previous not in all_ids:
                raise AmbiguousReferenceError('old identity must be visibly rejected before correction')
        return chosen,evidence
