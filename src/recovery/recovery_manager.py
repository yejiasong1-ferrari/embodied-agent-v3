from .failure_classifier import FailureType


class RecoveryManager:
    def __init__(self,config,replanner,log):
        self.config,self.replanner,self.log=config,replanner,log
        self.history=[]; self.failed_candidates=set()

    def decide(self,failure,context):
        kind=failure.failure
        candidate=context.candidate.id if context.candidate else None
        row={"failure":kind.value,"candidate":candidate,"evidence":failure.evidence,"skill":context.current_skill,
             "target_pose":None if context.source is None else context.source.position.tolist(),
             "contact":context.robot.get_contact_state().__dict__,
             "strategy":None if context.candidate is None else getattr(context.candidate,"grasp_type",None),
             "region":None if context.candidate is None else getattr(context.candidate,"contact_region",None),
             "approach":None if context.candidate is None else getattr(context.candidate,"approach_direction",None)}
        self.history.append(row)
        def recorded(action):
            row['action']=action
            row['attempt']=len(self.history)
            self.log.event('recovery',row)
            return action
        if candidate and kind in {FailureType.NO_CONTACT,FailureType.GRASP_FAILED,FailureType.IK_FAILED,FailureType.OBJECT_SLIPPED,FailureType.PATH_BLOCKED,FailureType.NO_VALID_PLACEMENT}:
            self.failed_candidates.add(candidate)
        if len(self.history)>self.config.recovery.max_recoveries: return recorded("ABORT")
        # A brush against furniture while only travelling to look at the object (nothing held, nothing grasped)
        # is a reason to back off and try another route, not to give up. Twice at most.
        retreatable=(kind==FailureType.COLLISION_RISK and not context.held and context.current_skill=='APPROACH_OBJECT'
                     and sum(h['failure']==kind.value for h in self.history)<=2)
        if retreatable:row['stop_reason']='backing off and re-planning the approach'
        if not retreatable and kind in {FailureType.COLLISION_RISK,FailureType.USER_STOP,
                    FailureType.AMBIGUOUS_REFERENCE_FRAME}: return recorded("ABORT")
        if failure.evidence.get("terminal"):return recorded("ABORT")
        if failure.evidence.get("unsupported"): return recorded("ABORT")
        if failure.evidence.get("detail_reviewed"): return recorded("ABORT")
        if failure.evidence.get("reason")=="distinct grasp candidate budget exhausted": return recorded("ABORT")
        if sum(h["failure"]==kind.value and h["candidate"]==candidate for h in self.history)>2: return recorded("ABORT")
        if failure.evidence.get('reason_code')=='IDENTITY_REVIEW_REQUIRED':
            # Repeating the same unresolved association is not new evidence: take the
            # likeliest remaining candidate rather than looping the same review.
            signature=(failure.evidence.get('reference'),tuple(sorted(failure.evidence.get('review',{}).get('rival_ids',[]))))
            previous=[h for h in self.history[:-1] if h['evidence'].get('reason_code')=='IDENTITY_REVIEW_REQUIRED']
            if any((h['evidence'].get('reference'),tuple(sorted(h['evidence'].get('review',{}).get('rival_ids',[]))))==signature for h in previous):
                row['stop_reason']='same identity conflict; committing the likeliest candidate'
                return recorded('REOBSERVE')
            return recorded('REOBSERVE')
        looking=context.current_skill in {'FIND_OBJECT','OBSERVE_SCENE','SCAN_SCENE','MOVE_CAMERA'}
        if kind in {FailureType.TARGET_NOT_FOUND,FailureType.TARGET_LOST,FailureType.LOW_CONFIDENCE,
                    FailureType.DEPTH_INVALID,FailureType.AMBIGUOUS_TARGET} or (looking and kind in {FailureType.IK_FAILED,FailureType.PATH_BLOCKED}):
            action="REOBSERVE"
        elif kind in {FailureType.NO_CONTACT,FailureType.GRASP_FAILED,FailureType.IK_FAILED,
                      FailureType.OBJECT_SLIPPED,FailureType.NO_VALID_PLACEMENT}: action="CHANGE_GRASP"
        else: action="REPLAN"
        if context.held and kind in {FailureType.IK_FAILED,FailureType.PATH_BLOCKED}:
            repeats=sum(h['failure']==kind.value and h.get('skill')==context.current_skill for h in self.history)
            action="CHANGE_GRASP" if repeats>1 else "REPLAN"
        if len(self.history)>=self.config.recovery.llm_after_failures and self.replanner.client.online:
            try:
                kwargs={}
                if context.config.strict_visual_identity and context.perception.last_frame is not None:
                    from PIL import Image,ImageDraw
                    import numpy as np
                    after=Image.fromarray(context.perception.last_frame.rgb).resize((640,480))
                    before=Image.fromarray(getattr(context,'skill_start_rgb',context.perception.last_frame.rgb)).resize((640,480))
                    sheet=Image.new('RGB',(1280,510));sheet.paste(before,(0,30));sheet.paste(after,(640,30))
                    draw=ImageDraw.Draw(sheet);draw.text((8,8),'BEFORE SKILL',fill='white');draw.text((648,8),'LATEST OBSERVATION',fill='white')
                    sheet.save(context.log.path/'images'/f'recovery_{len(self.history)}.png')
                    kwargs={'image':np.asarray(sheet)}
                result=self.replanner.reason(context.goal,context.world,kind.value,self.history,
                    {**failure.evidence,"grasp_strategy":candidate,"distance":context.target_distance(),
                     "relevant_image_description":[o.semantic_label for o in context.world.objects.values() if o.visible]},
                    {"last_failure":context.safety.last_failure,"carrying":context.held},**kwargs)
                if result.action!='ABORT' or kind in {FailureType.COLLISION_RISK,FailureType.USER_STOP}:
                    action=result.action
            except RuntimeError:
                if self.replanner.client.config.mode=="required" and kind in {FailureType.COLLISION_RISK,FailureType.USER_STOP}:
                    action="ABORT"
        if context.held and kind not in {FailureType.OBJECT_SLIPPED,FailureType.NO_VALID_PLACEMENT,
                                         FailureType.PATH_BLOCKED,FailureType.IK_FAILED} and action in {"CHANGE_GRASP","CHANGE_APPROACH"}:
            action="REPLAN"
        if context.held and action=="CHANGE_APPROACH":
            # Another approach while carrying is another route, not a mid-air release.
            action="REPLAN"
        if context.held and kind in {FailureType.PATH_BLOCKED,FailureType.IK_FAILED} and action=="CHANGE_GRASP":
            repeats=sum(h['failure']==kind.value and h.get('skill')==context.current_skill for h in self.history)
            if repeats<=1:action="REPLAN"
        return recorded(action)
