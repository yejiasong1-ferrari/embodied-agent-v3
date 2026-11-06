from .observation import ObserveSkill,ScanSceneSkill,MoveCameraSkill,FindObjectSkill
from .manipulation import ApproachSkill,GraspSkill,VerifyGraspSkill,LiftSkill,MoveObjectSkill,PlaceSkill,RetreatSkill,VerifyTaskSkill


class SkillRegistry:
    def __init__(self):
        self.skills={}
        for name,cls in (("OBSERVE_SCENE",ObserveSkill),("SCAN_SCENE",ScanSceneSkill),("MOVE_CAMERA",MoveCameraSkill),
                         ("FIND_OBJECT",FindObjectSkill),("APPROACH_OBJECT",ApproachSkill),("GRASP_OBJECT",GraspSkill),
                         ("VERIFY_GRASP",VerifyGraspSkill),("LIFT_OBJECT",LiftSkill),("MOVE_TO_TARGET",MoveObjectSkill),
                         ("PLACE_OBJECT",PlaceSkill),("RETREAT",RetreatSkill),("VERIFY_TASK",VerifyTaskSkill)):
            self.register(name,cls())
    def register(self,name,skill):
        if name in self.skills: raise ValueError(f"duplicate skill {name}")
        self.skills[name]=skill
    def execute(self,name,context):
        if name not in self.skills: raise ValueError(f"unauthorized skill {name}")
        skill=self.skills[name]; context.current_skill=name
        if context.config.strict_visual_identity and name not in {"OBSERVE_SCENE","SCAN_SCENE","MOVE_CAMERA","FIND_OBJECT"}:
            context.perception.identity_resolver.require_phase(context)
        if context.config.strict_visual_identity and context.perception.last_frame is not None:
            context.skill_start_rgb=context.perception.last_frame.rgb.copy()
        if context.config.strict_visual_identity and name=='PLACE_OBJECT':context.placement_reference_checked=False
        skill.check_preconditions(context)
        result=skill.execute(context)
        if context.config.strict_visual_identity and name=="VERIFY_TASK":
            resolver=context.perception.identity_resolver
            for reference in [context.goal.object,*context.goal.references]:
                if reference.category=='region':continue
                key=(reference.description,reference.color,reference.category)
                if resolver.anchors.get(key,{}).get('review_required'):
                    previous=context.world.reference_bindings.get(key)
                    from world.world_model import AmbiguousReferenceError,ReferenceNotFoundError
                    from recovery.failure_classifier import SkillFailure
                    try:checked=resolver.identify(reference,allow_search=False)
                    except (AmbiguousReferenceError,ReferenceNotFoundError) as exc:
                        raise SkillFailure('TARGET_LOST',{'reference':reference.description,
                            'reason_code':'IDENTITY_REVIEW_REQUIRED','identity_review_required':True,
                            'reason':'final identity review needs another observation',
                            'review':resolver.anchors.get(key,{}).get('review_reason',{})}) from exc
                    if checked.id!=previous:
                        from recovery.failure_classifier import SkillFailure
                        raise SkillFailure('AMBIGUOUS_TARGET',{'reference':reference.description,'reason':'instance changed in final identity check'})
            resolver.require_all(context.goal)
        skill.monitor(context)
        accepted=skill.validate(context,result)
        if context.config.strict_visual_identity and name=='VERIFY_TASK':
            context.log.event('final_validation',{
                **getattr(context,'final_geometry_evidence',{}),
                'accepted':bool(accepted),'identity_verified':True})
        return accepted
