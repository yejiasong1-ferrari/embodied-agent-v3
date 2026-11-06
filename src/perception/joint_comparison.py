"""Compare current image evidence jointly, with explicit structure consistency."""
import hashlib
import re
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from llm.schemas import VisualComparisonDecision
from recovery.failure_classifier import SkillFailure
from world.world_model import AmbiguousReferenceError


def requested_form(description):
    # These are definitions of explicit language constraints, not object IDs
    # or classifiers. All observed structures below must come from the VLM.
    for form,pattern in [('plain_cylinder',r'圆柱|\bcylinder\b|\bcylindrical solid\b'),
                         ('sphere',r'球体|圆球|\bsphere\b'),
                         ('cone',r'圆锥|\bcone\b'),
                         ('cuboid',r'正方体|长方体|立方体|方块|积木|\bcube\b|\bcuboid\b|\bblock\b')]:
        if re.search(pattern,description,re.I):return form
    return None


def prepare_joint(comparison,ref,labels):
    from .identity_guidance import reference_guidance
    p=comparison.resolver.p
    image=Image.new('RGB',(1280,960),(20,24,29));draw=ImageDraw.Draw(image)
    observations=[]
    for i,(label,row) in enumerate(labels.items()):
        width=1280//len(labels);x=i*width
        views=[row,*row.get('evidence_views',[])][:2]
        height=900//len(views)
        for j,view in enumerate(views):
            draw.text((x+8,10+j*height),f'{label} - VIEW {j+1}',fill='white')
            tile=ImageOps.contain(Image.fromarray(view['patch']),(width-16,height-38))
            image.paste(tile,(x+(width-tile.width)//2,38+j*height+(height-38-tile.height)//2))
        observations.append({'candidate_id':label,'views':[{'view':v['view'],'timestamp':v['timestamp']} for v in views],
                             'visible_rgbd_geometry':row.get('visible_geometry')})
    path=Path(p.log.path)/'images'/f'identity_joint_{comparison.serial:03d}.jpg'
    image.save(path,quality=88)
    payload={'user_command':p.task_description,'query':ref.description,'candidate_ids':list(labels),
             **reference_guidance(ref.description),'observations':observations,
             'instruction':'Report each whole structure first, then compare these candidates directly. A required_whole_form, when supplied, is an explicit user shape constraint. Do not substitute a cylindrical subpart or compound cylinder for a plain cylinder. When required_whole_form is null the reference is an everyday category: follow the guidance field, report whole_form as other unless the body is plainly a cylinder, cuboid, sphere or cone, and treat neck and shoulder as descriptive only. Apply identical requirements to every column. If a unique match is visible, identify it and explain the distinguishing structure. Otherwise request the missing view or return no selection.'}
    return image,payload,path


def joint_review(comparison,ref,labels):
    r=comparison.resolver;p=r.p
    if not r.can_spend(ref):
        raise SkillFailure('LOW_CONFIDENCE',{'reason':'identity budget exhausted before joint comparison',
            'reason_code':'IDENTITY_BUDGET_EXHAUSTED','reference':ref.description,'terminal':True})
    image,payload,path=prepare_joint(comparison,ref,labels)
    r.charge(ref)
    answer=p.client.complete(VisualComparisonDecision,'visual_comparison',payload,image=np.asarray(image))
    if {v.candidate_id for v in answer.candidates}!=set(labels):
        raise AmbiguousReferenceError('joint comparison must assess every offered candidate exactly once')
    selected=answer.selected_candidate_id if answer.status=='selected' else None
    reason=None
    if selected:
        reviews={v.candidate_id:v for v in answer.candidates}
        winner=reviews[selected];structure=next(v for v in answer.structures if v.candidate_id==selected)
        required=requested_form(ref.description)
        if answer.confidence<.85 or winner.confidence<.85 or any(
                v.verdict!='no' or v.confidence<.65 for k,v in reviews.items() if k!=selected):
            reason='comparison_not_unique_or_confident'
        elif any(v not in ('yes','not_specified') for v in (winner.shape_match,winner.color_match)):
            reason='missing_or_conflicting_attribute_checks'
        elif required and structure.whole_form!=required:
            reason='whole_form_contradicts_explicit_request'
        elif required=='plain_cylinder' and (structure.neck!='absent' or structure.shoulder!='absent'):
            reason='neck_or_shoulder_not_excluded_for_plain_cylinder'
        if reason:selected=None
    evidence={'query':ref.description,'response':answer.model_dump(),
              'selected':None if selected is None else labels[selected]['id'],
              'rejection_reason':reason,'label_to_object_id':{k:v['id'] for k,v in labels.items()},
              'review_image':str(path.relative_to(p.log.path)),
              'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    p.log.event('identity_joint_comparison',evidence)
    return selected,evidence


def contradicts_form(structure,required):
    """A reported structure that visibly violates an explicit geometric word.
    unknown means the view cannot tell, which is not a contradiction."""
    return (structure['whole_form'] not in (required,'unknown')
            or (required=='plain_cylinder' and 'present' in (structure['neck'],structure['shoulder'])))


def structural_survivor(ref,independent,response):
    """Explicit-form tie-break after an unresolved joint comparison.

    A user shape word such as cylinder is literal. If the side-by-side structures
    rule out every other candidate with high confidence, and the one that remains
    was independently judged a yes and also fits the form in the joint view, that
    candidate may go on to the fresh confirmation. Conflicting independent verdicts
    on an excluded candidate are outweighed only by this explicit structural
    contradiction; everyday category words never take this path.
    """
    required=requested_form(ref.description)
    if not required or len(independent)<2:return None
    joint={v['candidate_id']:v for v in response['candidates']}
    structure={v['candidate_id']:v for v in response['structures']}
    survivors=[k for k in independent if not (joint[k]['verdict']=='no' and joint[k]['confidence']>=.85
                                              and contradicts_form(structure[k],required))]
    if len(survivors)!=1:return None
    key=survivors[0];mine=independent[key]
    fits=structure[key]['whole_form']==required and (required!='plain_cylinder' or (
        structure[key]['neck']=='absent' and structure[key]['shoulder']=='absent'))
    if (mine.verdict=='yes' and mine.confidence>=.85 and joint[key]['verdict']!='no'
            and joint[key]['shape_match']=='yes' and joint[key]['color_match']!='no' and fits):return key
    return None

