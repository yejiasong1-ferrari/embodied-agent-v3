"""Per-reference wording for model identity requests.

An explicit geometric word (cylinder, sphere...) demands a literal whole-form
check. An everyday noun (cup, bottle, plate...) is recognised by its role cues,
and the simulator draws such objects as simplified solids. The wording below
only describes how to read the pixels; it never names an expected object.
"""
from .joint_comparison import requested_form

SUBJECT_RULE=(' Identify only the reference named in `query`. The user command may also mention other objects, for example '
    'the object being moved or the place it goes; they are context, never the thing to judge, and their category must not '
    'be applied to any candidate.')

EXPLICIT_FORM_RULE=('The user named an explicit geometric form. Check the entire outer silhouette first: list visible '
    'protrusions or steps and whether the outer width changes. A cylindrical subpart does not make a stepped or necked '
    'vessel a plain cylinder; then check colour allowing shadow.'+SUBJECT_RULE)

EVERYDAY_CATEGORY_RULE=('The user named an everyday object category, not a geometric form. Do not demand a plain-cylinder, '
    'neck or shoulder check. Judge the category from its role cues: overall body proportion, a side handle, a rim or lip, a '
    'flat base. This scene is rendered with simplified solids, so a cup or mug can be a closed upright body with a thin '
    'handle bar on one side and no visible opening or hollow interior; the missing opening does not contradict the '
    'category. A thin protrusion from the side of a wider body is a handle, not a neck, a shoulder or a second tier. '
    'A plate or dish is a shallow disc or thin slab, never a tall rectangular block. A block or cube is a rectangular '
    'prism with comparable height, never a thin disc. Reject a candidate only for a visible contradiction, for example a '
    'flat slab, a box-shaped body or a sphere for a vessel category. Then check the colour allowing shadow. Judge each '
    'candidate on its own cues, with the same criteria for all. If two candidates remain plausible, pick the one whose '
    'whole-body proportion better matches the named category.'+SUBJECT_RULE)


def reference_guidance(description):
    form=requested_form(description)
    if form:return {'reference_kind':'explicit_geometry','required_whole_form':form,'guidance':EXPLICIT_FORM_RULE}
    return {'reference_kind':'everyday_category','required_whole_form':None,'guidance':EVERYDAY_CATEGORY_RULE}
