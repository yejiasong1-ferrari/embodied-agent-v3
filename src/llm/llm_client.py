import base64
import hashlib
import io
import json
import os
from pathlib import Path
from urllib.request import Request,urlopen
from PIL import Image
from urllib.error import HTTPError
import re


class ModelRequestError(RuntimeError):
    def __init__(self, status, details):
        self.http_status=status
        self.details=details
        self.reason_code=('MODEL_CONTENT_REJECTED' if 'data_inspection_failed' in
            str(details.get('code','')).lower() else 'MODEL_REQUEST_FAILED')
        self.retryable=status is None or status in {408,429} or status>=500
        super().__init__(f'model request failed (HTTP {status})')


def safe_error_details(exc, api_key):
    # Keep only diagnostic fields, never echo a provider's request/image body.
    try:
        body=json.loads(exc.read(16384))
        error=body.get('error',body)
        if not isinstance(error,dict):return {}
        result={}
        for key in ('code','type','message'):
            value=error.get(key)
            if not isinstance(value,str):continue
            if api_key:value=value.replace(api_key,'[REDACTED]')
            value=re.sub(r'data:image/[^\s"\']+|[A-Za-z0-9+/=]{80,}|https?://[^\s]+','[REDACTED]',value)
            result[key]=value[:400]
        return result
    except Exception:return {}


VISUAL_REFERENT_RULES = """
Ground the user's words in the visible whole object, not a single bright patch.
Use a shape-first, colour-second comparison. First assess the complete form and
structural features against the user's shape/category words, without using hue
or brightness to rank candidates in this first assessment. Then check the user's
explicit colour and other attributes among the structurally compatible candidates.
A colour match must never rescue a clearly contradictory shape. A shape match
alone must never override a clearly contradictory explicit colour. When these
checks conflict or remain uncertain, return uncertainty rather than forcing a
winner. If the user supplies no shape/category, do not invent one; use the actual
explicit attributes. Use the full command to understand the reference, but do not
substitute the destination for the source.
Compare complete silhouettes, proportions and visible structural features
(straight sides, shoulders, neck, rim, opening, handle, flat ends). A circular
top alone does not establish a cylinder. For a geometric description such as
'cylinder', assess the whole object's external form: roughly constant width,
straight sides and flat ends. A clearly visible shoulder and narrower neck
contradict a plain cylinder, even if the lower body is cylindrical. Do not
reinterpret an explicit geometric description as a loose colloquial name to
excuse that contradiction. This applies equally to spheres, cubes, cones and
other explicit shapes. Everyday category words such as cup or drinking vessel
may tolerate naming differences when visually justified; that flexibility does
not erase an explicit shape requirement. Minor surface details need not change
the overall shape. If a shoulder may instead be an occluder's edge, mark it
unknown and request a clear view; do not assign the occluder's contour to the
candidate. A visible contradiction requires no; missing evidence requires
uncertain. Never return yes while describing the requested whole shape as
contradicted in your evidence.
Separate material colour from illumination: a shaded side can look much darker
than the top of the same material. Darkness alone does not prove a different
colour. Equally, a bright cap, rim or highlight alone does not prove the whole
object has that colour. Compare visible lit and shaded regions; never invent
unseen colours, surfaces or features to resolve uncertainty.
Compare alternatives symmetrically using the same criteria. Candidate order,
letter, crop size and apparent brightness are not evidence of identity; crops
may be independently enlarged and cannot establish relative physical size.
If shape is unresolved, request a side/oblique view; if image detail is unresolved,
request a closer or higher-resolution view. Use the schema's uncertainty status
when the images cannot decide rather than forcing a choice or reporting high
confidence. State the missing visual evidence briefly in the evidence field.
Before returning, check that each verdict follows its own visible evidence and
that rejection criteria are applied consistently to competing candidates.
Return only concise observable evidence and the decision, not a long reasoning
trace. Begin each evidence with the visible shape match/mismatch, followed by
the colour check and any missing view. Respect all schema string length limits;
keep candidate evidence within
300 characters. Do not assume any fixed candidate is the correct answer.
"""

PROMPTS={
    "visual_identity":"Identify the user-referred physical instance from actual RGB pixels and the full instruction. Explicitly distinguish confirmed, not_visible, ambiguous, detail_needed and occluded. Never infer hidden objects or use simulator identities. Return normalized xyxy bbox for confirmed only and concrete visible evidence. Confidence is an estimate, not proof.",
    "visual_grounding":"Identify the queried target only from the supplied RGB pixels. Use a tight xyxy bbox, explicitly normalized or pixel. Return found=false for absent/ambiguous targets. No world poses or hidden objects.",
    "grasp_strategy":"Select only an offered grasp candidate by ID using task semantics and geometry summary. Never invent coordinates, forces, poses or candidates. Safety and feasibility remain local. Return structured strategy and reason_code.",
    "interpret":"Interpret the user's manipulation goal. For relative placement fill spatial_constraints: explicit numeric distances in metres; default edge_gap and operator frame. right/left are +/-X, 上边/下边 are +/-Y (upper_of/lower_of), 上方/下方 are +/-Z (above/below). on means physical support. between requires two references; preserve every min/max distance constraint. Manipulation explicit constraints only if user asked; distinguish inferred. Ambiguous object-front uses object frame. No unspecified numeric distances. Resolve pronouns and descriptions. Use symbolic object descriptions, colours and categories. Do not output coordinates, joint angles, motor commands or forces. For 'move things blocking the cup then take it out', use extract.",
    "plan":"Propose only the authorized skills needed to reach the symbolic goal. Include, in this exact mandatory relative order: OBSERVE_SCENE, FIND_OBJECT, APPROACH_OBJECT, GRASP_OBJECT, VERIFY_GRASP, LIFT_OBJECT, MOVE_TO_TARGET, PLACE_OBJECT, RETREAT, VERIFY_TASK. You may insert SCAN_SCENE or MOVE_CAMERA where observations require it. Never output poses or controller parameters.",
    "recover":"Diagnose the supplied failure evidence and history. Choose only REOBSERVE, CHANGE_GRASP, CHANGE_APPROACH, REPLAN or ABORT. Never override a safety rejection or emit motion values.",
    "visual_detection":"Identify objects from image pixels. Output normalized boxes and visible semantic labels. Never infer simulator object names or hidden poses.",
    "visual_disambiguation":"Compare every labelled candidate in the supplied camera images or comparison sheet with the user's full instruction and target phrase. Enlarged crops do not necessarily contain high-resolution detail. Determine which physical object the user refers to; everyday names can differ from an internal provisional category, which is not ground truth. Return one verdict per candidate_id. Use yes only when visible distinguishing evidence supports that referent. Colour alone, a roughly similar shape, or the absence of alternatives is insufficient. Mark unclear or occluded candidates uncertain. Do not infer simulator names, world coordinates or hidden geometry. Multiple yes and all uncertain are allowed when the image cannot decide. Return concrete visual evidence for each verdict.",
}
REFERENCE_KIND_RULE = '''
The payload field reference_kind selects the reading. explicit_geometry: the whole-form
rules above apply literally. everyday_category: the plain-cylinder, neck and shoulder
rules above do not apply; follow the payload guidance. In that case a thin protrusion
from the side of a wide body is a handle, and a missing visible opening or hollow
interior does not contradict a cup or mug in this simplified rendering. Reject only for
a visible contradiction, and keep the same criteria for every candidate.
'''
for _visual_purpose in ('visual_identity', 'visual_disambiguation'):
    PROMPTS[_visual_purpose] += VISUAL_REFERENT_RULES + REFERENCE_KIND_RULE
PROMPTS['visual_comparison'] = '''Compare the actual candidate images jointly against the complete requested description.
Each column is ONE candidate; rows in that column show the same tracked object from different views.
First report observable whole_form, neck and shoulder for EVERY candidate, then compare all candidates by the SAME literal criteria.
A plain cylinder has approximately constant outer width and straight sides. A two-tier or necked cylindrical vessel is stepped_cylindrical, not plain_cylinder, even when its lower part is a cylinder.
For an explicit whole geometric shape, matching a subpart does not satisfy the whole-object requirement. Everyday category naming flexibility cannot waive that requirement.
Compare shape before checking colour; shadows alone do not change material colour. Do not copy a neighbour or occluder's contour onto the target.
Select only a uniquely supported full-description match, reject visible contradictions, and allow ambiguous, none or need_view. Never choose merely because one candidate is relatively better if it still violates explicit requirements.
Missing parts are unknown; candidate letters, order, enlarged crop scale and prior model confidence are not identifying evidence.
Fill shape_match and color_match for every candidate. Each verdict must agree with its reported structure and evidence. No registry labels or expected answer are supplied.
Return concise visual evidence, not a reasoning trace.''' + VISUAL_REFERENT_RULES + REFERENCE_KIND_RULE
PROMPTS['visual_disambiguation'] += ' For every candidate fill shape_match and color_match separately (yes/no/uncertain/not_specified). not_specified means the user supplied no constraint of that kind. Evaluate shape_match against the whole requested form before checking colour. The final verdict must agree with these checks: a visible mismatch requires no, an unresolved required attribute requires uncertain, and yes requires every specified attribute to fit. Do not infer the shape of a hidden part from the edge of a robot link or neighbouring object.'
PROMPTS['interpret'] += ' For min_distance/max_distance/distance_to constraints, relation MUST be null (not min_distance or a phrase). Example additional constraint: {"type":"min_distance","relation":null,"reference":{"description":"blue box","color":"blue","category":"box"},"distance_m":0.08,"distance_semantics":"edge_gap","reference_frame":"operator"}.'
PROMPTS['interpret'] += ' Always include "avoid_collision" in constraints: it is mandatory local policy even when not explicitly requested. Never put it into manipulation.explicit unless the user said it.'
PROMPTS['interpret'] += ' Set manipulation.grasp_specific_region only when the user explicitly instructs a grasp at that part. A cup or mug does not imply a handle grasp; otherwise use null.'
PROMPTS['visual_grounding'] += ' Prefer coordinate_space="normalized" with decimal bbox coordinates between 0 and 1. Pixel coordinates MUST fit the supplied image_width and image_height; do not use 0..1000 coordinates as pixels.'


class LLMClient:
    def __init__(self,config,event_sink=None):
        self.config=config; self.calls=0; self.event_sink=event_sink or (lambda *args:None)
        env=Path(__file__).resolve().parents[2]/".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    key,value=line.split("=",1)
                    os.environ.setdefault(key.strip(),value.strip().strip("\"'"))
        self.api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url=os.getenv("VLM_BASE_URL",config.base_url)
        self.model=os.getenv("VLM_MODEL",config.model)

    @property
    def online(self): return self.config.mode!="offline" and bool(self.api_key)

    def complete(self,schema,purpose,payload,image=None):
        if not self.online: raise RuntimeError("LLM is not configured/enabled")
        if self.calls>=self.config.max_calls: raise RuntimeError("LLM call budget exhausted")
        system=PROMPTS[purpose]+" Return only JSON matching this schema: "+json.dumps(schema.model_json_schema())
        content=[{"type":"text","text":json.dumps(payload,ensure_ascii=False)}]
        if image is not None:
            buffer=io.BytesIO(); picture=Image.fromarray(image)
            detail=purpose in {'visual_disambiguation','visual_comparison','visual_identity','grasp_strategy','recover'}
            picture.thumbnail((1280,960) if detail else (640,480))
            picture.save(buffer,format="JPEG",quality=88 if detail else 75)
            content.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(buffer.getvalue()).decode()}})
        self.calls+=1
        self.event_sink("llm_request",{"purpose":purpose,"model":self.model,"call":self.calls,"payload":payload,"has_image":image is not None,
                                       "image_size":None if image is None else list(picture.size),
                                       "image_sha256":None if image is None else hashlib.sha256(buffer.getvalue()).hexdigest(),
                                       "endpoint":self.base_url.rstrip("/")+"/chat/completions"})
        request=Request(self.base_url.rstrip("/")+"/chat/completions",data=json.dumps({"model":self.model,"messages":[{"role":"system","content":system},{"role":"user","content":content}],"response_format":{"type":"json_object"},"temperature":0}).encode(),headers={"Authorization":"Bearer "+self.api_key,"Content-Type":"application/json"})
        try:
            with urlopen(request,timeout=self.config.timeout_s) as response:
                raw=json.loads(response.read())["choices"][0]["message"]["content"]
            result=schema.model_validate_json(raw)
        except Exception as exc:
            status=getattr(exc,'code',None)
            details=safe_error_details(exc,self.api_key) if isinstance(exc,HTTPError) else {}
            self.event_sink("llm_error",{"purpose":purpose,"error_type":type(exc).__name__,"http_status":status,'provider_error':details})
            if hasattr(exc,'errors'):
                self.event_sink('llm_schema_error',{'purpose':purpose,'errors':[
                    {'location':list(e['loc']),'type':e['type'],'message':e['msg']} for e in exc.errors()]})
            if isinstance(exc,HTTPError):raise ModelRequestError(status,details) from exc
            raise RuntimeError(f"{purpose} LLM call failed: {type(exc).__name__}") from exc
        self.event_sink("llm",{"purpose":purpose,"model":self.model,"response":result.model_dump()})
        return result
