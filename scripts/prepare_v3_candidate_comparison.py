"""Prepare a comparison from recorded images, entirely offline. No model call."""
import argparse,json,sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from config import AgentConfig
from llm.schemas import Reference
from agent.task_log import TaskLog
from world.world_model import WorldModel,WorldObject
from perception.camera import CameraFrame
from perception.visual_identity import VisualIdentityResolver


def main():
    parser=argparse.ArgumentParser();parser.add_argument('task',type=Path);parser.add_argument('--query',required=True);parser.add_argument('--previous-id');parser.add_argument('--before-sequence',type=int);parser.add_argument('--raw-pixels',action='store_true');args=parser.parse_args()
    cfg=AgentConfig.load();log=TaskLog('outputs/candidate_comparison','offline recorded perception replay',cfg)
    world=WorldModel(cfg.perception)
    events=[json.loads(l) for l in (args.task/'events.jsonl').open()]
    command=next(e['data']['payload']['command'] for e in events if e['type']=='llm_request' and e['data']['purpose']=='interpret')
    p=SimpleNamespace(config=cfg,world=world,log=log,task_description=command)
    resolver=VisualIdentityResolver(p)
    if args.raw_pixels:
        from perception.active_perception import ActivePerception
        perception=ActivePerception(SimpleNamespace(config=cfg.camera),world,cfg,SimpleNamespace(online=False),log)
        perception.task_description=command;resolver=perception.identity_resolver
    count=0;allowed={f.name for f in fields(WorldObject)}
    for e in events:
        if args.before_sequence is not None and e['sequence']>=args.before_sequence:break
        if e['type']!='observation':continue
        count+=1;image=args.task/'images'/f'{count:05d}_rgb.png';depth=image.with_name(f'{count:05d}_depth.npz')
        if not image.exists() or not depth.exists():continue
        data=np.load(depth);frame=CameraFrame(np.asarray(Image.open(image)),data['depth'],data['intrinsics'],data['camera_to_world'],float(data['timestamp']),e['data']['view'],data['robot_mask'] if 'robot_mask' in data else None)
        if args.raw_pixels:
            perception.observe(frame=frame)
            continue
        objects=[]
        for row in e['data']['objects']:
            values={k:v for k,v in row.items() if k in allowed}
            for key in ['position','orientation','size','observation_camera_pose']:
                if values.get(key) is not None:values[key]=np.asarray(values[key])
            values['view_ids']=set(values.get('view_ids',[]));obj=WorldObject(**values);world.objects[obj.id]=obj;objects.append(obj)
        resolver.comparison.remember(objects,frame)
    prepared=resolver.comparison.prepare(Reference(description=args.query),args.previous_id)
    if prepared is None:raise RuntimeError('not enough observed candidates')
    labels,_,payload,path=prepared
    report={'mode':'offline raw RGB-D pipeline' if args.raw_pixels else 'offline saved detections','comparison_image':str(path),
        'candidates':{k:{'tracking_id':v['id'],'position':v['position'].tolist(),'view':v['view']} for k,v in labels.items()},'model_payload':payload}
    (log.path/'prepared_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
