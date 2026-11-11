import argparse
from common import *
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--count',type=int,required=True);a=p.parse_args()
base=O/a.dataset/a.arm;system=SYSTEMS[a.arm];rows=json.loads((base/'render-plan.json').read_text())['rows'];src=base/'verbalization';out=base/'synthesis';ace=Path(os.environ['ACE_STEP_SOURCE']);ck=Path(os.environ['ACE_STEP_SNAPSHOT']);assert ck.parent.name=='snapshots'
immutable(out/'audio_manifest.json',dict(schema='genplaylist-list5-audio-v2',model='ACE-Step/ACE-Step-v1-3.5B',model_revision=ck.name,duration_seconds=30,inference_steps=60,guidance_scale=15.0,scheduler_type='euler',cfg_type='apg',omega_scale=10.0,dtype='bfloat16',format='mp3',seed_rule='42000 + original941_history_index + 941*item_position',render_plan_sha256=sha(base/'render-plan.json'),verbalization_manifest_sha256=sha(src/'verbalization_manifest.json'),lists=100,positions=5,examples=500))
sys.path.insert(0,str(ace));pipeline=None
for row in rows[a.start:a.start+a.count]:
 i=row['flat_index'];vp=src/system/f'{i:04d}.json';record=json.loads(vp.read_text());assert record['render_row']==row;dest=out/system/f'{i:04d}.mp3';mp=dest.with_suffix('.json');reuse=None
 if dest.exists() and mp.exists():
  old=json.loads(mp.read_text());assert old['render_row']==row and old['audio_sha256']==sha(dest) and old['verbalization_sha256']==sha(vp);continue
 oldroot=old_root(a.dataset,a.arm)
 if row['item_position']==0 and oldroot and record.get('reused_verbalization'):
  old=oldroot/'synthesis'/system/f"{row['source_history_index']:04d}.mp3";om=old.with_suffix('.json');am=oldroot/'synthesis/audio_manifest.json'
  if old.exists() and om.exists() and am.exists():
   q=json.loads(om.read_text());m=json.loads(am.read_text())['ace_step']
   if q['seed']==row['seed'] and q['audio_sha256']==sha(old) and q['verbalization_sha256']==record['reused_verbalization_sha256'] and all(m.get(k)==v for k,v in dict(model_revision=ck.name,duration_seconds=30,inference_steps=60,guidance_scale=15.0,scheduler_type='euler',cfg_type='apg',omega_scale=10.0,dtype='bfloat16',format='mp3').items()):reuse=old
 dest.parent.mkdir(parents=True,exist_ok=True)
 if reuse:
  import shutil
  tmp=dest.with_suffix('.tmp');shutil.copy2(reuse,tmp);os.replace(tmp,dest)
 else:
  if pipeline is None:
   from acestep.pipeline_ace_step import ACEStepPipeline
   pipeline=ACEStepPipeline(checkpoint_dir=str(ck),device_id=0,dtype='bfloat16',torch_compile=False)
  partial=out/'.partial'/f'{i:04d}.{os.getpid()}.mp3';partial.parent.mkdir(parents=True,exist_ok=True)
  result=pipeline(prompt=record['music_attributes'],lyrics=record['lyric_draft'],audio_duration=30,infer_step=60,guidance_scale=15.0,scheduler_type='euler',cfg_type='apg',omega_scale=10.0,manual_seeds=[row['seed']],format='mp3',save_path=str(partial))
  assert Path(result[0]).resolve()==partial.resolve();os.replace(partial,dest)
 atomic(mp,dict(result_schema='genplaylist-list5-audio-record-v2',system=system,example_index=i,render_row=row,seed=row['seed'],audio_sha256=sha(dest),verbalization_sha256=sha(vp),reused_audio=None if reuse is None else str(reuse)))
 print('AUDIO',a.dataset,a.arm,i,'reuse' if reuse else 'new',flush=True)
