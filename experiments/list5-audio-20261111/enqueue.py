import json,hashlib,fcntl,datetime,re
from pathlib import Path
R=Path('/storage_fast/ttu/GenPlaylist');E=R/'experiments/list5-audio-20261111';O=R/'outputs/list5-audio-20261111';Q=R/'experiments/two-dataset-fair-20261108/tasks.json';J=R/'outputs/two-dataset-fair-20261108/jobs'
expected='806aa4f3239de00d64f15f1a41fcf49bbbea23a1328ccdc6c32e73134d236f17'
arms=['ddbc_sid','tiger_sid','ddbc_joint','tiger_joint','direct','history_random','profile','drop','retrieved']
new=[];pilots=[];shards={};metrics={}
for arm in arms:
 for ds in ['mpd','m4a']:
  name=f'l5-{ds}-{arm}-pilot';pilots.append(name);new.append(dict(id=name,deps=[],cmd=f'bash {E}/audio.sh {ds} {arm} 0 5'));shards[ds,arm]=[name]
# All pilots must succeed before any bulk production.
for start in range(5,500,10):
 for arm in arms:
  for ds in ['mpd','m4a']:
   name=f'l5-{ds}-{arm}-{start:04d}';shards[ds,arm].append(name);new.append(dict(id=name,deps=pilots,cmd=f'bash {E}/audio.sh {ds} {arm} {start} {min(10,500-start)}'))
for ds in ['mpd','m4a']:
 metrics[ds]=[]
 for arm in arms:
  for metric in ['mert','clap','fad']:
   name=f'l5-{ds}-{arm}-{metric}';metrics[ds].append(name)
   env='environment/env.sh'
   cmd=f'cd {R} && source {R}/{env} && source {R}/envs/audio-eval-20261107/bin/activate && export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && '
   if metric=='fad':cmd+=f'flock {O}/{ds}/fad-reference.lock '
   cmd+=f'python {E}/metrics.py --dataset {ds} --arm {arm} --metric {metric}'
   new.insert(0,dict(id=name,deps=shards[ds,arm],cmd=cmd))
 new.insert(0,dict(id=f'l5-{ds}-report',deps=metrics[ds],cmd=f'{R}/envs/repro-torch26/bin/python {E}/report.py --dataset {ds}'))
with (Q.parent/'tasks-edit.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);raw=Q.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
 old=json.loads(raw);keep=[];retired=[]
 for t in old:
  name=t['id'];pending=not(J/(name+'.done.json')).exists()
  obsolete=bool(re.match(r'^(hc-(mpd|m4a)-(history_random|report)|pb-(mpd|m4a)-(profile|report)|tjc-(mpd|m4a)-(audio|metrics))',name))
  if pending and obsolete:
   assert not(J/(name+'.running.json')).exists(),name;retired.append(t)
  else:keep.append(t)
 assert retired and len({t['id'] for t in new+keep})==len(new+keep)
 ids={t['id'] for t in new+keep};assert all(d in ids for t in new+keep for d in t.get('deps',[]))
 (O/'queue-before.json').write_bytes(raw);(O/'retired-first-item-tasks.json').write_text(json.dumps(retired,indent=2))
 merged=new+keep;data=(json.dumps(merged,indent=2)+'\n').encode();tmp=Q.with_suffix('.tmp');tmp.write_bytes(data);tmp.replace(Q)
 audit=dict(time=datetime.datetime.now().astimezone().isoformat(),before_sha256=expected,after_sha256=hashlib.sha256(data).hexdigest(),before=len(old),retired_pending=len(retired),new=len(new),total=len(merged),reason='User changed generation evaluation to 100 lists x5 items per dataset. Completed old evidence retained; only unstarted superseded audio work retired.',all_pilot_ids=pilots)
 (O/'queue-transition.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))
