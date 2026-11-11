"""Read-only status and provenance snapshot; never starts or stops a job."""
import json,hashlib,datetime,subprocess
from pathlib import Path
R=Path('/storage_fast/ttu/GenPlaylist');E=R/'experiments/ddbc-preserve-20261111';O=R/'outputs/ddbc-preserve-20261111'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text()) if p.exists() else None
jobs=sorted(p.stem.removesuffix('.done') for p in (O/'jobs').glob('*.done.json'))
preflight={}
for ds in ['mpd','m4a']:
 d={}
 for arm in ['sid','plain','preserve']:
  p=E/'smoke-runs'/f'{ds}-{arm}';v=read(p/'complete-2.json')
  d[arm]={'completed':bool(v),'identity':v.get('identity') if v else None}
  f=p/'training-metrics.jsonl'
  if f.exists():d[arm]['first_batch']=json.loads(f.read_text().splitlines()[0])
 if all('first_batch' in d[x] for x in ['plain','preserve']):
  keys=['preserve/sid_nll','preserve/cue_nll','preserve/base_loss']
  assert all(d['plain']['first_batch'][k]==d['preserve']['first_batch'][k] for k in keys)
  d['joint_initial_batch_equal']=True
 preflight[ds]=d
q=R/'experiments/two-dataset-fair-20261108/tasks.json';J=R/'outputs/two-dataset-fair-20261108/jobs'
audio={'queue_sha256':sha(q),'running':{f.name:read(f) for f in J.glob('l5-*.running.json')},'done':len(list(J.glob('l5-*.done.json'))),'failed':len(list(J.glob('l5-*.failed.json')))}
assert audio['queue_sha256']=='7927f6bff3e1aeb519d1a7d82c12535e8e65881853189484cd982232e27ea1b2'
processes=subprocess.check_output(['ps','-eo','pid,ppid,etime,pcpu,rss,args'],text=True).splitlines()
processes=[v for v in processes if ('/ddbc-preserve-20261111/' in v or '/tiger-top5-20261111/' in v) and 'inspect.py' not in v]
result={'time':datetime.datetime.now().astimezone().isoformat(),'host':subprocess.check_output(['hostname'],text=True).strip(),'active':read(O/'active.json'),'failed':read(O/'failed.json'),'finished_jobs':jobs,'preflight':preflight,'report_smoke_passed':(O/'smoke-report.json').exists(),'audio':audio,'processes':processes,'gpu':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True),'source_hashes':{p.name:sha(p) for p in E.iterdir() if p.is_file() and p.suffix in ('.py','.sh','.json')}}
(O/'status-snapshot.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
