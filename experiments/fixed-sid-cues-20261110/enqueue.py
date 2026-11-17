import json,hashlib,os,fcntl,datetime
from pathlib import Path
R=Path('/storage_fast/ttu/GenPlaylist');E=R/'experiments/fixed-sid-cues-20261110';O=R/'outputs/fixed-sid-cues-20261110';Q=R/'experiments/two-dataset-fair-20261108/tasks.json';J=R/'outputs/two-dataset-fair-20261108/jobs'
lock=Q.with_name('tasks-edit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX);before=Q.read_bytes();old=json.loads(before);expected='7a0d0e12e76ce7f833fad26ecb3feb0f2d231123cdad974d9efce32697720777';assert hashlib.sha256(before).hexdigest()==expected
py=R/'envs/repro-torch26/bin/python';smoke=[];full=[]
for ds in ['mpd','m4a']:
 for scope in ['full_catalog','sampled100']:
  name=f'fixedsidcue-{ds}-{scope}';cmd=f'cd {R} && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 {py} -u {E}/run.py --dataset {ds} --scope {scope}'
  deps=[f'sampled100-{ds}-ddbc-joint-full'] if scope=='sampled100' else []
  smoke.append(dict(id=name+'-smoke',cmd=cmd+' --smoke',deps=deps));full.append(dict(id=name+'-full',cmd=cmd,deps=[]))
for t in full:t['deps']=[x['id'] for x in smoke]
new=smoke+full+[dict(id='fixedsidcue-report',cmd=f'cd {R} && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 {py} -u {E}/report.py',deps=[t['id'] for t in full])]
assert not {t['id'] for t in new}&{t['id'] for t in old};assert len(old)==1664
O.mkdir(exist_ok=True);(O/'queue-before.json').write_bytes(before);merged=new+old;t=Q.with_suffix('.fixedcue.tmp');t.write_text(json.dumps(merged,indent=2)+'\n');os.replace(t,Q)
audit=dict(registered_sgt=datetime.datetime.now().astimezone().isoformat(),old_tasks=len(old),new_tasks=len(new),total_tasks=len(merged),queue_before_sha256=expected,queue_after_sha256=hashlib.sha256(Q.read_bytes()).hexdigest(),old_task_records_and_order_preserved=merged[len(new):]==old,workers_started=0,training_tasks_added=0,reservation_event='2202853592',term_sgt='2026-11-11T01:18:30+08:00',current_jobs=[p.name for p in J.glob('*.running.json')]);(O/'enqueue.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
