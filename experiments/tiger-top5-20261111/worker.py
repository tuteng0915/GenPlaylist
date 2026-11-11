import os,json,subprocess,datetime,fcntl,sys,time
from pathlib import Path
R=Path('/storage_fast/ttu/GenPlaylist');E=R/'experiments/tiger-top5-20261111';O=R/'outputs/tiger-top5-20261111';O.mkdir(parents=True,exist_ok=True)
lock=(O/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
end=datetime.datetime.fromisoformat('2026-11-12T01:20:30+08:00').timestamp()
for phase in ['pilot','full']:
 for ds in ['mpd','m4a']:
  for arm in ['sid','joint']:
   for mode in ['ar5','top5']:
    name=f'{ds}-{arm}-{mode}';d=O/phase/name;d.mkdir(parents=True,exist_ok=True)
    if (d/'complete.json').exists():continue
    if time.time()>end-300:sys.exit('Finite deadline margin reached; resumable batches retained')
    cmd=[str(R/'envs/repro-torch26/bin/python'),'-u',str(E/'run.py'),'--dataset',ds,'--arm',arm,'--mode',mode]+(['--pilot'] if phase=='pilot' else [])
    record=dict(phase=phase,case=name,start=datetime.datetime.now().astimezone().isoformat(),pid=os.getpid(),cmd=cmd,host=os.uname().nodename,gpu_enabled=False)
    (O/'active.json').write_text(json.dumps(record,indent=2));print('START',phase,name,flush=True)
    with (d/'run.log').open('a') as f:rc=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
    if rc:
     record['returncode']=rc;(O/'failed.json').write_text(json.dumps(record,indent=2));sys.exit(rc)
    print('DONE',phase,name,flush=True)
 if phase=='pilot':
  for ds in ['mpd','m4a']:
   for arm in ['sid','joint']:
    aa=json.loads((O/'pilot'/f'{ds}-{arm}-ar5/plan.json').read_text());bb=json.loads((O/'pilot'/f'{ds}-{arm}-top5/plan.json').read_text())
    assert [r[0] for r in aa['predictions']['item_ids']]==[r[0] for r in bb['predictions']['item_ids']],'First-choice identity failed'
    if arm=='joint':assert [r[0] for r in aa['predictions']['cue_ids']]==[r[0] for r in bb['predictions']['cue_ids']]
  (O/'pilot-passed.json').write_text(json.dumps(dict(passed=True,cases=8,contexts_per_case=8,first_choice_and_joint_cues_identical=True)))
with (O/'report.log').open('a') as f:rc=subprocess.call([str(R/'envs/repro-torch26/bin/python'),str(E/'report.py')],stdout=f,stderr=subprocess.STDOUT)
if rc:sys.exit(rc)
(O/'complete.json').write_text(json.dumps(dict(completed=datetime.datetime.now().astimezone().isoformat(),cases=8,contexts_per_case=941)))
(O/'active.json').unlink(missing_ok=True);print('ALL COMPLETE',flush=True)
