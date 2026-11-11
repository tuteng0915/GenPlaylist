import os,json,subprocess,datetime,fcntl,sys,time
from pathlib import Path
R=Path('/storage_fast/ttu/GenPlaylist');E=R/'experiments/ddbc-preserve-20261111';O=R/'outputs/ddbc-preserve-20261111';O.mkdir(parents=True,exist_ok=True)
lock=(O/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
end=datetime.datetime.fromisoformat('2026-11-12T01:20:30+08:00').timestamp();py=str(R/'envs/repro-torch26/bin/python')
def execute(name,script,args):
 mark=O/'jobs'/f'{name}.done.json';mark.parent.mkdir(exist_ok=True)
 if mark.exists():return
 if time.time()>end-600:sys.exit('Finite deadline margin; checkpoints retained')
 cmd=[py,'-u',str(E/script)]+args;r=dict(name=name,cmd=cmd,pid=os.getpid(),start=datetime.datetime.now().astimezone().isoformat(),host=os.uname().nodename,gpu=0)
 (O/'active.json').write_text(json.dumps(r,indent=2));print('START',name,flush=True)
 with (O/'jobs'/f'{name}.log').open('a') as f:rc=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
 if rc:r['returncode']=rc;(O/'failed.json').write_text(json.dumps(r,indent=2));sys.exit(rc)
 mark.write_text(json.dumps(r,indent=2));print('DONE',name,flush=True)
for ds in ['mpd','m4a']:
 for alpha in ['0','1']:
  execute(f'smoke-{ds}-{alpha}','evaluate.py',['--dataset',ds,'--alpha',alpha,'--partition','smoke'])
for ds in ['mpd','m4a']:
 for arm in ['sid','plain','preserve']:
  execute(f'train-smoke-{ds}-{arm}','train.py',['--dataset',ds,'--arm',arm,'--steps','2','--smoke'])
for alpha in ['1','0','.25','.5']:
 for ds in ['mpd','m4a']:
  execute(f'cue-{ds}-{alpha}','evaluate.py',['--dataset',ds,'--alpha',alpha])
execute('cue-report','report.py',['--phase','cue'])
# Formal training starts only after cue-strength validation completes.
for step in [500,1000]:
 for arm in ['sid','plain','preserve']:
  for ds in ['mpd','m4a']:
   execute(f'train-{ds}-{arm}-{step}','train.py',['--dataset',ds,'--arm',arm,'--steps',str(step)])
   execute(f'eval-{ds}-{arm}-{step}','evaluate.py',['--dataset',ds,'--arm',arm,'--step',str(step),'--partition','monitor' if step==500 else 'validation'])
execute('training-report','report.py',['--phase','training'])
selection=json.loads((O/'alpha-selection.json').read_text())
for ds in ['mpd','m4a']:
 for alpha in sorted(set([1,selection[ds]['alpha']])):
  execute(f'test-cue-{ds}-{alpha:g}','evaluate.py',['--dataset',ds,'--alpha',str(alpha),'--partition','test'])
 for arm in ['sid','plain','preserve']:
  execute(f'test-train-{ds}-{arm}','evaluate.py',['--dataset',ds,'--arm',arm,'--partition','test'])
execute('test-report','report.py',['--phase','test'])
(O/'complete.json').write_text(json.dumps(dict(completed=datetime.datetime.now().astimezone().isoformat())));(O/'active.json').unlink(missing_ok=True)
print('ALL_COMPLETE',flush=True)
