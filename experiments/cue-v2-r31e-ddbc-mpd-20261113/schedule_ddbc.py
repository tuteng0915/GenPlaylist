import datetime,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
R=Path('/storage_fast/ttu/GenPlaylist');ds=sys.argv[1];gpu={'mpd':0,'m4a':1}[ds];tiger={'mpd':2145110,'m4a':2145111}[ds];root=R/'experiments'/('cue-v2-r31e-ddbc-'+ds+'-20261113');prep=R/'outputs/cue-v2-r31e-prepared-20261113'/ds/'prepared/prepared_manifest.json';deadline='2026-11-14T19:43:30+08:00';end=datetime.datetime.fromisoformat(deadline).timestamp();log=root/'queue.log'
lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def record(x):
 with log.open('a') as f:f.write(datetime.datetime.now().astimezone().isoformat()+' '+x+'\n')
def command(stage,mode):
 if ds=='mpd':
  script=root/('stage1.py' if stage==1 else 'stage2.py');variant='separate01' if stage==1 else 'conditional_mask'
 else:script=root/'train.py';variant='joint_stage1' if stage==1 else 'conditional_mask'
 return [str(R/'envs/main/bin/python'),str(script),'--root',str(root),'--variant',variant,'--steps',str(5000 if stage==1 else 1000),'--mode',mode]
try:
 while not prep.exists():
  if time.time()>end-1800:raise TimeoutError('prepared input missing near deadline')
  time.sleep(15)
 for stage in (1,2):
  if time.time()>end-1800:raise TimeoutError('insufficient lease margin')
  run=root/'runs'/('separate01' if ds=='mpd' and stage==1 else 'joint_stage1' if stage==1 else 'conditional_mask')
  last=run/'checkpoints/last.ckpt'
  if last.exists():
   import torch
   ck=torch.load(last,map_location='cpu',weights_only=False)
   if int(ck['global_step']) >= (5000 if stage==1 else 1000):record('stage '+str(stage)+' already complete');continue
  record('stage '+str(stage)+' preflight')
  with (root/('queue-preflight-'+str(stage)+'.log')).open('ab') as f:subprocess.run(command(stage,'preflight'),stdout=f,stderr=subprocess.STDOUT,check=True)
  while True:
   if time.time()>end-1800:raise TimeoutError('GPU not released in lease')
   p=Path('/proc')/str(tiger)/'cmdline';live=p.exists() and ('r31e-tiger-'+ds) in p.read_bytes().decode(errors='replace')
   used=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
   if not live and not used:break
   time.sleep(15)
  guard=[str(R/'envs/main/bin/python'),str(R/'experiments/cue-v2-stageA-20261112/guard.py'),'--gpu',str(gpu),'--name','r31e-ddbc-'+ds+'-'+str(stage),'--deadline',deadline,'--']+command(stage,'train')
  record('launch '+str(stage)+' '+json.dumps(guard))
  with (root/('stage'+str(stage)+'-train.log')).open('ab') as f:subprocess.run(guard,stdout=f,stderr=subprocess.STDOUT,check=True)
  assert last.exists(),'missing checkpoint'
  import torch
  c=torch.load(last,map_location='cpu',weights_only=False);assert int(c['global_step']) >= (5000 if stage==1 else 1000)
  record('stage '+str(stage)+' complete')
 record('both stages complete')
except BaseException:
 record('FAILURE '+traceback.format_exc());raise
