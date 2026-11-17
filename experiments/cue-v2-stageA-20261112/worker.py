"""Finite Stage A queue only; stops on any failure and never enters Stage B."""
import argparse,fcntl,subprocess,traceback
from common import *
PY={e:str(R/'envs'/e/'bin/python') for e in ['main','ace-step','audio-eval-20261107','cue-v2-vllm']}

def run(name,env,script,args=()):
 log=O/'logs'/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True);started=time.time()
 with log.open('a') as f:subprocess.run([PY[env],str(E/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)
 atomic(O/'done'/(name+'.json'),dict(elapsed_seconds=time.time()-started,started_epoch=started,script_sha256=sha(E/script),command=[PY[env],str(E/script),*map(str,args)]))

def queue():
 jobs=[]
 def add(name,env,script,args=(),deps=()):jobs.append(dict(name=name,env=env,script=script,args=list(args),deps=list(deps)))
 add('mmr-prepare','main','prepare_mmr_gate.py')
 add('judge','cue-v2-vllm','diagnostics.py',['judge'])
 add('diagnostics','main','diagnostics.py',['stats'])
 add('human-audit','main','diagnostics.py',['export'],['judge'])
 for start in range(0,200,25):
  for ds in ['mpd','m4a']:
   for arm in ['v2','v1','none','mmr']:
    base=f'{ds}-{arm}-{start:04d}';args=['--dataset',ds,'--arm',arm,'--start',str(start),'--count','25']
    add(base+'-text','main','gate.py',['verbal',*args],['mmr-prepare'] if arm=='mmr' else []);add(base+'-audio','ace-step','gate.py',['synth',*args],[base+'-text'])
 for ds in ['mpd','m4a']:
  add(ds+'-reference-vggish','audio-eval-20261107','metrics.py',['vggish','--dataset',ds,'--arm','reference'])
  textdeps=[f'{ds}-{arm}-{i:04d}-text' for arm in ['v2','v1','none','mmr'] for i in range(0,200,25)]
  add(ds+'-content-four-arm','main','metrics.py',['content','--dataset',ds],textdeps)
  metricdeps=[ds+'-reference-vggish',ds+'-content-four-arm']
  for arm in ['v2','v1','none','mmr']:
   deps=[f'{ds}-{arm}-{i:04d}-audio' for i in range(0,200,25)]
   for metric,env in [('mert','cue-v2-vllm'),('vggish','audio-eval-20261107')]:
    name=ds+'-'+arm+'-'+metric;add(name,env,'metrics.py',[metric,'--dataset',ds,'--arm',arm],deps);metricdeps.append(name)
  add(ds+'-report','main','metrics.py',['report','--dataset',ds],metricdeps)
 add('stage-A-verification-four-arm','ace-step','verify.py',[],[j['name'] for j in jobs])
 add('stage-A-delivery','main','export.py',[],['stage-A-verification-four-arm'])
 immutable(O/'jobs-four-arm.json',jobs)
 return jobs

def stopped(deadline):
 if time.time()>=deadline-120:raise RuntimeError('Reached booking margin; resume only within new checked lease')
 failed=list((O/'failed').glob('*.json'))
 if failed:raise RuntimeError('Stage A halted after failure: '+','.join(p.stem for p in failed))

def main(shard,deadline):
 name='extract-'+str(shard)
 if not (O/'done'/(name+'.json')).exists():run(name,'cue-v2-vllm','extract.py',['--shard',shard,'--shards','4'])
 while not all((O/'done'/f'extract-{i}.json').exists() for i in range(4)):stopped(deadline);time.sleep(10)
 lock=O/'locks/build.lock';lock.parent.mkdir(parents=True,exist_ok=True)
 with lock.open('a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);builder=True
  except BlockingIOError:builder=False
  if builder and not (O/'done/build-and-prepare.json').exists():
   run('build','main','build.py')
   for ds in ['mpd','m4a']:
    check=json.loads((O/ds/'v1-reconstruction-check.json').read_text())
    assert check['exact_top8_rows']==check['rows'],f'{ds} v1 reconstruction differs; investigate before MMR comparison'
   run('gate-prepare','main','gate.py',['prepare']);queue();atomic(O/'done/build-and-prepare.json',dict(completed_epoch=time.time()))
 while not (O/'done/build-and-prepare.json').exists():stopped(deadline);time.sleep(10)
 jobs=queue()
 while True:
  stopped(deadline);claimed=False
  if all((O/'done'/(j['name']+'.json')).exists() for j in jobs):break
  for j in jobs:
   name=j['name']
   if (O/'done'/(name+'.json')).exists() or not all((O/'done'/(d+'.json')).exists() for d in j['deps']):continue
   with (O/'locks'/(name+'.lock')).open('a') as f:
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:continue
    if (O/'done'/(name+'.json')).exists():continue
    claimed=True;atomic(O/'running'/f'worker-{shard}.json',dict(job=name,pid=os.getpid(),started_epoch=time.time()))
    run(name,j['env'],j['script'],j['args']);break
  if not claimed:time.sleep(10)
 atomic(O/'worker-complete'/f'{shard}.json',dict(completed_epoch=time.time(),stage_B_started=False))

if __name__=='__main__':
 import datetime
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--deadline',required=True);a=p.parse_args()
 try:main(a.shard,datetime.datetime.fromisoformat(a.deadline).timestamp())
 except BaseException:
  atomic(O/'failed'/f'worker-{a.shard}.json',dict(traceback=traceback.format_exc(),epoch=time.time()));raise
