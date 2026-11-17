"""Own GPU locks, identity records and lease-bound process group cleanup."""
import argparse,datetime,fcntl,json,os,signal,subprocess,time
from pathlib import Path
from common import O,atomic
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--name',required=True);p.add_argument('--deadline',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
deadline=datetime.datetime.fromisoformat(a.deadline).timestamp();assert deadline>time.time()+300
uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
locks=[]
for name in [f'ttu-gpu-{uuid}.lock',f'ttu-gpu-{a.gpu}.lock',f'genplaylist-cuev2-{a.name}.lock']:
 f=open('/tmp/'+name,'a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
assert not subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HOME='/storage_fast/ttu/GenPlaylist/cache/huggingface',TORCH_HOME='/storage_fast/ttu/GenPlaylist/cache/torch',VLLM_WORKER_MULTIPROC_METHOD='spawn')
cmd=a.command[1:] if a.command[0]=='--' else a.command
started=time.time();proc=subprocess.Popen(cmd,env=env,start_new_session=True)
atomic(O/'processes'/f'{a.name}.json',dict(gpu=a.gpu,uuid=uuid,guard_pid=os.getpid(),child_pid=proc.pid,deadline=a.deadline,command=cmd,started_epoch=started))
def cleanup(*_):
 try:os.killpg(proc.pid,signal.SIGTERM)
 except ProcessLookupError:pass
 limit=time.time()+20
 while time.time()<limit:
  proc.poll()
  try:os.killpg(proc.pid,0)
  except ProcessLookupError:break
  time.sleep(.2)
 try:os.killpg(proc.pid,signal.SIGKILL)
 except ProcessLookupError:pass
 proc.wait()
signal.signal(signal.SIGTERM,cleanup);signal.signal(signal.SIGINT,cleanup)
try:rc=proc.wait(timeout=max(1,deadline-time.time()))
except subprocess.TimeoutExpired:cleanup();rc=124
finally:
 # Also clean any descendant engine processes remaining after the direct child exits.
 cleanup()
atomic(O/'processes'/f'{a.name}-exit.json',dict(returncode=rc,elapsed_seconds=time.time()-started))
raise SystemExit(rc)
