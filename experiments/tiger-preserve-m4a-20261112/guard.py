"""Exclusive physical GPU lock and bounded process-group lifetime."""
import argparse,datetime,fcntl,json,os,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--arm',required=True);a=p.parse_args()
R=Path('/storage_fast/ttu/GenPlaylist');E=R/'experiments/tiger-preserve-m4a-20261112';O=R/'outputs/tiger-preserve-m4a-20261112';O.mkdir(exist_ok=True)
deadline=datetime.datetime.fromisoformat('2026-11-12T16:21:30+08:00').timestamp()
assert time.time()<deadline-600
uuid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
locks=[]
for name in [f'ttu-gpu-{uuid}.lock',f'ttu-gpu-{a.gpu}.lock']:
    f=open('/tmp/'+name,'a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
apps=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not apps,apps
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
base=[str(R/'envs/main/bin/python'),'-u',str(E/'trial.py'),'--arm',a.arm]
for phase,extra in [('smoke',['--steps','2','--smoke']),('full',[])]:
    proc=subprocess.Popen(base+extra,env=env,start_new_session=True)
    (O/f'{a.arm}-process.json').write_text(json.dumps(dict(arm=a.arm,gpu=a.gpu,uuid=uuid,guard_pid=os.getpid(),child_pid=proc.pid,phase=phase,deadline='2026-11-12T16:21:30+08:00'))+'\n')
    try:rc=proc.wait(timeout=max(1,deadline-time.time()))
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        (O/f'{a.arm}-deadline.json').write_text(json.dumps({'stopped':True,'phase':phase})+'\n');raise SystemExit(124)
    if rc:raise SystemExit(rc)
