"""Same fixed pilot diagnostic as original, with a new repair arm. CPU only."""
import os
if os.environ.get('CUE_FOLLOWUP_USE_GPU')!='1':os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,time,json,types,hashlib
from pathlib import Path
R=Path('/storage_fast/ttu/GenPlaylist');S=R/'outputs/cue-v2-stageA-20261112';O=R/'outputs/cue-v2-repair-r2-20261113';E=R/'experiments/cue-v2-stageA-20261112'
sys.path.insert(0,str(E));t=time.time()
while not (O/'diagnostic-funnel.json').exists():
 if time.time()-t>3600:raise TimeoutError('Repair not complete')
 time.sleep(10)
import torch
torch.set_num_threads(1)
source=(E/'diagnostics.py').read_text();m=types.ModuleType('repair_diagnostics_internal');m.__dict__['__name__']='repair_diagnostics_internal';exec(compile(source if os.environ.get('CUE_FOLLOWUP_USE_GPU')=='1' else source.replace("device='cuda'","device='cpu'"),str(E/'diagnostics.py'),'exec'),m.__dict__);oldarms=m.arms

def arms(ds):
 co,data=oldarms(ds);v=json.loads((O/'cue_vocab.json').read_text());mp=json.loads((O/ds/'item2cues.json').read_text());data['v2_repair']={i:[v[c] if c else '<unk>' for c in mp[i]] for i in co['pilot']};return co,data

def atomic(path,data):
 assert Path(path).name=='diagnostics.json';data['original_diagnostics_script_sha256']=hashlib.sha256(source.encode()).hexdigest();data['device']='cuda' if os.environ.get('CUE_FOLLOWUP_USE_GPU')=='1' else 'cpu';(O/'diagnostics.json').write_text(json.dumps(data,indent=2)+'\n')
m.arms=arms;m.atomic=atomic;m.stats();print('REPAIR_DIAGNOSTICS_DONE',flush=True)
