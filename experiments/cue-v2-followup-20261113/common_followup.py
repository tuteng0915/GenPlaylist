import os
if os.environ.get('CUE_FOLLOWUP_USE_GPU')!='1':os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
R=Path('/storage_fast/ttu/GenPlaylist');S=R/'outputs/cue-v2-stageA-20261112';O=R/'outputs/cue-v2-followup-20261113';O.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'experiments/cue-v2-stageA-20261112'))
from common import paths,lyric_path,lyric_quality,normalized,sha
from metrics import unit,interval
ARMS=['v1','mmr','v2','none']
def save(name,obj):
 p=O/name;p.parent.mkdir(exist_ok=True,parents=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def waiting(paths_to_wait,seconds=36000):
 start=time.time()
 while not all(p.exists() for p in paths_to_wait):
  if time.time()-start>seconds:raise TimeoutError('Waiting for '+str([str(p) for p in paths_to_wait if not p.exists()]))
  time.sleep(30)
def model_cpu(name):
 import torch
 torch.set_num_threads(1)
 from sentence_transformers import SentenceTransformer
 return SentenceTransformer(name,device='cuda' if os.environ.get('CUE_FOLLOWUP_USE_GPU')=='1' else 'cpu',local_files_only=True)
