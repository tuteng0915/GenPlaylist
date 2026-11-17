import hashlib,importlib.util,json,os,sys
from pathlib import Path
import numpy as np
R=Path('/storage_fast/ttu/GenPlaylist');E=R/'experiments/cue-v2-r31e-list5-audio-20261117';O=R/'outputs/cue-v2-r31e-list5-audio-20261117';CODE=R/'v2-recovery-20261106/code'
# TIGER+cues is deliberately excluded from this production pass.  The MPD
# standard TIGER checkpoint is also excluded until its SID-only input audit is
# complete; the corrected M4A SID-only plan is explicitly listed below.
ARMS=['ddbc_joint','ddbc_sid','ddbc_catalog','ddbc_nocue','tiger_sid']
SYSTEMS={a:('GenPlaylist' if a in ['ddbc_joint','ddbc_catalog','ddbc_nocue'] else 'DDBC-SFT') for a in ARMS}
REV='cdbee75f17c01a7cc42f958dc650907174af0554'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
def immutable(p,x):
 if Path(p).exists():assert json.loads(Path(p).read_text())==x,str(p)
 else:atomic(p,x)
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def paths(ds):
 return module('list5_path_common',R/'experiments/history-cue-baselines-20261110/common.py').paths(ds)

def v2_cues(ds):
 base=R/'outputs/cue-v2-r31e-20261113'
 return json.loads((base/ds/'item2cues.json').read_text()),json.loads((base/'cue_vocab.json').read_text())
def sample_indices(ds,n=941):return np.sort(np.random.default_rng(np.random.SeedSequence([20261111,1 if ds=='mpd' else 2])).choice(n,100,replace=False)).tolist()
def seed(i,j):return 42000+i+941*j

def source_plans(ds):
 # The selected DDBC result is the legal Beam4+dedup alpha=.2 re-decode, not
 # the older joint full-mask completion.  SID-only predictions are frozen
 # standard controls; their catalog cues are replaced by r3.1e catalog cues.
 sources={
  'ddbc_joint':R/'outputs/cue-v2-r31e-redecode-20261116'/ds/'alpha0.2-beam4d-count941.json',
  'ddbc_sid':(R/'experiments/two-dataset-fair-20261108/mpd-existing-plans/sid_matched.json' if ds=='mpd' else R/'outputs/two-dataset-fair-20261108/m4a/plans/sid_matched-audio.json'),
 }
 if ds=='m4a':
  sources['tiger_sid']=R/'outputs/cue-v2-r31e-tiger-sid-inputfix-r2-20261116/m4a/sid/audio/plan.json'
 return sources
