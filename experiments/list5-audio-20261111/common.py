import hashlib,importlib.util,json,os,sys
from pathlib import Path
import numpy as np
R=Path('/storage_fast/ttu/GenPlaylist');E=R/'experiments/list5-audio-20261111';O=R/'outputs/list5-audio-20261111';CODE=R/'v2-recovery-20261106/code'
ARMS=['ddbc_sid','tiger_sid','ddbc_joint','tiger_joint','direct','history_random','profile','drop','retrieved']
SYSTEMS={a:('GenPlaylist' if a in ['ddbc_joint','tiger_joint','drop'] else 'DDBC-SFT' if a in ['ddbc_sid','tiger_sid','retrieved'] else 'ACE-Step-Direct') for a in ARMS}
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
def sample_indices(ds,n=941):return np.sort(np.random.default_rng(np.random.SeedSequence([20261111,1 if ds=='mpd' else 2])).choice(n,100,replace=False)).tolist()
def seed(i,j):return 42000+i+941*j

def history_random(refs,itemc,vocab,ds,i,j):
 pool=sorted({v for s in refs for v in itemc[s][:8] if vocab[v].strip().lower() not in ['<unk>','[unk]','<pad>']});assert len(pool)>=8
 entropy=[17000,1 if ds=='mpd' else 2,i] if j==0 else [17000,1 if ds=='mpd' else 2,i,j]
 return sorted(np.random.default_rng(np.random.SeedSequence(entropy)).choice(pool,8,replace=False).tolist())
def source_plans(ds,setting):
 report=json.loads((R/'outputs/sampled-candidates-20261110'/ds/'report.json').read_text())
 # per-dataset report has the same metrics/sources schema as aggregate report.
 if ds in report:report=report[ds]
 src={a:Path(report['sources'][a.replace('_','-')+'/'+('full_catalog' if setting=='full_catalog' else 'sampled')]['path']) for a in ['ddbc_sid','tiger_sid','ddbc_joint','tiger_joint']}
 if setting=='full_catalog':src['ddbc_joint']=R/'outputs/fixed-sid-cues-20261110'/ds/'full_catalog/full/plan.json'
 else:
  src['ddbc_joint']=R/'outputs/unique-cues-20261111'/ds/'ddbc/repair/full/plan.json';src['tiger_joint']=R/'outputs/unique-cues-20261111'/ds/'tiger/no_repeat/full/plan.json'
 return src

def old_root(ds,arm):
 if arm=='ddbc_joint':return None # corrected final-SID cues differ from old renderer
 if arm=='tiger_joint':return R/'outputs/tiger-joint-20261108'/ds/'audio/joint'
 if arm=='history_random':return R/'outputs/history-cue-baselines-20261110'/ds/'audio/history_random'
 if arm=='profile':return R/'outputs/pipeline-baselines-20261110'/ds/'audio/profile'
 name={'ddbc_sid':'sft','tiger_sid':'tiger','drop':'drop','retrieved':'retrieved','direct':'direct'}[arm]
 if ds=='mpd' and arm!='tiger_sid':return R/'outputs/generation-final-20261107'/name
 return R/'outputs/two-dataset-fair-20261108'/ds/'audio'/name
