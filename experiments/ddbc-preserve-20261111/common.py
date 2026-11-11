import os,sys,json,hashlib,importlib.util
from pathlib import Path
R=Path('/storage_fast/ttu/GenPlaylist');E=R/'experiments/ddbc-preserve-20261111';O=R/'outputs/ddbc-preserve-20261111'
code=R/'experiments/training-tricks-20261106/code/src'
sys.path[:0]=[str(code/'03_backbone_recommender'),str(code),str(R/'experiments/literature-tricks-20261106')]
import torch
import main as engine
import diffusion
from omegaconf import OmegaConf
from datasets import load_from_disk
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def source(ds,arm):
 base='two-dataset-fair-20261108' if ds=='m4a' else ('combined-decoding-20261107' if arm=='sid' else 'literature-tricks-20261106')
 return R/'experiments'/base/'runs'/('sid_stage2' if arm=='sid' else 'conditional_mask')
def setup(ds,arm):
 src=source(ds,arm);cfg=OmegaConf.load(src/'resolved-config.yaml')
 cfg.loader.num_workers=0;cfg.eval.allow_protocol_override=True;cfg.eval.disable_ema=False;cfg.eval.max_test_examples=None;cfg.eval_batch_size=16;cfg.sampling.steps=256
 dataset=engine.AbstractDataset(cfg);dataset.split();tok=engine.dataloader.get_tokenizer(cfg,dataset)
 raw=load_from_disk(str(Path(cfg.prepared_dataset_path)/'raw_dataset'))
 sp=R/'experiments'/('two-dataset-fair-20261108' if ds=='m4a' else 'literature-tricks-20261106')/'split.json'
 split=json.loads(sp.read_text());assert cfg.optimization_trial.split_sha256==sha(sp)
 assert not set(split['training_indices'])&set(split['validation_indices'])
 return cfg,tok,raw,split,sp
def ema_load(model,ck):
 from itertools import chain
 state=torch.load(ck,map_location='cpu',weights_only=False);model.load_state_dict(state['state_dict'],strict=True)
 if state.get('ema'):
  model.ema.load_state_dict(state['ema']);model.ema.copy_to(chain(model.backbone.parameters(),model.noise.parameters()))
 return state
