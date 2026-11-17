import argparse, json, os, sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from datasets import load_from_disk

p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['mpd','m4a'],required=True);a=p.parse_args()
R=Path('/storage_fast/ttu/GenPlaylist'); root=R/f'experiments/cue-v2-r31e-ddbc-{a.dataset}-20261113'; run=root/'runs/conditional_mask'
sys.path[:0]=[str(R/'experiments/training-tricks-20261106/code/src'),str(R/'experiments/training-tricks-20261106/code/src/03_backbone_recommender'),str(root)]
import main as engine
import diffusion
from training_objectives import install
install(diffusion.Diffusion,diffusion.Loss,'conditional_mask')
cfg=OmegaConf.load(run/'resolved-config.yaml'); ck=run/'checkpoints/last.ckpt'; assert ck.exists()
cfg.eval.checkpoint_path=str(ck); out=R/'outputs/cue-v2-r31e-evaluation-20261115'/a.dataset/'ddbc-joint-941.json';out.parent.mkdir(parents=True,exist_ok=True)
if out.exists(): print('EXISTS',out);raise SystemExit
cfg.eval.results_path=str(out);cfg.eval_batch_size=32;cfg.sampling.steps=256;cfg.loader.num_workers=0;cfg.eval.allow_protocol_override=True;cfg.eval.disable_ema=False
engine.L.seed_everything(1);logger=engine.utils.get_logger('ddbc-r31e-eval');dataset=engine.AbstractDataset(cfg);dataset.split();tok=engine.dataloader.get_tokenizer(cfg,dataset)
prep=Path(cfg.prepared_dataset_path);raw=load_from_disk(str(prep/'raw_dataset'))['test']
indices=list(range(941)) if a.dataset=='mpd' else json.loads((R/'experiments/two-dataset-fair-20261108/cohort.json').read_text())['source_test_indices']
assert len(indices)==941;data=tok.tokenize({'test':raw.select(indices)})
os.chdir(run);engine._rec_eval(cfg,logger,tok,data)
d=json.loads(out.read_text());d['evaluation']['fixed_941_contexts']=True;d['evaluation']['source_test_indices']=indices;out.write_text(json.dumps(d,indent=2)+'\n');print('DONE',a.dataset,out)
