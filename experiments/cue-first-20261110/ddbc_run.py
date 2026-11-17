import argparse,json,sys,os,hashlib,fcntl,time
from pathlib import Path
from datasets import load_from_disk
from omegaconf import OmegaConf
import torch
from ddbc_order import install
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['mpd','m4a'],required=True)
p.add_argument('--order',choices=['joint','cue_first','sid_first'],required=True)
p.add_argument('--partition',choices=['smoke','pilot','audio'],required=True);a=p.parse_args()
root=Path(__file__).resolve().parent;R=root.parents[1]
code=R/'experiments/training-tricks-20261106/code/src'
sys.path[:0]=[str(code/'03_backbone_recommender'),str(code)]
import main as engine
import diffusion
source=R/'experiments'/('literature-tricks-20261106' if a.dataset=='mpd' else 'two-dataset-fair-20261108')/'runs/conditional_mask'
cfg=OmegaConf.load(source/'resolved-config.yaml');ck=source/'checkpoints/last.ckpt'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
expected={'mpd':'519bcd3eeb0d6c737b9b3ad89a79549ec9ca92460581ab92326af4c9144f6c51','m4a':'9bbf884bda1597c55ec2d489c34c70534daa2123742141789801ba03ae4eedbf'}
assert sha(ck)==expected[a.dataset]
dest=R/'outputs/cue-first-20261110/ddbc'/a.dataset/a.partition/a.order;dest.mkdir(parents=True,exist_ok=True)
lock=(dest/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (dest/'verified.json').exists():print('Verified result exists');sys.exit(0)
cfg.eval.checkpoint_path=str(ck);cfg.eval.results_path=str(dest/'result.json')
cfg.eval.allow_protocol_override=True;cfg.eval.disable_ema=False;cfg.eval_batch_size=32
cfg.sampling.steps=256;cfg.loader.num_workers=0
engine.L.seed_everything(1)
dataset=engine.AbstractDataset(cfg);dataset.split();tok=engine.dataloader.get_tokenizer(cfg,dataset)
raw=load_from_disk(str(Path(cfg.prepared_dataset_path)/'raw_dataset'))
split=json.loads((R/'experiments'/('literature-tricks-20261106' if a.dataset=='mpd' else 'two-dataset-fair-20261108')/'split.json').read_text())
if a.partition in ['smoke','pilot']:
    indices=split['pilot_validation_indices'][:(2 if a.partition=='smoke' else 128)]
    source_rows=raw['train']
else:
    indices=list(range(941)) if a.dataset=='mpd' else json.loads((R/'experiments/two-dataset-fair-20261108/cohort.json').read_text())['source_test_indices']
    source_rows=raw['test']
selected=source_rows.select(indices);data=tok.tokenize({'test':selected})
stats=install(diffusion.Diffusion,a.order);start=time.monotonic()
engine._rec_eval(cfg,engine.utils.get_logger('cue-order'),tok,data)
result=json.loads((dest/'result.json').read_text());pred=result['predictions']
truth=[[str(x) for x in row[15:]] for row in selected['item_seq']]
assert pred['target_item_ids']==truth and len(pred['item_ids'])==len(indices)
assert all(len(row)==len(set(row))==5 for row in pred['item_ids'])
assert all(len(row)==5 and all(len(c)==8 for c in row) for row in pred['cue_ids'])
identity=dict(checkpoint_sha256=sha(ck),order=a.order,source_indices=indices,
    partition=a.partition,selection_only=a.partition!='audio',seconds=time.monotonic()-start,stats=stats,
    code_sha256=sha(root/'ddbc_order.py'),fixed_budget='256 cue/joint reverse steps + legal confidence beam4 + common-score dedup')
(dest/'verified.json').write_text(json.dumps(identity,indent=2)+'\n')
print('VERIFIED',json.dumps(identity),flush=True)
