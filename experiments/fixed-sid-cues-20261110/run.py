import argparse,sys,fcntl,time,json,os,hashlib
from pathlib import Path
import torch
from cue_completion import cue_completion
R=Path('/storage_fast/ttu/GenPlaylist');E=R/'experiments/fixed-sid-cues-20261110';O=R/'outputs/fixed-sid-cues-20261110'
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['mpd','m4a'],required=True);p.add_argument('--scope',choices=['full_catalog','sampled100'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,p)
dest=O/a.dataset/a.scope/('smoke' if a.smoke else 'full');dest.mkdir(parents=True,exist_ok=True)
lock=(dest/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (dest/'complete.json').exists():print('Verified complete; skip');sys.exit(0)
source=R/'outputs/cue-first-20261110/ddbc'/a.dataset/'audio/joint/result.json' if a.scope=='full_catalog' else R/'outputs/sampled-candidates-20261110'/a.dataset/'ddbc-joint/full/plan.json'
baseline=json.loads(source.read_text());bp=baseline['predictions'];n=2 if a.smoke else 941
assert len(bp['item_ids'])==941
code=R/'experiments/training-tricks-20261106/code/src';sys.path[:0]=[str(code/'03_backbone_recommender'),str(code)]
import main as engine
import diffusion
from omegaconf import OmegaConf
from datasets import load_from_disk
run=R/'experiments'/('literature-tricks-20261106' if a.dataset=='mpd' else 'two-dataset-fair-20261108')/'runs/conditional_mask'
cfg=OmegaConf.load(run/'resolved-config.yaml');ck=run/'checkpoints/last.ckpt'
assert sha(ck)=={'mpd':'519bcd3eeb0d6c737b9b3ad89a79549ec9ca92460581ab92326af4c9144f6c51','m4a':'9bbf884bda1597c55ec2d489c34c70534daa2123742141789801ba03ae4eedbf'}[a.dataset]
indices=list(range(941)) if a.dataset=='mpd' else json.loads((R/'experiments/two-dataset-fair-20261108/cohort.json').read_text())['source_test_indices'];indices=indices[:n]
identity=dict(dataset=a.dataset,scope=a.scope,source_plan=str(source),source_plan_sha256=sha(source),checkpoint_sha256=sha(ck),source_indices=indices,seed=1,batch_size=32,steps=256,operation='Mask and regenerate all40cue tokens; retain final selectedSID20tokens and original history throughout; no beam search or SID update',code_sha256=sha(E/'cue_completion.py'))
if (dest/'identity.json').exists():assert json.loads((dest/'identity.json').read_text())==identity
else:dump(dest/'identity.json',identity)
cfg.eval.checkpoint_path=str(ck);cfg.eval.results_path=str(dest/'result.json');cfg.eval.allow_protocol_override=True;cfg.eval.disable_ema=False;cfg.eval_batch_size=32;cfg.sampling.steps=256;cfg.loader.num_workers=0
engine.L.seed_everything(1);dataset=engine.AbstractDataset(cfg);dataset.split();tok=engine.dataloader.get_tokenizer(cfg,dataset)
raw=load_from_disk(str(Path(cfg.prepared_dataset_path)/'raw_dataset'))['test'].select(indices);data=tok.tokenize({'test':raw});assert bp['target_item_ids'][:n]==[[str(i) for i in row[15:]] for row in raw['item_seq']]
cursor=0
@torch.no_grad()
def patched(self,masked_ids,completion_mask,num_steps,eps=1e-5,context_emb=None,mu_c=None,sigma_c2=None,sequence_mask=None):
 global cursor
 assert num_steps==256 and not self.config.sampling.cfg_enabled and not self.config.sampling.structure_conditioning
 x=masked_ids.to(self.device).clone();cm=completion_mask.to(self.device).bool();seq=sequence_mask.to(self.device).bool();original=x.clone()
 pos=torch.arange(x.shape[1],device=self.device);off=(pos-1)%tok.tokens_per_item;sid=cm&(pos>0)&(off>=1)&(off<=4);cue=cm&~sid
 assert sid.sum()==len(x)*20 and cue.sum()==len(x)*40
 selected=bp['item_ids'][cursor:cursor+len(x)];codes=torch.tensor([[tok.semantic_tokens[i] for i in row] for row in selected],device=self.device)
 x[sid]=codes.reshape(-1);x[cue]=self.mask_index;fixed=x.clone()
 x=cue_completion(self,x,cue,seq,steps=256,eps=eps)
 assert torch.equal(x[sid],codes.reshape(-1)) and torch.equal(x[~cm],original[~cm])
 assert torch.equal(x[~cue],fixed[~cue]) and not (x[cue]==self.mask_index).any()
 legal=torch.as_tensor(tok.make_type_mask(x.shape[1]),device=self.device,dtype=torch.bool)
 assert legal.unsqueeze(0).expand(len(x),-1,-1).gather(2,x.unsqueeze(-1)).squeeze(-1)[cm].all()
 cursor+=len(x);return x
diffusion.Diffusion.sample_masked_completion=patched;start=time.monotonic();engine._rec_eval(cfg,engine.utils.get_logger('fixed-sid-cues'),tok,data)
r=json.loads((dest/'result.json').read_text());pr=r['predictions'];assert cursor==n
assert pr['item_ids']==bp['item_ids'][:n] and pr['target_item_ids']==bp['target_item_ids'][:n]
assert all(len(row)==5 and all(len(q)==8 for q in row) for row in pr['cue_ids'])
r.update(identity=identity,source_test_indices=indices);dump(dest/'plan.json',r)
dump(dest/'complete.json',dict(rows=n,seconds=time.monotonic()-start,all_selected_items_and_order_identical=True,history_and_sids_frozen_every_denoising_step=True,plan_sha256=sha(dest/'plan.json'),source_plan_sha256=sha(source),cue_token_changed_fraction=sum(x!=y for aa,bb in zip(pr['cue_ids'],bp['cue_ids'][:n]) for a,b in zip(aa,bb) for x,y in zip(a,b))/(n*40)))
print('COMPLETE',a.dataset,a.scope,n,flush=True)
