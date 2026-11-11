"""Finite matched continuation: SID-only, SID-warmstarted joint, joint+teacher KL."""
import argparse,fcntl,shutil,random,time
from itertools import chain
import numpy as np
from common import *
from training_objectives import install,channel_masks
from warmstart import sync_ema_to_model
from lightning.pytorch.callbacks import Callback
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['mpd','m4a'],required=True);p.add_argument('--arm',choices=['sid','plain','preserve'],required=True);p.add_argument('--steps',type=int,default=1000);p.add_argument('--smoke',action='store_true');a=p.parse_args()
assert a.steps in (2,500,1000);assert (a.steps==2)==a.smoke
engine.L.seed_everything(1)
cfg,tok,raw,split,sp=setup(a.dataset,'sid' if a.arm=='sid' else 'joint')
run=E/'runs'/f'{a.dataset}-{a.arm}'
if a.smoke:run=E/'smoke-runs'/f'{a.dataset}-{a.arm}'
run.mkdir(parents=True,exist_ok=True);lock=(run/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
src=source(a.dataset,'sid')/'checkpoints/last.ckpt';last=run/'checkpoints/last.ckpt'
cfg.seed=1;cfg.optim.lr=3e-5;cfg.loader.batch_size=32;cfg.loader.global_batch_size=512;cfg.trainer.accumulate_grad_batches=16
cfg.trainer.max_steps=a.steps;cfg.trainer.enable_progress_bar=False;cfg.trainer.log_every_n_steps=1 if a.smoke else 25
cfg.training.log_grad_norm=True;cfg.training.layer_loss_weights.cue_weight=.1;cfg.training.layer_loss_weights.warmup.enabled=False
cfg.checkpointing.save_dir=str(run);cfg.checkpointing.resume_from_ckpt=last.exists();cfg.checkpointing.resume_ckpt_path=str(last)
cfg.checkpointing.warmstart_path=None if last.exists() else str(src)
cfg.callbacks.checkpoint_every_n_steps.dirpath=str(run/'checkpoints');cfg.callbacks.checkpoint_every_n_steps.every_n_train_steps=100;cfg.callbacks.checkpoint_every_n_steps.save_last=True
cfg.run_name=f'preserve-{a.dataset}-{a.arm}';cfg.lr_scheduler.num_warmup_steps=100
# Keep the existing scheduler family; restart identically for all new arms.
cfg.preservation_trial=dict(dataset=a.dataset,arm=a.arm,teacher_source_sha256=sha(src),kl_weight=.5 if a.arm=='preserve' else 0.,temperature=2.,split_sha256=sha(sp))
dataset=engine.AbstractDataset(cfg);dataset.split()
data,_=engine.load_prepared_tokenized_dataset(Path(cfg.prepared_dataset_path),cfg,dataset,tok)
train_indices=split['training_indices'];valid_indices=split['pilot_validation_indices'][:128]
data['train']=data['train'].select(train_indices);data['test']=tok.tokenize({'test':raw['train'].select(valid_indices)})['test']
def warmstart(model,path):
 c=ema_load(model,path);assert c['global_step']==1000
 sync_ema_to_model(model)
 if teacher is not None:
  tp=dict(teacher.named_parameters())
  for name,v in model.named_parameters():assert torch.equal(v.detach().cpu(),tp[name].detach().cpu()), ('Warmstart parameter mismatch',name)
 return {'source':str(path),'sha256':sha(path),'same_vocab_rows':True,'EMA':True,'cue_initialization':'unused SID model cue rows; identical plain/preserve','position_layout_changed':a.arm!='sid'}
engine.apply_ddbc_warmstart=warmstart
teacher=None
if a.arm=='preserve':
 tc,tt,_,_,_=setup(a.dataset,'sid');teacher=diffusion.Diffusion(tc,tt);ema_load(teacher,src);teacher=teacher.cuda().eval();teacher.requires_grad_(False)
 assert tt.tokens_per_item==5 and tok.tokens_per_item==13
 assert tt.item_id_to_row==tok.item_id_to_row
 assert set(tt.semantic_tokens)==set(tok.semantic_tokens)
 assert all(list(tt.semantic_tokens[k])==list(tok.semantic_tokens[k]) for k in tt.semantic_tokens), 'SID row mapping mismatch'
if a.arm!='sid':install(diffusion.Diffusion,diffusion.Loss,'conditional_mask')
old_forward=diffusion.Diffusion.forward;old_loss=diffusion.Diffusion._loss
def forward(self,x,sigma,*args,**kwargs):
 y=old_forward(self,x,sigma,*args,**kwargs)
 if self.training:self._preserve_live=(x,sigma,y,kwargs.get('sequence_mask'))
 return y
diffusion.Diffusion.forward=forward
def loss(self,x0,attention_mask,target_mask=None,**kwargs):
 result,out=old_loss(self,x0,attention_mask,target_mask=target_mask,**kwargs)
 if not self.training:return result,out
 xt,sigma,lp,seq=self._preserve_live;del self._preserve_live
 if teacher is not None:assert teacher.mask_index==self.mask_index
 sid,cue=channel_masks(x0.shape,self.tokenizer,x0.device);active=result.token_mask.bool()
 for name,mask in [('sid',sid),('cue',cue)]:
  m=active&mask
  if m.any():self.log('preserve/'+name+'_nll',result.unweighted_nlls[m].mean(),on_step=True)
 kl=lp.sum()*0
 if teacher is not None:
  length=x0.shape[1];assert (length-2)%13==0
  keep=torch.tensor([0]+[1+13*i+j for i in range((length-2)//13) for j in range(5)]+[length-1],device=x0.device)
  compact=xt[:,keep];sm=seq[:,keep] if seq is not None else None
  with torch.no_grad():tlp=old_forward(teacher,compact,sigma,sequence_mask=sm)
  masked=(xt[:,keep]==self.mask_index)&active[:,keep]&sid[keep]
  if masked.any():
   student=F.log_softmax(lp[:,keep][masked].float()/2.,dim=-1)
   truth=F.log_softmax(tlp[masked].float()/2.,dim=-1)
   kl=(truth.exp()*(truth-student)).sum(-1).mean()*4
   assert torch.isfinite(kl) and kl>=-1e-5
  self.log('preserve/teacher_kl',kl.detach(),on_step=True)
  assert not any(v.grad is not None for v in teacher.parameters())
 self.log('preserve/masked_sid_tokens',((xt==self.mask_index)&active&sid).sum().float(),on_step=True)
 self.log('preserve/base_loss',result.loss.detach(),on_step=True)
 total=result.loss+(.5*kl if teacher is not None else 0)
 assert torch.isfinite(total)
 return diffusion.Loss(loss=total,nlls=result.nlls,token_mask=result.token_mask,unweighted_nlls=result.unweighted_nlls),out
diffusion.Diffusion._loss=loss
class Monitor(Callback):
 def __init__(self):self.logged=set()
 def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
  s=trainer.global_step
  if (s%25==0 or a.smoke) and s not in self.logged:
   self.logged.add(s);v={'step':s,'lr':[g['lr'] for g in trainer.optimizers[0].param_groups]}
   v.update({k:float(x.detach().cpu()) for k,x in trainer.callback_metrics.items() if isinstance(x,torch.Tensor) and x.numel()==1})
   with (run/'training-metrics.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
   print('TRAIN_METRICS',json.dumps(v),flush=True)
orig_instantiate=engine.hydra.utils.instantiate
def instantiate(config,*args,**kwargs):
 if str(config.get('_target_',''))=='lightning.Trainer':kwargs['callbacks'].append(Monitor())
 return orig_instantiate(config,*args,**kwargs)
engine.hydra.utils.instantiate=instantiate
identity=dict(dataset=a.dataset,arm=a.arm,source_checkpoint_sha256=sha(src),split_sha256=sha(sp),train_rows=len(train_indices),validation_indices=valid_indices,source_step=6000,extra_steps=a.steps,global_batch=512,learning_rate=3e-5,warmup=100,teacher_temperature=2,teacher_kl_weight=.5 if teacher else 0,teacher_in_optimizer=False,seed=1,official_test_used=False,code_sha256=sha(E/'train.py'))
save(run/'identity.json',identity);os.chdir(run);OmegaConf.save(cfg,run/'resolved-config.yaml',resolve=True)
started=time.monotonic()
# Teacher construction must not change student initialization, shuffle or corruption RNG.
engine.L.seed_everything(1)
if not last.exists() or int(torch.load(last,map_location='cpu',weights_only=False)['global_step'])<a.steps:
 engine._train(cfg,engine.utils.get_logger('sid-preserve'),tok,data)
c=torch.load(last,map_location='cpu',weights_only=False);assert c['global_step']==a.steps
assert not any(k.startswith('teacher.') for k in c['state_dict'])
milestone=run/f'step-{a.steps}.ckpt'
if not milestone.exists():shutil.copy2(last,milestone)
save(run/f'complete-{a.steps}.json',dict(identity=identity,checkpoint_sha256=sha(milestone),teacher_has_grad=False,elapsed_seconds=time.monotonic()-started));print('TRAIN_COMPLETE',a.dataset,a.arm,a.steps,flush=True)
