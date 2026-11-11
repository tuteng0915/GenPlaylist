import argparse,fcntl,time
from common import *
from decoder import install
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['mpd','m4a'],required=True);p.add_argument('--alpha',type=float,default=1);p.add_argument('--partition',choices=['smoke','validation','test','monitor'],default='validation');p.add_argument('--arm',choices=['existing','sid','plain','preserve'],default='existing');p.add_argument('--step',type=int,default=1000);a=p.parse_args()
assert a.alpha in (0,.2,.25,.5,1)
engine.L.seed_everything(13001)
is_sid=a.arm=='sid';cfg,tok,raw,split,sp=setup(a.dataset,'sid' if is_sid else 'joint')
dest=O/'eval'/a.dataset/a.arm/f'step{a.step}'/a.partition/f'alpha{a.alpha:g}';dest.mkdir(parents=True,exist_ok=True)
lock=(dest/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (dest/'complete.json').exists():print('Already complete',dest);sys.exit(0)
ck=source(a.dataset,'joint')/'checkpoints/last.ckpt' if a.arm=='existing' else E/'runs'/f'{a.dataset}-{a.arm}'/f'step-{a.step}.ckpt'
if a.arm!='existing' and a.step==0:ck=source(a.dataset,'sid')/'checkpoints/last.ckpt'
cfg.eval.checkpoint_path=str(ck);cfg.eval.results_path=str(dest/'plan.json')
if a.partition=='test':
 indices=list(range(941)) if a.dataset=='mpd' else json.loads((R/'experiments/two-dataset-fair-20261108/cohort.json').read_text())['source_test_indices'];part='test'
else:indices=split['pilot_validation_indices'][:(2 if a.partition=='smoke' else 128 if a.partition=='monitor' else 512)];part='train'
rows=raw[part].select(indices);data=tok.tokenize({'test':rows})
if not is_sid:stats=install(diffusion.Diffusion,a.alpha)
else:
 old=module('preserve_sid_beam',R/'experiments/cue-first-20261110/ddbc_order.py');original=diffusion.Diffusion.sample_masked_completion;stats={'sid_only':True}
 @torch.no_grad()
 def sid_sample(self,masked_ids,completion_mask,num_steps,eps=1e-5,context_emb=None,mu_c=None,sigma_c2=None,sequence_mask=None):
  x=original(self,masked_ids,completion_mask,num_steps,eps=eps,sequence_mask=sequence_mask)
  ids=sorted(self.tokenizer.item_id_to_row,key=self.tokenizer.item_id_to_row.get);codes=torch.tensor([self.tokenizer.semantic_tokens[i] for i in ids],device=self.device)
  return old.beam4_dedup(self,x,completion_mask.to(self.device).bool(),sequence_mask.to(self.device).bool(),codes)
 diffusion.Diffusion.sample_masked_completion=sid_sample
identity=dict(dataset=a.dataset,arm=a.arm,alpha=a.alpha,step=a.step,partition=a.partition,indices=indices,checkpoint_sha256=sha(ck),split_sha256=sha(sp),full_catalog=True,code_sha256=sha(E/'decoder.py'),seed=13001,selection_only=a.partition!='test',cue_finalizer='fixed final SID256; RNG isolated',cuda_device=os.environ.get('CUDA_VISIBLE_DEVICES'))
save(dest/'identity.json',identity);start=time.monotonic();engine._rec_eval(cfg,engine.utils.get_logger('preserve-eval'),tok,data)
result=json.loads((dest/'plan.json').read_text());pred=result['predictions'];truth=[[str(i) for i in row[15:]] for row in rows['item_seq']]
assert pred['target_item_ids']==truth and len(pred['item_ids'])==len(indices)
assert all(len(row)==len(set(row))==5 for row in pred['item_ids'])
save(dest/'complete.json',dict(identity=identity,seconds=time.monotonic()-start,stats=stats,plan_sha256=sha(dest/'plan.json')));print('COMPLETE',dest,flush=True)
