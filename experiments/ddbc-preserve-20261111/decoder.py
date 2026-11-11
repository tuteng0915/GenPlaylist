"""Two-pass probability interpolation. Only target cues are suppressed."""
import torch
from common import R,module
old=module('preserve_old_order',R/'experiments/cue-first-20261110/ddbc_order.py')
finalizer=module('preserve_finalizer',R/'experiments/fixed-sid-cues-20261110/decoder_v2/finalize_cues.py')
def mix_log_probs(masked,conditioned,alpha):
 if alpha==0:return masked
 if alpha==1:return conditioned
 return torch.logaddexp(masked+__import__('math').log1p(-alpha),conditioned+__import__('math').log(alpha))
def install(cls,alpha):
 original=cls.sample_masked_completion
 stats=dict(alpha=alpha,beam_model_calls=0,endpoint_max_error=0.,batches=0)
 @torch.no_grad()
 def sample(self,masked_ids,completion_mask,num_steps,eps=1e-5,context_emb=None,mu_c=None,sigma_c2=None,sequence_mask=None):
  assert num_steps==256 and not self.config.sampling.cfg_enabled and not self.config.sampling.structure_conditioning
  x=masked_ids.to(self.device);cm=completion_mask.to(self.device).bool();seq=sequence_mask.to(self.device).bool();fixed=x.clone()
  pos=torch.arange(x.shape[1],device=x.device);off=(pos-1)%self.tokenizer.tokens_per_item
  sid=cm&(pos>0)&(off>=1)&(off<=4);cue=cm&~sid
  x=original(self,x,cm,256,eps=eps,sequence_mask=seq)
  ids=sorted(self.tokenizer.item_id_to_row,key=self.tokenizer.item_id_to_row.get)
  codes=torch.tensor([self.tokenizer.semantic_tokens[i] for i in ids],device=x.device)
  base_forward=self.forward
  # Context and output positions are identical across rows; beam expands rows.
  assert torch.equal(cue,cue[0].expand_as(cue))
  def mixed(z,sigma,**kwargs):
   suppressed=z.clone();suppressed[:,cue[0]]=self.mask_index
   a=base_forward(suppressed,sigma,**kwargs);b=base_forward(z,sigma,**kwargs)
   stats['beam_model_calls']+=2
   out=mix_log_probs(a,b,alpha)
   if alpha in (0,1):assert torch.equal(out,a if alpha==0 else b)
   return out
  self.forward=mixed
  try:result=old.beam4_dedup(self,x,sid,seq,codes)
  finally:self.forward=base_forward
  if stats['batches']==0 and alpha in (0,1):
   # Same real model and beam: endpoint equivalence, outside cost measurement.
   control=x.clone()
   if alpha==0:control[cue]=self.mask_index
   check=old.beam4_dedup(self,control,sid,seq,codes)
   assert torch.equal(result[sid],check[sid]),'Endpoint beam mismatch'
  result=finalizer.finalize_cues(self,result,cue,seq,steps=256,eps=eps)
  assert torch.equal(result[~cm],fixed[~cm]);stats['batches']+=1
  return result
 cls.sample_masked_completion=sample
 return stats
