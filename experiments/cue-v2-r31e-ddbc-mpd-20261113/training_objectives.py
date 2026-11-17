"""Single-mechanism adaptations on a fixed SID/cue checkpoint and tokenizer."""
import torch
import torch.nn.functional as F

VARIANTS=('control','sid_temperature','hard_negative','whole_item_mask','conditional_mask')

def channel_masks(shape,tokenizer,device):
    pos=torch.arange(shape[1],device=device);off=(pos-1)%tokenizer.tokens_per_item
    sid=(pos>0)&(off>=1)&(off<=tokenizer.n_digit+1)
    cue=(pos>0)&(off>tokenizer.n_digit+1)
    return sid,cue

def ranking_loss(lp,xt,x0,active,codes,mask_id,period):
    """Hard negatives from compatible catalog tuples; exclude ALL row positives.

    Known digits restrict support. This is a conditional surrogate objective,
    not a claim of exact joint item likelihood or a faithful LETTER loss.
    """
    losses=[]
    width=codes.shape[1]
    for start in range(2,x0.shape[1]-width+1,period):
        pos=torch.arange(start,start+width,device=x0.device)
        valid=active[:,pos].all(-1)&(xt[:,pos]==mask_id).any(-1)
        if not bool(valid.any()):continue
        truth=x0[valid][:,pos];partial=xt[valid][:,pos]
        compat=((partial[:,None,:]==mask_id)|(partial[:,None,:]==codes[None,:,:])).all(-1)
        scores=sum(lp[valid][:,pos[k],codes[:,k]] for k in range(width))
        positive=lp[valid][:,pos].gather(-1,truth[:,:,None]).squeeze(-1).sum(-1)
        # Exclude all known items in the complete training playlist, including
        # every one of the five targets; do not treat another positive as negative.
        full=x0[valid]
        for other in range(2,x0.shape[1]-width+1,period):
            matched=(codes[None,:,:]==full[:,None,other:other+width]).all(-1)
            compat &= ~matched
        negative=scores.masked_fill(~compat,-torch.inf)
        usable=compat.any(-1)
        if bool(usable.any()):
            hardest=negative[usable].max(-1).values
            losses.append(F.softplus(hardest-positive[usable]).mean())
    return torch.stack(losses).mean() if losses else lp.sum()*0

def install(cls,Loss,variant):
    if variant not in VARIANTS:raise ValueError(variant)
    original_loss=cls._loss;original_forward=cls.forward;original_q=cls.q_xt
    stats={'variant':variant,'loss_calls':0,'forward_calls':0,'finite':True}
    def forward(self,x,*args,**kwargs):
        lp=original_forward(self,x,*args,**kwargs)
        stats['forward_calls']+=1
        if self.training and variant=='sid_temperature':
            sid,_=channel_masks(x.shape,self.tokenizer,x.device)
            mask=(x==self.mask_index)&sid
            z=lp[mask]/.8;z=z-z.logsumexp(-1,keepdim=True)
            lp=lp.clone();lp[mask]=z
        if self.training and variant=='hard_negative':
            self._trial_lp=lp;self._trial_xt=x
        return lp
    def q(self,x,move):
        if not self.training or variant not in ('whole_item_mask','conditional_mask'):
            return original_q(self,x,move)
        if variant=='whole_item_mask':
            # Half examples preserve original independent token corruption;
            # half use one uniform random value per item (marginals unchanged).
            token_u=torch.rand(x.shape,device=x.device)
            group=((torch.arange(x.shape[1],device=x.device)-1)//self.tokenizer.tokens_per_item).clamp_min(0)
            item_u=torch.rand((len(x),int(group.max())+1),device=x.device)[:,group]
            use_item=torch.rand((len(x),1),device=x.device)<.5
            u=torch.where(use_item,item_u,token_u)
            return torch.where(u<move,self.mask_index,x)
        # Three equally likely channel tasks. These change the corruption law;
        # report as an auxiliary task mixture, not the original diffusion bound.
        sid,cue=channel_masks(x.shape,self.tokenizer,x.device)
        task=torch.randint(3,(len(x),1),device=x.device)
        enabled=((task==0)&sid)|((task==1)&cue)|(task==2)
        return original_q(self,x,move*enabled)
    def loss(self,x0,attention_mask,target_mask=None,**kwargs):
        result,out=original_loss(self,x0,attention_mask,target_mask=target_mask,**kwargs)
        if self.training:
            stats['loss_calls']+=1
            if variant=='hard_negative':
                if not hasattr(self,'_trial_catalog'):
                    ids=sorted(self.tokenizer.item_id_to_row,key=self.tokenizer.item_id_to_row.get)
                    self._trial_catalog=torch.tensor([self.tokenizer.semantic_tokens[i] for i in ids],device=x0.device)
                extra=ranking_loss(self._trial_lp,self._trial_xt,x0,result.token_mask,
                                   self._trial_catalog,self.mask_index,self.tokenizer.tokens_per_item)
                total=result.loss+.1*extra
                self.log('trial/ranking_loss',extra.detach(),on_step=True,on_epoch=False)
                result=Loss(loss=total,nlls=result.nlls,token_mask=result.token_mask,unweighted_nlls=result.unweighted_nlls)
            if not bool(torch.isfinite(result.loss)):
                stats['finite']=False;raise FloatingPointError('Nonfinite training loss')
        return result,out
    cls.forward=forward;cls.q_xt=q;cls._loss=loss
    return stats
