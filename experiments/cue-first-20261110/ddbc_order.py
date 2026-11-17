"""True first-channel completion; the other channel stays MASK until its turn."""
import torch

@torch.no_grad()
def cue_completion(model,x,cue_mask,seq,steps=256,eps=1e-5):
    fixed=x.clone();x=x.clone();ones=torch.ones(len(x),device=x.device,dtype=model.dtype)
    times=torch.linspace(1,eps,steps+1,device=x.device,dtype=model.dtype)
    for k in range(steps):
        t=times[k]*ones;s=times[k+1]*ones
        sigma,_=model.noise(t);next_sigma,_=model.noise(s)
        probs=model.forward(x,sigma,sequence_mask=seq).exp()
        move=(1-torch.exp(-sigma))[:,None,None]
        next_move=(1-torch.exp(-next_sigma))[:,None,None]
        q=probs*(move-next_move);q[:,:,model.mask_index]=next_move[:,:,0]
        # Match the original sampler's exponential-race categorical draw.
        draw=(q/(-(torch.rand_like(q)+1e-10).log())).argmax(-1)
        x=torch.where(cue_mask & (x==model.mask_index),draw,x)
        x=torch.where(cue_mask,x,fixed)
    remaining=cue_mask&(x==model.mask_index)
    if remaining.any():
        sigma,_=model.noise(eps*ones)
        last=model.forward(x,sigma,sequence_mask=seq).argmax(-1)
        x=torch.where(remaining,last,x)
    assert torch.equal(x[~cue_mask],fixed[~cue_mask])
    assert not (x[cue_mask]==model.mask_index).any()
    return x

@torch.no_grad()
def beam4_dedup(model,x,sid_mask,seq,codes):
    b=len(x);x=x.clone();x[sid_mask]=model.mask_index
    positions=torch.stack([torch.where(row)[0].reshape(5,4) for row in sid_mask])
    assert torch.equal(positions,positions[0].expand_as(positions));ps=positions[0]
    def forward(z,t,sm):
        sigma,_=model.noise(torch.full((len(z),),max(t,1e-5),device=z.device,dtype=model.dtype))
        return model.forward(z,sigma,sequence_mask=sm).float()
    lp=forward(x,1/3,seq)
    scores=torch.stack([sum(lp[:,ps[j,k],codes[:,k]] for k in range(4)) for j in range(5)],1)
    states=x[:,None,:].clone();acc=torch.zeros((b,1),device=x.device)
    for step in range(20):
        width=states.shape[1];flat=states.reshape(b*width,-1)
        lp=forward(flat,(20-step)/60,seq[:,None,:].expand(b,width,-1).reshape(b*width,-1))
        children=[];child_scores=[]
        for bi in range(b):
            xs=[];ss=[]
            for wi in range(width):
                row=flat[bi*width+wi];partial=row[ps]
                compat=((partial[:,None,:]==model.mask_index)|(partial[:,None,:]==codes[None,:,:])).all(-1)
                assert compat.any(-1).all()
                options=[]
                for j,k in torch.nonzero(partial==model.mask_index,as_tuple=False).tolist():
                    allowed=torch.unique(codes[compat[j],k]);v=lp[bi*width+wi,ps[j,k],allowed]
                    v=v-v.logsumexp(0);options.append((float(v.max()),j,k,allowed,v))
                _,j,k,allowed,v=max(options,key=lambda o:o[0])
                vals,idx=torch.topk(v,min(4,len(allowed)))
                for vv,ii in zip(vals,idx):
                    z=row.clone();z[ps[j,k]]=allowed[ii];xs.append(z);ss.append(acc[bi,wi]+vv)
            order=torch.argsort(torch.stack(ss),descending=True).tolist();keep=[];seen=set()
            for i in order:
                signature=tuple(xs[i][ps].reshape(-1).tolist())
                if signature not in seen:keep.append(i);seen.add(signature)
                if len(keep)==4:break
            while len(keep)<4:keep.append(keep[-1])
            children.append(torch.stack([xs[i] for i in keep]));child_scores.append(torch.stack([ss[i] for i in keep]))
        states=torch.stack(children);acc=torch.stack(child_scores)
    result=states[:,0,:].clone()
    lookup={tuple(row):i for i,row in enumerate(codes.cpu().tolist())}
    for bi in range(b):
        used=[]
        for j in range(5):
            candidate=lookup[tuple(result[bi,ps[j]].tolist())]
            sc=scores[bi,j].clone();sc[used]=-torch.inf
            if candidate in used:candidate=int(sc.argmax())
            used.append(candidate);result[bi,ps[j]]=codes[candidate]
        assert len(set(used))==5
    assert torch.equal(result[~sid_mask],x[~sid_mask])
    return result

def install(cls,order):
    assert order in ['joint','cue_first','sid_first']
    original=cls.sample_masked_completion;original_forward=cls.forward
    stats=dict(order=order,forward_calls=0,example_forward_calls=0,batches=0)
    def forward(self,x,*args,**kwargs):
        stats['forward_calls']+=1;stats['example_forward_calls']+=len(x)
        return original_forward(self,x,*args,**kwargs)
    cls.forward=forward
    @torch.no_grad()
    def sample(self,masked_ids,completion_mask,num_steps,eps=1e-5,context_emb=None,mu_c=None,sigma_c2=None,sequence_mask=None):
        assert num_steps==256 and not self.config.sampling.cfg_enabled and not self.config.sampling.structure_conditioning
        x=masked_ids.to(self.device);fixed=x.clone();cm=completion_mask.to(self.device).bool()
        seq=sequence_mask.to(self.device).bool()
        pos=torch.arange(x.shape[1],device=x.device);off=(pos-1)%self.tokenizer.tokens_per_item
        sid=cm&(pos>0)&(off>=1)&(off<=4);cue=cm&~sid
        assert sid.sum()==len(x)*20 and cue.sum()==len(x)*40
        ids=sorted(self.tokenizer.item_id_to_row,key=self.tokenizer.item_id_to_row.get)
        codes=torch.tensor([self.tokenizer.semantic_tokens[i] for i in ids],device=x.device)
        if order=='joint':
            x=original(self,x,cm,256,eps=eps,sequence_mask=seq)
            x=beam4_dedup(self,x,sid,seq,codes)
        elif order=='cue_first':
            x=cue_completion(self,x,cue,seq)
            assert (x[sid]==self.mask_index).all()
            x=beam4_dedup(self,x,sid,seq,codes)
        else:
            x=beam4_dedup(self,x,sid,seq,codes)
            assert (x[cue]==self.mask_index).all()
            x=cue_completion(self,x,cue,seq)
        assert torch.equal(x[~cm],fixed[~cm]) and not (x[cm]==self.mask_index).any()
        legal=torch.as_tensor(self.tokenizer.make_type_mask(x.shape[1]),device=x.device,dtype=torch.bool)
        assert legal.unsqueeze(0).expand(len(x),-1,-1).gather(2,x.unsqueeze(-1)).squeeze(-1)[cm].all()
        stats['batches']+=1
        return x
    cls.sample_masked_completion=sample
    return stats
