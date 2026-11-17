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
        assert torch.equal(x[~cue_mask],fixed[~cue_mask])
    remaining=cue_mask&(x==model.mask_index)
    if remaining.any():
        sigma,_=model.noise(eps*ones)
        last=model.forward(x,sigma,sequence_mask=seq).argmax(-1)
        x=torch.where(remaining,last,x)
    assert torch.equal(x[~cue_mask],fixed[~cue_mask])
    assert not (x[cue_mask]==model.mask_index).any()
    return x
