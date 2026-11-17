"""Check output permutation, per-channel weights, resume and lawful cue-first decoding."""
import copy,json
import numpy as np
import torch
import tiger_original as original
import tiger_cue_first as cuefirst
torch.set_num_threads(2)
codes=np.array([[2+i//8,20+(i//4)%2,30+(i//2)%2,40+i%2] for i in range(8)])
cues=np.array([[(i+j)%7+1 for j in range(8)] for i in range(8)])
old=original.Catalog(codes,cues,8);new=cuefirst.Catalog(codes,cues,8)
seq=torch.tensor([[i%8 for i in range(20)],[(i+3)%8 for i in range(20)]])
g1=torch.Generator().manual_seed(42);g2=torch.Generator().manual_seed(42)
x1,m1,y1,r1,s1=old.batch(seq,12,g1,True,'cpu');x2,m2,y2,r2,s2=new.batch(seq,12,g2,True,'cpu')
assert torch.equal(x1,x2) and torch.equal(m1,m2) and torch.equal(r1,r2) and torch.equal(s1,s2)
permutation=list(range(4,12))+list(range(4))+[12]
assert torch.equal(y2,y1[:,permutation])
logits=torch.randn(12,13,old.vocab_size)
l1,p1=original.loss_parts(logits,y1,True);l2,p2=cuefirst.loss_parts(logits[:,permutation],y2,True)
assert torch.allclose(l1,l2) and all(torch.allclose(p1[k],p2[k]) for k in p1)
torch.manual_seed(1);model=original.build_model(old.vocab_size,tiny=True)
torch.manual_seed(1);other=cuefirst.build_model(new.vocab_size,tiny=True)
assert original.fingerprint(model)==cuefirst.fingerprint(other)
loss,_=cuefirst.loss_parts(other(input_ids=x2,attention_mask=m2,labels=y2).logits,y2,True)
loss.backward();assert torch.isfinite(loss) and other.shared.weight.grad[new.cue_offset:].abs().sum()>0
pred,c=cuefirst.predict(other.eval(),new,seq[:,:15].numpy(),True,'cpu',beam=4)
assert pred.shape==(2,5) and c.shape==(2,5,8) and all(len(set(row))==5 for row in pred)
assert set(seq[0,:15].tolist())==set(range(8))
from transformers import Adafactor
def fresh():
    torch.manual_seed(7);m=cuefirst.build_model(new.vocab_size,tiny=True)
    return m,Adafactor(m.parameters(),lr=.001,scale_parameter=False,relative_step=False,warmup_init=False),torch.Generator().manual_seed(7)
def steps(m,opt,g,n):
    m.train()
    for _ in range(n):
        x,mask,y,_,_=new.batch(seq,4,g,True,'cpu');opt.zero_grad(set_to_none=True)
        loss,_=cuefirst.loss_parts(m(input_ids=x,attention_mask=mask,labels=y).logits,y,True)
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
m,o,g=fresh();steps(m,o,g,4);expected=cuefirst.fingerprint(m)
m,o,g=fresh();steps(m,o,g,2);state=copy.deepcopy((m.state_dict(),o.state_dict(),g.get_state(),torch.get_rng_state()))
m,o,g=fresh();m.load_state_dict(state[0]);o.load_state_dict(state[1]);g.set_state(state[2]);torch.set_rng_state(state[3]);steps(m,o,g,2)
assert cuefirst.fingerprint(m)==expected
print(json.dumps({'passed':True,'tests':['same encoder and sampled cases','only output positions permuted','same per-channel loss weights','same initial weights','cue gradients','legal unique SID search with cue prefix','history remains eligible','exact resume']}))
