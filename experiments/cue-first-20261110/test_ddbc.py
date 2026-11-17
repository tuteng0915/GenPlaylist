"""Small model contract checks, including keeping the unvisited channel masked."""
from types import SimpleNamespace
import json
import torch
from ddbc_order import install,cue_completion
torch.set_num_threads(2)
codes=torch.tensor([[2,3,4,5+i] for i in range(8)])
class Tok:
    tokens_per_item=13;n_digit=3
    item_id_to_row={str(i):i for i in range(8)}
    semantic_tokens={str(i):row for i,row in enumerate(codes.tolist())}
    def make_type_mask(self,length):
        legal=torch.zeros(length,23,dtype=torch.bool)
        for p in range(length):
            off=(p-1)%13
            if p==0:legal[p,0]=True
            elif p==length-1:legal[p,21]=True
            elif off==0:legal[p,1]=True
            elif off<=4:legal[p,torch.unique(codes[:,off-1])]=True
            else:legal[p,13:21]=True
        return legal
class Model:
    mask_index=22;device=torch.device('cpu');dtype=torch.float32;tokenizer=Tok()
    config=SimpleNamespace(sampling=SimpleNamespace(cfg_enabled=False,structure_conditioning=False))
    def noise(self,t):return -torch.log(1-.999*t),None
    def forward(self,x,sigma,sequence_mask=None):
        self.observed.append(x.clone())
        legal=self.tokenizer.make_type_mask(x.shape[1])
        logits=torch.zeros(len(x),x.shape[1],23).masked_fill(~legal[None,:,:],-1e9)
        return logits.log_softmax(-1)
    def sample_masked_completion(self,x,mask,steps,eps=1e-5,sequence_mask=None):
        return cue_completion(self,x,mask,sequence_mask,steps,eps)
x=torch.ones(2,80,dtype=torch.long);x[:,0]=0;x[:,-1]=21
for item in range(6):x[:,2+13*item:6+13*item]=codes[0];x[:,6+13*item:14+13*item]=torch.arange(13,21)
cm=torch.zeros_like(x,dtype=torch.bool)
for item in range(1,6):cm[:,2+13*item:14+13*item]=True
x[cm]=22;pos=torch.arange(80);off=(pos-1)%13;sid=cm&(off>=1)&(off<=4);cue=cm&~sid
for order in ['joint','cue_first','sid_first']:
    cls=type('Trial',(Model,),{});install(cls,order);model=cls();model.observed=[];torch.manual_seed(42)
    y=model.sample_masked_completion(x,cm,256,sequence_mask=torch.ones_like(cm))
    assert torch.equal(y[~cm],x[~cm]) and not (y[cm]==22).any()
    for row in y:
        got=[tuple(row[2+13*i:6+13*i].tolist()) for i in range(1,6)]
        assert len(set(got))==5 and all(c in set(map(tuple,codes.tolist())) for c in got)
    if order=='cue_first':
        assert all((z[sid]==22).all() for z in model.observed[:256])
    if order=='sid_first':
        assert all((z[:,torch.where(cue[0])[0]]==22).all() for z in model.observed[:21])
print(json.dumps({'passed':True,'checks':['known tokens unchanged','inactive channel remains MASK','all outputs complete and typed','five legal distinct songs in every order']}))
