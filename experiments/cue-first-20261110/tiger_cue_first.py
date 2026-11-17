"""Output-order intervention only: same encoder, initialization and training cases."""
import numpy as np
import torch
import torch.nn.functional as F
from tiger_original import Catalog as OriginalCatalog, build_model, fingerprint

class Catalog(OriginalCatalog):
    def batch(self,sequences,batch_size,generator,joint,device):
        assert joint
        x,mask,y,rows,slots=super().batch(sequences,batch_size,generator,True,device)
        return x,mask,torch.cat([y[:,4:12],y[:,:4],y[:,12:]],1),rows,slots

def loss_parts(logits,labels,joint):
    assert joint and labels.shape[1]==13
    ce=F.cross_entropy(logits.float().transpose(1,2),labels,reduction='none')
    cue=ce[:,:8].mean();sid=ce[:,8:12].mean();eos=ce[:,12].mean()
    return sid+.1*cue+.1*eos,dict(sid=sid,cue=cue,eos=eos)

@torch.no_grad()
def predict(model,cat,references,joint,device,beam=20):
    assert joint
    histories=np.asarray(references,dtype=np.int64);assert histories.shape[1]==15
    predictions=[];cues=[]
    for slot in range(5):
        x,mask=cat.inputs(histories,device)
        prefix=model.generate(input_ids=x,attention_mask=mask,do_sample=False,num_beams=1,
            min_new_tokens=8,max_new_tokens=8,eos_token_id=None,forced_eos_token_id=None,
            prefix_allowed_tokens_fn=lambda b,p:cat.cue_token_ids)
        assert prefix.shape==(len(histories),9)
        generated_cues=prefix[:,1:].cpu().numpy()-cat.cue_offset
        excluded=[set(int(p[i]) for p in predictions) for i in range(len(histories))]
        def allowed(b,tokens):
            # The eight already generated cue tokens are fixed conditioning,
            # not competing beam hypotheses. Rank exactly four new SID tokens.
            return cat.allowed_sid(tokens.tolist()[9:],excluded[b])
        full=model.generate(input_ids=x,attention_mask=mask,decoder_input_ids=prefix,
            do_sample=False,num_beams=beam,num_return_sequences=1,min_new_tokens=4,max_new_tokens=4,
            eos_token_id=None,forced_eos_token_id=None,prefix_allowed_tokens_fn=allowed)
        assert full.shape==(len(histories),13) and torch.equal(full[:,:9],prefix)
        selected=np.array([cat.lookup[tuple(row)] for row in full[:,9:].cpu().tolist()])
        assert generated_cues.min()>=0 and generated_cues.max()<len(cat.cue_token_ids)
        predictions.append(selected);cues.append(generated_cues)
        histories=np.concatenate([histories,selected[:,None]],1)
    pred=np.stack(predictions,1);assert all(len(set(row))==5 for row in pred)
    return pred,np.stack(cues,1)

@torch.no_grad()
def validation_probe(model,cat,heldout,device):
    state=model.training;model.eval()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            gen=torch.Generator().manual_seed(23001)
            values=[]
            for _ in range(4):
                x,mask,y,_,_=cat.batch(heldout,32,gen,True,device)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                    loss,parts=loss_parts(model(input_ids=x,attention_mask=mask,labels=y).logits,y,True)
                values.append(dict(loss=float(loss),**{k:float(v) for k,v in parts.items()}))
            return {k:float(np.mean([v[k] for v in values])) for k in values[0]}
    finally:model.train(state)
