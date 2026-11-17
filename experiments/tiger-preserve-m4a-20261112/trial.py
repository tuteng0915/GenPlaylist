"""Finite SID-warmstarted Music4All trial. No existing outputs are modified."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, ast, fcntl, hashlib, importlib.util, json, math, signal, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

R=Path('/storage_fast/ttu/GenPlaylist')
E=R/'experiments/tiger-preserve-m4a-20261112'
O=R/'outputs/tiger-preserve-m4a-20261112'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def project_aux(primary,aux):
    """Remove only auxiliary gradient's component opposing primary gradient."""
    dot=sum((a.float()*b.float()).sum() for a,b in zip(primary,aux))
    norm=sum((a.float()*a.float()).sum() for a in primary)
    anorm=sum((a.float()*a.float()).sum() for a in aux)
    coeff=torch.minimum(dot,torch.zeros_like(dot))/norm.clamp_min(1e-20)
    corrected=[b-coeff*a for a,b in zip(primary,aux)]
    cosine=dot/(norm*anorm).sqrt().clamp_min(1e-20)
    return corrected,cosine,dot

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['sid','warm','ramp','project'],required=True)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    dest=O/('smoke' if a.smoke else 'full')/a.arm;dest.mkdir(parents=True,exist_ok=True)
    lock=(dest/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.manual_seed(101);np.random.seed(101);torch.cuda.manual_seed_all(101)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    device='cuda';assert torch.cuda.is_available()
    core=module('preserve_joint_core',R/'experiments/tiger-joint-20261108/core.py')
    native_path=R/'v2-recovery-20261106/code/scripts/train_eval_tiger.py';sys.path.insert(0,str(native_path.parent))
    native=module('preserve_native',native_path)
    from datasets import load_from_disk
    from transformers import Adafactor
    from types import SimpleNamespace
    prep=R/'original-ncl/data/processed/genplaylist-music4all-onion-v3-u500-8cue-k5-u8-u3-cap16'
    sp=R/'experiments/two-dataset-fair-20261108/split.json';split=json.loads(sp.read_text())
    assert sha(sp)=='2d4443d88d2e153a6e0b6d1fc72d76568cd979d1f52fd0d9a161bd9deff8f2a7'
    assert sha(prep/'prepared_manifest.json')==split['source_manifest_sha256']
    ids=list(map(str,json.loads((prep/'vectors/catalog_item_ids.json').read_text())));lookup={s:i for i,s in enumerate(ids)}
    codes=np.load(prep/'vectors/catalog_semantic_tokens.npy');cues=np.load(prep/'vectors/catalog_active_cues.npy')
    cue_dir=R/'original-ncl/data/processed/cues-music4all-onion-v3-u500/latest'
    itemc=json.loads((cue_dir/'item2cues.json').read_text());assert cues.tolist()==[itemc[s][:8] for s in ids]
    cat=core.Catalog(codes,cues,len(json.loads((cue_dir/'cue_vocab.json').read_text())))
    ck=R/'outputs/m4a-v3-baselines-20261106/tiger.ckpt';state=torch.load(ck,map_location='cpu',weights_only=False)
    assert state['step']==20000
    cfg=dict(state['model_config']);cfg.setdefault('dropout',.1);oldv=cfg['vocab_size']
    assert cfg['semantic_token_shift']==cat.shift and oldv==cat.cue_offset
    joint=a.arm!='sid';model=native._build_model(SimpleNamespace(**cfg),oldv)
    model.load_state_dict(state['model_state_dict']);original=model.shared.weight.detach().clone()
    if joint:
        # Explicitly copy every old row. New cue rows are initialized independently.
        model.resize_token_embeddings(cat.vocab_size)
        with torch.no_grad():model.shared.weight[:oldv].copy_(original)
    assert torch.equal(model.shared.weight[:oldv],original)
    # Architecture and all non-vocabulary parameters must match the checkpoint.
    for k,v in state['model_state_dict'].items():
        now=model.state_dict()[k]
        assert torch.equal(now[:oldv] if now.shape!=v.shape else now,v),k
    del state,original
    model=model.to(device);params=list(model.parameters())
    optimizer=Adafactor(params,lr=1e-4,scale_parameter=False,relative_step=False,warmup_init=False,weight_decay=0.)
    raw=load_from_disk(str(prep/'raw_dataset'))
    trainidx=split['training_indices'];valididx=split['validation_indices'][:128]
    assert not set(trainidx)&set(valididx)
    sequences=torch.tensor([[lookup[str(s)] for s in row] for row in raw['train'].select(trainidx)['item_seq']])
    valid=np.array([[lookup[str(s)] for s in row] for row in raw['train'].select(valididx)['item_seq']])
    gen=torch.Generator().manual_seed(20261112)
    protocol=json.loads((E/'protocol.json').read_text())
    identity=dict(arm=a.arm,steps=a.steps,source_checkpoint_sha256=sha(ck),split_sha256=sha(sp),code_sha256=sha(__file__),protocol=protocol,smoke=a.smoke,old_vocab=oldv,new_vocab=model.config.vocab_size)
    ip=dest/'identity.json'
    if ip.exists():assert json.loads(ip.read_text())==identity
    else:dump(ip,identity)
    start=0;digest='00'*32;elapsed=0.;last=dest/'last.ckpt'
    if last.exists():
        c=torch.load(last,map_location='cpu',weights_only=False);assert c['identity']==identity
        model.load_state_dict(c['model']);optimizer.load_state_dict(c['optimizer']);gen.set_state(c['sampling_rng'])
        torch.set_rng_state(c['torch_rng']);torch.cuda.set_rng_state_all(c['cuda_rng']);start=c['step'];digest=c['digest'];elapsed=c['seconds'];del c
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    tic=time.monotonic()
    def checkpoint(step):
        t=last.with_suffix('.tmp');torch.save(dict(identity=identity,step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),sampling_rng=gen.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),digest=digest,seconds=elapsed+time.monotonic()-tic),t);os.replace(t,last)
        dump(dest/'progress.json',dict(step=step,steps=a.steps,seconds=elapsed+time.monotonic()-tic,digest=digest))
    def decode(refs):
        refs=np.asarray(refs);x=(cat.items[refs].reshape(len(refs),-1) if joint else torch.tensor(cat.codes[refs].reshape(len(refs),-1))).to(device);mask=torch.ones_like(x)
        trie=native.SemanticTrie(cat.codes) if not joint else None
        def allowed(b,p):return cat.allowed_sid(p.tolist()[1:],set()) if joint else trie.allowed(tuple(p.tolist()[1:][:4]),set())
        kw=dict(input_ids=x,attention_mask=mask,num_beams=20,num_return_sequences=5,do_sample=False,prefix_allowed_tokens_fn=allowed)
        kw.update(dict(min_new_tokens=4,max_new_tokens=4,eos_token_id=None,forced_eos_token_id=None) if joint else dict(max_new_tokens=5,early_stopping=True))
        g=model.generate(**kw);pred=np.array([cat.lookup[tuple(row)] for row in g[:,1:5].cpu().tolist()]).reshape(len(refs),5)
        assert all(len(set(row))==5 for row in pred)
        if joint:
            q=model.generate(input_ids=x.repeat_interleave(5,0),attention_mask=mask.repeat_interleave(5,0),decoder_input_ids=g[:,:5],num_beams=1,do_sample=False,min_new_tokens=8,max_new_tokens=8,eos_token_id=None,forced_eos_token_id=None,prefix_allowed_tokens_fn=lambda b,p:cat.cue_token_ids)
            cue=(q[:,5:13].cpu().numpy()-cat.cue_offset).reshape(len(refs),5,8)
        else:cue=cues[pred]
        return pred,cue
    def evaluate(seq,tag):
        model.eval();pred=[];cq=[]
        with torch.inference_mode():
            for i in range(0,len(seq),8):
                r,q=decode(seq[i:i+8,:15]);pred.extend(r.tolist());cq.extend(q.tolist())
        rows=np.array(ids)[np.array(pred)].tolist();truth=np.array(ids)[seq[:,15:]].tolist()
        recall=[];f1=[]
        for r,t,q in zip(rows,truth,cq):
            recall.append(sum((Counter(r)&Counter(t)).values())/5)
            ac=Counter(v for song in q for v in song);bc=Counter(v for s in t for v in itemc[s][:8]);f1.append(2*sum((ac&bc).values())/(sum(ac.values())+sum(bc.values())))
        metrics=dict(recall_at_5=float(np.mean(recall)),cue_f1=float(np.mean(f1)),diversity=len(set(s for r in rows for s in r))/len(ids),contexts=len(seq))
        dump(dest/(tag+'.json'),dict(metrics=metrics,predictions=dict(item_ids=rows,cue_ids=cq,target_item_ids=truth),per_context=dict(recall_at_5=recall,cue_f1=f1),checkpoint_sha256=sha(last) if last.exists() else None))
        print(json.dumps(dict(evaluation=tag,**metrics)),flush=True);model.train()
    model.train()
    for step in range(start+1,a.steps+1):
        bs=8 if a.smoke else 256
        x,mask,y,rr,ss=cat.batch(sequences,bs,gen,True,'cpu')
        if not joint:
            x=x.reshape(bs,19,12)[:,:,:4].reshape(bs,-1);mask=mask.reshape(bs,19,12)[:,:,:4].reshape(bs,-1);y=torch.cat([y[:,:4],torch.ones((bs,1),dtype=y.dtype)],1)
        x=x.to(device);mask=mask.to(device);y=y.to(device)
        digest=hashlib.sha256(bytes.fromhex(digest)+rr.numpy().tobytes()+ss.numpy().tobytes()).hexdigest()
        freeze=joint and step<=(1 if a.smoke else 200)
        lr=1e-4*min(1.,step/100)
        for group in optimizer.param_groups:group['lr']=lr
        weight=.1 if a.arm!='ramp' else .1*min(1.,max(0.,(step-200)/800))
        # Identical 200-step cue-row bootstrap for joint arms; ramp starts afterwards.
        if freeze or a.smoke:weight=.1
        optimizer.zero_grad(set_to_none=True)
        out=model(input_ids=x,attention_mask=mask,labels=y)
        ce=F.cross_entropy(out.logits.float().transpose(1,2),y,reduction='none')
        sid=ce[:,:4].mean();cue=ce[:,4:12].mean() if joint else sid.new_zeros(());eos=ce[:,-1].mean()
        primary=sid+.1*eos;aux=weight*cue;loss=primary+aux;cosine=None
        if freeze:
            aux.backward()
            for par in params:
                if par is model.shared.weight:par.grad[:oldv].zero_()
                else:par.grad=None
        elif joint and (a.arm=='project' or step%50==0):
            gs=torch.autograd.grad(primary,params,retain_graph=True,allow_unused=True)
            gc=torch.autograd.grad(aux,params,allow_unused=True)
            gs=[torch.zeros_like(p) if g is None else g for g,p in zip(gs,params)];gc=[torch.zeros_like(p) if g is None else g for g,p in zip(gc,params)]
            projected,co,dot=project_aux(gs,gc);cosine=float(co)
            for par,s,c in zip(params,gs,projected if a.arm=='project' else gc):par.grad=s+c
        else:loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
        if step%25==0 or a.smoke:
            row=dict(step=step,sid_loss=float(sid),cue_loss=float(cue),eos_loss=float(eos),cue_weight=weight,freeze_old_parameters=freeze,grad_cosine=cosine,grad_norm=float(norm),lr=lr,seconds=elapsed+time.monotonic()-tic)
            with (dest/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if step%250==0 or step==a.steps or stop[0]:checkpoint(step)
        if stop[0]:print('Saved checkpoint on deadline',flush=True);return
        if step%500==0 and not a.smoke:evaluate(valid,f'validation-step{step}')
    if a.smoke:
        evaluate(valid[:8],'smoke-eval');dump(dest/'complete.json',{'steps':a.steps});return
    manifest=json.loads((R/'outputs/sampled-candidates-20261110/m4a/candidates.json').read_text())
    test=np.array([[lookup[str(s)] for s in row] for row in raw['test'].select(manifest['indices'])['item_seq']])
    assert np.array(ids)[test[:,15:]].tolist()==manifest['truth'] and len(test)==941
    evaluate(test,'test-step2000');dump(dest/'complete.json',dict(steps=a.steps,checkpoint_sha256=sha(last),test_report_sha256=sha(dest/'test-step2000.json'),digest=digest))

if __name__=='__main__':main()
