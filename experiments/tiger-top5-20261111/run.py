"""Frozen-checkpoint AR5 vs one-pass top5. CPU-only; separate finite experiment."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse, json, hashlib, importlib.util, sys, time, fcntl
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
R=Path('/storage_fast/ttu/GenPlaylist');E=R/'experiments/tiger-top5-20261111';O=R/'outputs/tiger-top5-20261111'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def decode(model,cat,native,codes,refs,joint,mode):
 histories=np.asarray(refs).copy();pred=[];cues=[];sid_scores=[]
 for slot in range(5 if mode=='ar5' else 1):
  if joint:x,mask=cat.inputs(histories,'cpu')
  else:x=torch.tensor(codes[histories].reshape(len(histories),-1));mask=torch.ones_like(x)
  excluded=[set(int(row[b]) for row in pred) for b in range(len(histories))]
  def allowed(b,prefix):
   v=prefix.tolist()[1:]
   return cat.allowed_sid(v,excluded[b]) if joint else cat.allowed(tuple(v[:4]),excluded[b])
  n=1 if mode=='ar5' else 5
  kwargs=dict(input_ids=x,attention_mask=mask,do_sample=False,num_beams=20,num_return_sequences=n,
              prefix_allowed_tokens_fn=allowed,return_dict_in_generate=True,output_scores=True)
  if joint:kwargs.update(min_new_tokens=4,max_new_tokens=4,eos_token_id=None,forced_eos_token_id=None)
  else:kwargs.update(max_new_tokens=5,early_stopping=True)
  with torch.inference_mode():
   g=model.generate(**kwargs);gen=g.sequences
   lookup=cat.lookup if joint else cat.tuple_to_row
   chosen=np.array([lookup[tuple(row)] for row in gen[:,1:5].tolist()])
   scores=g.sequences_scores.cpu().numpy().reshape(len(histories),n)
   if joint:
    full=model.generate(input_ids=x.repeat_interleave(n,dim=0),attention_mask=mask.repeat_interleave(n,dim=0),
          decoder_input_ids=gen[:,:5],do_sample=False,num_beams=1,min_new_tokens=8,max_new_tokens=8,
          eos_token_id=None,forced_eos_token_id=None,prefix_allowed_tokens_fn=lambda b,p:cat.cue_token_ids)
    cq=(full[:,5:13].cpu().numpy()-cat.cue_offset).reshape(len(histories),n,8)
   else:cq=np.empty((len(histories),n,0),dtype=np.int64)
  chosen=chosen.reshape(len(histories),n)
  if mode=='top5':
   assert all(len(set(row))==5 for row in chosen),'Beam returned duplicate catalog IDs; do not silently resample'
   return chosen,cq,scores
  pred.append(chosen[:,0]);cues.append(cq[:,0]);sid_scores.append(scores[:,0])
  histories=np.concatenate([histories,chosen],axis=1)
 return np.stack(pred,axis=1),np.stack(cues,axis=1),np.stack(sid_scores,axis=1)

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['mpd','m4a'],required=True);p.add_argument('--arm',choices=['sid','joint'],required=True);p.add_argument('--mode',choices=['ar5','top5'],required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(1);torch.use_deterministic_algorithms(True)
 assert not torch.cuda.is_available(),'CUDA must remain invisible'
 name=f'{a.dataset}-{a.arm}-{a.mode}';dest=O/('pilot' if a.pilot else 'full')/name;dest.mkdir(parents=True,exist_ok=True)
 lock=(dest/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cpath=R/'outputs/sampled-candidates-20261110'/a.dataset/'candidates.json';c=json.loads(cpath.read_text());ids=c['catalog_ids'];lookup={s:i for i,s in enumerate(ids)}
 prep=R/'original-ncl/data/processed'/('genplaylist-v4-8cue-20item-joint-15to5' if a.dataset=='mpd' else 'genplaylist-music4all-onion-v3-u500-8cue-k5-u8-u3-cap16')
 assert json.loads((prep/'vectors/catalog_item_ids.json').read_text())==ids
 canonical=np.load(prep/'vectors/catalog_semantic_tokens.npy');cuearr=np.load(prep/'vectors/catalog_active_cues.npy')
 rp=R/'outputs/list5-audio-20261111'/a.dataset/('tiger_'+a.arm)/'render-plan.json';render=json.loads(rp.read_text());sp=Path(render['source_plan']);assert sha(sp)==render['source_plan_sha256'];source=json.loads(sp.read_text());assert source['predictions']['target_item_ids']==c['truth']
 joint=a.arm=='joint';native=None
 if joint:
  corepath=R/'experiments/tiger-joint-20261108/core.py';core=module('tiger_top5_core',corepath)
  cd=R/('original-ncl/src/02_creative_cues/outputs/production/20260804_172128' if a.dataset=='mpd' else 'original-ncl/data/processed/cues-music4all-onion-v3-u500/latest')
  cat=core.Catalog(canonical,cuearr,len(json.loads((cd/'cue_vocab.json').read_text())));codes=cat.codes
  ck=R/'outputs/tiger-joint-20261108'/a.dataset/'joint/last.ckpt';state=torch.load(ck,map_location='cpu',weights_only=False);model=core.build_model(cat.vocab_size);model.load_state_dict(state['model'])
  expected=source['inference']['checkpoint_sha256']
 else:
  corepath=R/'v2-recovery-20261106/code/scripts/train_eval_tiger.py';sys.path.insert(0,str(corepath.parent));import train_eval_tiger as native
  ck=R/('original-ncl/src/03_backbone_recommender/outputs/baselines/tiger-15to5-lr1e-3-step20000.ckpt' if a.dataset=='mpd' else 'outputs/m4a-v3-baselines-20261106/tiger.ckpt')
  state=torch.load(ck,map_location='cpu',weights_only=False);codes,shift=native._reserve_special_tokens(canonical);cfg=dict(state['model_config']);assert cfg.get('semantic_token_shift',0)==shift;cfg.setdefault('dropout',.1)
  model=native._build_model(SimpleNamespace(**cfg),int(codes.max())+1);model.load_state_dict(state['model_state_dict']);cat=native.SemanticTrie(codes);expected=source['checkpoint_sha256']
 assert sha(ck)==expected and state['step']==20000;del state
 model=model.cpu().float().eval();n=8 if a.pilot else 941
 identity=dict(dataset=a.dataset,arm=a.arm,mode=a.mode,checkpoint_sha256=expected,steps=20000,
     source_plan=str(sp),source_plan_sha256=sha(sp),candidate_manifest_sha256=sha(cpath),code_sha256=sha(Path(__file__)),core_sha256=sha(corepath),
     contexts=n,indices=c['indices'][:n],candidate_setting='full_catalog',candidate_count=len(ids),beam_width=20,
     returned_per_call=1 if a.mode=='ar5' else 5,device='cpu',dtype='float32',threads=2,batch_size=8,
     original_history_items=15,history_allowed=True,distinct_outputs=5,cue_decode='greedy8 after fixed SID; original full cue vocabulary' if joint else 'retrieved catalog cues for evaluation only; model input/output SID only',
     eos='rank4 SID tokens; cue and EOS do not rerank SID' if joint else 'native4 SID plus EOS',
     fairness='same checkpoint, precision, device, width and support; changed conditioning/number of searches. Runtime differs and is reported, not a FLOP-matched comparison.')
 ip=dest/'identity.json'
 if ip.exists():assert json.loads(ip.read_text())==identity
 else:save(ip,identity)
 tic=time.monotonic();refs=np.array([[lookup[s] for s in row] for row in c['histories'][:n]])
 for start in range(0,n,8):
  f=dest/f'batch-{start:04d}.npz'
  if f.exists():continue
  t=time.monotonic();pr,cq,ss=decode(model,cat,native,codes,refs[start:start+8],joint,a.mode)
  assert pr.shape==(min(8,n-start),5) and all(len(set(row))==5 for row in pr)
  tmp=f.with_suffix('.tmp')
  with tmp.open('wb') as out:np.savez(out,pred=pr,cues=cq,scores=ss,seconds=time.monotonic()-t)
  os.replace(tmp,f);print(name,start+len(pr),n,flush=True)
 batches=[np.load(dest/f'batch-{s:04d}.npz') for s in range(0,n,8)];pr=np.concatenate([b['pred'] for b in batches]);cq=np.concatenate([b['cues'] for b in batches]);ss=np.concatenate([b['scores'] for b in batches]);elapsed=sum(float(b['seconds']) for b in batches)
 payload=dict(identity=identity,inference_seconds=elapsed,seconds_this_invocation=time.monotonic()-tic,
       predictions=dict(item_ids=np.array(ids)[pr].tolist(),target_item_ids=c['truth'][:n],cue_ids=cq.tolist(),semantic_token_ids=canonical[pr].tolist()),sid_sequence_scores=ss.tolist())
 if a.mode=='ar5':payload['comparison_to_historical_gpu']=dict(identical_lists=sum(x==y for x,y in zip(payload['predictions']['item_ids'],source['predictions']['item_ids'])),total=n,note='Diagnostic only; primary contrast uses fresh CPU AR5 vs CPU top5.')
 save(dest/'plan.json',payload);save(dest/'complete.json',dict(contexts=n,plan_sha256=sha(dest/'plan.json'),inference_seconds=elapsed));print('COMPLETE',name,elapsed,flush=True)

if __name__=='__main__':main()
