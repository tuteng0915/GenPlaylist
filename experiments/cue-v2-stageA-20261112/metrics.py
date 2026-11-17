"""Stage A gate metrics. No CLAP. Every score retains the 200 fixed contexts."""
import argparse
import numpy as np
from common import *
from gate import ARMS,SYSTEM,validate
CODE=R/'v2-recovery-20261106/code'
MERT_REV='12af15fef9d0ac838c3f475bfbbf26d2060dd4f5'

def unit(x):
 x=np.asarray(x,dtype=np.float64);norm=np.linalg.norm(x,axis=-1,keepdims=True)
 assert np.isfinite(x).all() and np.all(norm>1e-12)
 return x/norm

def oas_one_to_five(predicted,target):
 p,t=unit(predicted),unit(target)
 assert p.ndim==2 and t.shape==(len(p),5,p.shape[1])
 sims=np.clip(np.einsum('nd,nkd->nk',p,t),-1,1)
 # Exactly DDBC rectangular Hungarian cost / number of predictions (one).
 return sims.max(1),sims

def interval(v,seed=42,repetitions=10000):
 v=np.asarray(v,dtype=float);assert v.shape==(200,) and np.isfinite(v).all()
 rng=np.random.default_rng(seed);means=[]
 for k in range(0,repetitions,200):means.extend(v[rng.integers(200,size=(min(200,repetitions-k),200))].mean(1))
 return np.quantile(means,[.025,.975]).tolist()

def stats(v):return dict(mean=float(np.mean(v)),ci95=interval(v),contexts=len(v))

def mert(ds,arm,limit=None):
 import torch
 from transformers import AutoModel,Wav2Vec2FeatureExtractor
 sys.path[:0]=[str(CODE/'scripts'),str(CODE/'src')]
 from extract_mert_embeddings import _load_center_audio,_pool_valid_frames
 started=time.time();b=O/ds/'gate'/arm;rows=json.loads((b/'render-plan.json').read_text())['rows'];model=processor=None
 if limit:rows=rows[:limit]
 # Audio decode validation uses the existing ACE environment before this task.
 for row in rows:
  i=row['local_index'];audio=b/'synthesis'/SYSTEM/f'{i:04d}.mp3';dest=b/'metrics/mert-rows'/f'{i:04d}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
  identity=dict(audio_sha256=sha(audio),revision=MERT_REV,sample_rate=24000,clip_seconds=30,render_row=row,pooling='attention-masked final-layer mean',normalization='L2')
  if dest.exists():assert json.loads(str(np.load(dest)['identity']))==identity;continue
  if model is None:
   processor=Wav2Vec2FeatureExtractor.from_pretrained('m-a-p/MERT-v1-95M',revision=MERT_REV,trust_remote_code=True);model=AutoModel.from_pretrained('m-a-p/MERT-v1-95M',revision=MERT_REV,trust_remote_code=True).eval().cuda()
  wave=_load_center_audio(audio,24000,720000);batch=processor([wave],sampling_rate=24000,padding=True,return_attention_mask=True,return_tensors='pt');batch={k:v.cuda() for k,v in batch.items()}
  with torch.inference_mode():h=model(**batch).last_hidden_state;v=torch.nn.functional.normalize(_pool_valid_frames(model,h,batch.get('attention_mask')).float(),dim=-1)[0].cpu().numpy()
  assert v.shape==(768,) and np.isfinite(v).all();tmp=dest.with_suffix('.tmp')
  with tmp.open('wb') as f:np.savez(f,embedding=v,identity=json.dumps(identity,sort_keys=True))
  os.replace(tmp,dest)
 atomic(b/'metrics/mert-complete.json',dict(rows=len(rows),elapsed_seconds=time.time()-started))

def vggish(ds,arm,limit=None):
 import torch,soundfile as sf,librosa
 from frechet_audio_distance import FrechetAudioDistance
 started=time.time();evaluator=None;co=json.loads((O/ds/'cohort.json').read_text());audio_paths=json.loads((O/ds/'audio-paths.json').read_text());jobs=[]
 if arm=='reference':
  for row in co['gate_rows']:jobs.append((row,Path(audio_paths[row['condition_item']])))
 else:
  for row in co['gate_rows']:jobs.append((row,O/ds/'gate'/arm/'synthesis'/SYSTEM/f"{row['local_index']:04d}.mp3"))
 if limit:jobs=jobs[:limit]
 for row,p in jobs:
  dest=O/ds/'vggish'/arm/f"{row['local_index']:04d}.npz";dest.parent.mkdir(parents=True,exist_ok=True);identity=dict(audio_sha256=sha(p),source_path=str(p),source_history_index=row['source_history_index'],sample_rate=16000,clip_seconds=30,center_crop=True,reference='first target, same as frozen first-item FAD protocol',model='VGGish',use_pca=False,use_activation=False)
  if dest.exists():assert json.loads(str(np.load(dest)['identity']))==identity;continue
  if evaluator is None:evaluator=FrechetAudioDistance(model_name='vggish',sample_rate=16000,channels=1,use_pca=False,use_activation=False,verbose=False,audio_load_worker=1)
  wave,sr=sf.read(p,dtype='float32',always_2d=True);wave=wave.mean(1)
  if len(wave)>30*sr:start=(len(wave)-30*sr)//2;wave=wave[start:start+30*sr]
  if sr!=16000:wave=librosa.resample(wave,orig_sr=sr,target_sr=16000)
  if len(wave)<480000:wave=np.pad(wave,(0,480000-len(wave)))
  with torch.inference_mode():emb=np.asarray(evaluator.get_embeddings([wave],16000))
  assert emb.ndim==2 and emb.shape[1]==128 and np.isfinite(emb).all();tmp=dest.with_suffix('.tmp')
  with tmp.open('wb') as f:np.savez(f,embedding=emb,identity=json.dumps(identity,sort_keys=True))
  os.replace(tmp,dest)
 atomic(O/ds/'vggish'/arm/'complete.json',dict(elapsed_seconds=time.time()-started,rows=len(jobs),checkpoint_sha256={p.name:sha(p) for p in Path(torch.hub.get_dir()).glob('checkpoints/vggish*.pth')}))

def text_embedding(model,texts):
 # Use every text chunk; avoid silently dropping lyrics after long attributes.
 chunks=[];spans=[]
 for text in texts:
  tokens=model.tokenizer.encode(text,add_special_tokens=False);start=len(chunks)
  chunks.extend(model.tokenizer.decode(tokens[k:k+300],skip_special_tokens=True) for k in range(0,len(tokens),300))
  if len(chunks)==start:chunks.append('')
  spans.append((start,len(chunks)))
 e=unit(model.encode(chunks,batch_size=64,convert_to_numpy=True))
 return unit(np.stack([e[s:t].mean(0) for s,t in spans]))

def content(ds):
 from sentence_transformers import SentenceTransformer
 started=time.time();model=SentenceTransformer('sentence-transformers/all-mpnet-base-v2',device='cuda');co=json.loads((O/ds/'cohort.json').read_text());refs={};none={}
 for version in ['v1','v2']:
  vocab=json.loads(((paths(ds)['cues']/'cue_vocab.json') if version=='v1' else O/'v2/cue_vocab.json').read_text());mapping=json.loads(((paths(ds)['cues']/'item2cues.json') if version=='v1' else O/ds/'v2-item2cues.json').read_text());texts=[];mask=[]
  for row in co['gate_rows']:
   for i in row['targets']:
    terms=[vocab[c] for c in mapping[i][:8] if c and not vocab[c].startswith('<')];texts.append(', '.join(terms));mask.append(bool(terms))
  refs[version]=text_embedding(model,texts).reshape(200,5,-1);none[version]=np.asarray(mask).reshape(200,5)
 result={}
 for arm in ARMS:
  rows=[json.loads((O/ds/'gate'/arm/'verbalization'/SYSTEM/f'{i:04d}.json').read_text()) for i in range(200)]
  attr=text_embedding(model,[r['music_attributes'] for r in rows]);lyric=text_embedding(model,[r['lyric_draft'] for r in rows]);q=unit(attr+lyric)
  for version in ['v1','v2']:
   sims=np.einsum('nd,nkd->nk',q,refs[version]);sims=np.where(none[version],sims,0.)
   # No evidence is scored zero, never removed from denominator.
   result[arm+'__content_fit_'+version]=sims.mean(1);result[arm+'__content_fit_'+version+'_first']=sims[:,0]
 dest=O/ds/'content-per-context.npz';np.savez_compressed(dest,**result,**{v+'__reference_nonempty':m for v,m in none.items()})
 atomic(O/ds/'content-config.json',dict(encoder='sentence-transformers/all-mpnet-base-v2',chunks='300 tokenizer tokens, encode all chunks; normalized mean',query='equal-weight normalized lyric and attribute embeddings',reference='mean similarity across all5 target cue texts; v1/v2 scored separately; missing cue evidence contributes zero',elapsed_seconds=time.time()-started,sha256=sha(dest)))

def fad(emb):
 # sqrt(C1) C2 sqrt(C1) is symmetric PSD; avoids complex sqrtm artefacts.
 x,y=emb;mx,my=x.mean(0),y.mean(0);cx,cy=np.cov(x,rowvar=False),np.cov(y,rowvar=False);vals,vecs=np.linalg.eigh(cx);root=(vecs*np.sqrt(np.maximum(vals,0)))@vecs.T;product=root@cy@root;trace=np.sqrt(np.maximum(np.linalg.eigvalsh((product+product.T)/2),0)).sum()
 return float(max(0,np.sum((mx-my)**2)+np.trace(cx)+np.trace(cy)-2*trace))

def content_scores(content_rows, arm):
 # Presence masks are 200x5 metadata, never scalar score vectors.
 prefix=arm+'__content_fit_'
 return {k.split('__',1)[1]:content_rows[k] for k in content_rows.files if k.startswith(prefix)}

def report(ds):
 started=time.time();x=paths(ds);co=json.loads((O/ds/'cohort.json').read_text());cat=x['catalog'];manifest=json.loads((cat/'mert_manifest.json').read_text());assert manifest['model_revision']==MERT_REV
 for f in ['catalog_item_ids.json','catalog_mert_embeddings_l2.npy']:assert sha(cat/f)==manifest['outputs'][f]
 lookup={str(i):j for j,i in enumerate(json.loads((cat/'catalog_item_ids.json').read_text()))};emb=unit(np.load(cat/'catalog_mert_embeddings_l2.npy'));truth=emb[[[lookup[i] for i in r['targets']] for r in co['gate_rows']]];history=emb[[[lookup[i] for i in r['history']] for r in co['gate_rows']]];per={};out=dict(dataset=ds,contexts=200,definitions=dict(mert_oas='1x5 rectangular optimal assignment cosine; not 5x5 playlist evaluation',history_fit='mean cosine against15 original history songs',fad='VGGish global distance200 generated clips vs200 first-target occurrences;30-second center crop; no matching'),arms={},paired={});content_rows=np.load(O/ds/'content-per-context.npz')
 vg={a:[np.load(O/ds/'vggish'/a/f'{i:04d}.npz')['embedding'].astype(np.float64) for i in range(200)] for a in ARMS+['reference']}
 for arm in ARMS:
  b=O/ds/'gate'/arm;rows=[]
  for i in range(200):
   q=np.load(b/'metrics/mert-rows'/f'{i:04d}.npz');assert json.loads(str(q['identity']))['audio_sha256']==sha(b/'synthesis'/SYSTEM/f'{i:04d}.mp3');rows.append(q['embedding'])
  m=unit(np.stack(rows));scores,sims=oas_one_to_five(m,truth);per[arm]=dict(mert_oas=scores,mert_target_mean=sims.mean(1),mert_first_target=sims[:,0],history_fit=np.einsum('nd,nkd->nk',m,history).mean(1),**content_scores(content_rows,arm))
  out['arms'][arm]={k:stats(v) for k,v in per[arm].items()};out['arms'][arm]['fad']=fad((np.concatenate(vg[arm]),np.concatenate(vg['reference'])))
 # Resample CONTEXTS with common indices across all arms and real audio, retaining frames.
 rng=np.random.default_rng(42);fd={a:[] for a in ARMS};repetitions=1000
 for _ in range(repetitions):
  inds=rng.integers(200,size=200);ref=np.concatenate([vg['reference'][i] for i in inds])
  for a in ARMS:fd[a].append(fad((np.concatenate([vg[a][i] for i in inds]),ref)))
 for a in ARMS:out['arms'][a]['fad_ci95']=np.quantile(fd[a],[.025,.975]).tolist()
 for comparator in ['none','v1','mmr']:
  key='v2 minus '+comparator;out['paired'][key]={k:stats(per['v2'][k]-per[comparator][k]) for k in per['v2']};out['paired'][key]['fad']=dict(delta=out['arms']['v2']['fad']-out['arms'][comparator]['fad'],ci95=np.quantile(np.asarray(fd['v2'])-fd[comparator],[.025,.975]).tolist())
 np.savez_compressed(O/ds/'gate-per-context.npz',**{a+'__'+k:v for a,d in per.items() for k,v in d.items()},**{a+'__fad_bootstrap':v for a,v in fd.items()})
 out.update(elapsed_seconds=time.time()-started,bootstrap=dict(unit='context',paired=True,scalar_repetitions=10000,fad_repetitions=repetitions,seed=42),cohort_sha256=sha(O/ds/'cohort.json'),per_context_sha256=sha(O/ds/'gate-per-context.npz'))
 atomic(O/ds/'gate-report.json',out)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['mert','vggish','content','report']);p.add_argument('--dataset',required=True);p.add_argument('--arm');p.add_argument('--limit',type=int);a=p.parse_args()
 if a.action in ['mert','vggish']:globals()[a.action](a.dataset,a.arm,a.limit)
 else:globals()[a.action](a.dataset)
