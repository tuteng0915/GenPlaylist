"""Frozen first-target-item gate. Same SID/context/renderer/seed; only cue condition varies."""
import argparse
import numpy as np
from common import *
CODE=R/'v2-recovery-20261106/code'
ARMS=['v1','mmr','v2','none']
SYSTEM='GenPlaylist'

def prepare(controls_only=False, only_arm=None):
 for ds in ['mpd','m4a']:
  x=paths(ds);co=json.loads((O/ds/'cohort.json').read_text());source=json.loads((R/'outputs/sampled-candidates-20261110'/ds/'candidates.json').read_text());codes=np.load(x['prepared']/'vectors/catalog_semantic_tokens.npy');code_ids=json.loads((x['prepared']/'vectors/catalog_item_ids.json').read_text());assert len(codes)==len(code_ids) and set(code_ids)==set(source['catalog_ids']);code_map=dict(zip(code_ids,codes.tolist()))
  v1=json.loads((x['cues']/'item2cues.json').read_text());v2={} if controls_only else json.loads((O/ds/'v2-item2cues.json').read_text())
  for arm in ([only_arm] if only_arm else (['v1','none'] if controls_only else ARMS)):
   mmr=json.loads((O/ds/'v1-mmr-gate-item2cues.json').read_text()) if arm=='mmr' else {}
   rows=[]
   for row in co['gate_rows']:
    i=row['condition_item'];rows.append(dict(row,arm=arm,semantic_token_ids=code_map[i],cue_ids=[] if arm=='none' else (v1[i][:8] if arm=='v1' else (mmr[i] if arm=='mmr' else v2[i]))))
   immutable(O/ds/'gate'/arm/'render-plan.json',dict(schema='cue-v2-first-target-gate-v1',rows=rows,cohort_sha256=sha(O/ds/'cohort.json'),conditional_target_access=True,not_end_to_end_recommendation=True,seed='42000 + original941context index',examples=200))

def verbal(ds,arm,start,count):
 started=time.time();x=paths(ds);b=O/ds/'gate'/arm;plan=json.loads((b/'render-plan.json').read_text());rows=plan['rows'][start:start+count];v=module('cuev2_frozen_verbal',CODE/'scripts/run_end_to_end_verbalization.py');vb=v._load_verbalization_module()
 lookup={str(i):int(j) for i,j in json.loads((x['art']/'item_id_to_row.json').read_text()).items()};emb=np.load(x['art']/'catalog_item_embeddings.npy').astype(np.float32);weights=np.load(x['art']/'rvq_codebook_weights.npy')
 meta=v.CatalogItem.load_catalog(str(x['data']/'catalog_metadata.json'));meta={m.item_id:m for m in meta} if isinstance(meta,list) else meta;ordered=[None]*len(lookup)
 for i,j in lookup.items():ordered[j]=meta[i]
 itemc=json.loads((x['cues']/'item2cues.json').read_text());vocab=json.loads(((O/'v2/cue_vocab.json') if arm=='v2' else (x['cues']/'cue_vocab.json')).read_text());caller=None
 immutable(b/'verbalization/manifest.json',dict(model='Qwen/Qwen3-4B-Instruct-2507',revision=REV,decoding='greedy',seed=42,renderer_sha256=sha(CODE/'scripts/run_end_to_end_verbalization.py'),render_plan_sha256=sha(b/'render-plan.json'),cue_vocab_sha256=sha((O/'v2/cue_vocab.json') if arm=='v2' else x['cues']/'cue_vocab.json'),unknown_cues='omit NONE/padding strings from renderer, retain IDs in records'))
 for row in rows:
  i=row['local_index'];dest=b/'verbalization'/SYSTEM/f'{i:04d}.json'
  if dest.exists():assert json.loads(dest.read_text())['render_row']==row;continue
  refs=row['history'];mu,sigma=v._reference_structure(refs,lookup,emb)
  tiny={'predictions':{'item_ids':[[row['condition_item']]],'semantic_token_ids':[[row['semantic_token_ids']]],'cue_ids':[[itemc[row['condition_item']][:8] if arm=='none' else row['cue_ids']]]}}
  g=v._generated_item(system=SYSTEM,example_index=0,reference_ids=refs,mu_c=mu,sigma_c2=sigma,plan=tiny,codebook_weights=weights,item2cues=itemc,cue_intervention='drop' if arm=='none' else 'none')
  neighbors=vb.knn_verbalize(g.z_hat_emb,emb,ordered,5);style=vb.knn_verbalize(g.mu_c_emb,emb,ordered,5);terms=[vocab[c] for c in row['cue_ids'] if c and not vocab[c].startswith('<')]
  if caller is None:
   import torch,random
   random.seed(42);np.random.seed(42);torch.manual_seed(42);caller=v.LocalQwen(model_name='Qwen/Qwen3-4B-Instruct-2507',revision=REV,device='cuda');assert caller.model_revision==REV
  kwargs=dict(reference_music=[meta[t] for t in refs],cue_terms=terms,user_instruction=g.context_prefix.raw_input.strip() if g.context_prefix else '',llm_call=caller)
  attr=vb.generate_music_attributes(neighbors,style,g.sigma_c2,**kwargs);lyr=vb.generate_lyrics(neighbors,style,attr,g.sigma_c2,**kwargs)
  atomic(dest,dict(render_row=row,seed=row['seed'],reference_item_ids=refs,retrieved_item_id=row['condition_item'],cue_ids=row['cue_ids'],cue_terms=terms,neighbor_item_ids=[t.item_id for t in neighbors],style_item_ids=[t.item_id for t in style],music_attributes=attr,lyric_draft=lyr,offensive_screen=bool(OFFENSIVE.search(attr+' '+lyr))))
 print('VERBAL_DONE',ds,arm,start,count,time.time()-started,flush=True)

def synth(ds,arm,start,count):
 started=time.time();b=O/ds/'gate'/arm;plan=json.loads((b/'render-plan.json').read_text());rows=plan['rows'][start:start+count];ace=R/'vendor/ACE-Step';ck=R/'cache/huggingface/hub/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c';sys.path.insert(0,str(ace));pipeline=None
 immutable(b/'synthesis/audio_manifest.json',dict(model='ACE-Step/ACE-Step-v1-3.5B',revision=ck.name,duration_seconds=30,inference_steps=60,guidance_scale=15.,scheduler_type='euler',cfg_type='apg',omega_scale=10.,dtype='bfloat16',format='mp3',seed_rule='42000+original941context_index',render_plan_sha256=sha(b/'render-plan.json')))
 for row in rows:
  i=row['local_index'];vp=b/'verbalization'/SYSTEM/f'{i:04d}.json';record=json.loads(vp.read_text());assert record['render_row']==row;dest=b/'synthesis'/SYSTEM/f'{i:04d}.mp3';mp=dest.with_suffix('.json')
  if dest.exists() and mp.exists():
   d=json.loads(mp.read_text());assert d['audio_sha256']==sha(dest) and d['verbalization_sha256']==sha(vp);continue
  dest.parent.mkdir(parents=True,exist_ok=True)
  if pipeline is None:
   from acestep.pipeline_ace_step import ACEStepPipeline
   pipeline=ACEStepPipeline(checkpoint_dir=str(ck),device_id=0,dtype='bfloat16',torch_compile=False)
  tmp=dest.with_name(f'{i:04d}.{os.getpid()}.partial.mp3');result=pipeline(prompt=record['music_attributes'],lyrics=record['lyric_draft'],audio_duration=30,infer_step=60,guidance_scale=15.,scheduler_type='euler',cfg_type='apg',omega_scale=10.,manual_seeds=[row['seed']],format='mp3',save_path=str(tmp));assert Path(result[0]).resolve()==tmp.resolve();os.replace(tmp,dest)
  atomic(mp,dict(render_row=row,seed=row['seed'],audio_sha256=sha(dest),verbalization_sha256=sha(vp)))
 validate(ds,arm,start,count);print('SYNTH_DONE',ds,arm,start,count,time.time()-started,flush=True)

def validate(ds,arm,start=0,count=200):
 import soundfile as sf
 b=O/ds/'gate'/arm;rows=json.loads((b/'render-plan.json').read_text())['rows'][start:start+count];checked=[]
 for row in rows:
  p=b/'synthesis'/SYSTEM/f"{row['local_index']:04d}.mp3";m=json.loads(p.with_suffix('.json').read_text());vp=b/'verbalization'/SYSTEM/(p.stem+'.json');assert m['render_row']==row and m['seed']==row['seed'] and m['audio_sha256']==sha(p) and m['verbalization_sha256']==sha(vp)
  w,sr=sf.read(p);assert np.isfinite(w).all() and 29<=len(w)/sr<=31 and np.max(np.abs(w))>0
  checked.append(dict(index=row['local_index'],sha256=m['audio_sha256'],duration=len(w)/sr))
 atomic(b/'validation'/f'{start:04d}-{count}.json',dict(records=checked))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','prepare-controls','verbal','synth','validate']);p.add_argument('--dataset');p.add_argument('--arm',choices=ARMS);p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=200);a=p.parse_args()
 if a.action in ['prepare','prepare-controls']:prepare(a.action=='prepare-controls')
 else:globals()[a.action](a.dataset,a.arm,a.start,a.count)
