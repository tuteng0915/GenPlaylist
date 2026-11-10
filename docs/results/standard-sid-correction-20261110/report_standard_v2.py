"""Same scorers and paired cases; preserve negative/null outcomes."""
import argparse,ast,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
P=argparse.ArgumentParser();P.add_argument('--dataset',required=True);P.add_argument('--mode',choices=['catalog','audio'],required=True);a=P.parse_args()
E=Path(__file__).resolve().parent;R=E.parents[1];O=R/'outputs/tiger-joint-20261108'/a.dataset;F=R/'experiments/two-dataset-fair-20261108';B=R/'outputs/two-dataset-fair-20261108'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ci(v):
 rng=np.random.default_rng(42);n=len(v);samples=[]
 for _ in range(100):samples.extend(v[rng.integers(n,size=(100,n))].mean(1))
 return np.quantile(samples,[.025,.975]).tolist()
def save(name,d):
 (O/name).write_text(json.dumps(d,indent=2)+'\n')
if a.mode=='catalog':
 from datasets import load_from_disk
 prepared=R/'original-ncl/data/processed'/('genplaylist-v4-8cue-20item-joint-15to5' if a.dataset=='mpd' else 'genplaylist-music4all-onion-v3-u500-8cue-k5-u8-u3-cap16')
 raw=load_from_disk(str(prepared/'raw_dataset'))['test'];seq=raw['item_seq'];truth=[[str(x) for x in row[15:]] for row in seq];refs=[[str(x) for x in row[:15]] for row in seq]
 ids=list(map(str,json.loads((prepared/'vectors/catalog_item_ids.json').read_text())))
 cp=R/('original-ncl/src/02_creative_cues/outputs/production/20260804_172128/item2cues.json' if a.dataset=='mpd' else 'original-ncl/data/processed/cues-music4all-onion-v3-u500/latest/item2cues.json');item_cues=json.loads(cp.read_text())
 # Reuse precisely the verified common scorer, without running its report entrypoint.
 source=F/'catalog_report.py';tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='score')
 env={'np':np,'Counter':Counter,'set_ids':set(ids)};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),env);score=env['score']
 plans={}
 for arm in ['joint']:
  ds=[json.loads((O/arm/part/'plan.json').read_text()) for part in ['audio','remaining']]
  indices=[i for d in ds for i in d['source_test_indices']];assert sorted(indices)==list(range(len(seq)))
  order=np.argsort(indices);plans[arm]={k:[v[i] for i in order] for k in ['item_ids','target_item_ids','cue_ids'] for v in [[v for d in ds for v in d['predictions'][k]]]}
 baseline_path=B/a.dataset/'tiger/full-plan.json'
 baseline=json.loads(baseline_path.read_text());assert baseline['checkpoint_steps']==20000 and baseline['native_beam_width']==20 and baseline['allow_visible_history']
 plans['sid']=baseline['predictions']
 if a.dataset=='mpd':plans['genplaylist']=json.loads((F/'mpd-existing-plans/conditional_mask.json').read_text())['predictions']
 else:
  ds=[json.loads((B/'m4a/plans'/f'conditional_mask-{part}.json').read_text()) for part in ['audio','remaining']];indices=[i for d in ds for i in d['source_test_indices']];assert sorted(indices)==list(range(len(seq)));order=np.argsort(indices)
  plans['genplaylist']={k:[v[i] for i in order] for k in ['item_ids','target_item_ids','cue_ids'] for v in [[v for d in ds for v in d['predictions'][k]]]}
 metrics={};per={}
 for arm,p in plans.items():
  assert p['target_item_ids']==truth
  cues=p['cue_ids'] if arm!='sid' else [[item_cues[str(song)][:8] for song in row] for row in p['item_ids']]
  metrics[arm],per[arm]=score(p['item_ids'],truth,refs,ids,cues,item_cues)
  if cues is not None:
   flat=np.array(cues);metrics[arm]['cue_unknown_rate']=float((flat==0).mean());metrics[arm]['cue_unique_ratio']=float(np.mean([len(set(row))/8 for row in flat.reshape(-1,8)]))
 paired={}
 for left,right in [('joint','sid'),('genplaylist','joint')]:
  paired[left+'-'+right]={}
  for k in sorted(set(per[left])&set(per[right])):
   d=per[left][k]-per[right][k];paired[left+'-'+right][k]={'mean_difference':float(d.mean()),'ci95':ci(d)}
 save('catalog-standard-baseline-report.json',{'status':'complete','dataset':a.dataset,'test_histories':len(seq),'catalog_items':len(ids),'metrics':metrics,'paired':paired,'common_scorer_sha256':sha(source),'baseline_plan_sha256':sha(baseline_path),'scope':'standard TIGER SID->SID versus joint SID+cue->SID+cue. Both fixed20k checkpoints, same test cases/catalog/beam20/unique-list policy; not identical initialization or training implementation. Standard TIGER retrieved-cue grounding is 1 by construction.'})
else:
 base=R/'outputs/generation-final-20261107' if a.dataset=='mpd' else B/'m4a/audio'
 roots={'joint':O/'audio/full','sid':B/a.dataset/'audio/tiger','genplaylist':base/'full'}
 catalog=R/('experiments/generation-final-20261107/mert-v1-95m-catalog-v1' if a.dataset=='mpd' else 'v2-recovery-20261106/mert-v1-95m-music4all-u500-audio-covered-v1')
 manifest=json.loads((catalog/'mert_manifest.json').read_text())
 for n in ['catalog_item_ids.json','catalog_mert_embeddings_l2.npy']:assert sha(catalog/n)==manifest['outputs'][n]
 ids=json.loads((catalog/'catalog_item_ids.json').read_text());idx={str(x):i for i,x in enumerate(ids)};cat=np.load(catalog/'catalog_mert_embeddings_l2.npy')
 per={};metrics={};reference=None
 for arm,root in roots.items():
  slug='ddbc_sft' if arm=='sid' else 'genplaylist';system='DDBC-SFT' if arm=='sid' else 'GenPlaylist';record=[json.loads((root/'verbalization'/system/f'{i:04d}.json').read_text()) for i in range(941)]
  refs=[[str(x) for x in d['reference_item_ids']] for d in record];nxt=[str(d['target_item_id']) for d in record]
  if reference is None:reference=(refs,nxt)
  assert reference==(refs,nxt)
  emb=[]
  for i in range(941):
   f=root/'metrics/mert-rows'/f'{i:04d}.npz';v=np.load(f,allow_pickle=False);identity=json.loads(str(v['identity']));assert identity['history_index']==i and identity['revision']==manifest['model_revision'];emb.append(v['embedding'])
  gen=np.stack(emb);assert gen.shape==(941,768) and np.isfinite(gen).all() and np.allclose(np.linalg.norm(gen,axis=1),1,atol=1e-4)
  sims=np.einsum('bd,bkd->bk',gen,cat[np.array([[idx[x] for x in row] for row in refs])])
  aa=np.load(root/'metrics/clap'/f'{slug}_clap_audio_l2.npy');tt=np.load(root/'metrics/clap'/f'{slug}_clap_attribute_l2.npy');assert aa.shape==tt.shape==(941,512)
  clap=json.loads((root/'metrics/clap'/f'{slug}_clap.json').read_text());assert clap['examples']==941
  per[arm]={'history_fit':sims.mean(1),'next_similarity':(gen*cat[[idx[x] for x in nxt]]).sum(1),'clap_own_prompt':(aa*tt).sum(1),'catalog_max_similarity':np.concatenate([(gen[i:i+64]@cat.T).max(1) for i in range(0,941,64)])}
  fad=json.loads((root/'metrics/fad.json').read_text());assert fad['examples_per_corpus']==941
  metrics[arm]={k:{'mean':float(v.mean()),'ci95':ci(v)} for k,v in per[arm].items()}
  metrics[arm].update(fad_vggish=next(iter(fad['fad'].values())),cross_history_diversity=float(1-(np.square(gen.sum(0)).sum()-941)/(941*940)))
 paired={}
 for left,right in [('joint','sid'),('genplaylist','joint')]:
  paired[left+'-'+right]={k:{'difference':float((per[left][k]-per[right][k]).mean()),'ci95':ci(per[left][k]-per[right][k])} for k in per[left]}
 save('audio-standard-baseline-report.json',{'status':'complete','histories':941,'metrics':metrics,'paired':paired,'bootstrap':{'samples':10000,'seed':42},'baseline':'standard TIGER SID->SID; its catalog cues enter only the downstream renderer, never its recommender input','interpretation':'CLAP own-prompt is adherence, not listener preference. FAD is corpus-level, not paired per-history. Follow-up contrasts do not replace original primary family.'})
print('Saved TIGER joint report',a.dataset,a.mode)
