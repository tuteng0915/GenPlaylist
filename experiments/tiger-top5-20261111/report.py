import json,sys,ast,itertools
from collections import Counter
from pathlib import Path
import numpy as np
from run import R,O,sha,save,module
from datasets import load_from_disk
oas=module('top5_oas',R/'experiments/list5-audio-20261111/oas.py')
tree=ast.parse((R/'experiments/two-dataset-fair-20261108/catalog_report.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='score')
all_results={}
for ds in ['mpd','m4a']:
 cpath=R/'outputs/sampled-candidates-20261110'/ds/'candidates.json';c=json.loads(cpath.read_text());ids=c['catalog_ids']
 prep=R/'original-ncl/data/processed'/('genplaylist-v4-8cue-20item-joint-15to5' if ds=='mpd' else 'genplaylist-music4all-onion-v3-u500-8cue-k5-u8-u3-cap16')
 cp=R/('original-ncl/src/02_creative_cues/outputs/production/20260804_172128/item2cues.json' if ds=='mpd' else 'original-ncl/data/processed/cues-music4all-onion-v3-u500/latest/item2cues.json');itemc=json.loads(cp.read_text())
 art=Path('/storage/ttu/GenPlaylist/runtime/mpd/data/dataset') if ds=='mpd' else R/'original-ncl/data/artifacts-music4all-onion-v3-u500'
 manifest=json.loads((art/'wpd_artifact_manifest.json').read_text());vp=art/'catalog_item_embeddings.npy';ip=art/'item_id_to_row.json'
 for p in [vp,ip]:assert sha(p)==manifest['outputs'][p.name]['sha256']
 emb=np.load(vp);mapping=json.loads(ip.read_text());truth=c['truth']
 target=emb[np.array([[mapping[s] for s in row] for row in truth])]
 ns=dict(np=np,Counter=Counter,set_ids=set(ids));exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen_catalog_score','exec'),ns)
 raw=load_from_disk(str(prep/'raw_dataset'))['test'].select(c['indices']);groups=[str(x).split(':joint5:')[0] for x in raw['bundle']] if ds=='mpd' else list(map(str,c['indices']))
 unique=list(dict.fromkeys(groups));assign=np.array([unique.index(g) for g in groups]);counts=np.bincount(assign)
 def ci(v):
  rng=np.random.default_rng(42);sums=np.bincount(assign,weights=v);vals=[]
  for _ in range(100):
   ix=rng.integers(len(unique),size=(100,len(unique)));vals.extend(sums[ix].sum(1)/counts[ix].sum(1))
  return np.quantile(vals,[.025,.975]).tolist()
 out=dict(dataset=ds,contexts=941,catalog=len(ids),candidate_setting='full_catalog',beam_width=20,device='cpu',metrics={},paired={},sources={},ci_unit='source playlist/user cluster',clusters=len(unique),rank_caveat='NDCG/MRR auxiliary: AR emission order vs top5 beam ranking; Recall and OAS are order invariant',cue_caveat='SID-only uses retrieved catalog cues; joint uses predicted cues; no audio-quality inference from cue-only metrics')
 per={};arrays={};checkpoints={}
 for arm in ['sid','joint']:
  for mode in ['ar5','top5']:
   key=arm+'-'+mode;p=O/'full'/f'{ds}-{arm}-{mode}'/'plan.json';plan=json.loads(p.read_text());d=plan['predictions'];identity=plan['identity'];assert d['target_item_ids']==truth and identity['contexts']==941
   checkpoints.setdefault(arm,identity['checkpoint_sha256']);assert checkpoints[arm]==identity['checkpoint_sha256']
   cues=d['cue_ids'] if arm=='joint' else [[itemc[s][:8] for s in row] for row in d['item_ids']]
   scores,vals=ns['score'](d['item_ids'],truth,c['histories'],ids,cues,itemc)
   scores.pop('cue_song_grounding_f1');vals.pop('cue_song_grounding_f1')
   # Independent multiset Recall and cue F1 calculations.
   rr=[];ff=[];dup=[]
   for row,t,cs in zip(d['item_ids'],truth,cues):
    rr.append(sum(min(row.count(s),t.count(s)) for s in set(t))/5)
    a=[v for csong in cs for v in csong];b=[v for s in t for v in itemc[s][:8]]
    ff.append(2*sum(min(a.count(v),b.count(v)) for v in set(a))/(len(a)+len(b)))
    dup.append(np.mean([1-len(set(song))/8 for song in cs]))
   np.testing.assert_allclose(rr,vals['recall_at_5'],atol=1e-12);np.testing.assert_allclose(ff,vals['cue_f1'],atol=1e-12)
   features=emb[np.array([[mapping[s] for s in row] for row in d['item_ids']])];match=oas.assignment_oas(features,target)
   # Exhaustive assignment verification on every context, independent search.
   perms=np.array(list(itertools.permutations(range(5))));brute=np.array([m[np.arange(5)[None,:],perms].mean(1).max() for m in match['pairwise_cosine']]);np.testing.assert_allclose(brute,match['similarity'],atol=1e-12)
   vals.update(clhe_oas=match['similarity'],cue_duplicate_rate=np.array(dup));scores.update(clhe_oas=float(brute.mean()),cue_duplicate_rate=float(np.mean(dup)))
   scores['diversity']=scores.pop('catalog_coverage');per[key]=vals
   out['metrics'][key]=dict(scores=scores,ci95={k:ci(v) for k,v in vals.items()},inference_seconds=plan['inference_seconds'],historical_gpu_check=plan.get('comparison_to_historical_gpu'))
   out['sources'][key]=dict(path=str(p),sha256=sha(p),identity=identity)
   arrays.update({key+'__'+k:v for k,v in vals.items()})
 for left,right in [('sid-top5','sid-ar5'),('joint-top5','joint-ar5'),('joint-top5','sid-top5'),('joint-ar5','sid-ar5')]:
  out['paired'][left+' minus '+right]={k:dict(mean=float((per[left][k]-per[right][k]).mean()),ci95=ci(per[left][k]-per[right][k])) for k in ['recall_at_5','clhe_oas','cue_f1','cue_duplicate_rate']}
 out['verification']='Independent Recall/cue multiset checks and all120 assignment permutations passed'
 out['provenance']={str(p):sha(p) for p in [cpath,cp,vp,ip]}
 dest=O/'report'/ds;dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'per-context.npz',**arrays);save(dest/'report.json',out);all_results[ds]=out
save(O/'report/report.json',all_results);print(json.dumps({ds:d['paired'] for ds,d in all_results.items()},indent=2))
