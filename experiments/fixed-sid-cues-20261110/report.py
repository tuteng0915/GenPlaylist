import ast,json,hashlib,os
from pathlib import Path
from collections import Counter
import numpy as np
from datasets import load_from_disk
R=Path('/storage_fast/ttu/GenPlaylist');O=R/'outputs/fixed-sid-cues-20261110';scorer=R/'experiments/two-dataset-fair-20261108/catalog_report.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
fn=next(n for n in ast.parse(scorer.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='score');report={}
for ds in ['mpd','m4a']:
 cohort=json.loads((R/'outputs/sampled-candidates-20261110'/ds/'candidates.json').read_text());ids=cohort['catalog_ids']
 cp=R/('original-ncl/src/02_creative_cues/outputs/production/20260804_172128/item2cues.json' if ds=='mpd' else 'original-ncl/data/processed/cues-music4all-onion-v3-u500/latest/item2cues.json');itemc=json.loads(cp.read_text())
 ns={'np':np,'Counter':Counter,'set_ids':set(ids)};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(scorer),'exec'),ns)
 prep=R/'original-ncl/data/processed'/('genplaylist-v4-8cue-20item-joint-15to5' if ds=='mpd' else 'genplaylist-music4all-onion-v3-u500-8cue-k5-u8-u3-cap16');raw=load_from_disk(str(prep/'raw_dataset'))['test'].select(cohort['indices'])
 groups=[str(b).split(':joint5:')[0] for b in raw['bundle']] if ds=='mpd' else list(map(str,cohort['indices']));unique=list(dict.fromkeys(groups));assign=np.array([unique.index(g) for g in groups]);counts=np.bincount(assign);rng=np.random.default_rng(42);draws=rng.integers(len(unique),size=(10000,len(unique)))
 report[ds]={}
 for scope in ['full_catalog','sampled100']:
  p=O/ds/scope/'full/plan.json';end=json.loads(p.with_name('complete.json').read_text());assert sha(p)==end['plan_sha256'];new=json.loads(p.read_text());sp=Path(new['identity']['source_plan']);assert sha(sp)==end['source_plan_sha256'];old=json.loads(sp.read_text());nn=new['predictions'];oo=old['predictions'];assert nn['item_ids']==oo['item_ids'] and nn['target_item_ids']==oo['target_item_ids']==cohort['truth']
  mm={};per={}
  for label,d in [('original',oo),('refreshed',nn)]:
   mm[label],per[label]=ns['score'](d['item_ids'],d['target_item_ids'],cohort['histories'],ids,d['cue_ids'],itemc)
   mm[label].pop('cue_song_grounding_f1');per[label].pop('cue_song_grounding_f1')
   vals=[]
   for target,cues in zip(d['target_item_ids'],d['cue_ids']):
    aa=[v for q in cues for v in q];bb=[v for t in target for v in itemc[t][:8]];vals.append(sum(min(aa.count(v),bb.count(v)) for v in set(aa))/40)
   assert abs(np.mean(vals)-mm[label]['cue_f1'])<1e-12
   flat=np.asarray(d['cue_ids']);mm[label]['cue_unique_ratio']=float(np.mean([len(set(row))/8 for row in flat.reshape(-1,8)]));mm[label]['unknown_cue_rate']=float((flat==0).mean())
  for k in ['recall_at_5','precision_at_5','hit_at_5','mrr_at_5','ndcg_at_5','history_repeat_rate','catalog_coverage','catalog_entropy','distinct_songs','top10_song_share']:assert mm['original'][k]==mm['refreshed'][k]
  delta=per['refreshed']['cue_f1']-per['original']['cue_f1'];sums=np.bincount(assign,weights=delta);boot=sums[draws].sum(1)/counts[draws].sum(1)
  report[ds][scope]=dict(contexts=941,metrics=mm,cue_f1_delta=dict(mean=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()),recommendation_metrics_exactly_unchanged=True,cue_token_changed_fraction=end['cue_token_changed_fraction'],source_plan_sha256=sha(sp),refreshed_plan_sha256=sha(p),independent_cue_f1_check=True,interpretation='Cue prediction only; no audio-quality result. Single generation seed; bootstrap excludes generation-seed uncertainty.')
tmp=O/'report.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');os.replace(tmp,O/'report.json');print(json.dumps(report,indent=2))
