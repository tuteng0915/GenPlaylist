import argparse,ast,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from datasets import load_from_disk
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['mpd','m4a'],required=True);p.add_argument('--backbone',choices=['ddbc','tiger'],required=True);a=p.parse_args()
R=Path(__file__).resolve().parents[2];O=R/'outputs/cue-first-20261110'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
prepared=R/'original-ncl/data/processed'/('genplaylist-v4-8cue-20item-joint-15to5' if a.dataset=='mpd' else 'genplaylist-music4all-onion-v3-u500-8cue-k5-u8-u3-cap16')
raw=load_from_disk(str(prepared/'raw_dataset'))['test']
indices=list(range(941)) if a.dataset=='mpd' else json.loads((R/'experiments/two-dataset-fair-20261108/cohort.json').read_text())['source_test_indices']
selected=raw.select(indices);seq=selected['item_seq'];truth=[[str(x) for x in row[15:]] for row in seq];refs=[[str(x) for x in row[:15]] for row in seq]
ids=list(map(str,json.loads((prepared/'vectors/catalog_item_ids.json').read_text())))
cuefile=R/('original-ncl/src/02_creative_cues/outputs/production/20260804_172128/item2cues.json' if a.dataset=='mpd' else 'original-ncl/data/processed/cues-music4all-onion-v3-u500/latest/item2cues.json')
item2cues=json.loads(cuefile.read_text())
# Use the same frozen metric implementation without executing its report main.
scorer=R/'experiments/two-dataset-fair-20261108/catalog_report.py'
tree=ast.parse(scorer.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='score')
ns={'np':np,'Counter':Counter,'set_ids':set(ids)};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(scorer),'exec'),ns)
if a.backbone=='ddbc':
    paths={k:O/'ddbc'/a.dataset/'audio'/k/'result.json' for k in ['joint','cue_first','sid_first']}
    for path in paths.values():assert json.loads(path.with_name('verified.json').read_text())['source_indices']==indices
    baseline='joint';contrasts=[('cue_first','sid_first'),('cue_first','joint'),('sid_first','joint')]
else:
    paths={'sid_first':R/'outputs/tiger-joint-20261108'/a.dataset/'joint/audio/plan.json','cue_first':O/'tiger'/a.dataset/'joint/audio/plan.json'}
    old=json.loads((paths['sid_first'].parents[1]/'train-complete.json').read_text());new=json.loads((paths['cue_first'].parents[1]/'train-complete.json').read_text())
    assert old['case_digest']==new['case_digest'] and old['initial_sha256']==new['initial_sha256']
    baseline='sid_first';contrasts=[('cue_first','sid_first')]
metrics={};per={};payloads={};internal={}
for name,path in paths.items():
    payload=json.loads(path.read_text());pred=payload['predictions'];payloads[name]=pred
    assert pred['target_item_ids']==truth
    if 'source_test_indices' in payload:assert payload['source_test_indices']==indices
    assert np.asarray(pred['cue_ids']).shape==(941,5,8)
    metrics[name],per[name]=ns['score'](pred['item_ids'],truth,refs,ids,pred['cue_ids'],item2cues)
    internal[name]={'cue_song_grounding_f1':metrics[name].pop('cue_song_grounding_f1')}
    c=np.array(pred['cue_ids']);metrics[name]['cue_unknown_rate']=float((c==0).mean())
    metrics[name]['cue_unique_ratio']=float(np.mean([len(set(row))/8 for row in c.reshape(-1,8)]))
# Preserve cluster units if multiple MPD contexts share a source playlist.
groups=[str(x).split(':joint5:')[0] for x in selected['bundle']] if a.dataset=='mpd' else list(map(str,indices))
unique=list(dict.fromkeys(groups));assignment=np.array([unique.index(g) for g in groups]);counts=np.bincount(assignment)
rng=np.random.default_rng(42);draws=rng.integers(len(unique),size=(10000,len(unique)));paired={}
for left,right in contrasts:
    key=left+' minus '+right;paired[key]={}
    for metric in ['recall_at_5','cue_f1','history_repeat_rate']:
        delta=per[left][metric]-per[right][metric];sums=np.bincount(assignment,weights=delta)
        boot=sums[draws].sum(1)/counts[draws].sum(1)
        paired[key][metric]={'mean_difference':float(delta.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist()}
    paired[key]['catalog_coverage_difference']=metrics[left]['catalog_coverage']-metrics[right]['catalog_coverage']
    paired[key]['changed_song_fraction']=float(np.mean(np.array(payloads[left]['item_ids'])!=np.array(payloads[right]['item_ids'])))
out=O/a.backbone/a.dataset/'order-comparison.json'
report={'dataset':a.dataset,'backbone':a.backbone,'contexts':941,'catalog_size':len(ids),'metrics':metrics,'paired':paired,
        'internal_diagnostic':internal,'sources':{k:{'path':str(v),'sha256':sha(v)} for k,v in paths.items()},
        'scorer_sha256':sha(scorer),'bootstrap':{'n':10000,'seed':42,'clusters':len(unique),'scope':'exploratory, checkpoint conditional; no training-seed uncertainty'},
        'no_audio_quality_claim':'Cue prediction metrics only; no new audio generation results.'}
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
