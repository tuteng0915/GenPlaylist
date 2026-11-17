"""Independent evidence and candidate-capacity checks; no claim of final mapping coverage."""
import sys,time,json,datetime,collections
from pathlib import Path
R=Path('/storage_fast/ttu/GenPlaylist');O=R/'outputs/cue-v2-repair-r3-20261113';E=R/'experiments/cue-v2-repair-r3-20261113';sys.path.insert(0,str(E))
from pilot_extract import postfilter,evidence_units,paths,sha,atomic
limit=datetime.datetime.fromisoformat('2026-11-13T11:11:30+08:00').timestamp()
while not (O/'pilot-extraction-report.json').exists():
 if time.time()>=limit:raise TimeoutError('Pilot incomplete at booking deadline')
 time.sleep(20)
cfg=json.loads((O/'pilot-config.json').read_text());out={}
for ds,ids in cfg['selected'].items():
 meta=json.loads((paths(ds)['data']/'catalog_metadata.json').read_text());n=0;capacity=0;category=collections.Counter();records=[]
 for i in ids:
  p=O/ds/'candidates'/f'{i}.json';q=json.loads(p.read_text());raw=json.loads((O/ds/'raw-responses'/f'{i}.json').read_text());assert raw['finish_reason']!='length';a,r=postfilter(json.loads(raw['raw']),q['input'],meta[i],evidence_units(q['input']));assert a==q['accepted'] and r==q['rejected'];assert q['config_sha256']==sha(O/'pilot-config.json');category.update(c['category'] for c in a)
  if q['input']['lyrics']:n+=1;capacity+=min(8,len({c['phrase'] for c in a}))
  records.append(dict(item=i,candidate_sha256=sha(p),accepted=len(a)))
 reported=json.loads((O/'pilot-extraction-report.json').read_text())['datasets'][ds]['new_candidate_capacity'];assert capacity/(8*n)<=reported+1e-12
 out[ds]=dict(songs_with_lyrics=n,unique_phrase_capacity_slots=capacity,capacity=capacity/(8*n),accepted_categories=dict(category),records=records)
atomic(O/'pilot-independent-check.json',dict(passed=True,datasets=out,scope='Only evidence provenance/filter reproducibility/candidate capacity; semantic faithfulness and final shared-vocabulary coverage still required',audio_started=False,stage_B_started=False))
print('PILOT_INDEPENDENT_CHECK_PASS',flush=True)
