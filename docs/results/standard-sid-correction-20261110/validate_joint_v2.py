import argparse,json,hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--count',type=int,required=True);a=p.parse_args()
E=Path(__file__).resolve().parent;R=E.parents[1];O=R/'outputs/tiger-joint-20261108'/a.dataset;A=O/'audio'
base=R/'outputs/generation-final-20261107/full'
vm=json.loads((base/'verbalization/verbalization_manifest.json').read_text());am=json.loads((base/'synthesis/audio_manifest.json').read_text())
plans={k:json.loads((O/v/'audio/plan.json').read_text()) for k,v in [('full','joint')]}
if a.dataset=='m4a':cohort=json.loads((R/'experiments/two-dataset-fair-20261108/cohort.json').read_text())
cue_dir=R/('original-ncl/src/02_creative_cues/outputs/production/20260804_172128' if a.dataset=='mpd' else 'original-ncl/data/processed/cues-music4all-onion-v3-u500/latest')
item_cues=json.loads((cue_dir/'item2cues.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
records=[]
for arm,system in [('full','GenPlaylist')]:
 V=A/arm/'verbalization';S=A/arm/'synthesis';v=json.loads((V/'verbalization_manifest.json').read_text());m=json.loads((S/'audio_manifest.json').read_text())
 for k in ['model_name','model_revision','decoding','seed','examples','reference_items','selected_prediction']:assert v[k]==vm[k]
 for k in ['ace_step','seed','examples']:assert m[k]==am[k]
 assert m['verbalization_manifest_sha256']==sha(V/'verbalization_manifest.json')
 for i in range(a.start,a.start+a.count):
  vr=V/system/f'{i:04d}.json';audio=S/system/f'{i:04d}.mp3';d=json.loads(vr.read_text());meta=json.loads(audio.with_suffix('.json').read_text())
  if a.dataset=='m4a':ref=list(map(str,cohort['sequences'][i]));assert plans[arm]['source_test_indices']==cohort['source_test_indices']
  else:
   old=json.loads((base/'verbalization/GenPlaylist'/f'{i:04d}.json').read_text());ref=old['reference_item_ids']+[old['target_item_id']]
  assert d['reference_item_ids']==ref[:15] and str(d['target_item_id'])==ref[15] and d['system']==system and d['example_index']==i
  pred=plans[arm]['predictions'];song=str(pred['item_ids'][i][0]);assert str(d['retrieved_item_id'])==song
  assert d['cue_ids']==(pred['cue_ids'][i][0] if arm=='full' else item_cues[song][:8]) and len(d['cue_ids'])==8
  assert meta['seed']==42000+i and meta['audio_sha256']==sha(audio) and meta['verbalization_sha256']==sha(vr)
  wave,sr=sf.read(audio,always_2d=True);assert 29.9<=len(wave)/sr<=30.1 and np.isfinite(wave).all() and np.any(wave)
  records.append(dict(arm=arm,index=i,audio_sha256=sha(audio)))
(A/f'validation-{a.start}-{a.count}.json').write_text(json.dumps({'status':'passed','histories':a.count,'scope':'joint-only; standard SID baseline is reused from verified two-dataset outputs','records':records},indent=2));print('Validated joint TIGER audio',a.dataset,a.start,a.count)
