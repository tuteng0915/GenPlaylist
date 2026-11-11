import argparse
import soundfile as sf
from common import *
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--arm',required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--count',type=int,required=True);a=p.parse_args();b=O/a.dataset/a.arm;rows=json.loads((b/'render-plan.json').read_text())['rows'];sysname=SYSTEMS[a.arm];records=[]
for row in rows[a.start:a.start+a.count]:
 i=row['flat_index'];v=b/'verbalization'/sysname/f'{i:04d}.json';f=b/'synthesis'/sysname/f'{i:04d}.mp3';meta=json.loads(f.with_suffix('.json').read_text());ver=json.loads(v.read_text());wave,sr=sf.read(f)
 assert meta['render_row']==ver['render_row']==row and meta['seed']==seed(row['source_history_index'],row['item_position'])
 assert sha(f)==meta['audio_sha256'] and sha(v)==meta['verbalization_sha256'] and ver['cue_ids']==row['cue_ids'] and ver['retrieved_item_id']==row['predicted_item_id']
 assert wave.size and np.isfinite(wave).all() and abs(len(wave)/sr-30)<.1
 records.append(dict(flat_index=i,source_history_index=row['source_history_index'],item_position=row['item_position'],seed=meta['seed'],audio_sha256=meta['audio_sha256'],verbalization_sha256=meta['verbalization_sha256'],duration=len(wave)/sr))
atomic(b/f'validation-{a.start:04d}-{a.count}.json',dict(passed=True,records=records));print('VALIDATED',a.dataset,a.arm,a.start,a.count)
