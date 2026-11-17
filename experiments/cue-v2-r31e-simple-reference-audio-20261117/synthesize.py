import argparse
import json
import os
import sys
from pathlib import Path

from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=('mpd', 'm4a'), required=True)
parser.add_argument('--arm', choices=ARMS, required=True)
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--count', type=int, required=True)
args = parser.parse_args()
base = O / args.dataset / args.arm
rows = json.loads((base / 'render-plan.json').read_text())['rows']
source = base / 'verbalization'
output = base / 'synthesis'
ace = Path(os.environ['ACE_STEP_SOURCE'])
checkpoint = Path(os.environ['ACE_STEP_SNAPSHOT'])
assert checkpoint.parent.name == 'snapshots'
immutable(output / 'audio_manifest.json', {
    'schema': 'cue-v2-r31e-simple-reference-audio-v1', 'model': 'ACE-Step/ACE-Step-v1-3.5B',
    'model_revision': checkpoint.name, 'duration_seconds': 30, 'inference_steps': 60,
    'guidance_scale': 15.0, 'scheduler_type': 'euler', 'cfg_type': 'apg', 'omega_scale': 10.0,
    'seed_rule': '42000 + original941_history_index + 941*item_position',
    'render_plan_sha256': sha(base / 'render-plan.json'),
    'verbalization_manifest_sha256': sha(source / 'verbalization_manifest.json'), 'examples': len(rows),
})
sys.path.insert(0, str(ace))
pipeline = None
for row in rows[args.start:args.start + args.count]:
    index = row['flat_index']
    verbal = source / SYSTEM / f'{index:04d}.json'
    record = json.loads(verbal.read_text())
    assert record['render_row'] == row and record['cue_ids'] == []
    destination = output / SYSTEM / f'{index:04d}.mp3'
    meta = destination.with_suffix('.json')
    if destination.exists() and meta.exists():
        old = json.loads(meta.read_text())
        assert old['render_row'] == row and old['audio_sha256'] == sha(destination)
        continue
    destination.parent.mkdir(parents=True, exist_ok=True)
    if pipeline is None:
        from acestep.pipeline_ace_step import ACEStepPipeline
        pipeline = ACEStepPipeline(checkpoint_dir=str(checkpoint), device_id=0, dtype='bfloat16', torch_compile=False)
    partial = output / '.partial' / f'{index:04d}.{os.getpid()}.mp3'
    partial.parent.mkdir(parents=True, exist_ok=True)
    result = pipeline(prompt=record['music_attributes'], lyrics=record['lyric_draft'], audio_duration=30,
        infer_step=60, guidance_scale=15.0, scheduler_type='euler', cfg_type='apg', omega_scale=10.0,
        manual_seeds=[row['seed']], format='mp3', save_path=str(partial))
    assert Path(result[0]).resolve() == partial.resolve()
    os.replace(partial, destination)
    atomic(meta, {'result_schema': 'cue-v2-r31e-simple-reference-audio-record-v1', 'system': SYSTEM,
        'example_index': index, 'render_row': row, 'seed': row['seed'], 'audio_sha256': sha(destination),
        'verbalization_sha256': sha(verbal)})
    print('AUDIO', args.dataset, args.arm, index, flush=True)
