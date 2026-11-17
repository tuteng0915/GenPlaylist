import argparse
import json

import numpy as np
import soundfile as sf

from common import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=('mpd', 'm4a'), required=True)
parser.add_argument('--arm', choices=ARMS, required=True)
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--count', type=int, required=True)
args = parser.parse_args()
base = O / args.dataset / args.arm
rows = json.loads((base / 'render-plan.json').read_text())['rows']
records = []
for row in rows[args.start:args.start + args.count]:
    index = row['flat_index']
    verbal = base / 'verbalization' / SYSTEM / f'{index:04d}.json'
    audio = base / 'synthesis' / SYSTEM / f'{index:04d}.mp3'
    meta = json.loads(audio.with_suffix('.json').read_text())
    text = json.loads(verbal.read_text())
    waveform, sample_rate = sf.read(audio)
    assert meta['render_row'] == text['render_row'] == row
    assert text['cue_ids'] == [] and text['retrieved_item_ids'] == row['reference_item_ids']
    assert meta['seed'] == clip_seed(row['source_history_index'], row['item_position'])
    assert meta['audio_sha256'] == sha(audio) and meta['verbalization_sha256'] == sha(verbal)
    assert waveform.size and np.isfinite(waveform).all() and abs(len(waveform) / sample_rate - 30) < 0.1
    records.append({'flat_index': index, 'audio_sha256': meta['audio_sha256'], 'duration': len(waveform) / sample_rate})
atomic(base / f'validation-{args.start:04d}-{args.count}.json', {'passed': True, 'records': records})
print('VALIDATED', args.dataset, args.arm, args.start, args.count)
