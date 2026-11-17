"""Create immutable full-catalog Random and Popularity Top-5 plans for audio.

The historical simple-baseline writer used colliding filenames across methods,
so this script reconstructs the two frozen source plans directly from the
dataset and preserves them under this experiment's own evidence root.
"""
import argparse
from collections import Counter
import json

import numpy as np
from datasets import load_from_disk

from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=('mpd', 'm4a'), required=True)
args = parser.parse_args()
dataset = args.dataset
candidate_path = R / 'outputs/sampled-candidates-20261110' / dataset / 'candidates.json'
candidates = json.loads(candidate_path.read_text())
prepared = R / 'original-ncl/data/processed' / (
    'genplaylist-v4-8cue-20item-joint-15to5' if dataset == 'mpd'
    else 'genplaylist-music4all-onion-v3-u500-8cue-k5-u8-u3-cap16'
)
data = load_from_disk(str(prepared / 'raw_dataset'))
test = data['test'].select(candidates['indices'])
sequences = [[str(item) for item in row] for row in test['item_seq']]
assert [row[:15] for row in sequences] == candidates['histories']
assert [row[15:] for row in sequences] == candidates['truth']
catalog = [str(item) for item in candidates['catalog_ids']]
counts = Counter(str(item) for row in data['train']['item_seq'] for item in row[15:])
order = {item: index for index, item in enumerate(catalog)}
popularity = sorted(catalog, key=lambda item: (-counts[item], order[item]))
random_rows = np.random.default_rng(0).choice(catalog, size=(941, 5), replace=False).tolist()
popularity_rows = [popularity[:5] for _ in range(941)]
for name, seed, rows, description in (
    ('uniform-random-seed0.json', 0, random_rows, 'Uniform full-catalog draws without replacement; history items allowed.'),
    ('train-popularity.json', None, popularity_rows, 'Training target-position popularity, catalog-order tie break; history items allowed.'),
):
    assert all(len(row) == 5 and len(set(row)) == 5 and set(row) <= set(catalog) for row in rows)
    value = {
        'schema': 'cue-v2-r31e-simple-retrieval-plan-v1', 'dataset': dataset,
        'candidate_setting': 'full_catalog', 'method': 'uniform_random' if seed is not None else 'train_popularity',
        'seed': seed, 'rule': description, 'source_candidates_sha256': sha(candidate_path),
        'source_history_indices': candidates['indices'], 'predictions': {
            'item_ids': rows, 'target_item_ids': candidates['truth'],
        },
    }
    immutable(O / dataset / 'input-plans' / name, value)
print(json.dumps({'dataset': dataset, 'random_seed': 0, 'top_popularity': popularity[:5]}, indent=2))
