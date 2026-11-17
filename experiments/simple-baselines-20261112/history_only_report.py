"""Immutable v2 full-catalog History-only recommendation report.

History-only returns the five most recent distinct songs in a listener's fixed
15-item history.  If fewer than five distinct songs remain, training-target
popularity fills the list.  It has no learned or supplied cue output.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from datasets import load_from_disk


ROOT = Path('/storage_fast/ttu/GenPlaylist')
OUTPUT = ROOT / 'outputs/cue-v2-r31e-evaluation-20261115/history-only'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main(dataset):
    candidate_path = ROOT / 'outputs/sampled-candidates-20261110' / dataset / 'candidates.json'
    candidates = json.loads(candidate_path.read_text())
    processed = ROOT / 'original-ncl/data/processed' / (
        'genplaylist-v4-8cue-20item-joint-15to5' if dataset == 'mpd'
        else 'genplaylist-music4all-onion-v3-u500-8cue-k5-u8-u3-cap16'
    )
    catalog = [str(item) for item in candidates['catalog_ids']]
    base_rows = [list(dict.fromkeys(str(item) for item in reversed(history))) for history in candidates['histories']]
    # The frozen 941-context cohorts happen to have at least five distinct
    # history songs each. Only scan the full training split when the documented
    # popularity fallback is actually needed.
    if any(len(row) < 5 for row in base_rows):
        data = load_from_disk(str(processed / 'raw_dataset'))
        order = {item: index for index, item in enumerate(catalog)}
        counts = Counter(str(item) for row in data['train']['item_seq'] for item in row[15:])
        popularity = sorted(catalog, key=lambda item: (-counts[item], order[item]))
    else:
        popularity = []
    predictions = []
    recall = []
    for row, target in zip(base_rows, candidates['truth']):
        row.extend(item for item in popularity if item not in set(row))
        row = row[:5]
        assert len(row) == len(set(row)) == 5 and set(row) <= set(catalog)
        predictions.append(row)
        recall.append(len(set(row) & set(target)) / 5)
    report = {
        'schema': 'cue-v2-r31e-history-only-full-catalog-v1', 'dataset': dataset,
        'contexts': len(predictions), 'candidate_setting': 'full_catalog',
        'method': 'five most recent distinct history songs; popularity fills shortage',
        'source_candidates': str(candidate_path), 'source_candidates_sha256': sha(candidate_path),
        'metrics': {
            'recall_at_5': sum(recall) / len(recall),
            'diversity': len({item for row in predictions for item in row}) / len(catalog),
            'unique_recommended_songs': len({item for row in predictions for item in row}),
            'catalog_size': len(catalog),
        },
        'predictions': {'item_ids': predictions, 'target_item_ids': candidates['truth']},
        'per_context': {'recall_at_5': recall},
        'cue_output': None,
    }
    atomic(OUTPUT / dataset / 'history-only-full-catalog.json', report)
    print(json.dumps({'dataset': dataset, **report['metrics']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('mpd', 'm4a'), required=True)
    main(parser.parse_args().dataset)
