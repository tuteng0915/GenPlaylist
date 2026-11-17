"""Fixed-cohort evaluation for the two TIGER+cues warm-start trials."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk


R = Path('/storage_fast/ttu/GenPlaylist')
E = R / 'experiments/cue-v2-r31e-tiger-tune-20261117'
O = R / 'outputs/cue-v2-r31e-tiger-tune-20261117'
PERMUTATIONS = list(itertools.permutations(range(5)))


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.new')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(specification)
    sys.modules[name] = loaded
    specification.loader.exec_module(loaded)
    return loaded


def multiset_f1(predicted, target, category=None, categories=None):
    if category is not None:
        predicted = [cue for cue in predicted if categories[str(cue)] == category]
        target = [cue for cue in target if categories[str(cue)] == category]
    overlap = sum((Counter(predicted) & Counter(target)).values())
    total = len(predicted) + len(target)
    return 0.0 if total == 0 else 2 * overlap / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cue-weight', choices=('0.02', '0.05'), required=True)
    args = parser.parse_args()
    arm = 'warm-ramp-' + args.cue_weight.replace('.', 'p')
    run = O / 'm4a' / arm
    checkpoint = run / 'last.ckpt'
    identity = json.loads((run / 'identity.json').read_text())
    complete = json.loads((run / 'complete.json').read_text())
    assert sha(checkpoint) == complete['checkpoint_sha256']

    core = module('tiger_tune_joint_core', R / 'experiments/cue-v2-r31e-tiger-20261113/core.py')
    prepared = R / 'original-ncl/data/processed/genplaylist-music4all-onion-v3-u500-8cue-k5-u8-u3-cap16'
    vectors = prepared / 'vectors'
    ids = list(map(str, json.loads((vectors / 'catalog_item_ids.json').read_text())))
    row_for = {item: row for row, item in enumerate(ids)}
    canonical = np.load(vectors / 'catalog_semantic_tokens.npy').astype(np.int64)
    cue_root = R / 'outputs/cue-v2-r31e-20261113'
    item2cues = json.loads((cue_root / 'm4a/item2cues.json').read_text())
    cues = np.asarray([item2cues[item][:8] for item in ids], dtype=np.int64)
    vocabulary = json.loads((cue_root / 'cue_vocab.json').read_text())
    categories = json.loads((cue_root / 'cue_categories.json').read_text())
    catalog = core.Catalog(canonical, cues, len(vocabulary))
    model = core.build_model(catalog.vocab_size)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    assert state['identity'] == identity
    del state
    device = torch.device('cuda')
    model = model.to(device).eval()

    candidates_path = R / 'outputs/sampled-candidates-20261110/m4a/candidates.json'
    candidates = json.loads(candidates_path.read_text())
    raw = load_from_disk(str(prepared / 'raw_dataset'))['test']
    selected = raw.select(candidates['indices'])['item_seq']
    history = np.asarray([[row_for[str(item)] for item in row[:15]] for row in selected], dtype=np.int64)
    targets = [[str(item) for item in row[15:]] for row in selected]
    assert targets == candidates['truth'] and len(history) == 941

    target_identity = {
        'schema': 'tiger-v2-warm-start-evaluation-v1',
        'checkpoint_sha256': sha(checkpoint),
        'training_identity_sha256': sha(run / 'identity.json'),
        'candidate_cohort_sha256': sha(candidates_path),
        'contexts': 941,
        'decoder': 'one-pass Top-5 beam20, four legal SID tokens then greedy unrestricted eight cue tokens',
        'history': '15 SID4+v2 catalog-cue8 items; no target leakage',
    }
    destination = run / 'evaluation'
    if (destination / 'identity.json').exists():
        assert json.loads((destination / 'identity.json').read_text()) == target_identity
    else:
        dump(destination / 'identity.json', target_identity)

    predictions = []
    generated_cues = []
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(history), 16):
            batch_path = destination / f'batch-{start:04d}.npz'
            if batch_path.exists():
                saved = np.load(batch_path, allow_pickle=False)
                predicted, predicted_cues = saved['pred'], saved['cues']
            else:
                predicted, predicted_cues = core.predict(model, catalog, history[start:start + 16], True, device)
                temporary = batch_path.with_suffix('.new')
                with temporary.open('wb') as handle:
                    np.savez(handle, pred=predicted, cues=predicted_cues)
                os.replace(temporary, batch_path)
            predictions.append(predicted)
            generated_cues.append(predicted_cues)
    prediction_rows = np.concatenate(predictions)
    cue_rows = np.concatenate(generated_cues)
    assert prediction_rows.shape == (941, 5) and cue_rows.shape == (941, 5, 8)
    assert all(len(set(row)) == 5 for row in prediction_rows)
    item_rows = np.asarray(ids)[prediction_rows].tolist()

    recalls = []
    cue_scores = {name: [] for name in ('overall', 'sound', 'emotion', 'theme')}
    for predicted_items, predicted_cue_rows, target_items in zip(item_rows, cue_rows.tolist(), targets):
        recalls.append(sum((Counter(predicted_items) & Counter(target_items)).values()) / 5)
        target_cue_rows = [item2cues[item][:8] for item in target_items]
        edge = np.asarray([
            [multiset_f1(predicted_cue_rows[i], target_cue_rows[j]) for j in range(5)] for i in range(5)
        ])
        pairing = max(PERMUTATIONS, key=lambda permutation: sum(edge[i, j] for i, j in enumerate(permutation)))
        for name in cue_scores:
            category = None if name == 'overall' else name
            cue_scores[name].append(float(np.mean([
                multiset_f1(predicted_cue_rows[i], target_cue_rows[j], category, categories)
                for i, j in enumerate(pairing)
            ])))
    result = {
        'schema': 'tiger-v2-warm-start-evaluation-result-v1',
        'identity': target_identity,
        'seconds': time.monotonic() - started,
        'metrics': {
            'recall_at_5': float(np.mean(recalls)),
            'diversity': len(set(item for row in item_rows for item in row)) / len(ids),
            'cue_f1_one_to_one': {name: float(np.mean(value)) for name, value in cue_scores.items()},
            'duplicate_rate': 0.0,
        },
        'per_context': {'recall_at_5': recalls, 'cue_f1_one_to_one': cue_scores},
        'predictions': {
            'item_ids': item_rows,
            'semantic_token_ids': canonical[prediction_rows].tolist(),
            'cue_ids': cue_rows.tolist(),
            'target_item_ids': targets,
        },
    }
    dump(destination / 'result.json', result)
    print(json.dumps(result['metrics'], indent=2))


if __name__ == '__main__':
    main()
