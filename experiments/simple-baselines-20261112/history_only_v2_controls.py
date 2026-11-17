"""Evaluate the frozen History-only control under the v2 main-table protocol.

History-only returns the five most recent distinct items in the 15-item
history.  It has no cue decoder.  Its Cue F1 is therefore a *pooled-cue
control*, not a generated-cue score: we draw forty distinct cues from the
union of the visible history and returned songs, split them into five ordered
sets of eight, and optimally match those sets to the five target cue sets.
This is exactly the cue-control rule used for SID-only rows.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path('/storage_fast/ttu/GenPlaylist')
EVAL = ROOT / 'outputs/cue-v2-r31e-evaluation-20261115'
V2 = ROOT / 'outputs/cue-v2-r31e-20261113'
PERMUTATIONS = tuple(itertools.permutations(range(5)))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def f1(left, right, categories, category=None):
    if category is not None:
        left = [cue for cue in left if categories[cue] == category]
        right = [cue for cue in right if categories[cue] == category]
    left, right = Counter(left), Counter(right)
    denominator = sum(left.values()) + sum(right.values())
    return 0.0 if denominator == 0 else 2 * sum((left & right).values()) / denominator


def match_scores(predicted, targets, categories):
    score = np.asarray([[f1(a, b, categories) for b in targets] for a in predicted])
    permutation = max(PERMUTATIONS, key=lambda order: sum(score[i, j] for i, j in enumerate(order)))
    row = {'overall': float(np.mean([score[i, j] for i, j in enumerate(permutation)]))}
    for category in ('sound', 'emotion', 'theme'):
        row[category] = float(np.mean([
            f1(predicted[i], targets[j], categories, category)
            for i, j in enumerate(permutation)
        ]))
    return row


def cue_draw(pool, rng):
    """Apply the frozen SID-only pooled-cue rule to one context."""
    replacement = len(pool) < 40
    return rng.choice(pool, size=40, replace=replacement).astype(int).reshape(5, 8).tolist(), replacement


def evaluate(dataset):
    candidates_path = ROOT / 'outputs/sampled-candidates-20261110' / dataset / 'candidates.json'
    history_path = EVAL / 'history-only' / dataset / 'history-only-full-catalog.json'
    item_cues_path = V2 / dataset / 'item2cues.json'
    categories_path = V2 / 'cue_categories.json'
    common = load_module(
        'history_only_common',
        ROOT / 'experiments/history-cue-baselines-20261110/common.py',
    )
    oas = load_module(
        'history_only_oas',
        ROOT / 'experiments/list5-audio-20261111/oas.py',
    )
    candidates = json.loads(candidates_path.read_text())
    history = json.loads(history_path.read_text())
    item_cues = {str(key): value[:8] for key, value in json.loads(item_cues_path.read_text()).items()}
    categories = json.loads(categories_path.read_text())
    categories = {int(key): value for key, value in categories.items()} if isinstance(categories, dict) else categories
    predicted = [[str(item) for item in row] for row in history['predictions']['item_ids']]
    targets = [[str(item) for item in row] for row in candidates['truth']]
    histories = [[str(item) for item in row] for row in candidates['histories']]
    assert len(predicted) == len(targets) == len(histories) == 941
    assert history['predictions']['target_item_ids'] == candidates['truth']
    catalog = [str(item) for item in candidates['catalog_ids']]
    assert all(len(row) == len(set(row)) == 5 for row in predicted)
    assert all(set(row) <= set(catalog) for row in predicted)

    # Preference metrics, all independent of cue draw.
    recalls = np.asarray([len(set(row) & set(target)) / 5 for row, target in zip(predicted, targets)])
    hits = (recalls > 0).astype(float)
    repeats = np.asarray([sum(item in set(hist) for item in row) / 5 for row, hist in zip(predicted, histories)])
    path_info = common.paths(dataset)
    embeddings_path = path_info['art'] / 'catalog_item_embeddings.npy'
    item_row_path = path_info['art'] / 'item_id_to_row.json'
    embeddings = np.load(embeddings_path)
    item_rows = {str(key): int(value) for key, value in json.loads(item_row_path.read_text()).items()}
    selected_embeddings = embeddings[np.asarray([[item_rows[item] for item in row] for row in predicted])]
    target_embeddings = embeddings[np.asarray([[item_rows[item] for item in row] for row in targets])]
    clhe = oas.assignment_oas(selected_embeddings, target_embeddings)['similarity']

    # Ten fixed draws, analogous to the SID-only controls.  The RNG sequence
    # is committed in the report; no seed is selected after scoring.
    run_rows, diagnostics = [], []
    for draw_index in range(10):
        rng = np.random.default_rng(np.random.SeedSequence([20261117, 1 if dataset == 'mpd' else 2, draw_index]))
        rows = []
        for index, (hist, returned, target) in enumerate(zip(histories, predicted, targets)):
            pool = sorted({
                int(cue)
                for item in hist + returned
                for cue in item_cues[item]
                if 0 < int(cue) < len(categories)
            })
            assert pool, (dataset, index)
            cue_sets, replacement = cue_draw(pool, rng)
            rows.append(match_scores(cue_sets, [item_cues[item] for item in target], categories))
            if draw_index == 0:
                diagnostics.append({'context_index': index, 'pool_size': len(pool), 'replacement': replacement})
        run_rows.append(rows)
    run_values = {
        metric: np.asarray([[row[metric] for row in run] for run in run_rows])
        for metric in ('overall', 'sound', 'emotion', 'theme')
    }
    per_context = {metric: values.mean(axis=0).tolist() for metric, values in run_values.items()}
    result = {
        'schema': 'cue-v2-r31e-history-only-pooled-cue-control-v1',
        'dataset': dataset,
        'contexts': len(predicted),
        'candidate_setting': 'full_catalog',
        'method': 'five most recent distinct history songs; popularity fallback if necessary',
        'cue_output': 'none; pooled-cue control only',
        'cue_protocol': {
            'matching': 'maximum-weight one-to-one matching of five predicted cue sets to five target cue sets; edge score is within-item cue multiset F1',
            'draw': '10 fixed uniform draws of 40 cues from the distinct union of 15 history songs and five returned songs; partitioned in draw order into five eight-cue sets',
            'sparse_pool_rule': 'without replacement when pool has at least 40 distinct cues; otherwise with replacement only within that pool',
            'rng_seed_sequences': [[20261117, 1 if dataset == 'mpd' else 2, i] for i in range(10)],
        },
        'metrics': {
            'recall_at_5': float(recalls.mean()),
            'hit_at_5': float(hits.mean()),
            'diversity': len({item for row in predicted for item in row}) / len(catalog),
            'history_repeat_rate': float(repeats.mean()),
            'clhe_oas': float(clhe.mean()),
            'cue_f1_pooled_control': {metric: float(values.mean()) for metric, values in run_values.items()},
            'cue_f1_seed_sd_of_context_means': {
                metric: float(values.mean(axis=1).std(ddof=1)) for metric, values in run_values.items()
            },
        },
        'per_context': {
            'recall_at_5': recalls.tolist(), 'hit_at_5': hits.tolist(),
            'history_repeat_rate': repeats.tolist(), 'clhe_oas': clhe.tolist(),
            'cue_f1_pooled_control': per_context,
        },
        'pool_diagnostics': {
            'min': min(row['pool_size'] for row in diagnostics),
            'mean': float(np.mean([row['pool_size'] for row in diagnostics])),
            'contexts_below_40': sum(row['pool_size'] < 40 for row in diagnostics),
            'seed0': diagnostics,
        },
        'sources': {
            str(path): sha256(path) for path in [
                candidates_path, history_path, item_cues_path, categories_path,
                embeddings_path, item_row_path,
                ROOT / 'experiments/list5-audio-20261111/oas.py',
            ]
        },
    }
    output = EVAL / dataset / 'history-only-pool-one-to-one.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'dataset': dataset,
        'recall_at_5': result['metrics']['recall_at_5'],
        'hit_at_5': result['metrics']['hit_at_5'],
        'diversity': result['metrics']['diversity'],
        'history_repeat_rate': result['metrics']['history_repeat_rate'],
        'clhe_oas': result['metrics']['clhe_oas'],
        'cue_f1_pooled_control': result['metrics']['cue_f1_pooled_control'],
    }, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('mpd', 'm4a'), required=True)
    evaluate(parser.parse_args().dataset)
