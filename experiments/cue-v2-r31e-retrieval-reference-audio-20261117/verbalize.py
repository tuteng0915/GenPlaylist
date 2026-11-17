"""Qwen verbalization conditioned only on a retriever's five returned songs."""
import argparse
import json
import random

import numpy as np

from common import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=('mpd', 'm4a'), required=True)
parser.add_argument('--arm', choices=ARMS, required=True)
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--count', type=int, required=True)
args = parser.parse_args()

base = O / args.dataset / args.arm
plan = json.loads((base / 'render-plan.json').read_text())
rows = plan['rows']
assert 0 <= args.start < args.start + args.count <= len(rows) == 500
x = paths(args.dataset)
runner = module('retrieval_reference_verbal_runner', CODE / 'scripts/run_end_to_end_verbalization.py')
verbal = runner._load_verbalization_module()
lookup = {str(item): int(row) for item, row in json.loads((x['art'] / 'item_id_to_row.json').read_text()).items()}
embeddings = np.load(x['art'] / 'catalog_item_embeddings.npy').astype(np.float32)
metadata = runner.CatalogItem.load_catalog(str(x['data'] / 'catalog_metadata.json'))
metadata = {item.item_id: item for item in metadata} if isinstance(metadata, list) else metadata
ordered_metadata = [None] * len(lookup)
for item, row in lookup.items():
    ordered_metadata[row] = metadata[item]
assert all(value is not None for value in ordered_metadata)
assets = {
    str(x['data'] / 'catalog_metadata.json'): sha(x['data'] / 'catalog_metadata.json'),
    str(x['art'] / 'catalog_item_embeddings.npy'): sha(x['art'] / 'catalog_item_embeddings.npy'),
    str(CODE / 'scripts/run_end_to_end_verbalization.py'): sha(CODE / 'scripts/run_end_to_end_verbalization.py'),
}
root = base / 'verbalization'
immutable(root / 'verbalization_manifest.json', {
    'schema': 'cue-v2-r31e-retrieval-reference-verbalization-v1',
    'model_name': 'Qwen/Qwen3-4B-Instruct-2507',
    'model_revision': REV,
    'seed': 42,
    'decoding': 'greedy',
    'examples': 500,
    'lists': 100,
    'positions': 5,
    'reference_items': 5,
    'cue_conditioning': 'none',
    'render_plan_sha256': sha(base / 'render-plan.json'),
    'assets': assets,
})
random.seed(42)
np.random.seed(42)
import torch
torch.manual_seed(42)
caller = None
for row in rows[args.start:args.start + args.count]:
    index = row['flat_index']
    destination = root / SYSTEM / f'{index:04d}.json'
    if destination.exists():
        previous = json.loads(destination.read_text())
        assert previous['render_row'] == row
        continue
    references = row['reference_item_ids']
    mu, sigma = runner._reference_structure(references, lookup, embeddings)
    generated = runner._generated_item(
        system=SYSTEM,
        example_index=0,
        reference_ids=references,
        mu_c=mu,
        sigma_c2=sigma,
        plan=None,
        codebook_weights=np.empty((0, 0), dtype=np.float32),
        item2cues={},
    )
    if caller is None:
        caller = runner.LocalQwen(
            model_name='Qwen/Qwen3-4B-Instruct-2507', revision=REV, device='cuda'
        )
        assert caller.model_revision == REV
    result = verbal.verbalize(
        generated=generated,
        catalog_embs=embeddings,
        catalog_metadata=ordered_metadata,
        k=5,
        cue_vocab=None,
        llm_call=caller,
    )
    record = {
        'result_schema': 'cue-v2-r31e-retrieval-reference-verbalization-record-v1',
        'system': SYSTEM,
        'example_index': index,
        'render_row': row,
        'seed': row['seed'],
        'retrieved_item_ids': references,
        'cue_ids': [],
        'cue_terms': result['cue_terms'],
        'neighbor_item_ids': [item.item_id for item in result['neighbors']],
        'style_item_ids': [item.item_id for item in result['style_summary']],
        'music_attributes': result['music_attributes'],
        'lyric_draft': result['lyric_draft'],
    }
    atomic(destination, record)
    print('VERBAL', args.dataset, args.arm, index, flush=True)
