"""build_catalog_clap_embs.py — One-time script: encode all catalog songs as CLAP text embeddings.

Produces (5119, 512) float32 L2-normalised matrix for comparison against
catalog_text_embs.npy (all-MiniLM-L6-v2, 384-dim).

Uses the SAME text strings as build_catalog_text_embs.py — only the encoder changes.
This isolates the encoder quality from the text representation choice.

Run once from project root:
  HF_HOME=.hf_cache python src/01_input_normalization/build_catalog_clap_embs.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

CATALOG_PATH = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'catalog_metadata.json')
EMBS_PATH    = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'catalog_clap_embs.npy')
IDS_PATH     = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'catalog_clap_ids.json')

BATCH_SIZE = 32  # CLAP is heavier than ST; smaller batches

sys.path.insert(0, SCRIPT_DIR)
from build_catalog_text_embs import build_text_string  # same text format, different encoder
from encoders import CLAPTextEncoder


def main() -> None:
    hf_cache = os.path.join(PROJECT_ROOT, '.hf_cache')
    os.makedirs(hf_cache, exist_ok=True)
    os.environ.setdefault('HF_HOME', hf_cache)

    print(f"Loading catalog  →  {CATALOG_PATH}")
    with open(CATALOG_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # Preserve insertion order — must match load_catalog_from_dict()
    item_ids = list(raw.keys())
    texts    = [build_text_string(raw[iid]) for iid in item_ids]

    print(f"Items: {len(item_ids)}")
    print(f"Sample text string:\n  [{item_ids[0]}] {texts[0]}\n")

    print(f"Loading {CLAPTextEncoder.MODEL_ID} ...")
    encoder = CLAPTextEncoder()

    print("Encoding catalog ...")
    embs = encoder.encode(
        texts,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
    )

    print(f"\nEmbedding matrix: {embs.shape}  dtype={embs.dtype}")

    np.save(EMBS_PATH, embs)
    with open(IDS_PATH, 'w', encoding='utf-8') as f:
        json.dump(item_ids, f)

    print(f"Saved embeddings  →  {EMBS_PATH}")
    print(f"Saved ID list     →  {IDS_PATH}")

    # Sanity check with the same queries used in export_examples.py
    test_queries = [
        "upbeat indie rock songs for a road trip",
        "sad breakup ballads",
        "hip hop motivation for working out",
    ]
    print("\nSanity check:")
    for query in test_queries:
        q_emb    = encoder.encode(query, normalize_embeddings=True)
        sims     = embs @ q_emb
        top3_idx = sims.argsort()[::-1][:3]
        print(f"\n  Query: \"{query}\"")
        for rank, idx in enumerate(top3_idx, 1):
            iid    = item_ids[idx]
            title  = raw[iid]['title']
            artist = raw[iid]['artist']
            print(f"    {rank}. {title} — {artist}  (sim={sims[idx]:.3f})")


if __name__ == '__main__':
    main()
