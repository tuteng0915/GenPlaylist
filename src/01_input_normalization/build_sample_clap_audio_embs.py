"""build_sample_clap_audio_embs.py — Encode downloaded audio sample with CLAP audio encoder.

Reads:  data/audio/spotify/{item_id}.mp3   (downloaded by download_audio.py)
Writes: data/audio/catalog_clap_audio_embs.npy   (N, 512) float32, L2-normalised
        data/audio/catalog_clap_audio_ids.json    ordered item_id list matching rows

N = number of successfully encoded files (49 out of 50 expected).
Embeddings are in the same 512-dim CLAP space as CLAPTextEncoder, so a text query
can be compared directly against audio embeddings via cosine similarity.

Run:
  HF_HOME=.hf_cache python src/01_input_normalization/build_sample_clap_audio_embs.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

AUDIO_DIR  = os.path.join(PROJECT_ROOT, 'data', 'audio', 'spotify')
EMBS_PATH  = os.path.join(PROJECT_ROOT, 'data', 'audio', 'catalog_clap_audio_embs.npy')
IDS_PATH   = os.path.join(PROJECT_ROOT, 'data', 'audio', 'catalog_clap_audio_ids.json')

CATALOG_PATH = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'catalog_metadata.json')

sys.path.insert(0, SCRIPT_DIR)
from encoders import CLAPAudioEncoder, CLAPTextEncoder


def main() -> None:
    hf_cache = os.path.join(PROJECT_ROOT, '.hf_cache')
    os.makedirs(hf_cache, exist_ok=True)
    os.environ.setdefault('HF_HOME', hf_cache)

    # Collect downloaded audio files
    audio_dir = Path(AUDIO_DIR)
    audio_exts = {'.mp3', '.wav', '.flac', '.m4a', '.webm', '.opus', '.ogg'}
    files = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in audio_exts)
    print(f"Found {len(files)} audio files in {AUDIO_DIR}")

    if not files:
        print("No audio files found. Run download_audio.py first.")
        return

    # Load catalog for metadata display
    with open(CATALOG_PATH) as f:
        catalog = json.load(f)

    # item_id is the stem of each file (e.g. "241333.mp3" → "241333")
    file_map = {p.stem: p for p in files}
    print(f"Item IDs with audio: {list(file_map.keys())[:5]} ...")

    # Encode
    print(f"\nLoading {CLAPAudioEncoder.MODEL_ID} (audio encoder) ...")
    encoder = CLAPAudioEncoder()

    paths_ordered = [str(file_map[iid]) for iid in sorted(file_map, key=lambda x: files.index(file_map[x]))]
    print(f"Encoding {len(paths_ordered)} files ...")
    embs, good_paths = encoder.encode_files(
        paths_ordered,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Derive item_ids from successfully encoded paths
    item_ids = [Path(p).stem for p in good_paths]

    print(f"\nEncoded: {len(item_ids)} / {len(paths_ordered)} files")
    print(f"Embedding matrix: {embs.shape}  dtype={embs.dtype}")

    np.save(EMBS_PATH, embs)
    with open(IDS_PATH, 'w') as f:
        json.dump(item_ids, f)

    print(f"Saved embeddings  →  {EMBS_PATH}")
    print(f"Saved ID list     →  {IDS_PATH}")

    # Sanity check — same queries as other encoders
    print("\nSanity check (text query vs audio embeddings):")
    text_enc = CLAPTextEncoder()
    test_queries = [
        "upbeat indie rock songs for a road trip",
        "sad breakup ballads",
        "hip hop motivation for working out",
    ]
    for query in test_queries:
        q_emb    = text_enc.encode(query, normalize_embeddings=True)
        sims     = embs @ q_emb
        top3_idx = sims.argsort()[::-1][:3]
        print(f"\n  Query: \"{query}\"")
        for rank, idx in enumerate(top3_idx, 1):
            iid    = item_ids[idx]
            meta   = catalog.get(iid, {})
            title  = meta.get('title', iid)
            artist = meta.get('artist', '?')
            print(f"    {rank}. {title} — {artist}  (sim={sims[idx]:.3f})")


if __name__ == '__main__':
    main()
