"""build_feature_table.py — Pre-compute a multi-modal feature vector per song.

Each song gets a fixed-length numeric vector combining:
  - Mood (one-hot, 6 dims)
  - Tempo/BPM (normalized, 1 dim)
  - Musical key: root (one-hot, 12 dims) + mode (major/minor, 2 dims)
  - Genre (one-hot over top-20 genres, 20 dims; 0-vector when missing)
  - CLHE CF embedding (64 dims) — co-occurrence signal
  - ST text embedding (384 dims) — semantic/metadata signal
  - CLAP text embedding (512 dims) — music-aware text signal

Total: 6 + 1 + 12 + 2 + 20 + 64 + 384 + 512 = 1001 dims

The table enables fast song→song distance via pre-computed cosine or L2
without invoking any encoder at query time.

Outputs
-------
  data/dataset/feature_table.npy      (5119, 1001) float32
  data/dataset/feature_table_ids.json ordered item_id list
  data/dataset/feature_table_meta.json schema: {field: (start_col, end_col)}

Run:
  HF_HOME=.hf_cache python src/01_input_normalization/build_feature_table.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

CATALOG_PATH  = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'catalog_metadata.json')
CLHE_PATH     = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'clhe.npy')
ST_EMBS_PATH  = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'catalog_text_embs.npy')
ST_IDS_PATH   = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'catalog_text_ids.json')
CLAP_EMBS_PATH= os.path.join(PROJECT_ROOT, 'data', 'dataset', 'catalog_clap_embs.npy')
CLAP_IDS_PATH = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'catalog_clap_ids.json')

OUT_EMBS_PATH = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'feature_table.npy')
OUT_IDS_PATH  = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'feature_table_ids.json')
OUT_META_PATH = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'feature_table_meta.json')

# Vocab for one-hot encoding
MOOD_VOCAB  = ["aggressive", "calm", "euphoric", "intense", "melancholic", "neutral"]
KEY_ROOTS   = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KEY_MODES   = ["major", "minor"]
TOP_N_GENRE = 20   # one-hot over top-N genres by frequency; 0-vector for unknown

TEMPO_MIN, TEMPO_MAX = 39.0, 235.0   # observed range in catalog


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

def one_hot(val: str | None, vocab: list[str]) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    if val and val.lower() in vocab:
        vec[vocab.index(val.lower())] = 1.0
    return vec


def encode_mood(item: dict) -> np.ndarray:
    return one_hot(item.get("mood"), MOOD_VOCAB)


def encode_tempo(item: dict) -> np.ndarray:
    bpm = float(item.get("tempo") or 0)
    if bpm <= 0:
        return np.array([0.5], dtype=np.float32)  # mean imputation
    norm = (bpm - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN)
    return np.array([float(np.clip(norm, 0.0, 1.0))], dtype=np.float32)


def encode_key(item: dict) -> np.ndarray:
    """Parse 'D major' / 'F# minor' → (12-dim root one-hot) + (2-dim mode one-hot)."""
    key_str = (item.get("key") or "").strip()
    root = mode = None
    if key_str:
        parts = key_str.split()
        if len(parts) >= 2:
            root = parts[0]
            mode = parts[1].lower()
        elif len(parts) == 1:
            root = parts[0]
    root_vec = one_hot(root, KEY_ROOTS)
    mode_vec = one_hot(mode, KEY_MODES)
    return np.concatenate([root_vec, mode_vec])


def get_top_genres(catalog: dict, n: int) -> list[str]:
    from collections import Counter
    counts = Counter(
        v.get("genre", "").lower().strip()
        for v in catalog.values()
        if v.get("genre")
    )
    return [g for g, _ in counts.most_common(n)]


def encode_genre(item: dict, genre_vocab: list[str]) -> np.ndarray:
    return one_hot(item.get("genre", "").lower().strip(), genre_vocab)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading catalog ...")
    with open(CATALOG_PATH) as f:
        catalog = json.load(f)
    item_ids = list(catalog.keys())
    print(f"  {len(item_ids)} items")

    # Genre vocab
    genre_vocab = get_top_genres(catalog, TOP_N_GENRE)
    print(f"  Top-{TOP_N_GENRE} genres: {genre_vocab}")

    # Load pre-computed embeddings
    print("\nLoading embeddings ...")
    clhe = np.load(CLHE_PATH).astype(np.float32)
    with open(ST_IDS_PATH)   as f: st_ids   = json.load(f)
    with open(CLAP_IDS_PATH) as f: clap_ids = json.load(f)
    st_embs   = np.load(ST_EMBS_PATH).astype(np.float32)
    clap_embs = np.load(CLAP_EMBS_PATH).astype(np.float32)

    # Build ID-to-row maps
    clhe_id2row = {iid: i for i, iid in enumerate(st_ids)}   # CLHE uses same order as ST
    st_id2row   = {iid: i for i, iid in enumerate(st_ids)}
    clap_id2row = {iid: i for i, iid in enumerate(clap_ids)}

    # Compute feature dimensions
    mood_dim  = len(MOOD_VOCAB)      # 6
    tempo_dim = 1
    key_dim   = len(KEY_ROOTS) + len(KEY_MODES)   # 14
    genre_dim = TOP_N_GENRE          # 20
    clhe_dim  = clhe.shape[1]        # 64
    st_dim    = st_embs.shape[1]     # 384
    clap_dim  = clap_embs.shape[1]   # 512
    total_dim = mood_dim + tempo_dim + key_dim + genre_dim + clhe_dim + st_dim + clap_dim

    print(f"\nFeature dimensions:")
    print(f"  mood:       {mood_dim}")
    print(f"  tempo:      {tempo_dim}")
    print(f"  key:        {key_dim}")
    print(f"  genre:      {genre_dim}")
    print(f"  CLHE:       {clhe_dim}")
    print(f"  ST text:    {st_dim}")
    print(f"  CLAP text:  {clap_dim}")
    print(f"  TOTAL:      {total_dim}")

    # Build schema for column slices
    col = 0
    schema = {}
    for name, dim in [
        ("mood", mood_dim), ("tempo", tempo_dim), ("key", key_dim),
        ("genre", genre_dim), ("CLHE", clhe_dim), ("ST", st_dim), ("CLAP_text", clap_dim),
    ]:
        schema[name] = {"start": col, "end": col + dim, "dim": dim}
        col += dim

    # Build feature matrix
    print(f"\nBuilding feature table ({len(item_ids)} × {total_dim}) ...")
    feature_matrix = np.zeros((len(item_ids), total_dim), dtype=np.float32)

    for i, iid in enumerate(item_ids):
        item = catalog[iid]
        parts = [
            encode_mood(item),
            encode_tempo(item),
            encode_key(item),
            encode_genre(item, genre_vocab),
        ]

        # CLHE
        if iid in clhe_id2row:
            parts.append(clhe[clhe_id2row[iid]])
        else:
            parts.append(np.zeros(clhe_dim, dtype=np.float32))

        # ST
        if iid in st_id2row:
            parts.append(st_embs[st_id2row[iid]])
        else:
            parts.append(np.zeros(st_dim, dtype=np.float32))

        # CLAP text
        if iid in clap_id2row:
            parts.append(clap_embs[clap_id2row[iid]])
        else:
            parts.append(np.zeros(clap_dim, dtype=np.float32))

        feature_matrix[i] = np.concatenate(parts)

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(item_ids)}")

    print(f"  Done. Shape: {feature_matrix.shape}  dtype={feature_matrix.dtype}")
    print(f"  Memory: {feature_matrix.nbytes / 1e6:.1f} MB")

    # Save
    np.save(OUT_EMBS_PATH, feature_matrix)
    with open(OUT_IDS_PATH, 'w') as f:
        json.dump(item_ids, f)
    with open(OUT_META_PATH, 'w') as f:
        json.dump({
            "total_dim": total_dim,
            "genre_vocab": genre_vocab,
            "mood_vocab": MOOD_VOCAB,
            "key_roots": KEY_ROOTS,
            "key_modes": KEY_MODES,
            "columns": schema,
        }, f, indent=2)

    print(f"\nSaved → {OUT_EMBS_PATH}")
    print(f"Saved → {OUT_IDS_PATH}")
    print(f"Saved → {OUT_META_PATH}")

    # Sanity check: find nearest neighbors for a sample song
    print("\nSanity check — nearest neighbors by full feature vector:")
    # Normalize to make cosine distance meaningful across dimensions
    norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True) + 1e-9
    normed = feature_matrix / norms
    id2row = {iid: i for i, iid in enumerate(item_ids)}

    for test_id in [item_ids[0], item_ids[100], item_ids[500]]:
        q = normed[id2row[test_id]]
        sims = normed @ q
        sims[id2row[test_id]] = -1  # exclude self
        top5 = np.argsort(sims)[::-1][:5]
        meta = catalog[test_id]
        print(f"\n  Seed: {meta['title']} — {meta['artist']} (mood={meta.get('mood')}, BPM={meta.get('tempo')})")
        for rank, idx in enumerate(top5, 1):
            iid2 = item_ids[idx]
            m = catalog[iid2]
            print(f"    {rank}. {m['title']} — {m['artist']} (mood={m.get('mood')}, BPM={m.get('tempo')}, sim={sims[idx]:.3f})")


if __name__ == '__main__':
    main()
