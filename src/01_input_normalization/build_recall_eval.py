"""build_recall_eval.py — Recall@K / Precision@K evaluation for WP-A encoders.

Protocol
--------
For each playlist in the unified original-val + original-test source:
  1. Require every song to be present in the frozen catalog.
  2. Keep playlists with at least 20 songs and retain the first 20.
  3. Split: songs 1-15 = query context, songs 16-20 = ground truth.
  4. For each encoder:
       - Compute query embedding = L2-normalised centroid of query songs' embeddings.
       - Retrieve top-K from full catalog, excluding query songs.
       - Compute Recall@K = |retrieved ∩ ground_truth| / |ground_truth|.
  5. Report mean Recall@K and mean Precision@K across all playlists for K ∈ {5,10,20,50}.

Encoders evaluated
------------------
  ST        : SentenceTransformer text embeddings  (catalog_text_embs.npy)
  CLAP_text : CLAP text embeddings                 (catalog_clap_embs.npy)
  CLAP_audio: CLAP audio embeddings                (catalog_clap_audio_embs.npy)
  CLHE      : Collaborative-filtering embeddings   (clhe.npy)

Run:
  HF_HOME=.hf_cache python src/01_input_normalization/build_recall_eval.py
  HF_HOME=.hf_cache python src/01_input_normalization/build_recall_eval.py --n-playlists 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from shared.protocol import FROZEN_NEXT_SONG_PROTOCOL

DEFAULT_DATA_DIR = os.environ.get(
    'GENPLAYLIST_DATA_ROOT', os.path.join(PROJECT_ROOT, 'data', 'dataset'))
DEFAULT_ARTIFACT_DIR = os.environ.get(
    'GENPLAYLIST_ARTIFACT_ROOT', os.path.join(PROJECT_ROOT, 'data', 'dataset'))
DEFAULT_AUDIO_DIR = os.environ.get(
    'GENPLAYLIST_AUDIO_ROOT', os.path.join(PROJECT_ROOT, 'data', 'audio'))


def encoder_specs(data_dir: str, artifact_dir: str, audio_dir: str) -> dict[str, dict]:
    """Resolve canonical CLHE plus optional WP-A baseline artifacts."""
    return {
        'CLHE': {
            'embs': os.path.join(artifact_dir, 'catalog_item_embeddings.npy'),
            'ids': os.path.join(artifact_dir, 'item_id_to_row.json'),
            'required': True,
        },
        'ST': {
            'embs': os.path.join(data_dir, 'catalog_text_embs.npy'),
            'ids': os.path.join(data_dir, 'catalog_text_ids.json'),
        },
        'CLAP_text': {
            'embs': os.path.join(data_dir, 'catalog_clap_embs.npy'),
            'ids': os.path.join(data_dir, 'catalog_clap_ids.json'),
        },
        'CLAP_audio': {
            'embs': os.path.join(audio_dir, 'catalog_clap_audio_embs.npy'),
            'ids': os.path.join(audio_dir, 'catalog_clap_audio_ids.json'),
        },
        'LARP_audio': {
            'embs': os.path.join(data_dir, 'larp_audio_embs.npy'),
            'ids': os.path.join(data_dir, 'larp_ids.json'),
        },
        'LARP_caption': {
            'embs': os.path.join(data_dir, 'larp_caption_embs.npy'),
            'ids': os.path.join(data_dir, 'larp_ids.json'),
        },
        'LARP_fused': {
            'embs': os.path.join(data_dir, 'larp_fused_embs.npy'),
            'ids': os.path.join(data_dir, 'larp_ids.json'),
        },
    }

K_VALUES = [5, 10, 20, 50]
OUT_PATH = os.path.join(SCRIPT_DIR, 'outputs', 'recall_eval_15to5.json')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test_playlists(
    paths: str | os.PathLike | list[str | os.PathLike],
    catalog_ids: set[str],
) -> list[list[str]]:
    """Load the unified catalog-aligned test rows in deterministic source order."""
    playlists = []
    source_paths = [paths] if isinstance(paths, (str, os.PathLike)) else paths
    for path in source_paths:
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                fields = [value.strip() for value in line.strip().split(',') if value.strip()]
                if not fields:
                    continue
                songs = fields[1:]  # first field is the playlist ID
                unknown = [item_id for item_id in songs if item_id not in catalog_ids]
                if unknown:
                    raise ValueError(
                        f"{path}:{line_no}: item IDs missing from catalog: {unknown[:5]}")
                if len(songs) >= FROZEN_NEXT_SONG_PROTOCOL.eval_total_items:
                    references, targets = FROZEN_NEXT_SONG_PROTOCOL.split_evaluation_items(songs)
                    playlists.append([*references, *targets])
    return playlists


def split_playlist(songs: list[str]) -> tuple[list[str], list[str]]:
    """Apply the immutable first-20, 15-reference/5-target split."""
    return FROZEN_NEXT_SONG_PROTOCOL.split_evaluation_items(songs)


def load_row_ids(path: str | os.PathLike) -> list[str]:
    """Load either an ordered ID list or the canonical item_id -> row mapping."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, list):
        ids = [str(item_id) for item_id in raw]
    elif isinstance(raw, dict):
        rows = {str(item_id): int(row) for item_id, row in raw.items()}
        expected = set(range(len(rows)))
        if set(rows.values()) != expected:
            raise ValueError(f"{path}: row mapping must be contiguous 0..{len(rows) - 1}")
        ids = [""] * len(rows)
        for item_id, row in rows.items():
            ids[row] = item_id
    else:
        raise ValueError(f"{path}: expected an ID list or item_id-to-row object")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate item IDs")
    return ids


def validate_encoder_alignment(
    name: str,
    embeddings: np.ndarray,
    id_list: list[str],
    catalog_ids: set[str],
) -> None:
    if embeddings.ndim != 2:
        raise ValueError(f"[{name}] embeddings must be 2-D, got {embeddings.shape}")
    if embeddings.shape[0] != len(id_list):
        raise ValueError(
            f"[{name}] embedding rows ({embeddings.shape[0]}) != IDs ({len(id_list)})")
    ids = set(id_list)
    missing = sorted(catalog_ids - ids)
    extra = sorted(ids - catalog_ids)
    if missing or extra:
        raise ValueError(
            f"[{name}] must cover the complete frozen catalog; "
            f"missing={missing[:5]}, extra={extra[:5]}")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"[{name}] embeddings contain NaN or infinity")
    if np.any(np.linalg.norm(embeddings, axis=1) <= np.finfo(np.float32).eps):
        raise ValueError(f"[{name}] embeddings contain zero-norm catalog rows")


def centroid_embedding(
    item_ids: list[str],
    embs_matrix: np.ndarray,
    id_to_row: dict[str, int],
) -> np.ndarray | None:
    """L2-normalised centroid of embeddings for a list of item IDs."""
    rows = [id_to_row[iid] for iid in item_ids if iid in id_to_row]
    if not rows:
        return None
    vecs = embs_matrix[rows].astype(np.float32)
    c = vecs.mean(axis=0)
    norm = np.linalg.norm(c)
    return c / (norm + 1e-9)


def retrieve_top_k(
    query_emb: np.ndarray,
    embs_matrix: np.ndarray,
    all_ids: list[str],
    exclude: set[str],
    k: int,
) -> list[str]:
    """Cosine similarity retrieval, excluding query songs."""
    sims = embs_matrix @ query_emb           # (N,)
    order = np.argsort(sims)[::-1]
    results = []
    for idx in order:
        iid = all_ids[idx]
        if iid not in exclude:
            results.append(iid)
        if len(results) == k:
            break
    return results


def recall_at_k(retrieved: list[str], ground_truth: set[str], k: int) -> float:
    if not ground_truth:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in ground_truth)
    return hits / len(ground_truth)


def precision_at_k(retrieved: list[str], ground_truth: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in ground_truth)
    return hits / k


def reciprocal_rank(retrieved: list[str], ground_truth: set[str]) -> float:
    """Mean Reciprocal Rank contribution: 1/rank of the FIRST correct song, else 0."""
    for rank, iid in enumerate(retrieved, start=1):
        if iid in ground_truth:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR,
                        help='Dataset root containing catalog_metadata.json and splits/')
    parser.add_argument('--artifact-dir', default=DEFAULT_ARTIFACT_DIR,
                        help='Canonical CLHE/item_id_to_row artifact root')
    parser.add_argument('--audio-dir', default=DEFAULT_AUDIO_DIR,
                        help='Optional CLAP-audio artifact root')
    parser.add_argument('--n-playlists', type=int, default=None,
                        help='Evaluate on first N eligible test playlists (default: all 941)')
    parser.add_argument('--output', default=OUT_PATH,
                        help='Output JSON path')
    parser.add_argument('--out-suffix', type=str, default='',
                        help='Suffix appended to output filename (e.g. "_split30")')
    args = parser.parse_args()

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    artifact_dir = os.path.abspath(os.path.expanduser(args.artifact_dir))
    audio_dir = os.path.abspath(os.path.expanduser(args.audio_dir))
    catalog_path = os.path.join(data_dir, 'catalog_metadata.json')
    split_paths = [
        os.path.join(data_dir, 'splits', 'val.txt'),
        os.path.join(data_dir, 'splits', 'test.txt'),
    ]

    # Load catalog
    print("Loading catalog ...")
    with open(catalog_path) as f:
        catalog = json.load(f)
    catalog_ids = set(catalog.keys())
    print(f"  {len(catalog_ids)} catalog songs")

    # Load test playlists
    print("Loading test playlists ...")
    playlists = load_test_playlists(split_paths, catalog_ids)
    if args.n_playlists:
        playlists = playlists[:args.n_playlists]
    print(f"  {len(playlists)} playlists (min {min(len(p) for p in playlists)}, "
          f"max {max(len(p) for p in playlists)}, "
          f"avg {sum(len(p) for p in playlists)/len(playlists):.1f} songs)")

    # Load embeddings for each encoder
    encoders: dict[str, dict] = {}
    for name, paths in encoder_specs(data_dir, artifact_dir, audio_dir).items():
        embs_path = paths['embs']
        ids_path  = paths['ids']
        if not os.path.exists(embs_path) or not os.path.exists(ids_path):
            message = f"[{name}] missing embeddings/IDs: {embs_path}, {ids_path}"
            if paths.get('required', False):
                raise FileNotFoundError(message)
            print(f"  {message} — SKIP optional baseline")
            continue
        embs = np.load(embs_path).astype(np.float32)
        id_list = load_row_ids(ids_path)
        validate_encoder_alignment(name, embs, id_list, catalog_ids)
        # L2-normalise rows (idempotent if already normalised)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        embs = embs / norms

        id_to_row = {iid: i for i, iid in enumerate(id_list)}
        encoders[name] = {'embs': embs, 'id_list': id_list, 'id_to_row': id_to_row}
        print(f"  [{name}] {embs.shape}  —  {len(id_list)} IDs")

    # Evaluate
    results: dict[str, dict] = {}

    for enc_name, enc in encoders.items():
        embs      = enc['embs']
        id_list   = enc['id_list']
        id_to_row = enc['id_to_row']

        recall    = {k: [] for k in K_VALUES}
        precision = {k: [] for k in K_VALUES}
        rr_list   = []
        skipped   = 0

        for pl in playlists:
            query_songs, gt_songs = split_playlist(pl)
            gt_set = set(gt_songs)
            exclude = set(query_songs)

            q_emb = centroid_embedding(query_songs, embs, id_to_row)
            if q_emb is None:
                skipped += 1
                continue

            # Sort the complete non-reference catalog once. Recall/precision use
            # their requested cutoffs; MRR uses the true full-catalog rank.
            retrieved = retrieve_top_k(
                q_emb, embs, id_list, exclude, len(id_list) - len(exclude))

            for k in K_VALUES:
                recall[k].append(recall_at_k(retrieved, gt_set, k))
                precision[k].append(precision_at_k(retrieved, gt_set, k))
            rr_list.append(reciprocal_rank(retrieved, gt_set))

        n_eval = len(playlists) - skipped
        results[enc_name] = {
            'n_playlists': n_eval,
            'skipped': skipped,
            'protocol': 'unified_val_test_first20_15ref_5target',
            'reference_items': FROZEN_NEXT_SONG_PROTOCOL.eval_reference_items,
            'target_items': FROZEN_NEXT_SONG_PROTOCOL.eval_target_items,
            'recall':    {k: float(np.mean(v)) for k, v in recall.items()},
            'precision': {k: float(np.mean(v)) for k, v in precision.items()},
            'mrr_full_catalog': float(np.mean(rr_list)) if rr_list else 0.0,
        }
        print(f"\n[{enc_name}]  n={n_eval}  (15 references -> 5 targets)")
        print(f"  {'K':>4}  {'Recall@K':>10}  {'Precision@K':>12}")
        for k in K_VALUES:
            r = results[enc_name]['recall'][k]
            p = results[enc_name]['precision'][k]
            print(f"  {k:>4}  {r:>10.4f}  {p:>12.4f}")
        print(f"   MRR  {results[enc_name]['mrr_full_catalog']:>10.4f} (full catalog)")

    # Save (suffix in filename for split-ratio ablation)
    out_path = os.path.abspath(os.path.expanduser(args.output))
    if args.out_suffix:
        base, ext = os.path.splitext(out_path)
        out_path = f"{base}{args.out_suffix}{ext}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
