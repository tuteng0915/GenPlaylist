"""build_catalog_text_embs.py — One-time script: encode all catalog songs as text embeddings.

Each song is converted to a descriptive text string, then encoded by
SentenceTransformer('all-MiniLM-L6-v2') into a 384-dim vector.

The result is a (5119, 384) matrix saved to data/dataset/catalog_text_embs.npy.
A companion file catalog_text_ids.json records which item_id maps to each row,
so callers can verify alignment before using the matrix.

Row order matches the insertion order of catalog_metadata.json, which is the
same order load_catalog_from_dict() returns — so catalog_id_list[i] == text_ids[i].

Run once from project root:
  python src/01_input_normalization/build_catalog_text_embs.py
"""

from __future__ import annotations

import json
import os

import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

CATALOG_PATH = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'catalog_metadata.json')
EMBS_PATH    = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'catalog_text_embs.npy')
IDS_PATH     = os.path.join(PROJECT_ROOT, 'data', 'dataset', 'catalog_text_ids.json')

MODEL_NAME   = 'all-MiniLM-L6-v2'
BATCH_SIZE   = 64

# Template D vocabulary maps (from run_template_ablation.py §12.7 ablation winner)
_MOOD_MAP = {
    "aggressive":  ("aggressive, intense, high-energy, raw",
                    "workouts, hard runs, intense exercise, pumping up"),
    "calm":        ("calm, peaceful, relaxing, mellow, soothing",
                    "studying, reading, unwinding, background listening, meditation"),
    "euphoric":    ("euphoric, uplifting, joyful, feel-good, celebratory",
                    "parties, dancing, celebrating, road trips, happy moments"),
    "intense":     ("intense, powerful, emotionally charged, dramatic",
                    "deep listening, emotional moments, focus sessions"),
    "melancholic": ("melancholic, sad, emotional, wistful, heartfelt",
                    "reflective moods, heartbreak, late nights, rainy days"),
    "neutral":     ("easy-going, accessible, versatile, laid-back",
                    "casual listening, background music, everyday playlists"),
}


def _tempo_word(bpm: float) -> str:
    if bpm < 70:   return "very slow, ballad-paced"
    if bpm < 100:  return "slow to moderate, relaxed"
    if bpm < 120:  return "moderate, mid-tempo"
    if bpm < 140:  return "upbeat, lively"
    return "fast-paced, energetic"


def build_text_string(item: dict) -> str:
    """Template D: query-mirroring format (ablation winner from §12.7).

    Writes activity/context vocabulary directly into the catalog string so
    intent queries ("songs for studying", "gym playlist") overlap lexically
    with catalog entries. Only format where intent avg > specific avg for
    both ST and CLAP encoders.

    Format: "Title by Artist. Good for: <activities>. <mood_adj>, <tempo>, <key>. <lyric>"
    """
    mood = item.get('mood', 'neutral')
    bpm  = float(item.get('tempo') or 0)
    key  = item.get('key', '')
    adj, act = _MOOD_MAP.get(mood, ("versatile", "general listening"))
    tp = _tempo_word(bpm) if bpm else ""

    out = f"{item['title']} by {item['artist']}. "
    out += f"Good for: {act}. "
    out += adj
    if tp:
        out += f", {tp}"
    if key:
        out += f", {key}"
    out += "."
    lyric = item.get('lyric_excerpt', '')
    if lyric:
        out += f" {lyric.replace(chr(10), ' ')[:130]}"
    return out


def main() -> None:
    from sentence_transformers import SentenceTransformer

    print(f"Loading catalog  →  {CATALOG_PATH}")
    with open(CATALOG_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # Preserve JSON insertion order — must match load_catalog_from_dict()
    item_ids = list(raw.keys())
    texts    = [build_text_string(raw[iid]) for iid in item_ids]

    print(f"Items: {len(item_ids)}")
    print(f"Sample text string:\n  [{item_ids[0]}] {texts[0]}\n")

    print(f"Loading encoder: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding catalog ...")
    embs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        # Pre-normalize so that dot product == cosine similarity at query time
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    print(f"\nEmbedding matrix: {embs.shape}  dtype={embs.dtype}")

    np.save(EMBS_PATH, embs)
    with open(IDS_PATH, 'w', encoding='utf-8') as f:
        json.dump(item_ids, f)

    print(f"Saved embeddings  →  {EMBS_PATH}")
    print(f"Saved ID list     →  {IDS_PATH}")

    # Sanity check: run three test queries and show top-3 results
    test_queries = [
        "upbeat indie rock songs for a road trip",
        "sad breakup ballads",
        "hip hop motivation for working out",
    ]
    print("\nSanity check:")
    for query in test_queries:
        q_emb    = model.encode(query, normalize_embeddings=True)
        sims     = embs @ q_emb
        top3_idx = sims.argsort()[::-1][:3]
        print(f"\n  Query: \"{query}\"")
        for rank, idx in enumerate(top3_idx, 1):
            iid   = item_ids[idx]
            title = raw[iid]['title']
            artist = raw[iid]['artist']
            print(f"    {rank}. {title} — {artist}  (sim={sims[idx]:.3f})")


if __name__ == '__main__':
    main()
