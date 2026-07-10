"""run_wpb.py — Run WP-B cue mining on the full 5,119-song Spotify catalog.

Usage:
    python run_wpb.py              # full 5,119 songs
    python run_wpb.py --limit 200  # first 200 songs only
"""

import argparse
import os
import sys

sys.path.insert(0, "src/02_creative_cues")
sys.path.insert(0, "src/00_data_schema")

from schema import CatalogItem, CUE_VOCAB_SIZE
from cue_mining import (
    build_vocab,
    run_full_dataset,
    export_outputs,
    write_report,
    compute_coverage_stats,
)

CATALOG_PATH = "data/dataset/catalog_metadata.json"
LYRICS_DIR   = "data/lyrics/spotify"
OUTPUT_DIR   = "src/02_creative_cues/outputs"

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=None,
                    help="Only process the first N songs (default: all)")
parser.add_argument("--vocab-method", default="tfidf",
                    choices=["tfidf", "keybert", "yake"],
                    help="How to build the cue vocabulary (default: tfidf)")
parser.add_argument("--assign-method", default="pmi",
                    choices=["pmi", "tfidf"],
                    help="How to assign cues per song (default: pmi)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load catalog
# ---------------------------------------------------------------------------
import json
with open(CATALOG_PATH, encoding="utf-8") as f:
    raw = json.load(f)
catalog = [
    CatalogItem(**{k: v for k, v in entry.items()
                   if k in CatalogItem.__dataclass_fields__})
    for entry in raw.values()
]
if args.limit:
    catalog = catalog[:args.limit]
print(f"[run_wpb] Loaded {len(catalog)} catalog items.")

# ---------------------------------------------------------------------------
# Load lyrics
# ---------------------------------------------------------------------------
lyrics_dict = {}
for item in catalog:
    path = os.path.join(LYRICS_DIR, f"{item.item_id}.txt")
    if os.path.isfile(path):
        lyrics_dict[item.item_id] = open(path, encoding="utf-8", errors="ignore").read()
print(f"[run_wpb] Lyrics loaded for {len(lyrics_dict)} / {len(catalog)} items.")

# ---------------------------------------------------------------------------
# Build vocabulary (TF-IDF over all text fields + lyrics)
# ---------------------------------------------------------------------------
print(f"[run_wpb] Building vocab ({args.vocab_method}) …")
vocab = build_vocab(
    catalog,
    lyrics_dict,
    vocab_size=CUE_VOCAB_SIZE,
    extraction_method=args.vocab_method,
)
assert vocab[0] == "<unk>" and len(vocab) == CUE_VOCAB_SIZE
print(f"[run_wpb] Vocab built: {len(vocab)} entries. Sample [1:6]: {vocab[1:6]}")

# ---------------------------------------------------------------------------
# Assign 6 cues per song (PMI + diversity regularization)
# ---------------------------------------------------------------------------
print(f"[run_wpb] Assigning cues ({args.assign_method}) …")
item2cues = run_full_dataset(catalog, vocab, lyrics_dict, assignment_method=args.assign_method)
print(f"[run_wpb] Assigned cues for {len(item2cues)} items.")

# ---------------------------------------------------------------------------
# Stats + export
# ---------------------------------------------------------------------------
stats = compute_coverage_stats(item2cues, vocab)
print(f"[run_wpb] Coverage stats: {stats}")

export_outputs(vocab, item2cues, OUTPUT_DIR)
write_report(stats, OUTPUT_DIR)
print("[run_wpb] Done. Outputs in", OUTPUT_DIR)
