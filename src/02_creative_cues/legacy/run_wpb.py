"""legacy/run_wpb.py — Run the legacy Week-1 cue mining on the full catalog.

**Superseded** by run_production.py, which uses the current pipeline (multiple
extraction methods, shared cleaning, MMR assignment) instead of this original
single-method TF-IDF/PMI approach. Kept only for reference / historical runs.

Usage:
    python legacy/run_wpb.py              # full catalog
    python legacy/run_wpb.py --limit 200  # first 200 songs only
"""

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent            # .../02_creative_cues/legacy
_CUE_DIR = _HERE.parent                            # .../02_creative_cues
_REPO_ROOT = _CUE_DIR.parents[1]                   # repo root

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_CUE_DIR))
sys.path.insert(0, str(_CUE_DIR.parent / "00_data_schema"))

from schema import CatalogItem, CUE_VOCAB_SIZE  # noqa: E402
from cue_mining_legacy import build_vocab, run_full_dataset  # noqa: E402
from cue_export import export_outputs, write_report, compute_coverage_stats  # noqa: E402

CATALOG_PATH = _REPO_ROOT / "data" / "dataset" / "catalog_metadata.json"
LYRICS_DIR = _REPO_ROOT / "data" / "lyrics" / "spotify"
OUTPUT_DIR = _CUE_DIR / "outputs" / "legacy_reports" / "run_wpb"

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
    path = LYRICS_DIR / f"{item.item_id}.txt"
    if path.is_file():
        lyrics_dict[item.item_id] = path.read_text(encoding="utf-8", errors="ignore")
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

export_outputs(vocab, item2cues, str(OUTPUT_DIR))
write_report(stats, str(OUTPUT_DIR))
print("[run_wpb] Done. Outputs in", OUTPUT_DIR)
