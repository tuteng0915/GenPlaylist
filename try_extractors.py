"""try_extractors.py — quick look at cues from each extraction method.

Runs: extract -> normalize, then prints the cleaned cue vocab for each method.
Keep --limit small (llm + keybert run per-song).

Usage (from project root, with venv):
    .venv/Scripts/python.exe try_extractors.py
    .venv/Scripts/python.exe try_extractors.py --limit 80 --methods tfidf,yake,keybert,llm
"""

import argparse
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "src", "02_creative_cues"))
sys.path.insert(0, os.path.join(BASE, "src", "00_data_schema"))

import cue_extractors           # noqa: E402
import cue_normalize            # noqa: E402
from schema import CatalogItem  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=60)
parser.add_argument("--methods", default="tfidf,yake,keybert,llm")
parser.add_argument("--top-n", type=int, default=20, help="raw cues per song")
parser.add_argument("--show", type=int, default=50, help="how many cues to print")
parser.add_argument("--force", action="store_true", help="ignore cached raw cues")
parser.add_argument("--min-df", type=int, default=5,
                    help="min songs a cue must appear in (lower for small samples)")
args = parser.parse_args()

# --- load catalog + lyrics ---
with open(os.path.join(BASE, "data", "dataset", "catalog_metadata.json"), encoding="utf-8") as f:
    raw = json.load(f)
items = [
    CatalogItem(**{k: v for k, v in e.items() if k in CatalogItem.__dataclass_fields__})
    for e in list(raw.values())[:args.limit]
]
lyrics = {}
for it in items:
    p = os.path.join(BASE, "data", "lyrics", "spotify", f"{it.item_id}.txt")
    if os.path.isfile(p):
        lyrics[it.item_id] = open(p, encoding="utf-8", errors="ignore").read()
print(f"[try] {len(items)} songs, {len(lyrics)} with lyrics\n")

# --- run each method ---
for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
    print(f"================ {method} ================")
    try:
        rc = cue_extractors.extract_raw_cues(items, lyrics, method=method,
                                             force=args.force, top_n=args.top_n)
        res = cue_normalize.build_vocab_normalized(rc, vocab_size=2048,
                                                   min_df=args.min_df, verbose=True)
        real = [c for c in res["vocab"][1:] if not c.startswith("<pad_")]
        # save full cleaned vocab for browsing
        outp = os.path.join(BASE, "src", "02_creative_cues", "outputs", method, "cue_vocab_preview.json")
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        json.dump(real, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[try] {len(real)} real cues -> {outp}")
        print(f"[try] first {args.show}: {real[:args.show]}\n")
    except Exception as e:
        print(f"[try] {method} FAILED: {type(e).__name__}: {e}\n")
