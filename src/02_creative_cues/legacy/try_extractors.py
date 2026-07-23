"""legacy/try_extractors.py — quick look at cues from each extraction method.

Runs: extract -> normalize, then prints the cleaned cue vocab for each method.
Keep --limit small (llm + keybert run per-song). Ad-hoc debug script, not part
of the production or comparison pipelines.

Usage (from project root, with venv):
    .venv/Scripts/python.exe src/02_creative_cues/legacy/try_extractors.py
    .venv/Scripts/python.exe src/02_creative_cues/legacy/try_extractors.py --limit 80 --methods tfidf,yake,keybert,llm
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))               # 02_creative_cues/ (siblings)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "00_data_schema"))

import cue_extractors           # noqa: E402
import cue_normalize            # noqa: E402
import data_loading              # noqa: E402

CUE_DIR = Path(__file__).resolve().parents[1]

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
items, lyrics = data_loading.load_catalog_and_lyrics(args.limit)
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
        outp = CUE_DIR / "outputs" / method / "cue_vocab_preview.json"
        outp.parent.mkdir(parents=True, exist_ok=True)
        json.dump(real, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[try] {len(real)} real cues -> {outp}")
        print(f"[try] first {args.show}: {real[:args.show]}\n")
    except Exception as e:
        print(f"[try] {method} FAILED: {type(e).__name__}: {e}\n")
