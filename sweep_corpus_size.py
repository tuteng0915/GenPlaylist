"""sweep_corpus_size.py — vocabulary size vs number of songs (corpus size).

For each --limit (number of training songs) and each method, extracts raw cues
(cached) and runs the cleaning step, reporting the resulting real vocabulary size.
Shows how the cue vocabulary grows with corpus size. Additive: imports the pipeline
functions and changes nothing else.

Usage:
  .venv/Scripts/python.exe sweep_corpus_size.py --methods tfidf,yake \
      --limits 100,300,500,1000 --min-df 5 --dedup-threshold 0.92 \
      --lyrics-mode dedup --lyrics-cap 2000 --top-n 60

Note: uncached limits trigger extraction. tfidf/yake are fast; keybert/llm are
expensive per song, so only sweep those over limits you have cached (or expect cost).
"""

import argparse
import json
import os
import sys
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "src", "02_creative_cues"))
sys.path.insert(0, os.path.join(BASE, "src", "00_data_schema"))

import cue_extractors   # noqa: E402
import cue_normalize    # noqa: E402
import cue_lyrics       # noqa: E402
import cue_io           # noqa: E402
from schema import CatalogItem  # noqa: E402

OUT_DIR = os.path.join(BASE, "src", "02_creative_cues", "outputs", "experiments")
_CATALOG = None


def _catalog():
    global _CATALOG
    if _CATALOG is None:
        with open(os.path.join(BASE, "data", "dataset", "catalog_metadata.json"), encoding="utf-8") as f:
            _CATALOG = list(json.load(f).items())
    return _CATALOG


def load_data(limit):
    items = [CatalogItem(**{k: v for k, v in e.items() if k in CatalogItem.__dataclass_fields__})
             for _iid, e in _catalog()[:limit]]
    lyrics = {}
    for it in items:
        p = os.path.join(BASE, "data", "lyrics", "spotify", f"{it.item_id}.txt")
        if os.path.isfile(p):
            lyrics[it.item_id] = open(p, encoding="utf-8", errors="ignore").read()
    return items, lyrics


def run_one(method, limit, args, tag):
    items, lyrics = load_data(limit)
    lyrics_proc = {iid: cue_lyrics.preprocess_lyrics(t, args.lyrics_mode, args.lyrics_cap)
                   for iid, t in lyrics.items()}
    block = cue_normalize.build_block_tokens(items)
    raw = cue_extractors.extract_raw_cues(items, lyrics_proc, method=method,
                                          force=args.force, top_n=args.top_n, cache_tag=tag)
    norm = cue_normalize.build_vocab_normalized(
        raw, vocab_size=2048, min_df=args.min_df, dedup_threshold=args.dedup_threshold,
        block_tokens=block, verbose=False)
    return norm["stats"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="tfidf,yake")
    ap.add_argument("--limits", default="100,300,500,1000", help="comma-separated song counts")
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--dedup-threshold", type=float, default=0.92)
    ap.add_argument("--lyrics-mode", default="dedup", choices=list(cue_lyrics.MODES))
    ap.add_argument("--lyrics-cap", type=int, default=cue_lyrics.DEFAULT_CAP)
    ap.add_argument("--top-n", type=int, default=60)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    limits = [int(x) for x in args.limits.split(",") if x.strip()]
    tag = cue_lyrics.cache_tag(args.lyrics_mode, args.lyrics_cap)
    print(f"[sweep] methods={methods} limits={limits} min_df={args.min_df} "
          f"dedup={args.dedup_threshold} lyrics-mode={args.lyrics_mode}")

    # results[method][limit] = stats
    results = {m: {} for m in methods}
    for limit in limits:
        for method in methods:
            s = run_one(method, limit, args, tag)
            results[method][limit] = s
            print(f"  N={limit:>5} {method:<8} -> real vocab {s['after_dedup']} "
                  f"(raw {s['raw_candidates']})")

    _write_report(results, methods, limits, args)


def _write_report(results, methods, limits, args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"sweep_corpus_{stamp}.md")

    lines = [
        "# Vocabulary size vs corpus size\n",
        f"_Generated {stamp} · min_df {args.min_df} · dedup {args.dedup_threshold} · "
        f"lyrics-mode {args.lyrics_mode} (cap {args.lyrics_cap}) · top_n {args.top_n}_\n",
        "\n`real vocab` = distinct cues surviving all cleaning filters (rest of the 2,048 "
        "slots are `<pad_*>`). This shows how many usable cues the corpus yields as it grows.\n",
        "\n## Real vocabulary size (rows = songs N, columns = method)\n",
        "| songs (N) | " + " | ".join(methods) + " | fill% (best) |\n",
        "|-----------|" + "----|" * len(methods) + "----|\n",
    ]
    for limit in limits:
        cells = []
        best = 0
        for m in methods:
            v = results[m].get(limit, {}).get("after_dedup", "-")
            cells.append(str(v))
            if isinstance(v, int):
                best = max(best, v)
        fill = int(round(min(best, 2047) / 2047 * 100))
        lines.append(f"| {limit} | " + " | ".join(cells) + f" | {fill}% |\n")

    lines += ["\n## Raw candidate cues before cleaning (rows = N, columns = method)\n",
              "| songs (N) | " + " | ".join(methods) + " |\n",
              "|-----------|" + "----|" * len(methods) + "\n"]
    for limit in limits:
        cells = [str(results[m].get(limit, {}).get("raw_candidates", "-")) for m in methods]
        lines.append(f"| {limit} | " + " | ".join(cells) + " |\n")

    # per-method funnel at the largest N (where cleaning cuts are clearest)
    biggest = max(limits)
    lines += [f"\n## Cleaning funnel at N={biggest}\n",
              "| method | raw | after df-band | after POS | after blocklist | real vocab |\n",
              "|--------|-----|---------------|-----------|-----------------|-----------|\n"]
    for m in methods:
        s = results[m].get(biggest, {})
        lines.append(f"| {m} | {s.get('raw_candidates','-')} | {s.get('after_df_band','-')} | "
                     f"{s.get('after_pos','-')} | {s.get('after_blocklist','-')} | "
                     f"**{s.get('after_dedup','-')}** |\n")

    lines += ["\n## How to read this\n",
              "- Read **down a column** to see how vocab grows with more songs.\n",
              "- If vocab plateaus well below 2,047 as N grows, the ceiling is cleaning "
              "(min_df / POS), not corpus size — cross-check with `sweep_cleaning.py`.\n",
              "- `min_df` is an absolute count, so its effect strengthens as N grows "
              "(a fixed min_df filters a smaller *fraction* of a larger corpus).\n"]
    cue_io.atomic_write_text(path, "".join(lines))
    print(f"\n[sweep] report -> {path}")


if __name__ == "__main__":
    main()
