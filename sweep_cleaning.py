"""sweep_cleaning.py — parameter sweep over min_df x semantic-dedup threshold.

Answers "how do min_df and the dedup threshold affect the vocabulary size?" by
re-running only the (cheap) cleaning step across a grid of configs. Extraction is
shared/cached, so this is fast and makes zero changes to run_compare or the
pipeline modules — it just imports their functions.

Usage:
  .venv/Scripts/python.exe sweep_cleaning.py --method tfidf --limit 1000 \
      --lyrics-mode dedup --lyrics-cap 2000 --top-n 60 \
      --min-df 2,3,5,10 --dedup-threshold 0.90,0.92,0.95,1.0
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


def load_data(limit):
    with open(os.path.join(BASE, "data", "dataset", "catalog_metadata.json"), encoding="utf-8") as f:
        raw = json.load(f)
    items = [CatalogItem(**{k: v for k, v in e.items() if k in CatalogItem.__dataclass_fields__})
             for e in list(raw.values())[:limit]]
    lyrics = {}
    for it in items:
        p = os.path.join(BASE, "data", "lyrics", "spotify", f"{it.item_id}.txt")
        if os.path.isfile(p):
            lyrics[it.item_id] = open(p, encoding="utf-8", errors="ignore").read()
    return items, lyrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="tfidf")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--lyrics-mode", default="dedup", choices=list(cue_lyrics.MODES))
    ap.add_argument("--lyrics-cap", type=int, default=cue_lyrics.DEFAULT_CAP)
    ap.add_argument("--top-n", type=int, default=60)
    ap.add_argument("--min-df", default="2,3,5,10", help="comma-separated min_df values")
    ap.add_argument("--dedup-threshold", default="0.90,0.92,0.95,1.0",
                    help="comma-separated dedup thresholds (1.0 = skip dedup)")
    ap.add_argument("--force", action="store_true", help="re-extract raw cues")
    args = ap.parse_args()

    min_dfs = [int(x) for x in args.min_df.split(",") if x.strip()]
    dedups = [float(x) for x in args.dedup_threshold.split(",") if x.strip()]

    items, lyrics = load_data(args.limit)
    lyrics_proc = {iid: cue_lyrics.preprocess_lyrics(t, args.lyrics_mode, args.lyrics_cap)
                   for iid, t in lyrics.items()}
    block = cue_normalize.build_block_tokens(items)
    tag = cue_lyrics.cache_tag(args.lyrics_mode, args.lyrics_cap)
    print(f"[sweep] method={args.method} limit={args.limit} lyrics-mode={args.lyrics_mode} "
          f"grid: min_df={min_dfs} x dedup={dedups}")

    raw = cue_extractors.extract_raw_cues(items, lyrics_proc, method=args.method,
                                          force=args.force, top_n=args.top_n, cache_tag=tag)

    rows = []
    for md in min_dfs:
        for dt in dedups:
            norm = cue_normalize.build_vocab_normalized(
                raw, vocab_size=2048, min_df=md, dedup_threshold=dt,
                block_tokens=block, verbose=False)
            s = norm["stats"]
            real = s["after_dedup"]
            rows.append({
                "min_df": md, "dedup": dt,
                "raw": s["raw_candidates"], "df_band": s["after_df_band"],
                "pos": s["after_pos"], "block": s["after_blocklist"],
                "real": real, "fill": min(real, 2047) / 2047,
            })
            print(f"  min_df={md:>3} dedup={dt:<4} -> real vocab {real}")

    _write_report(rows, args)


def _write_report(rows, args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"sweep_{args.method}_{stamp}.md")

    lines = [
        f"# Cleaning parameter sweep — `{args.method}`\n",
        f"_Generated {stamp} · {args.limit} songs · lyrics-mode {args.lyrics_mode} "
        f"(cap {args.lyrics_cap}) · top_n {args.top_n}_\n",
        "\nEach row is one **(min_df, dedup-threshold)** config. Extraction is shared/cached; "
        "only the cleaning step re-runs. `real vocab` = distinct cues surviving every filter "
        "(remaining slots up to 2,048 are `<pad_*>`). `fill%` = real vocab / 2,047.\n",
        "\n## Full grid\n",
        "| min_df | dedup thr | raw cand | after df-band | after POS | after blocklist | real vocab | fill% |\n",
        "|--------|-----------|----------|---------------|-----------|-----------------|-----------|-------|\n",
    ]
    for r in rows:
        lines.append(f"| {r['min_df']} | {r['dedup']:.2f} | {r['raw']} | {r['df_band']} | "
                     f"{r['pos']} | {r['block']} | **{r['real']}** | {int(round(r['fill']*100))}% |\n")

    # Marginal views: hold one variable, vary the other
    min_dfs = sorted({r["min_df"] for r in rows})
    dedups = sorted({r["dedup"] for r in rows})
    lines += ["\n## Effect of min_df (real vocab; columns = dedup threshold)\n",
              "| min_df | " + " | ".join(f"{d:.2f}" for d in dedups) + " |\n",
              "|--------|" + "----|" * len(dedups) + "\n"]
    for md in min_dfs:
        cells = []
        for d in dedups:
            v = next((r["real"] for r in rows if r["min_df"] == md and r["dedup"] == d), "-")
            cells.append(str(v))
        lines.append(f"| {md} | " + " | ".join(cells) + " |\n")

    lines += [
        "\n## How to read this\n",
        "- Read **down a column** to see the `min_df` effect (dominant lever); read **across a row** "
        "for the dedup-threshold effect (usually smaller).\n",
        "- `dedup thr = 1.00` skips semantic dedup entirely (keeps all near-duplicates).\n",
        "- The gap between `after df-band` and `real vocab` is what the POS + blocklist + dedup "
        "filters remove — inspect it to judge whether cleaning is too aggressive.\n",
    ]
    cue_io.atomic_write_text(path, "".join(lines))
    print(f"\n[sweep] report -> {path}")


if __name__ == "__main__":
    main()
