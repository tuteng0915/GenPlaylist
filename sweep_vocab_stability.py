"""sweep_vocab_stability.py — does the vocabulary converge as the corpus grows?

The vocab size is capped at 2,048, so "convergence" is really about set STABILITY:
do the same cues keep getting chosen as more songs are added? This builds the
top-2,047 vocabulary at several corpus sizes and measures the Jaccard overlap
between them.

  Jaccard(A, B) = |A ∩ B| / |A ∪ B|   (1.0 = identical sets, 0 = disjoint)

Rising Jaccard between consecutive sizes (and toward the largest size) = the
vocabulary is stabilizing/converging. Low, flat Jaccard = it keeps churning.

Additive: imports the pipeline functions, changes nothing. tfidf/yake are fast
and API-free; keybert/llm cost per uncached corpus size.

Usage:
  .venv/Scripts/python.exe sweep_vocab_stability.py --methods tfidf,yake \
      --limits 500,1000,2000,3000,5000 --min-df 2 --lyrics-mode dedup --lyrics-cap 2000 --top-n 60
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

import cue_extractors    # noqa: E402
import cue_normalize     # noqa: E402
import cue_lyrics        # noqa: E402
import cue_io            # noqa: E402
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


def vocab_set(method, limit, args, tag):
    items, lyrics = load_data(limit)
    lyrics_proc = {iid: cue_lyrics.preprocess_lyrics(t, args.lyrics_mode, args.lyrics_cap)
                   for iid, t in lyrics.items()}
    block = cue_normalize.build_block_tokens(items)
    raw = cue_extractors.extract_raw_cues(items, lyrics_proc, method=method,
                                          force=args.force, top_n=args.top_n, cache_tag=tag)
    norm = cue_normalize.build_vocab_normalized(
        raw, vocab_size=2048, min_df=args.min_df, dedup_threshold=args.dedup_threshold,
        block_tokens=block, verbose=False)
    vocab = norm["vocab"]
    return {c for c in vocab if c != "<unk>" and not c.startswith("<pad_")}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="tfidf,yake")
    ap.add_argument("--limits", default="500,1000,2000,3000,5000")
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--dedup-threshold", type=float, default=0.92)
    ap.add_argument("--lyrics-mode", default="dedup", choices=list(cue_lyrics.MODES))
    ap.add_argument("--lyrics-cap", type=int, default=cue_lyrics.DEFAULT_CAP)
    ap.add_argument("--top-n", type=int, default=60)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    limits = sorted(int(x) for x in args.limits.split(",") if x.strip())
    tag = cue_lyrics.cache_tag(args.lyrics_mode, args.lyrics_cap)
    print(f"[sweep] stability: methods={methods} limits={limits} min_df={args.min_df}")

    sets = {m: {} for m in methods}
    for m in methods:
        for n in limits:
            s = vocab_set(m, n, args, tag)
            sets[m][n] = s
            print(f"  {m:<8} N={n:>5} -> vocab {len(s)}")

    _write_report(sets, methods, limits, args)


def _write_report(sets, methods, limits, args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"sweep_stability_{stamp}.md")
    final = limits[-1]

    lines = [
        "# Vocabulary set stability (convergence) vs corpus size\n",
        f"_Generated {stamp} · min_df {args.min_df} · dedup {args.dedup_threshold} · "
        f"lyrics-mode {args.lyrics_mode} · top_n {args.top_n}_\n",
        "\nVocab size is capped at 2,048, so this measures whether the *chosen cue set* "
        "stabilizes as songs are added. **Jaccard** = overlap between two vocab sets "
        "(1.0 = identical, 0 = disjoint). Rising toward 1.0 = converging.\n",
    ]

    # Consecutive-size Jaccard + retention
    lines += ["\n## Consecutive-size overlap (per method)\n",
              "| method | N_from -> N_to | vocab_from | vocab_to | Jaccard | retention |\n",
              "|--------|----------------|-----------|----------|---------|-----------|\n"]
    for m in methods:
        for a, b in zip(limits, limits[1:]):
            va, vb = sets[m][a], sets[m][b]
            j = jaccard(va, vb)
            ret = (len(va & vb) / len(va)) if va else 0.0
            lines.append(f"| {m} | {a} -> {b} | {len(va)} | {len(vb)} | "
                         f"{j:.3f} | {ret:.3f} |\n")

    # Overlap of each size with the largest (final) vocab
    lines += [f"\n## Overlap with the largest vocab (N={final}) — convergence curve\n",
              "| method | " + " | ".join(f"N={n}" for n in limits) + " |\n",
              "|--------|" + "----|" * len(limits) + "\n"]
    for m in methods:
        cells = [f"{jaccard(sets[m][n], sets[m][final]):.3f}" for n in limits]
        lines.append(f"| {m} | " + " | ".join(cells) + " |\n")

    lines += ["\n## How to read this\n",
              "- **Consecutive Jaccard rising toward 1.0** = the vocabulary is settling; each "
              "added batch of songs changes it less.\n",
              "- **retention** = fraction of the smaller vocab that survives into the larger one; "
              "high retention means cues persist rather than churn out.\n",
              "- **Overlap-with-final rising toward 1.0** = smaller-corpus vocabs increasingly "
              "resemble the full-corpus vocab (the vocab is converging to a stable set).\n",
              "- A method whose numbers stay **low/flat** produces a churning, unstable vocabulary; "
              "one that **climbs** produces a convergent, canonical vocabulary.\n"]
    cue_io.atomic_write_text(path, "".join(lines))
    print(f"\n[sweep] report -> {path}")


if __name__ == "__main__":
    main()
