"""sweep_embedder_dedup.py — recalibrate the semantic-dedup threshold per embedder.

Motivation (see PROGRESS_REPORT_2.md §6.5): swapping the default `minilm` embedder
for `qwen3-0.6b` at the *same* --dedup-threshold (0.92, tuned against MiniLM's cosine
distribution) collapsed within-item cue diversity (intra_cos_mean 0.30 -> 0.66, near
the paper's 0.7 collapse ceiling) with no reconstruction-quality gain to offset it.
That report flagged, but never ran, the obvious next step: re-sweep the threshold
against each embedder's own cosine distribution instead of reusing MiniLM's tuning.

This script runs the full extract(shared/cached) -> clean -> assign -> health-check
pipeline once per (embedder, dedup-threshold) grid point, via the same
pipeline.build_vocab_and_assign used by run_production.py, and reports
intra_cos_mean (diversity; paper target < 0.7) + vocab/coverage stats side by side
across embedders so you can pick a per-embedder threshold that actually gets
diversity back down near the MiniLM baseline instead of assuming 0.92 transfers.

The embedder model itself is loaded once per embedder (not once per grid point) via
a local cache around cue_normalize._make_embedder, since re-instantiating
SentenceTransformer for qwen3-0.6b/4b on every threshold is the expensive part;
each grid point still re-embeds its own (smaller, threshold-dependent) cue set and
re-embeds the song corpus, since both vocab and song text change per iteration.

Usage:
  python src/02_creative_cues/sweeps/sweep_embedder_dedup.py \
      --embedders minilm,qwen3-0.6b --dedup-threshold 0.90,0.92,0.94,0.96,0.98 \
      --limit 800 --num-cues 18

  # Narrow in on just qwen once you know roughly where MiniLM's baseline sits
  python src/02_creative_cues/sweeps/sweep_embedder_dedup.py \
      --embedders qwen3-0.6b --dedup-threshold 0.95,0.96,0.97,0.98,0.99 --limit 800
"""

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "00_data_schema"))

import cue_extractors    # noqa: E402
import cue_normalize     # noqa: E402
import cue_eval          # noqa: E402
import cue_export        # noqa: E402
import cue_lyrics        # noqa: E402
import cue_io            # noqa: E402
import data_loading      # noqa: E402
import pipeline          # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "experiments")

DIVERSITY_TARGET = 0.7   # paper's within-item cue-diversity collapse ceiling


def _install_embedder_cache():
    """Monkeypatch cue_normalize._make_embedder to load each model once per
    process instead of once per grid point. Cheap encode() calls still happen
    every iteration (vocab/song text differ per threshold) — this only removes
    the repeated model-loading cost, which dominates for qwen3-0.6b/4b."""
    cache = {}
    original = cue_normalize._make_embedder

    def cached(token_ids_of=None, embedder=cue_normalize.DEFAULT_EMBEDDER):
        if embedder not in cache:
            cache[embedder] = original(token_ids_of, embedder=embedder)
        return cache[embedder]

    cue_normalize._make_embedder = cached


def run_one(embedder, threshold, raw, items, lyrics_proc, block, out_dir, args):
    vocab, item2cues, norm_stats, cue_emb = pipeline.finish_from_raw(
        args.method, raw, items, lyrics_proc, out_dir,
        min_df=args.min_df, max_df_frac=args.max_df_frac, dedup_threshold=threshold,
        rank_by=args.rank_by, num_cues=args.num_cues, vocab_size=args.vocab_size,
        embedder=embedder, block_tokens=block, verbose=False,
    )
    cov = cue_export.compute_coverage_stats(item2cues, vocab)
    div = cue_eval.within_item_diversity(item2cues, cue_emb)
    return {
        "real_vocab": norm_stats["after_dedup"],
        "coverage_rate": cov.get("coverage_rate"),
        "unk_rate": cov.get("unk_rate"),
        "vocab_coverage": cov.get("vocab_coverage"),
        "intra_cos_mean": div.get("intra_cos_mean"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedders", default="minilm,qwen3-0.6b",
                    help="comma-separated cue_normalize.EMBEDDER_MODELS keys (or raw HF model ids)")
    ap.add_argument("--dedup-threshold", default="0.90,0.92,0.94,0.96,0.98",
                    help="comma-separated cosine thresholds to sweep per embedder")
    ap.add_argument("--method", default="tfidf", help="cue extraction method (tfidf = free/fast, "
                    "since this sweep is about the embedder, not extraction quality)")
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--max-df-frac", type=float, default=0.3)
    ap.add_argument("--rank-by", default="idf")
    ap.add_argument("--num-cues", type=int, default=18,
                    help="cues/song (default 18, matching the production preset that surfaced "
                         "this issue — not the WP-D schema default of 6)")
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--lyrics-mode", default="dedup", choices=list(cue_lyrics.MODES))
    ap.add_argument("--lyrics-cap", type=int, default=2000)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--force", action="store_true", help="bypass extraction/cleaning caches")
    ap.add_argument("--keep-artifacts", action="store_true",
                    help="keep each grid point's cue_vocab.json/item2cues.json "
                         "(default: only the summary report is kept, per-point dirs are deleted)")
    args = ap.parse_args()

    embedders = [e.strip() for e in args.embedders.split(",") if e.strip()]
    thresholds = [float(x) for x in args.dedup_threshold.split(",") if x.strip()]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(OUT_DIR, f"sweep_embedder_dedup_{stamp}")
    os.makedirs(sweep_dir, exist_ok=True)

    _install_embedder_cache()

    items, lyrics = data_loading.load_catalog_and_lyrics(args.limit)
    lyrics_proc = data_loading.build_lyrics_proc(lyrics, args.lyrics_mode, args.lyrics_cap)
    block = cue_normalize.build_block_tokens(items)
    tag = cue_lyrics.cache_tag(args.lyrics_mode, args.lyrics_cap)
    print(f"[sweep] {len(items)} songs · embedders={embedders} · thresholds={thresholds} · "
          f"method={args.method} · num_cues={args.num_cues}")

    raw = cue_extractors.extract_raw_cues(items, lyrics_proc, method=args.method,
                                          force=args.force, top_n=args.top_n, cache_tag=tag)

    # results[embedder][threshold] = stats
    results = {e: {} for e in embedders}
    for embedder in embedders:
        for dt in thresholds:
            point_dir = os.path.join(sweep_dir, f"{embedder}_dt{dt:.2f}")
            stats = run_one(embedder, dt, raw, items, lyrics_proc, block, point_dir, args)
            results[embedder][dt] = stats
            if not args.keep_artifacts:
                shutil.rmtree(point_dir, ignore_errors=True)
            print(f"  {embedder:<14} dt={dt:.2f} -> vocab {stats['real_vocab']:>5}  "
                  f"unk={stats['unk_rate']}  intra_cos={stats['intra_cos_mean']}")

    _write_report(results, embedders, thresholds, sweep_dir, stamp, args)


def _write_report(results, embedders, thresholds, sweep_dir, stamp, args):
    path = os.path.join(sweep_dir, "report.md")

    lines = [
        "# Embedder x dedup-threshold sweep\n",
        f"_Generated {stamp} · {args.limit} songs · method {args.method} · min_df {args.min_df} · "
        f"num_cues {args.num_cues} · vocab_size {args.vocab_size} · lyrics-mode {args.lyrics_mode} "
        f"(cap {args.lyrics_cap})_\n",
        "\nContext: production runs on `minilm`. This sweep exists to find whether some "
        "`dedup-threshold` recovers non-`minilm` embedders (esp. `qwen3-0.6b`) to a comparable "
        f"within-item diversity, instead of assuming MiniLM's tuned `0.92` transfers — see "
        "`PROGRESS_REPORT_2.md` §6.5.\n",
        "\n## Within-item cue diversity — `intra_cos_mean` (lower = more diverse; "
        f"paper collapse ceiling = {DIVERSITY_TARGET})\n",
        "| dedup thr | " + " | ".join(embedders) + " |\n",
        "|-----------|" + "----|" * len(embedders) + "\n",
    ]
    for dt in thresholds:
        cells = []
        for e in embedders:
            v = results[e].get(dt, {}).get("intra_cos_mean")
            if v is None:
                cells.append("-")
            else:
                flag = " ⚠️" if v >= DIVERSITY_TARGET else ""
                cells.append(f"{v:.3f}{flag}")
        lines.append(f"| {dt:.2f} | " + " | ".join(cells) + " |\n")

    lines += ["\n## Vocabulary size / coverage (real vocab, incl. `<unk>`)\n",
              "| dedup thr | " + " | ".join(embedders) + " |\n",
              "|-----------|" + "----|" * len(embedders) + "\n"]
    for dt in thresholds:
        cells = [str(results[e].get(dt, {}).get("real_vocab", "-")) for e in embedders]
        lines.append(f"| {dt:.2f} | " + " | ".join(cells) + " |\n")

    lines += ["\n## UNK slot rate (fraction of assigned cue slots that fell back to `<unk>`)\n",
              "| dedup thr | " + " | ".join(embedders) + " |\n",
              "|-----------|" + "----|" * len(embedders) + "\n"]
    for dt in thresholds:
        cells = []
        for e in embedders:
            v = results[e].get(dt, {}).get("unk_rate")
            cells.append(f"{v:.1%}" if v is not None else "-")
        lines.append(f"| {dt:.2f} | " + " | ".join(cells) + " |\n")

    # Best (lowest intra_cos) threshold per embedder
    lines += ["\n## Best threshold per embedder (lowest `intra_cos_mean` in this grid)\n",
              "| embedder | best threshold | intra_cos_mean | vocab size |\n",
              "|----------|-----------------|-----------------|------------|\n"]
    for e in embedders:
        scored = [(dt, s["intra_cos_mean"]) for dt, s in results[e].items()
                  if s.get("intra_cos_mean") is not None]
        if not scored:
            lines.append(f"| {e} | - | - | - |\n")
            continue
        best_dt, best_v = min(scored, key=lambda x: x[1])
        vs = results[e][best_dt]["real_vocab"]
        lines.append(f"| {e} | {best_dt:.2f} | {best_v:.3f} | {vs} |\n")

    lines += [
        "\n## How to read this\n",
        f"- Rows marked ⚠️ are at/above the {DIVERSITY_TARGET} collapse ceiling — cues assigned "
        "to the same song are, on average, near-duplicates in embedding space.\n",
        "- If no threshold in this grid gets a non-minilm embedder's `intra_cos_mean` down near "
        "minilm's, that embedder's cosine distribution is compressed enough that `dedup-threshold` "
        "alone can't fix it — narrow the grid toward higher thresholds (e.g. 0.97-0.999) and re-run, "
        "or treat that embedder as unsuitable for this pipeline's MMR-based diversity mechanism.\n",
        "- Vocabulary size and UNK rate are shown so a threshold that \"fixes\" diversity by "
        "starving the vocab (real vocab collapsing toward 0, most slots `<unk>`) isn't mistaken "
        "for a win — check both tables together.\n",
        "- `--keep-artifacts` was " + ("on" if args.keep_artifacts else "off") +
        " for this run; per-point `cue_vocab.json`/`item2cues.json` " +
        ("were kept under each grid point's subdirectory." if args.keep_artifacts
         else "were discarded after scoring — re-run with `--keep-artifacts` to inspect actual "
              "cues chosen at a specific threshold.") + "\n",
    ]
    cue_io.atomic_write_text(path, "".join(lines))
    print(f"\n[sweep] report -> {path}")


if __name__ == "__main__":
    main()
