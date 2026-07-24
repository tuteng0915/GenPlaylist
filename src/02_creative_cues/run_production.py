"""run_production.py — build the production cue vocabulary + mapping. No evaluation.

extract -> clean -> assign -> export ONE method's cue_vocab.json + item2cues.json,
using a named, fixed preset from config.py instead of a long CLI flag list.
Held-out splits, retrieval, grounding, and reconstruction (Level 1-3) are
evaluation-only concerns and never run here — for those, use run_compare.py.

Usage:
    .venv/Scripts/python.exe src/02_creative_cues/run_production.py
    .venv/Scripts/python.exe src/02_creative_cues/run_production.py --config tfidf
    .venv/Scripts/python.exe src/02_creative_cues/run_production.py --limit 500   # smoke test
    .venv/Scripts/python.exe src/02_creative_cues/run_production.py --force       # bypass caches

Output:
    outputs/production/<timestamp>/  {cue_vocab.json, item2cues.json, run_config.json,
                                       health_report.md (unless --skip-health-check)}
    outputs/production/latest/       the same files, overwritten each run
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import asdict
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cue_config     # noqa: E402
import data_loading             # noqa: E402
import pipeline                 # noqa: E402
import cue_normalize            # noqa: E402
import cue_lyrics               # noqa: E402
import cue_eval                 # noqa: E402
import cue_export               # noqa: E402
import cue_io                   # noqa: E402

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "production")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="default",
                    help=f"named preset from config.py (available: {', '.join(sorted(cue_config.PRESETS))})")
    ap.add_argument("--limit", type=int, default=None,
                    help="override the preset's song count (e.g. for a smoke test)")
    ap.add_argument("--force", action="store_true",
                    help="override the preset's cache setting; re-extract/re-clean from scratch")
    ap.add_argument("--vocab-size", type=int, default=None,
                    help="override the preset's vocab_size (default from config.py preset, "
                         "normally CUE_VOCAB_SIZE=2048 — the WP-D schema contract). A different "
                         "value still runs and exports fine, but CueMappingEntry.validate()/"
                         "load_mapping() must be called with a matching vocab_size to read the "
                         "resulting cue_vocab.json/item2cues.json back correctly.")
    ap.add_argument("--skip-health-check", action="store_true",
                    help="skip the free coverage/diversity sanity stats (on by default; no LLM calls)")
    args = ap.parse_args()

    cfg = cue_config.get_preset(args.config)
    if args.limit is not None:
        cfg = cue_config.replace(cfg, limit=args.limit)
    if args.force:
        cfg = cue_config.replace(cfg, force=True)
    if args.vocab_size is not None:
        cfg = cue_config.replace(cfg, vocab_size=args.vocab_size)
        if args.vocab_size != cue_config.CUE_VOCAB_SIZE:
            print(f"[production] WARNING: --vocab-size={args.vocab_size} overrides the "
                  f"CUE_VOCAB_SIZE={cue_config.CUE_VOCAB_SIZE} schema contract; the resulting "
                  f"cue_vocab.json/item2cues.json need load_mapping(path, "
                  f"vocab_size={args.vocab_size}) to validate/read back, and WP-D expects "
                  f"exactly {cue_config.CUE_VOCAB_SIZE}.")

    print(f"[production] preset={args.config} -> {cfg}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUT_ROOT, stamp)
    latest_dir = os.path.join(OUT_ROOT, "latest")
    os.makedirs(run_dir, exist_ok=True)

    items, lyrics_raw = data_loading.load_catalog_and_lyrics(cfg.limit)
    lyrics_proc = data_loading.build_lyrics_proc(lyrics_raw, cfg.lyrics_mode, cfg.lyrics_cap)
    block_tokens = cue_normalize.build_block_tokens(items)
    cache_tag = cue_lyrics.cache_tag(cfg.lyrics_mode, cfg.lyrics_cap)
    print(f"[production] {len(items)} songs, {len(lyrics_raw)} with lyrics, "
          f"{len(block_tokens)} blocked artist tokens")

    vocab, item2cues, norm_stats, cue_emb = pipeline.build_vocab_and_assign(
        cfg.method, items, lyrics_proc, run_dir,
        force=cfg.force, top_n=cfg.top_n, cache_tag=cache_tag, block_tokens=block_tokens,
        min_df=cfg.min_df, max_df_frac=cfg.max_df_frac, dedup_threshold=cfg.dedup_threshold,
        rank_by=cfg.rank_by, num_cues=cfg.num_cues, vocab_size=cfg.vocab_size,
    )

    run_config = {"preset": args.config, "generated": stamp, **asdict(cfg),
                  "n_items": len(items), "n_with_lyrics": len(lyrics_raw)}
    cue_io.atomic_write_json(os.path.join(run_dir, "run_config.json"), run_config)

    if not args.skip_health_check:
        # Cheap, API-free sanity stats only — no retrieval/grounding/reconstruction.
        cov = cue_export.compute_coverage_stats(item2cues, vocab)
        div = cue_eval.within_item_diversity(item2cues, cue_emb)
        cue_export.write_report({**norm_stats, **cov, **div}, run_dir)
        os.replace(os.path.join(run_dir, "cue_report.md"),
                   os.path.join(run_dir, "health_report.md"))
        print(f"[production] health check: coverage={cov.get('coverage_rate')} "
              f"unk_rate={cov.get('unk_rate')} vocab_util={cov.get('vocab_coverage')} "
              f"intra_cos={div.get('intra_cos_mean')}")

    # "latest" mirrors the most recent run_dir so callers don't need to know the stamp.
    if os.path.isdir(latest_dir):
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)

    print(f"[production] done -> {run_dir}")
    print(f"[production] latest -> {latest_dir}")


if __name__ == "__main__":
    main()
