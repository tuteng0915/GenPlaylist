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
    outputs/production/<timestamp>/  {cue_vocab.json, item2cues.json, cue_manifest.json,
                                       run_config.json,
                                       health_report.md (unless --skip-health-check)}
    outputs/production/latest/       the same files, overwritten each run
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

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
import cue_assign               # noqa: E402
from schema import CUE_CANDIDATES_PER_ITEM, CUE_TOKENS   # noqa: E402

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
                         "normally CUE_VOCAB_SIZE=2048 — the WP-C schema contract). A different "
                         "value still runs and exports fine, but CueMappingEntry.validate()/"
                         "load_mapping() must be called with a matching vocab_size to read the "
                         "resulting cue_vocab.json/item2cues.json back correctly.")
    ap.add_argument(
        "--fixed-vocab", default=None,
        help="reuse an existing cue_vocab.json and regenerate only the ranked "
             "per-song table; extraction and vocabulary rebuilding are skipped")
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
                  f"vocab_size={args.vocab_size}) to validate/read back, and WP-C expects "
                  f"exactly {cue_config.CUE_VOCAB_SIZE}.")

    print(f"[production] preset={args.config} -> {cfg}")
    if cfg.num_cues not in {CUE_TOKENS, CUE_CANDIDATES_PER_ITEM}:
        print(f"[production] WARNING: preset '{args.config}' is an experiment with "
              f"{cfg.num_cues} stored cues/item rather than the frozen master width "
              f"{CUE_CANDIDATES_PER_ITEM}; cue_manifest.json will mark it as "
              "wp_d_compatible=false.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUT_ROOT, stamp)
    latest_dir = os.path.join(OUT_ROOT, "latest")

    items, lyrics_raw = data_loading.load_catalog_and_lyrics(cfg.limit)
    if items and not lyrics_raw:
        raise RuntimeError(
            "No lyrics were found for the production catalog. Set CUE_LYRICS_DIR "
            "to the shared lyrics/spotify directory instead of silently building "
            "a metadata-only cue table.")
    os.makedirs(run_dir, exist_ok=False)
    lyrics_proc = data_loading.build_lyrics_proc(lyrics_raw, cfg.lyrics_mode, cfg.lyrics_cap)
    block_tokens = cue_normalize.build_block_tokens(items)
    cache_tag = cue_lyrics.cache_tag(cfg.lyrics_mode, cfg.lyrics_cap)
    print(f"[production] {len(items)} songs, {len(lyrics_raw)} with lyrics, "
          f"{len(block_tokens)} blocked artist tokens")

    # min_df is corpus-relative (see config.ProductionConfig.min_df_frac), resolved
    # here against the corpus this run actually builds from (respects --limit).
    resolved_min_df = cue_normalize.resolve_min_df(cfg.min_df_frac, len(items))
    print(f"[production] min_df: frac={cfg.min_df_frac} x {len(items)} songs "
          f"-> min_df={resolved_min_df}")

    if args.fixed_vocab:
        fixed_vocab_path = str(Path(args.fixed_vocab).expanduser().resolve())
        vocab = cue_export.load_vocab(fixed_vocab_path)
        embed_fn, backend = cue_normalize._make_embedder(embedder=cfg.embedder)
        cue_emb = embed_fn(vocab[1:])
        for row, cue in enumerate(vocab[1:]):
            if cue.startswith("<pad_"):
                cue_emb[row] = 0.0
        item2cues, score_mapping = cue_assign.assign_all(
            items, lyrics_proc, vocab, cue_emb, n_cues=cfg.num_cues,
            candidate_k=cfg.candidate_k, strategy=cfg.assignment_strategy,
            embed_fn=embed_fn, embedder=cfg.embedder, return_scores=True)
        cue_export.export_outputs(
            vocab, item2cues, run_dir, score_mapping=score_mapping,
            assignment_metadata={
                "strategy": cfg.assignment_strategy,
                "score": "cosine_similarity",
                "ordering": "relevance_desc_then_cue_id_asc",
                "candidate_k": cfg.candidate_k,
                "embedder": cfg.embedder,
                "embedder_backend": backend,
                "fallback": "expand_to_n_then_unk_tail",
                "song_text_fields": [
                    "title", "genre", "mood", "lyric_excerpt", "tags", "lyrics"],
                "lyrics_mode": cfg.lyrics_mode,
                "lyrics_cap": cfg.lyrics_cap,
                "fixed_vocab_source": fixed_vocab_path,
                "fixed_vocab_source_sha256": hashlib.sha256(
                    Path(fixed_vocab_path).read_bytes()).hexdigest(),
            })
        norm_stats = {
            "fixed_vocab": True,
            "selected": sum(not cue.startswith("<pad_") for cue in vocab[1:]),
            "final_vocab": len(vocab),
        }
    else:
        vocab, item2cues, norm_stats, cue_emb = pipeline.build_vocab_and_assign(
            cfg.method, items, lyrics_proc, run_dir,
            force=cfg.force, top_n=cfg.top_n, cache_tag=cache_tag,
            block_tokens=block_tokens, min_df=resolved_min_df,
            max_df_frac=cfg.max_df_frac, dedup_threshold=cfg.dedup_threshold,
            rank_by=cfg.rank_by, num_cues=cfg.num_cues,
            vocab_size=cfg.vocab_size, embedder=cfg.embedder,
            assignment_strategy=cfg.assignment_strategy,
            candidate_k=cfg.candidate_k, export_scores=True)

    run_config = {"preset": args.config, "generated": stamp, **asdict(cfg),
                  "min_df": resolved_min_df,   # resolved from min_df_frac x n_items above;
                                               # unused (vocab reused as-is) if fixed_vocab is set
                  "fixed_vocab": str(Path(args.fixed_vocab).expanduser().resolve())
                  if args.fixed_vocab else None,
                  "catalog_path": str(data_loading.CATALOG_PATH.resolve()),
                  "lyrics_dir": str(data_loading.LYRICS_DIR.resolve()),
                  "n_items": len(items), "n_with_lyrics": len(lyrics_raw)}
    cue_io.atomic_write_json(os.path.join(run_dir, "run_config.json"), run_config)

    if not args.skip_health_check:
        # Cheap, API-free sanity stats only — no retrieval/grounding/reconstruction.
        active_cov = cue_export.compute_coverage_stats(
            item2cues, vocab, cue_limit=cfg.active_cues)
        stored_cov = cue_export.compute_coverage_stats(item2cues, vocab)
        active_div = cue_eval.within_item_diversity(
            item2cues, cue_emb, cue_limit=cfg.active_cues)
        stored_div = cue_eval.within_item_diversity(item2cues, cue_emb)
        active_stats = {**active_cov, **active_div}
        stored_stats = {**stored_cov, **stored_div}
        cue_export.write_report(
            {**norm_stats, **stored_stats}, run_dir,
            active_stats=active_stats, stored_stats=stored_stats)
        os.replace(os.path.join(run_dir, "cue_report.md"),
                   os.path.join(run_dir, "health_report.md"))
        print(
            f"[production] active@{cfg.active_cues}: "
            f"coverage={active_cov.get('coverage_rate')} "
            f"unk_rate={active_cov.get('unk_rate')} "
            f"vocab_util={active_cov.get('vocab_coverage')} "
            f"intra_cos={active_div.get('intra_cos_mean')}")
        print(
            f"[production] stored@{cfg.num_cues}: "
            f"coverage={stored_cov.get('coverage_rate')} "
            f"unk_rate={stored_cov.get('unk_rate')} "
            f"vocab_util={stored_cov.get('vocab_coverage')} "
            f"intra_cos={stored_div.get('intra_cos_mean')}")

    # "latest" mirrors the most recent run_dir so callers don't need to know the stamp.
    if os.path.isdir(latest_dir):
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)

    print(f"[production] done -> {run_dir}")
    print(f"[production] latest -> {latest_dir}")


if __name__ == "__main__":
    main()
