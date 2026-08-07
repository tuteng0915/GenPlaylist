"""sweep_min_df.py — min_df ablation at a fixed corpus size.

Same corpus (default the full 5,000-song catalog), same everything else — only
`min_df` (the df-band floor) varies. Reuses run_compare.py's own pipeline calls
and report writer directly (run_method / evaluate_method / run_level3_conditions /
write_report), so the Level 2 retrieval, Level 3 reconstruction, and LLM judge
numbers are produced by the exact same code path as your run_compare.py reports
— this script only adds a loop over `min_df` instead of over `methods`.

min_df values are given as FRACTIONS of `--limit` (not raw counts), e.g. `0.001`
means `min_df = round(0.001 * limit)`. This keeps the grid meaningful if you change
`--limit` without having to hand-recompute absolute counts. Resolved min_df is
never allowed below 2, regardless of how small a fraction produces. The report
labels each arm by the fraction you passed in (not a converted percentage).

Usage:
  .venv/Scripts/python.exe src/02_creative_cues/sweeps/sweep_min_df.py \
      --methods tfidf --limit 5000 --min-df-fracs 0.0004,0.001,0.002,0.005 \
      --top-n 100 --rank-by idf --lyrics-mode dedup --lyrics-cap 2000 \
      --num-cues 18 --vocab-size 2048 --held-out-eval --test-frac 0.15 --split-seed 42
"""

import argparse
import os
import sys
import uuid
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "..", "00_data_schema"))

import run_compare       # noqa: E402
import cue_normalize     # noqa: E402
import cue_lyrics        # noqa: E402
import cue_eval          # noqa: E402
import cue_clients       # noqa: E402
import cue_io            # noqa: E402
from schema import CUE_TOKENS, CUE_VOCAB_SIZE   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5000, help="fixed corpus size (same for every arm)")
    ap.add_argument("--methods", default="tfidf", help="comma-separated extraction methods")
    ap.add_argument("--min-df-fracs", default="0.0004,0.001,0.002,0.005",
                    help="comma-separated min_df values as FRACTIONS of --limit "
                         "(e.g. 0.001 -> min_df = round(0.001 * limit)); resolved min_df is "
                         "floored at 2 regardless of how small a fraction requests")
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--dedup-threshold", type=float, default=0.92)
    ap.add_argument("--rank-by", default="idf", choices=list(cue_normalize.RANK_METHODS))
    ap.add_argument("--lyrics-mode", default="dedup", choices=list(cue_lyrics.MODES))
    ap.add_argument("--lyrics-cap", type=int, default=cue_lyrics.DEFAULT_CAP)
    ap.add_argument("--score-chars", type=int, default=2000)
    ap.add_argument("--eval-sample", type=int, default=150)
    ap.add_argument("--num-cues", type=int, default=CUE_TOKENS)
    ap.add_argument("--vocab-size", type=int, default=CUE_VOCAB_SIZE)
    ap.add_argument("--embedder", default=cue_normalize.DEFAULT_EMBEDDER)
    ap.add_argument("--level3", action=argparse.BooleanOptionalAction, default=True,
                    help="run the LLM reconstruction ablation (on by default)")
    ap.add_argument("--llm-judge", action=argparse.BooleanOptionalAction, default=True,
                    help="run the LLM-judge grounding score (on by default)")
    ap.add_argument("--held-out-eval", action="store_true")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    min_df_fracs = [float(x) for x in args.min_df_fracs.split(",") if x.strip()]
    # resolved against the FIXED --limit corpus size; never below 2 regardless of frac*limit.
    resolved_min_dfs = [(frac, max(2, round(frac * args.limit))) for frac in min_df_fracs]

    run_id = "sweep_min_df_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
    sweep_dir = os.path.join(run_compare.OUT_ROOT, "runs", run_id)
    os.makedirs(sweep_dir, exist_ok=True)
    cue_io.atomic_write_json(os.path.join(sweep_dir, "run_config.json"),
                             {"run_id": run_id, "resolved_min_dfs": resolved_min_dfs, **vars(args)})
    print(f"[sweep] run folder: {sweep_dir}")
    print(f"[sweep] methods={methods} limit={args.limit} "
          f"min_df_fracs->resolved={[(f, md) for f, md in resolved_min_dfs]}")

    cue_eval.set_max_target_lines(0 if args.lyrics_mode == "full" else cue_eval.DEFAULT_MAX_TARGET_LINES)
    cue_eval.set_score_chars(args.score_chars)

    items, lyrics_raw = run_compare.load_data(args.limit)
    catalog_by_id = {it.item_id: it for it in items}
    block_tokens = cue_normalize.build_block_tokens(items)
    cache_tag = cue_lyrics.cache_tag(args.lyrics_mode, args.lyrics_cap)
    lyrics_proc = run_compare.data_loading.build_lyrics_proc(lyrics_raw, args.lyrics_mode, args.lyrics_cap)
    print(f"[sweep] {len(items)} songs, {len(lyrics_raw)} with lyrics")

    train_items = test_items = test_id_set = None
    if args.held_out_eval:
        train_items, test_items = cue_normalize.split_items(
            items, test_frac=args.test_frac, seed=args.split_seed)
        test_id_set = {it.item_id for it in test_items}
        print(f"[sweep] held-out eval: {len(train_items)} train / {len(test_items)} test items")

    eval_pool = test_id_set if test_id_set is not None else set(catalog_by_id)
    sample_ids = [iid for iid in catalog_by_id
                 if iid in eval_pool and lyrics_raw.get(iid, "").strip()][:args.eval_sample]

    def _eval_view(item2cues):
        if test_id_set is None:
            return item2cues
        return {iid: e for iid, e in item2cues.items() if iid in test_id_set}

    run_judge = args.llm_judge
    if run_judge and not cue_clients.is_available():
        print("[sweep] LLM judge requested but no API key; skipping.")
        run_judge = False
    run_level3 = args.level3
    if run_level3 and not cue_clients.is_available():
        print("[sweep] Level 3 reconstruction requested but no API key; skipping.")
        run_level3 = False

    rows, vocabs, mappings = {}, {}, {}
    for method in methods:
        for frac, md in resolved_min_dfs:
            label = (f"{method}(min_df={md}, frac={frac})" if len(methods) > 1
                    else f"min_df={md} (frac={frac})")
            arm_dirname = f"{method}_md{md}_frac{frac}".replace(".", "_")
            run_compare.RUN_DIR = os.path.join(sweep_dir, "arms", arm_dirname)
            print(f"\n########## {label} ##########")
            vocab, item2cues, nstats, cue_emb = run_compare.run_method(
                method, items, lyrics_proc, md, args.force, block_tokens, args.top_n, cache_tag,
                dedup_threshold=args.dedup_threshold, rank_by=args.rank_by,
                vocab_items=train_items, num_cues=args.num_cues,
                vocab_size=args.vocab_size, embedder=args.embedder)
            row = run_compare.evaluate_method(
                label, vocab, _eval_view(item2cues), nstats, cue_emb, catalog_by_id,
                lyrics_raw, lyrics_proc, sample_ids, run_judge=run_judge)
            rows[label] = row
            vocabs[label] = vocab
            mappings[label] = item2cues

    floor = oracle = oracle_cues = metadata_floor = None
    if run_level3:
        floor, method_l3, oracle, oracle_cues, metadata_floor = run_compare.run_level3_conditions(
            rows, vocabs, mappings, catalog_by_id, lyrics_raw, sample_ids,
            use_batch=False, num_cues=args.num_cues, poll_s=30, timeout_s=86400)
        for label, res in method_l3.items():
            rows[label]["level3"] = res

    split_info = None
    if args.held_out_eval:
        split_info = {"n_train": len(train_items), "n_test": len(test_items),
                      "test_frac": args.test_frac, "seed": args.split_seed}

    # write_report reads a few args fields this script's own CLI doesn't define
    # (args.min_df -- there's no single value, that's the whole point of this sweep;
    # args.level2 -- a deprecated run_compare-only alias). Patch them onto a copy.
    report_args = argparse.Namespace(**vars(args))
    report_args.min_df = "swept — see table"
    report_args.level2 = False

    run_compare.RUN_DIR = sweep_dir
    run_compare.write_report(rows, vocabs, mappings, sample_ids, catalog_by_id, floor, oracle,
                             report_args, run_id, split_info=split_info, oracle_cues=oracle_cues,
                             metadata_floor=metadata_floor)


if __name__ == "__main__":
    main()
