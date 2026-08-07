"""run_compare.py — WP-B Phase 5: end-to-end multi-method comparison + report.

For each extraction method:
    extract raw cues -> normalize vocab -> assign N cues/song -> export -> evaluate
    (N = --num-cues; production stores 16 and WP-C activates the first 8)

Then writes one comparison report:
    src/02_creative_cues/outputs/runs/<run_id>/comparison_report.md

This script does the FULL evaluation (grounding, retrieval, optionally
reconstruction) across multiple methods — for a single-method production
vocab build with no evaluation, use run_production.py instead. The two share
their extract/clean/assign/export logic via pipeline.py.

Usage (from project root, with venv):
    .venv/Scripts/python.exe src/02_creative_cues/run_compare.py --limit 1000 --methods tfidf,yake
    .venv/Scripts/python.exe src/02_creative_cues/run_compare.py --methods tfidf,yake,keybert,llm \
        --eval-sample 200 --level3
    .venv/Scripts/python.exe src/02_creative_cues/run_compare.py --methods tfidf,yake,keybert,llm \
        --llm-batch
    .venv/Scripts/python.exe src/02_creative_cues/run_compare.py --methods tfidf,yake \
        --held-out-eval --test-frac 0.15
"""

import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# This script lives alongside its sibling cue_*.py modules, so they're already
# importable (Python puts the running script's own directory on sys.path[0]).
# Only 00_data_schema, one level up from the repo root's src/, needs adding.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_data_schema"))

import uuid                     # noqa: E402
import cue_extractors          # noqa: E402
import cue_normalize           # noqa: E402
import cue_assign               # noqa: E402
import cue_eval                 # noqa: E402
import cue_clients              # noqa: E402
import cue_lyrics               # noqa: E402
import cue_export               # noqa: E402
import cue_io                   # noqa: E402
import data_loading             # noqa: E402
import pipeline                 # noqa: E402
from schema import CUE_TOKENS, CUE_VOCAB_SIZE   # noqa: E402
from datetime import datetime   # noqa: E402

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
# Per-run output folder (set in main). Shared caches stay under OUT_ROOT; every
# run's own artifacts (reports + per-method vocab/item2cues) go under RUN_DIR so
# concurrent runs never collide.
RUN_DIR = OUT_ROOT


def load_data(limit):
    return data_loading.load_catalog_and_lyrics(limit)


def finish_method_from_raw(method, raw, items, lyrics_proc, min_df, block_tokens,
                           dedup_threshold=0.92, rank_by="idf", num_cues=CUE_TOKENS,
                           vocab_size=CUE_VOCAB_SIZE, embedder=cue_normalize.DEFAULT_EMBEDDER):
    """normalize -> assign -> export, via pipeline.py. See pipeline.finish_from_raw."""
    out_dir = os.path.join(RUN_DIR, "methods", method)
    return pipeline.finish_from_raw(method, raw, items, lyrics_proc, out_dir,
                                    min_df=min_df, dedup_threshold=dedup_threshold,
                                    rank_by=rank_by, num_cues=num_cues,
                                    vocab_size=vocab_size, embedder=embedder,
                                    block_tokens=block_tokens)


def run_method(method, items, lyrics_proc, min_df, force, block_tokens, top_n, cache_tag,
               dedup_threshold=0.92, rank_by="idf", vocab_items=None, num_cues=CUE_TOKENS,
               vocab_size=CUE_VOCAB_SIZE, embedder=cue_normalize.DEFAULT_EMBEDDER):
    """extract -> normalize -> assign -> export, via pipeline.py. See pipeline.build_vocab_and_assign.

    vocab_items: corpus used for vocabulary building (raw cue extraction + df/idf/dedup).
    Defaults to `items`. Pass a train-only split for --held-out-eval so the vocabulary
    never sees held-out test songs; cues are still assigned to every item in `items`
    (item2cues.json stays the full deliverable either way).
    """
    out_dir = os.path.join(RUN_DIR, "methods", method)
    return pipeline.build_vocab_and_assign(
        method, items, lyrics_proc, out_dir,
        force=force, top_n=top_n, cache_tag=cache_tag, vocab_items=vocab_items,
        min_df=min_df, dedup_threshold=dedup_threshold, rank_by=rank_by,
        num_cues=num_cues, vocab_size=vocab_size, embedder=embedder, block_tokens=block_tokens)


def evaluate_method(method, vocab, item2cues, nstats, cue_emb, catalog_by_id, lyrics_ref,
                    lyrics_proc, sample_ids, run_judge=False):
    """Coverage + within-item diversity + Level 1 grounding + Level 2 retrieval (+ LLM judge)."""
    cov = cue_export.compute_coverage_stats(item2cues, vocab)
    # Within-item cue diversity (paper target < 0.7), using the assignment embeddings.
    div = cue_eval.within_item_diversity(item2cues, cue_emb)
    # Level 1 grounding: cues vs the REAL (unprocessed) lyrics.
    l1 = cue_eval.level1_intrinsic(item2cues, vocab, lyrics_ref)
    # Level 2 retrieval: same processed song representation used at assignment time.
    l2 = cue_eval.level2_semantic_retrieval(
        catalog_by_id, item2cues, vocab, lyrics_proc, sample_ids)
    row = {**nstats, **cov, **div, **l1, **l2}
    # LLM judge: cues vs the REAL lyrics, scored 1-5 on the eval sample.
    if run_judge:
        row.update(cue_eval.llm_judge_grounding(item2cues, vocab, lyrics_ref, sample_ids))
    return row


def run_level3_conditions(rows, vocabs, mappings, catalog_by_id, lyrics_ref,
                          sample_ids, use_batch, poll_s, timeout_s, num_cues=CUE_TOKENS):
    """Bracketed Level 3 ablation: metadata-only floor < random floor < methods < oracle ceiling.

    Returns (floor_metrics, method_metrics_by_name, oracle_metrics, oracle_cues_by_id,
    metadata_floor_metrics). All conditions share the decoder (title/artist withheld)
    and the same cue budget (num_cues); only the cue content differs. oracle_cues_by_id
    is returned (not just its aggregate score) so callers can display the oracle's
    actual cue strings alongside each method's assigned cues.

    metadata_floor uses ZERO cues (only genre/mood reach the decoder) — a purer floor
    than `random`, which still gives the decoder 6 real (just irrelevant) cues. The gap
    between metadata_floor and random isolates the value of having ANY cues at all,
    separate from having the RIGHT cues.
    """
    first_vocab = next(iter(vocabs.values()))
    conds = {
        "random": cue_eval.random_cues_by_id(first_vocab, sample_ids, n_cues=num_cues),
        "metadata_only": cue_eval.no_cues_by_id(sample_ids),
        "oracle": cue_eval.oracle_cues_by_id(catalog_by_id, lyrics_ref, sample_ids, n_cues=num_cues),
    }
    for m in rows:
        conds[m] = cue_eval.assigned_cues_by_id(mappings[m], vocabs[m], sample_ids)

    results: dict[str, dict] = {}
    if use_batch:
        print("[compare] submitting Level 3 batches (random, metadata-only, methods, oracle)...")
        jobs = {tag: cue_eval.submit_reconstruction_batch(
                    catalog_by_id, cues, lyrics_ref, sample_ids, tag)
                for tag, cues in conds.items()}
        print("[compare] collecting Level 3 batches...")
        for tag, job in jobs.items():
            results[tag] = cue_eval.collect_reconstruction_batch(job, poll_s, timeout_s)
    else:
        for tag, cues in conds.items():
            print(f"[compare] Level 3 reconstruction ({tag}) on {len(sample_ids)} songs...")
            results[tag] = cue_eval.level3_reconstruction(
                catalog_by_id, cues, lyrics_ref, sample_ids)

    floor = results.pop("random")
    metadata_floor = results.pop("metadata_only")
    oracle = results.pop("oracle")
    oracle_cues = conds["oracle"]
    return floor, results, oracle, oracle_cues, metadata_floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--methods", default="tfidf,yake")
    ap.add_argument("--top-n", type=int, default=40,
                    help="raw candidate cues kept per song before cleaning")
    ap.add_argument("--min-df", type=int, default=5)
    ap.add_argument("--dedup-threshold", type=float, default=0.92,
                    help="cosine above which near-duplicate cues are merged in cleaning "
                         "(higher = fewer merges = larger vocabulary)")
    ap.add_argument("--no-semantic-dedup", action="store_true",
                    help="skip the semantic dedup step entirely (keep all cues)")
    ap.add_argument("--rank-by", default="idf", choices=list(cue_normalize.RANK_METHODS),
                    help="stage-5 vocabulary selection rule")
    ap.add_argument("--lyrics-mode", default="cap", choices=list(cue_lyrics.MODES),
                    help="how lyrics feed extraction/assignment: cap | full | dedup | summarize")
    ap.add_argument("--lyrics-cap", type=int, default=cue_lyrics.DEFAULT_CAP,
                    help="char cap for cap/dedup modes")
    ap.add_argument("--score-chars", type=int, default=2000,
                    help="common window (chars) applied to both gen and real lyrics "
                         "before Level 3 metrics, so lengths are comparable (0 = full)")
    ap.add_argument("--eval-sample", type=int, default=150)
    ap.add_argument("--level3", action=argparse.BooleanOptionalAction, default=True,
                    help="run the LLM reconstruction ablation (costs API calls when "
                         "OPENAI_API_KEY is set; skipped automatically otherwise). "
                         "On by default — use --no-level3 to skip.")
    ap.add_argument("--level2", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--llm-judge", action=argparse.BooleanOptionalAction, default=True,
                    help="run an LLM-judge grounding score: for each eval-sample song, an LLM "
                         "rates 1-5 how well the assigned cues match the real lyrics (costs API "
                         "calls when OPENAI_API_KEY is set; skipped automatically otherwise). "
                         "On by default — use --no-llm-judge to skip.")
    ap.add_argument("--llm-batch", action="store_true",
                    help="submit LLM extraction through the OpenAI Batch API")
    ap.add_argument("--recon-batch", action="store_true",
                    help="run Level 3 reconstruction through the OpenAI Batch API")
    ap.add_argument("--recon-report-samples", type=int, default=15,
                    help="songs shown in the original-vs-regenerated lyrics report")
    ap.add_argument("--recon-report-lyric-chars", type=int, default=0,
                    help="chars of each lyric shown in the reconstruction report (0 = full)")
    ap.add_argument("--llm-batch-poll-seconds", type=int, default=30,
                    help="seconds between OpenAI Batch status checks")
    ap.add_argument("--llm-batch-timeout-seconds", type=int, default=24 * 60 * 60,
                    help="maximum seconds to wait for the OpenAI Batch job")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--note", default="",
                    help="free-text note to self, stamped at the top of comparison_report.md "
                         "(e.g. what this run is testing, caveats to remember)")
    ap.add_argument("--held-out-eval", action="store_true",
                    help="build the vocabulary from a train-only item split and score every "
                         "metric on a disjoint held-out test split, so results reflect "
                         "generalization instead of in-sample fit. Song-level split (NOT the "
                         "playlist-level data/dataset/splits/ files, which have ~100%% item "
                         "overlap between their train/test — see cue_normalize.split_items). "
                         "Default off: unset, behavior is unchanged from before.")
    ap.add_argument("--test-frac", type=float, default=0.15,
                    help="fraction of songs held out for --held-out-eval")
    ap.add_argument("--split-seed", type=int, default=42,
                    help="seed for the --held-out-eval item split")
    ap.add_argument("--num-cues", type=int, default=CUE_TOKENS,
                    help=f"cues assigned per song (default {CUE_TOKENS}, the CUE_TOKENS schema "
                         "contract WP-C expects). A different value still runs, validates, and "
                         "exports item2cues.json fine, but CueMappingEntry.load_mapping() must "
                         "be called with a matching n_cues to read a non-default file back.")
    ap.add_argument("--vocab-size", type=int, default=CUE_VOCAB_SIZE,
                    help=f"total vocab entries incl. <unk> (default {CUE_VOCAB_SIZE}, the "
                         "CUE_VOCAB_SIZE schema contract WP-C expects). A different value still "
                         "runs, validates, and exports fine, but CueMappingEntry.validate()/"
                         "load_mapping() must be called with a matching vocab_size to read a "
                         "non-default file back without falsely rejecting in-range cue IDs.")
    ap.add_argument("--embedder", default=cue_normalize.DEFAULT_EMBEDDER,
                    help=f"sentence-transformers model for semantic dedup + assignment relevance "
                         f"(default '{cue_normalize.DEFAULT_EMBEDDER}'). One of "
                         f"{sorted(cue_normalize.EMBEDDER_MODELS)}, or any raw HF model id. Does "
                         "NOT affect evaluation encoders (Level 2 retrieval and Level 3 STS-cosine "
                         "deliberately use a different, independent encoder either way).")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    # threshold >= 1.0 makes build_vocab_normalized skip semantic dedup entirely
    eff_dedup = 1.0 if args.no_semantic_dedup else args.dedup_threshold

    # Per-run output folder: datetime + short random id (unique even for parallel
    # same-second runs). Shared caches stay under OUT_ROOT; this run's artifacts go here.
    global RUN_DIR
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
    RUN_DIR = os.path.join(OUT_ROOT, "runs", run_id)
    os.makedirs(RUN_DIR, exist_ok=True)
    cue_io.atomic_write_json(os.path.join(RUN_DIR, "run_config.json"),
                             {"run_id": run_id, **vars(args)})
    print(f"[compare] run folder: {RUN_DIR}")

    # In full mode, don't clamp the decode target length — match the real song.
    cue_eval.set_max_target_lines(0 if args.lyrics_mode == "full"
                                  else cue_eval.DEFAULT_MAX_TARGET_LINES)
    # Score generated vs real lyrics over the same character window (consistency).
    cue_eval.set_score_chars(args.score_chars)
    items, lyrics_raw = load_data(args.limit)     # ORIGINAL lyrics (Level 1/3 references)
    catalog_by_id = {it.item_id: it for it in items}
    block_tokens = cue_normalize.build_block_tokens(items)

    # preprocessed lyrics fed into extraction / assignment / retrieval
    cache_tag = cue_lyrics.cache_tag(args.lyrics_mode, args.lyrics_cap)
    print(f"[compare] lyrics-mode={args.lyrics_mode} (cap={args.lyrics_cap}) tag={cache_tag}")
    lyrics_proc = data_loading.build_lyrics_proc(lyrics_raw, args.lyrics_mode, args.lyrics_cap)
    print(f"[compare] {len(items)} songs, {len(lyrics_raw)} with lyrics, methods={methods}")
    print(f"[compare] {len(block_tokens)} blocked artist tokens")
    if args.num_cues != CUE_TOKENS:
        print(f"[compare] WARNING: --num-cues={args.num_cues} overrides the CUE_TOKENS="
              f"{CUE_TOKENS} schema contract; item2cues.json from this run needs "
              f"load_mapping(path, n_cues={args.num_cues}) to read back, and WP-C expects "
              f"exactly {CUE_TOKENS}.")
    if args.vocab_size != CUE_VOCAB_SIZE:
        print(f"[compare] WARNING: --vocab-size={args.vocab_size} overrides the CUE_VOCAB_SIZE="
              f"{CUE_VOCAB_SIZE} schema contract; cue_vocab.json/item2cues.json from this run "
              f"need load_mapping(path, vocab_size={args.vocab_size}) to validate/read back, and "
              f"WP-C expects exactly {CUE_VOCAB_SIZE}.")

    # --held-out-eval: vocabulary is built from train_items only; evaluation (below) is
    # restricted to test_items so every reported metric reflects generalization to songs
    # the vocab never saw. vocab_items stays None (= no split) otherwise.
    train_items = test_items = None
    test_id_set = None
    if args.held_out_eval:
        train_items, test_items = cue_normalize.split_items(
            items, test_frac=args.test_frac, seed=args.split_seed)
        test_id_set = {it.item_id for it in test_items}
        print(f"[compare] held-out eval: {len(train_items)} train / {len(test_items)} test "
              f"items (test_frac={args.test_frac}, seed={args.split_seed}); vocab built from "
              f"train only, all metrics scored on test only")

    # deterministic eval sample (songs that have real lyrics, restricted to the held-out
    # test split if enabled)
    eval_pool = test_id_set if test_id_set is not None else set(catalog_by_id)
    sample_ids = [iid for iid in catalog_by_id
                  if iid in eval_pool and lyrics_raw.get(iid, "").strip()][:args.eval_sample]

    def _eval_view(item2cues):
        """Restrict item2cues to the held-out test split for evaluation (no-op if disabled)."""
        if test_id_set is None:
            return item2cues
        return {iid: e for iid, e in item2cues.items() if iid in test_id_set}

    rows = {}
    vocabs = {}
    mappings = {}
    run_level3 = args.level3 or args.level2
    run_judge = args.llm_judge
    if run_judge and not cue_clients.is_available():
        print("[compare] LLM judge requested but no API key; skipping.")
        run_judge = False
    llm_batch_job = None
    run_methods = methods
    if args.llm_batch and "llm" in methods:
        print("[compare] submitting LLM extraction batch before other methods...")
        llm_batch_job = cue_extractors.submit_llm_batch(
            train_items if args.held_out_eval else items, lyrics_proc,
            top_n=args.top_n, force=args.force, cache_tag=cache_tag)
        run_methods = [m for m in methods if m != "llm"]

    for method in run_methods:
        vocab, item2cues, nstats, cue_emb = run_method(method, items, lyrics_proc, args.min_df,
                                                       args.force, block_tokens, args.top_n, cache_tag,
                                                       dedup_threshold=eff_dedup, rank_by=args.rank_by,
                                                       vocab_items=train_items, num_cues=args.num_cues,
                                                       vocab_size=args.vocab_size, embedder=args.embedder)
        row = evaluate_method(method, vocab, _eval_view(item2cues), nstats, cue_emb, catalog_by_id,
                              lyrics_raw, lyrics_proc, sample_ids, run_judge=run_judge)
        rows[method] = row
        vocabs[method] = vocab
        mappings[method] = item2cues

    if llm_batch_job is not None:
        print("[compare] collecting LLM batch results after other methods...")
        raw = cue_extractors.collect_llm_batch(
            llm_batch_job,
            poll_seconds=args.llm_batch_poll_seconds,
            timeout_seconds=args.llm_batch_timeout_seconds,
        )
        print("\n########## METHOD: llm ##########")
        vocab, item2cues, nstats, cue_emb = finish_method_from_raw(
            "llm", raw, items, lyrics_proc, args.min_df, block_tokens,
            dedup_threshold=eff_dedup, rank_by=args.rank_by, num_cues=args.num_cues,
            vocab_size=args.vocab_size, embedder=args.embedder)
        row = evaluate_method("llm", vocab, _eval_view(item2cues), nstats, cue_emb, catalog_by_id,
                              lyrics_raw, lyrics_proc, sample_ids, run_judge=run_judge)
        rows["llm"] = row
        vocabs["llm"] = vocab
        mappings["llm"] = item2cues

    # Level 3 bracketed ablation: random floor < methods < oracle ceiling.
    # References are the REAL (unprocessed) lyrics — lyrics_raw.
    floor = oracle = oracle_cues = metadata_floor = None
    if run_level3 and not cue_clients.is_available():
        print("[compare] reconstruction requested but no API key; skipping Level 3.")
    elif run_level3:
        floor, method_l3, oracle, oracle_cues, metadata_floor = run_level3_conditions(
            rows, vocabs, mappings, catalog_by_id, lyrics_raw, sample_ids,
            use_batch=args.recon_batch, num_cues=args.num_cues,
            poll_s=args.llm_batch_poll_seconds, timeout_s=args.llm_batch_timeout_seconds)
        for m, res in method_l3.items():
            rows[m]["level3"] = res

    split_info = None
    if args.held_out_eval:
        split_info = {"n_train": len(train_items), "n_test": len(test_items),
                      "test_frac": args.test_frac, "seed": args.split_seed}
    write_report(rows, vocabs, mappings, sample_ids, catalog_by_id, floor, oracle,
                 args, run_id, split_info=split_info, oracle_cues=oracle_cues,
                 metadata_floor=metadata_floor)

    # Separate side-by-side original-vs-regenerated lyrics report (Level 3 only)
    if run_level3 and cue_clients.is_available():
        write_reconstruction_report(rows, vocabs, mappings, sample_ids,
                                    catalog_by_id, lyrics_raw, args, run_id)


def write_reconstruction_report(rows, vocabs, mappings, sample_ids, catalog_by_id,
                                lyrics, args, stamp):
    """Side-by-side: original lyrics vs LLM-regenerated lyrics, per method."""
    n = args.recon_report_samples
    show_ids = sample_ids[:n]
    lyric_cap = args.recon_report_lyric_chars  # chars shown per lyric block (0 = full)

    lines = ["# WP-B — Reconstruction Samples (original vs regenerated)\n",
             f"Generated: {stamp} | lyrics-mode: {args.lyrics_mode} | "
             f"Sample: {len(show_ids)} songs | decoder: {cue_clients.CHAT_MODEL}\n",
             "\nEach song shows its real lyrics, then the lyrics an LLM regenerates "
             f"from ONLY the {args.num_cues} cues + minimal metadata, per method. "
             "Closer = the cues "
             "carried more of the song's content.\n"]

    for iid in show_ids:
        item = catalog_by_id[iid]
        real = lyrics.get(iid, "").strip()
        if not real:
            continue
        _clip = (lambda s: s if not lyric_cap else s[:lyric_cap])
        lines.append(f"\n---\n\n## {item.title} — {item.artist} _( {item.genre} )_\n")
        lines.append("\n**Original lyrics:**\n\n```\n" + _clip(real) + "\n```\n")
        for m in rows:
            v, mp = vocabs[m], mappings[m]
            cues_by_id = cue_eval.assigned_cues_by_id(mp, v, [iid])
            gens = cue_eval.generate_lyrics_for_sample(
                catalog_by_id, cues_by_id, lyrics, [iid])
            if not gens:
                continue
            _, cue_strings, gen = gens[0]
            lines.append(f"\n**`{m}` cues:** {', '.join(cue_strings) if cue_strings else '(none)'}\n")
            lines.append("\n```\n" + _clip(gen.strip()) + "\n```\n")

    path = os.path.join(RUN_DIR, "reconstruction_report.md")
    cue_io.atomic_write_text(path, "".join(lines))
    print(f"[compare] reconstruction samples -> {path}")


def _bold_best(rows, key, higher=True):
    """Return {method: formatted-cell} with the leading value bolded."""
    vals = {m: r.get(key) for m, r in rows.items()}
    nums = [v for v in vals.values() if isinstance(v, (int, float))]
    best = (max if higher else min)(nums) if nums else None
    out = {}
    for m, v in vals.items():
        s = "-" if v is None else str(v)
        if best is not None and v == best:
            s = f"**{s}**"
        out[m] = s
    return out


def _table(rows, headers, cols):
    """Assemble a markdown table. headers=[str]; cols=[(key, higher_better)] aligned after Method."""
    bolded = [_bold_best(rows, key, higher) for key, higher in cols]
    out = ["| Method | " + " | ".join(headers) + " |\n",
           "|--------|" + "---|" * len(headers) + "\n"]
    for m in rows:
        cells = " | ".join(b[m] for b in bolded)
        out.append(f"| {m} | {cells} |\n")
    return out


def _best_method(rows, key, higher=True):
    cand = {m: r.get(key) for m, r in rows.items() if isinstance(r.get(key), (int, float))}
    if not cand:
        return None
    return (max if higher else min)(cand, key=cand.get)


def write_report(rows, vocabs, mappings, sample_ids, catalog_by_id, floor, oracle, args, stamp,
                 split_info=None, oracle_cues=None, metadata_floor=None):
    _mode_str = (f"{args.lyrics_mode} (cap {args.lyrics_cap})"
                 if args.lyrics_mode in ("cap", "dedup") else args.lyrics_mode)

    # ---- 1. Header ----
    lines = ["# WP-B Cue Extraction — Method Comparison\n",
             f"_Generated {stamp} · {args.limit} songs · eval sample {len(sample_ids)} · "
             f"min_df {args.min_df} · top_n {args.top_n} · num_cues {args.num_cues} · "
             f"vocab_size {args.vocab_size} · embedder {args.embedder} · "
             f"lyrics-mode {_mode_str}_\n"]

    # ---- num-cues override banner (optional, --num-cues != CUE_TOKENS) ----
    if args.num_cues != CUE_TOKENS:
        lines.append(
            f"\n> **Non-default cue count:** this run assigned {args.num_cues} cues/song "
            f"(schema default is {CUE_TOKENS}). item2cues.json here needs "
            f"`CueMappingEntry.load_mapping(path, n_cues={args.num_cues})` to read back.\n")

    # ---- vocab-size override banner (optional, --vocab-size != CUE_VOCAB_SIZE) ----
    if args.vocab_size != CUE_VOCAB_SIZE:
        lines.append(
            f"\n> **Non-default vocab size:** this run built a {args.vocab_size}-entry vocab "
            f"(schema default is {CUE_VOCAB_SIZE}). cue_vocab.json/item2cues.json here need "
            f"`CueMappingEntry.load_mapping(path, vocab_size={args.vocab_size})` to validate "
            f"correctly on read-back.\n")

    # ---- embedder override banner (optional, --embedder != DEFAULT_EMBEDDER) ----
    if args.embedder != cue_normalize.DEFAULT_EMBEDDER:
        lines.append(
            f"\n> **Non-default embedder:** semantic dedup and assignment relevance both used "
            f"`{args.embedder}` instead of `{cue_normalize.DEFAULT_EMBEDDER}` in this run. "
            "Evaluation encoders (Level 2 retrieval, Level 3 STS-cosine) are unaffected — they "
            "deliberately use an independent encoder regardless of this setting.\n")

    # ---- LLM judge / Level 3 skipped banners (disabled via flag, or no API key) ----
    if not args.llm_judge:
        lines.append("\n> **LLM judge skipped:** run with `--no-llm-judge`.\n")
    elif not any(r.get("llm_judge_n") for r in rows.values()):
        lines.append("\n> **LLM judge skipped:** no `OPENAI_API_KEY` was available.\n")
    if not (args.level3 or args.level2):
        lines.append("\n> **Level 3 reconstruction skipped:** run with `--no-level3`.\n")
    elif not (oracle or any(r.get("level3") for r in rows.values())):
        lines.append("\n> **Level 3 reconstruction skipped:** no `OPENAI_API_KEY` was available.\n")

    # ---- Held-out eval banner (optional, --held-out-eval) ----
    if split_info:
        lines.append(
            f"\n> **Held-out eval:** vocabulary built from {split_info['n_train']} train "
            f"songs only; every metric below is scored on {split_info['n_test']} disjoint "
            f"test songs the vocab never saw (item-level split, test_frac="
            f"{split_info['test_frac']}, seed={split_info['seed']}). Not the playlist-level "
            f"`data/dataset/splits/` files.\n")

    # ---- Note to self (optional, --note) ----
    if args.note.strip():
        lines.append(f"\n> **Note:** {args.note.strip()}\n")

    # ---- 2. TL;DR ----
    has_l3 = bool(oracle) or any(r.get("level3") for r in rows.values())
    has_judge = any(r.get("llm_judge_n") for r in rows.values())
    best_ret = _best_method(rows, "retrieval_mrr", higher=True)
    best_ground = _best_method(rows, "level1_rouge1_recall", higher=True)
    best_div = _best_method(rows, "intra_cos_mean", higher=False)
    best_judge = _best_method(rows, "llm_judge_mean", higher=True)
    lines += ["\n## TL;DR\n"]
    if best_ret:
        lines.append(f"- **Retrieval (identity):** best is `{best_ret}` "
                     f"(MRR {rows[best_ret].get('retrieval_mrr')}).\n")
    if best_ground:
        lines.append(f"- **Grounding (cues↔lyrics):** best is `{best_ground}` "
                     f"(L1 ROUGE-1 {rows[best_ground].get('level1_rouge1_recall')}).\n")
    if best_div:
        lines.append(f"- **Cue diversity (intra-cos, <0.7):** most diverse is `{best_div}` "
                     f"({rows[best_div].get('intra_cos_mean')}).\n")
    if best_judge:
        lines.append(f"- **LLM judge (cues↔lyrics):** best is `{best_judge}` "
                     f"(mean score {rows[best_judge].get('llm_judge_mean')}/5).\n")
    if has_l3:
        best_recon = None
        cand = {m: r["level3"].get("sts_cosine") for m, r in rows.items()
                if r.get("level3") and isinstance(r["level3"].get("sts_cosine"), (int, float))}
        if cand:
            best_recon = max(cand, key=cand.get)
        if best_recon:
            oc = oracle.get("sts_cosine") if oracle else None
            fl = floor.get("sts_cosine") if floor else None
            frac = ""
            if isinstance(oc, (int, float)) and isinstance(fl, (int, float)) and oc != fl:
                frac = f", {round(100*(cand[best_recon]-fl)/(oc-fl))}% of the way from floor to oracle"
            lines.append(f"- **Reconstruction (downstream):** best is `{best_recon}` "
                         f"(cosine {cand[best_recon]}{frac}).\n")

    # ---- Methods legend ----
    lines += ["\n**Methods:** `tfidf` statistical term weighting · `yake` statistical "
              "keyphrases · `keybert` embedding keyphrases · `llm` LLM imagery extraction. "
              "Arrows in headers: (↑) higher is better, (↓) lower is better.\n"]

    # ---- 3. Vocabulary & assignment health ----
    lines += ["\n## 1. Vocabulary & assignment health\n"]
    lines += _table(
        rows,
        ["Vocab(real) ↑", "Coverage ↑", "UNK rate ↓", "Vocab util ↑", "Entropy ↑"],
        [("after_dedup", True), ("coverage_rate", True), ("unk_rate", False),
         ("vocab_coverage", True), ("cue_entropy_bits", True)])
    lines.append("\n_Is the vocabulary well-formed and broadly used? Vocab util scales with "
                 "corpus size — low on small samples is expected._\n")

    # ---- 4. Cue quality ----
    lines += ["\n## 2. Cue quality\n"]
    lines += _table(rows, ["Intra-cos ↓ (target <0.7)"], [("intra_cos_mean", False)])
    lines.append(f"\n_Within-item pairwise cosine of each song's {args.num_cues} cues; lower = "
                 "more diverse, less redundant cue sets._\n")

    # ---- 5. Grounding & retrieval ----
    _enc = next((r.get("retrieval_encoder") for r in rows.values() if r.get("retrieval_encoder")), "n/a")
    _indep = next((r.get("retrieval_independent") for r in rows.values()), False)
    _indep_note = ("independent of the T5 assignment encoder, so it is not circular."
                   if _indep else
                   "**WARNING: fell back to the assignment encoder — circular.** Set an "
                   "OpenAI key or install sentence-transformers for an independent encoder.")
    lines += ["\n## 3. Grounding & retrieval\n",
              "Do the cues identify their source song? Cue strings are embedded and ranked "
              "against artist-free song texts.\n\n",
              f"_Retrieval encoder: `{_enc}` — {_indep_note}_\n"]
    lines += _table(
        rows,
        ["L1 ROUGE-1 ↑", "L1 ROUGE-L ↑", "R@1 ↑", "R@5 ↑", "R@10 ↑", "MRR ↑", "Median rank ↓"],
        [("level1_rouge1_recall", True), ("level1_rougeL_recall", True),
         ("retrieval_r1", True), ("retrieval_r5", True), ("retrieval_r10", True),
         ("retrieval_mrr", True), ("retrieval_median_rank", False)])

    # ---- 6. LLM judge (cue<->lyric match) ----
    if has_judge:
        _judge_model = next((r.get("llm_judge_model") for r in rows.values()
                             if r.get("llm_judge_model")), cue_clients.CHAT_MODEL)
        lines += ["\n## 4. LLM judge (cue↔lyric match)\n",
                  "An LLM is shown each song's assigned cues alongside its real lyrics and rates "
                  "1 (unrelated) to 5 (clearly drawn from the lyrics) how well the cues match the "
                  "lyrics' content. Scored on the eval sample; independent of the ROUGE-based "
                  "Level 1 grounding metric above.\n\n",
                  f"_Judge model: `{_judge_model}` · temperature 0._\n\n",
                  "**System prompt:**\n\n```\n" + cue_eval.JUDGE_SYSTEM_PROMPT + "\n```\n\n",
                  "**User prompt template** (per song, `{lyrics}`/`{cues}` filled in; "
                  f"lyrics truncated to {cue_eval._JUDGE_LYRIC_CHARS} chars):\n\n",
                  "```\nLyrics:\n{lyrics}\n\nCreative cues: {cues}\n\n"
                  "On a scale of 1-5, how well do these cues match the content of these lyrics? "
                  "Respond with ONLY the integer.\n```\n\n"]
        lines += _table(
            rows, ["Judge score (1-5) ↑", "n scored"],
            [("llm_judge_mean", True), ("llm_judge_n", True)])

    # ---- 7. Reconstruction (Level 3) ----
    if has_l3:
        _sts_enc = next((r.get("level3", {}).get("sts_encoder") for r in rows.values()
                         if r.get("level3", {}).get("sts_encoder")), "all-mpnet-base-v2")
        _win = (f"both lyrics scored over the same first {args.score_chars} chars"
                if args.score_chars else "full lyrics scored (no length window)")
        lines += ["\n## 5. Reconstruction comparison (downstream usefulness)\n",
                  "Decode lyrics from cues (title/artist withheld), score vs real lyrics. "
                  "Bracketed **no-cues floor < random floor < methods < oracle ceiling** — "
                  "read deltas above the floors.\n\n",
                  f"_{_win}. Cosine uses `{_sts_enc}` (document-level, length-robust). "
                  "BERTScore is baseline-rescaled, so near/below 0 is expected — read rankings._\n\n",
                  "| Condition | ROUGE-1 ↑ | ROUGE-2 ↑ | ROUGE-L ↑ | BLEU ↑ | BERTScore ↑ | Cosine ↑ |\n",
                  "|-----------|-----------|-----------|-----------|--------|-------------|----------|\n"]

        def _l3_row(label, d):
            return (f"| {label} | {d.get('rouge1_f')} | {d.get('rouge2_f')} | "
                    f"{d.get('rougeL_f')} | {d.get('bleu')} | "
                    f"{d.get('bertscore_f1')} | {d.get('sts_cosine')} |\n")

        if metadata_floor:
            lines.append(_l3_row("no cues (metadata-only floor)", metadata_floor))
        if floor:
            lines.append(_l3_row("random (floor)", floor))
        for m, r in rows.items():
            if r.get("level3"):
                lines.append(_l3_row(m, r["level3"]))
        if oracle:
            lines.append(_l3_row("oracle (ceiling)", oracle))

    # ---- 8. Qualitative ----
    _heading = "## 6. Qualitative samples (cues per method"
    _heading += ", + oracle ceiling)\n" if oracle_cues is not None else ")\n"
    lines += [f"\n{_heading}"]
    for iid in sample_ids[:12]:
        item = catalog_by_id[iid]
        lines.append(f"\n**{item.title}** — {item.artist} _( {item.genre} )_\n\n")
        for m in rows:
            v, mp = vocabs[m], mappings[m]
            entry = mp.get(iid)
            cues = [v[c] for c in entry.cue_ids if c != 0] if entry else []
            lines.append(f"- `{m}`: {', '.join(cues) if cues else '(unk)'}\n")
        if oracle_cues is not None:
            ocues = oracle_cues.get(iid, [])
            lines.append(f"- `oracle` _(ceiling — LLM cues from real lyrics)_: "
                         f"{', '.join(ocues) if ocues else '(none)'}\n")

    lines += ["\n### Top cues per method (most-assigned)\n"]
    import collections
    for m in rows:
        v, mp = vocabs[m], mappings[m]
        cnt = collections.Counter()
        for entry in mp.values():
            for c in entry.cue_ids:
                if c != 0:
                    cnt[c] += 1
        top = ", ".join(f"{v[c]}({n})" for c, n in cnt.most_common(15))
        lines.append(f"- **{m}**: {top}\n")

    # ---- 8. Appendix: metric definitions ----
    lines += [
        "\n## Appendix — metric definitions\n",
        "- **Vocab(real):** non-placeholder cue phrases surviving df-band, POS, blocklist, and semantic dedup.\n",
        "- **Coverage:** fraction of songs with zero `<unk>` slots — every assigned cue is real (target >=40%).\n",
        "- **UNK rate:** fraction of all cue slots that are `<unk>` (target <=60%).\n",
        "- **Vocab util:** fraction of non-`<unk>` vocab entries used by >=1 song (target >=50%; scales with corpus size).\n",
        f"- **Intra-cos:** mean within-item pairwise cosine of a song's {args.num_cues} cue "
        "embeddings (target <0.7; lower=more diverse).\n",
        "- **Entropy:** Shannon entropy (bits) of cue-usage distribution; higher=less collapse onto a few cues.\n",
        "- **L1 ROUGE-1/L:** ROUGE recall of assigned cue text vs real lyrics; lexical grounding sanity check.\n",
        "- **Retrieval R@K / MRR / rank:** rank of the true song when its cues query artist-free song texts, via an independent encoder (non-circular).\n",
        "- **LLM judge:** an LLM (see prompt in section 4) rates 1-5 how well a song's assigned cues match its real lyrics, averaged over the eval sample; independent of the lexical Level 1 ROUGE metric.\n",
        "- **Level 3 ROUGE/BLEU/BERTScore/Cosine:** decode lyrics from cues, score vs real. Cosine is document-level and length-robust; BERTScore is baseline-rescaled.\n",
        f"- **no-cues / random floor / oracle ceiling:** reconstruction from zero cues "
        f"(genre/mood only), from {args.num_cues} random vocab cues, and from {args.num_cues} "
        f"cues extracted from the real lyrics (ceiling), respectively.\n",
    ]

    path = os.path.join(RUN_DIR, "comparison_report.md")
    cue_io.atomic_write_text(path, "".join(lines))
    print(f"\n[compare] report -> {path}")


if __name__ == "__main__":
    main()
