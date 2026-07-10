"""run_compare.py — WP-B Phase 5: end-to-end multi-method comparison + report.

For each extraction method:
    extract raw cues -> normalize vocab -> assign 6 cues/song -> export -> evaluate

Then writes one comparison report:
    src/02_creative_cues/outputs/comparison_report.md

Usage (from project root, with venv):
    .venv/Scripts/python.exe run_compare.py --limit 1000 --methods tfidf,yake
    .venv/Scripts/python.exe run_compare.py --methods tfidf,yake,keybert,llm \
        --eval-sample 200 --level3
    .venv/Scripts/python.exe run_compare.py --methods tfidf,yake,keybert,llm \
        --llm-batch
"""

import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "src", "02_creative_cues"))
sys.path.insert(0, os.path.join(BASE, "src", "00_data_schema"))

import cue_extractors          # noqa: E402
import cue_normalize           # noqa: E402
import cue_assign              # noqa: E402
import cue_eval                # noqa: E402
import cue_clients             # noqa: E402
import cue_lyrics              # noqa: E402
import cue_mining              # noqa: E402
from schema import CatalogItem  # noqa: E402
from datetime import datetime   # noqa: E402

OUT_ROOT = os.path.join(BASE, "src", "02_creative_cues", "outputs")


def load_data(limit):
    with open(os.path.join(BASE, "data", "dataset", "catalog_metadata.json"), encoding="utf-8") as f:
        raw = json.load(f)
    items = [
        CatalogItem(**{k: v for k, v in e.items() if k in CatalogItem.__dataclass_fields__})
        for e in list(raw.values())[:limit]
    ]
    lyrics = {}
    for it in items:
        p = os.path.join(BASE, "data", "lyrics", "spotify", f"{it.item_id}.txt")
        if os.path.isfile(p):
            lyrics[it.item_id] = open(p, encoding="utf-8", errors="ignore").read()
    return items, lyrics


def finish_method_from_raw(method, raw, items, lyrics_proc, min_df, block_tokens,
                           dedup_threshold=0.92):
    """normalize -> assign -> export. Returns (vocab, item2cues, norm_stats, cue_emb).

    lyrics_proc = preprocessed lyrics used for cue assignment.
    dedup_threshold: cosine above which two cues are merged in semantic dedup
                     (higher = fewer merges = larger vocabulary).
    """
    norm = cue_normalize.build_vocab_normalized(raw, vocab_size=2048, min_df=min_df,
                                                dedup_threshold=dedup_threshold,
                                                block_tokens=block_tokens, verbose=True)
    vocab, cue_emb = norm["vocab"], norm["embeddings"]
    item2cues = cue_assign.assign_all(items, lyrics_proc, vocab, cue_emb, verbose=True)
    out_dir = os.path.join(OUT_ROOT, method)
    cue_mining.export_outputs(vocab, item2cues, out_dir)
    return vocab, item2cues, norm["stats"], cue_emb


def run_method(method, items, lyrics_proc, min_df, force, block_tokens, top_n, cache_tag,
               dedup_threshold=0.92):
    """extract -> normalize -> assign -> export. Returns (vocab, item2cues, norm_stats, cue_emb)."""
    print(f"\n########## METHOD: {method} ##########")
    raw = cue_extractors.extract_raw_cues(items, lyrics_proc, method=method, force=force,
                                          top_n=top_n, cache_tag=cache_tag)
    return finish_method_from_raw(method, raw, items, lyrics_proc, min_df, block_tokens,
                                  dedup_threshold=dedup_threshold)


def evaluate_method(method, vocab, item2cues, nstats, cue_emb, catalog_by_id, lyrics_ref,
                    lyrics_proc, sample_ids):
    """Coverage + within-item diversity + Level 1 grounding + Level 2 retrieval."""
    cov = cue_mining.compute_coverage_stats(item2cues, vocab)
    # Within-item cue diversity (paper target < 0.7), using the assignment embeddings.
    div = cue_eval.within_item_diversity(item2cues, cue_emb)
    # Level 1 grounding: cues vs the REAL (unprocessed) lyrics.
    l1 = cue_eval.level1_intrinsic(item2cues, vocab, lyrics_ref)
    # Level 2 retrieval: same processed song representation used at assignment time.
    l2 = cue_eval.level2_semantic_retrieval(
        catalog_by_id, item2cues, vocab, lyrics_proc, sample_ids)
    return {**nstats, **cov, **div, **l1, **l2}


def run_level3_conditions(rows, vocabs, mappings, catalog_by_id, lyrics_ref,
                          sample_ids, use_batch, poll_s, timeout_s):
    """Bracketed Level 3 ablation: random floor < methods < oracle ceiling.

    Returns (floor_metrics, method_metrics_by_name, oracle_metrics). All conditions
    share the decoder (title/artist withheld); only the cue content differs.
    """
    first_vocab = next(iter(vocabs.values()))
    conds = {
        "random": cue_eval.random_cues_by_id(first_vocab, sample_ids),
        "oracle": cue_eval.oracle_cues_by_id(catalog_by_id, lyrics_ref, sample_ids),
    }
    for m in rows:
        conds[m] = cue_eval.assigned_cues_by_id(mappings[m], vocabs[m], sample_ids)

    results: dict[str, dict] = {}
    if use_batch:
        print("[compare] submitting Level 3 batches (random, methods, oracle)...")
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
    oracle = results.pop("oracle")
    return floor, results, oracle


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
    ap.add_argument("--lyrics-mode", default="cap", choices=list(cue_lyrics.MODES),
                    help="how lyrics feed extraction/assignment: cap | full | dedup | summarize")
    ap.add_argument("--lyrics-cap", type=int, default=cue_lyrics.DEFAULT_CAP,
                    help="char cap for cap/dedup modes")
    ap.add_argument("--score-chars", type=int, default=2000,
                    help="common window (chars) applied to both gen and real lyrics "
                         "before Level 3 metrics, so lengths are comparable (0 = full)")
    ap.add_argument("--eval-sample", type=int, default=150)
    ap.add_argument("--level3", action="store_true", help="run LLM reconstruction ablation")
    ap.add_argument("--level2", action="store_true", help=argparse.SUPPRESS)
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
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
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
    lyrics_proc = {iid: cue_lyrics.preprocess_lyrics(t, args.lyrics_mode, args.lyrics_cap)
                   for iid, t in lyrics_raw.items()}
    print(f"[compare] {len(items)} songs, {len(lyrics_raw)} with lyrics, methods={methods}")
    print(f"[compare] {len(block_tokens)} blocked artist tokens")

    # deterministic eval sample (songs that have real lyrics)
    sample_ids = [iid for iid in catalog_by_id if lyrics_raw.get(iid, "").strip()][:args.eval_sample]

    rows = {}
    vocabs = {}
    mappings = {}
    run_level3 = args.level3 or args.level2
    llm_batch_job = None
    run_methods = methods
    if args.llm_batch and "llm" in methods:
        print("[compare] submitting LLM extraction batch before other methods...")
        llm_batch_job = cue_extractors.submit_llm_batch(
            items, lyrics_proc, top_n=args.top_n, force=args.force, cache_tag=cache_tag)
        run_methods = [m for m in methods if m != "llm"]

    for method in run_methods:
        vocab, item2cues, nstats, cue_emb = run_method(method, items, lyrics_proc, args.min_df,
                                                       args.force, block_tokens, args.top_n, cache_tag,
                                                       dedup_threshold=args.dedup_threshold)
        row = evaluate_method(method, vocab, item2cues, nstats, cue_emb, catalog_by_id,
                              lyrics_raw, lyrics_proc, sample_ids)
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
            dedup_threshold=args.dedup_threshold)
        row = evaluate_method("llm", vocab, item2cues, nstats, cue_emb, catalog_by_id,
                              lyrics_raw, lyrics_proc, sample_ids)
        rows["llm"] = row
        vocabs["llm"] = vocab
        mappings["llm"] = item2cues

    # Level 3 bracketed ablation: random floor < methods < oracle ceiling.
    # References are the REAL (unprocessed) lyrics — lyrics_raw.
    floor = oracle = None
    if run_level3 and not cue_clients.is_available():
        print("[compare] reconstruction requested but no API key; skipping Level 3.")
    elif run_level3:
        floor, method_l3, oracle = run_level3_conditions(
            rows, vocabs, mappings, catalog_by_id, lyrics_raw, sample_ids,
            use_batch=args.recon_batch,
            poll_s=args.llm_batch_poll_seconds, timeout_s=args.llm_batch_timeout_seconds)
        for m, res in method_l3.items():
            rows[m]["level3"] = res

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_report(rows, vocabs, mappings, sample_ids, catalog_by_id, floor, oracle,
                 args, stamp)

    # Separate side-by-side original-vs-regenerated lyrics report (Level 3 only)
    if run_level3 and cue_clients.is_available():
        write_reconstruction_report(rows, vocabs, mappings, sample_ids,
                                    catalog_by_id, lyrics_raw, args, stamp)


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
             "from ONLY the 6 cues + minimal metadata, per method. Closer = the cues "
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

    stamped = os.path.join(OUT_ROOT, f"reconstruction_report_{stamp}.md")
    latest = os.path.join(OUT_ROOT, "reconstruction_report.md")
    for path in (stamped, latest):
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    print(f"[compare] reconstruction samples -> {stamped} (and latest)")


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


def write_report(rows, vocabs, mappings, sample_ids, catalog_by_id, floor, oracle, args, stamp):
    _mode_str = (f"{args.lyrics_mode} (cap {args.lyrics_cap})"
                 if args.lyrics_mode in ("cap", "dedup") else args.lyrics_mode)

    # ---- 1. Header ----
    lines = ["# WP-B Cue Extraction — Method Comparison\n",
             f"_Generated {stamp} · {args.limit} songs · eval sample {len(sample_ids)} · "
             f"min_df {args.min_df} · top_n {args.top_n} · lyrics-mode {_mode_str}_\n"]

    # ---- 2. TL;DR ----
    has_l3 = bool(oracle) or any(r.get("level3") for r in rows.values())
    best_ret = _best_method(rows, "retrieval_mrr", higher=True)
    best_ground = _best_method(rows, "level1_rouge1_recall", higher=True)
    best_div = _best_method(rows, "intra_cos_mean", higher=False)
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
    lines.append("\n_Within-item pairwise cosine of each song's 6 cues; lower = more diverse, "
                 "less redundant cue sets._\n")

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

    # ---- 6. Reconstruction (Level 3) ----
    if has_l3:
        _sts_enc = next((r.get("level3", {}).get("sts_encoder") for r in rows.values()
                         if r.get("level3", {}).get("sts_encoder")), "all-mpnet-base-v2")
        _win = (f"both lyrics scored over the same first {args.score_chars} chars"
                if args.score_chars else "full lyrics scored (no length window)")
        lines += ["\n## 4. Reconstruction comparison (downstream usefulness)\n",
                  "Decode lyrics from cues (title/artist withheld), score vs real lyrics. "
                  "Bracketed **random floor < methods < oracle ceiling** — read deltas above "
                  "the floor.\n\n",
                  f"_{_win}. Cosine uses `{_sts_enc}` (document-level, length-robust). "
                  "BERTScore is baseline-rescaled, so near/below 0 is expected — read rankings._\n\n",
                  "| Condition | ROUGE-1 ↑ | ROUGE-2 ↑ | ROUGE-L ↑ | BLEU ↑ | BERTScore ↑ | Cosine ↑ |\n",
                  "|-----------|-----------|-----------|-----------|--------|-------------|----------|\n"]

        def _l3_row(label, d):
            return (f"| {label} | {d.get('rouge1_f')} | {d.get('rouge2_f')} | "
                    f"{d.get('rougeL_f')} | {d.get('bleu')} | "
                    f"{d.get('bertscore_f1')} | {d.get('sts_cosine')} |\n")

        if floor:
            lines.append(_l3_row("random (floor)", floor))
        for m, r in rows.items():
            if r.get("level3"):
                lines.append(_l3_row(m, r["level3"]))
        if oracle:
            lines.append(_l3_row("oracle (ceiling)", oracle))

    # ---- 7. Qualitative ----
    lines += ["\n## 5. Qualitative samples (cues per method)\n"]
    for iid in sample_ids[:12]:
        item = catalog_by_id[iid]
        lines.append(f"\n**{item.title}** — {item.artist} _( {item.genre} )_\n\n")
        for m in rows:
            v, mp = vocabs[m], mappings[m]
            entry = mp.get(iid)
            cues = [v[c] for c in entry.cue_ids if c != 0] if entry else []
            lines.append(f"- `{m}`: {', '.join(cues) if cues else '(unk)'}\n")

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
        "- **Coverage:** fraction of songs with >=1 non-`<unk>` assigned cue (target >=40%).\n",
        "- **UNK rate:** fraction of all cue slots that are `<unk>` (target <=60%).\n",
        "- **Vocab util:** fraction of non-`<unk>` vocab entries used by >=1 song (target >=50%; scales with corpus size).\n",
        "- **Intra-cos:** mean within-item pairwise cosine of a song's 6 cue embeddings (target <0.7; lower=more diverse).\n",
        "- **Entropy:** Shannon entropy (bits) of cue-usage distribution; higher=less collapse onto a few cues.\n",
        "- **L1 ROUGE-1/L:** ROUGE recall of assigned cue text vs real lyrics; lexical grounding sanity check.\n",
        "- **Retrieval R@K / MRR / rank:** rank of the true song when its cues query artist-free song texts, via an independent encoder (non-circular).\n",
        "- **Level 3 ROUGE/BLEU/BERTScore/Cosine:** decode lyrics from cues, score vs real. Cosine is document-level and length-robust; BERTScore is baseline-rescaled.\n",
        "- **random floor / oracle ceiling:** reconstruction from 6 random vocab cues (floor) and from 6 cues extracted from the real lyrics (ceiling).\n",
    ]

    stamped = os.path.join(OUT_ROOT, f"comparison_report_{stamp}.md")
    latest = os.path.join(OUT_ROOT, "comparison_report.md")
    for path in (stamped, latest):
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    print(f"\n[compare] report -> {stamped} (and latest)")


if __name__ == "__main__":
    main()
