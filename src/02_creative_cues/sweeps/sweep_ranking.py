"""sweep_ranking.py — controlled comparison of stage-5 vocabulary ranking rules.

Stages 0-4 of cleaning run ONCE per corpus size; all ranking arms then select from
that identical candidate pool, so the arms differ only in stage 5. Includes a
`random` arm (ranking removed) as a genuine ablation floor.

Measures three axes:
  1. Stability   — Jaccard overlap of the vocabulary across corpus sizes (free)
  2. Vocab health— effective vocab, utilisation, entropy, cues used <5x, intra-cos (free)
  3. Retrieval   — Level 2 cue-to-song retrieval (cheap API, one corpus size)

Level 3 reconstruction is intentionally NOT run for every arm (expensive); use
run_compare.py --rank-by <winner> --level3 for the leading arms.

Usage:
  .venv/Scripts/python.exe src/02_creative_cues/sweeps/sweep_ranking.py --method tfidf \
      --limits 500,1000,2000,3000,5000 --retrieval-limit 1000 \
      --vocab-size 2048 --min-df 2 --dedup-threshold 0.92 --num-cues 18 \
      --lyrics-mode dedup --lyrics-cap 2000 --top-n 60 --eval-sample 100
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "00_data_schema"))

import numpy as np       # noqa: E402
import cue_extractors    # noqa: E402
import cue_normalize     # noqa: E402
import cue_lyrics        # noqa: E402
import cue_eval          # noqa: E402
import cue_clients       # noqa: E402
import cue_io            # noqa: E402
import data_loading      # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "experiments")


def _song_text(item, lyrics, cap=600):
    meta = " ".join(p for p in [item.title, item.genre, item.mood, item.lyric_excerpt, *item.tags] if p)
    return f"{meta} {lyrics[:cap]}" if lyrics else meta


def build_pool(limit, args, tag):
    """Stages 0-4 once for a corpus size -> shared candidate pool."""
    items, lyrics = data_loading.load_catalog_and_lyrics(limit)
    lyrics_proc = data_loading.build_lyrics_proc(lyrics, args.lyrics_mode, args.lyrics_cap)
    block = cue_normalize.build_block_tokens(items)
    raw = cue_extractors.extract_raw_cues(items, lyrics_proc, method=args.method,
                                          force=args.force, top_n=args.top_n, cache_tag=tag)
    pool = cue_normalize.clean_candidates(raw, min_df=args.min_df,
                                          dedup_threshold=args.dedup_threshold,
                                          block_tokens=block, verbose=False)
    return items, lyrics, lyrics_proc, pool


def assign_n(song_emb, vocab, valid_idx, valid_emb, n, lam=0.7, cand_k=40):
    """MMR-select n cues for one song (same rule as cue_assign, arbitrary n)."""
    if n <= 0 or len(valid_idx) == 0:
        return []
    rel = valid_emb @ song_emb
    k = min(len(valid_idx), max(cand_k, 2 * n))
    cand = np.argpartition(rel, -k)[-k:]
    cand = list(cand[np.argsort(rel[cand])[::-1]])
    sel = []
    for _ in range(min(n, len(cand))):
        best_s, best = -1e9, None
        for c in cand:
            if c in sel:
                continue
            div = float((valid_emb[sel] @ valid_emb[c]).max()) if sel else 0.0
            s = lam * float(rel[c]) - (1 - lam) * div
            if s > best_s:
                best_s, best = s, c
        if best is None:
            break
        sel.append(best)
    return sel  # indices into valid_idx


def vocab_health(vocab, cue_emb, assignments, num_cues):
    """Effective vocab / utilisation / entropy / rare-cue count / intra-item cosine."""
    import collections
    import math
    counts = collections.Counter()
    for sel in assignments.values():
        for i in sel:
            counts[i] += 1
    real_slots = sum(1 for c in vocab[1:] if not c.startswith("<pad_"))
    effective = len(counts)
    util = effective / max(real_slots, 1)
    total = sum(counts.values())
    entropy = 0.0
    for v in counts.values():
        p = v / max(total, 1)
        entropy -= p * math.log2(p + 1e-12)
    rare = sum(1 for v in counts.values() if v < 5)

    # intra-item diversity (mean pairwise cosine within each song's cue set)
    sims = []
    for sel in assignments.values():
        if len(sel) < 2:
            continue
        V = cue_emb[sel]
        V = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)
        S = V @ V.T
        iu = np.triu_indices(len(sel), k=1)
        sims.append(float(S[iu].mean()))
    intra = round(float(np.mean(sims)), 4) if sims else None
    return {"effective": effective, "real_slots": real_slots, "util": round(util, 4),
            "entropy": round(entropy, 4), "rare_lt5": rare, "intra_cos": intra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="tfidf")
    ap.add_argument("--arms", default="random,df,idf,df_idf,band,cluster")
    ap.add_argument("--limits", default="500,1000,2000,3000,5000")
    ap.add_argument("--retrieval-limit", type=int, default=1000,
                    help="corpus size used for the health + retrieval measurements")
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--dedup-threshold", type=float, default=0.92)
    ap.add_argument("--num-cues", type=int, default=18)
    ap.add_argument("--eval-sample", type=int, default=100)
    ap.add_argument("--lyrics-mode", default="dedup", choices=list(cue_lyrics.MODES))
    ap.add_argument("--lyrics-cap", type=int, default=cue_lyrics.DEFAULT_CAP)
    ap.add_argument("--top-n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-retrieval", action="store_true", help="Phase 1 only (no API)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    limits = sorted(int(x) for x in args.limits.split(",") if x.strip())
    tag = cue_lyrics.cache_tag(args.lyrics_mode, args.lyrics_cap)
    print(f"[rank] method={args.method} arms={arms} limits={limits} K={args.vocab_size} "
          f"min_df={args.min_df} dedup={args.dedup_threshold} cues/song={args.num_cues}")

    # corpus size used for health + retrieval (decided up front so Phase 1 can cache it)
    rl = args.retrieval_limit if args.retrieval_limit in limits else limits[-1]

    # ---- Phase 1: stability across corpus sizes (free) ----
    vocab_sets = {a: {} for a in arms}
    assembled = {}       # arm -> assemble_vocab output at N=rl (reused in Phase 2)
    rl_data = None       # only the rl pool is retained (others freed each iteration)
    for n in limits:
        print(f"[rank] cleaning pool at N={n} ...")
        items, lyrics, lyrics_proc, pool = build_pool(n, args, tag)
        for a in arms:
            out = cue_normalize.assemble_vocab(pool, vocab_size=args.vocab_size,
                                               rank_by=a, seed=args.seed, verbose=False)
            vocab_sets[a][n] = {c for c in out["vocab"]
                                if c != "<unk>" and not c.startswith("<pad_")}
            if n == rl:
                assembled[a] = out          # reuse instead of re-running (k-means is costly)
            print(f"    {a:<8} -> vocab {len(vocab_sets[a][n])}")
        if n == rl:
            rl_data = (items, lyrics, lyrics_proc, pool)

    # ---- Phase 2: health + retrieval at one corpus size ----
    items, lyrics, lyrics_proc, pool = rl_data
    catalog_by_id = {it.item_id: it for it in items}
    sample_ids = [iid for iid in catalog_by_id if lyrics.get(iid, "").strip()][:args.eval_sample]
    embed_fn, _backend = cue_normalize._make_embedder()
    song_emb = embed_fn([_song_text(catalog_by_id[i], lyrics_proc.get(i, "")) for i in sample_ids])
    song_emb = song_emb / np.clip(np.linalg.norm(song_emb, axis=1, keepdims=True), 1e-9, None)

    health, retrieval = {}, {}
    for a in arms:
        out = assembled[a]                  # cached from Phase 1
        vocab, cue_emb = out["vocab"], out["embeddings"]
        norms = np.linalg.norm(cue_emb, axis=1)
        valid_idx = np.array([k for k in range(len(cue_emb))
                              if norms[k] > 0 and not vocab[k + 1].startswith("<pad_")])
        if len(valid_idx) == 0:
            continue
        valid_emb = cue_emb[valid_idx] / norms[valid_idx][:, None]

        assignments = {iid: assign_n(song_emb[r], vocab, valid_idx, valid_emb, args.num_cues)
                       for r, iid in enumerate(sample_ids)}
        health[a] = vocab_health(vocab, valid_emb, assignments, args.num_cues)

        if not args.skip_retrieval and cue_clients.is_available():
            cues_by_id = {iid: [vocab[int(valid_idx[c]) + 1] for c in sel]
                          for iid, sel in assignments.items()}
            retrieval[a] = _retrieval(catalog_by_id, cues_by_id, lyrics_proc, sample_ids)
        print(f"[rank] {a:<8} health={health[a]}")

    _write_report(args, arms, limits, vocab_sets, health, retrieval, rl)


def _retrieval(catalog_by_id, cues_by_id, lyrics_proc, sample_ids):
    """Level-2 style retrieval using an independent encoder, on arbitrary cue lists."""
    cand = [i for i in sample_ids if lyrics_proc.get(i, "").strip()]
    qids, qdocs = [], []
    for i in cand:
        cues = cues_by_id.get(i, [])
        if cues:
            qids.append(i)
            qdocs.append(", ".join(cues))
    if not qids:
        return {}
    tdocs = [cue_eval._retrieval_song_text(catalog_by_id[i], lyrics_proc.get(i, "")) for i in cand]
    embed_fn, _name, _indep = cue_eval._make_retrieval_embedder()
    allv = embed_fn(qdocs + tdocs)
    q, t = allv[:len(qdocs)], allv[len(qdocs):]
    pos = {i: k for k, i in enumerate(cand)}
    sims = q @ t.T
    ranks = []
    for r, i in enumerate(qids):
        order = np.argsort(sims[r])[::-1]
        ranks.append(int(np.where(order == pos[i])[0][0]) + 1)
    ra = np.asarray(ranks, dtype=np.float32)
    return {"r1": round(float(np.mean(ra <= 1)), 4), "r5": round(float(np.mean(ra <= 5)), 4),
            "r10": round(float(np.mean(ra <= 10)), 4), "mrr": round(float(np.mean(1 / ra)), 4),
            "median": round(float(np.median(ra)), 2)}


def _jac(a, b):
    return 1.0 if (not a and not b) else len(a & b) / len(a | b)


def _write_report(args, arms, limits, vocab_sets, health, retrieval, rl):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"sweep_ranking_{args.method}_{stamp}.md")
    final = limits[-1]

    L = [
        "# Vocabulary Ranking Strategy — Controlled Comparison\n",
        f"_Generated {stamp} · method {args.method} · corpus {'/'.join(map(str,limits))} · "
        f"K (vocab) {args.vocab_size} · min_df {args.min_df} · dedup {args.dedup_threshold} · "
        f"cues/song {args.num_cues} · lyrics-mode {args.lyrics_mode} (cap {args.lyrics_cap}) · "
        f"top_n {args.top_n}_\n",

        "\n## 1. Why we are experimenting with the ranking step\n",
        "\nThe cleaning pipeline reduces raw candidate cues to the final vocabulary in six stages:\n",
        "\n```\n0 normalize → 1 df-band → 2 POS → 3 embed → 4 semantic dedup → 5 RANK + CUT\n```\n",
        "\nStages 0-4 are *filters*: they remove cues that are malformed, too rare or too common, "
        "grammatically wrong, boilerplate, or near-duplicates. After them many more candidates "
        "survive than the K slots the vocabulary has. **Stage 5 is therefore the only stage that "
        "decides *which* cues actually become the vocabulary**, and it is the least examined.\n",
        "\nThis matters because stage 5 currently ranks by IDF (rarest first), and that choice "
        "produced a measured problem: the vocabulary does not converge. Across corpus sizes the "
        "chosen cue set churned 35-60% per step (Jaccard 0.26-0.48), and a 1,000-song vocabulary "
        "shared under 10% of its cues with the 5,000-song vocabulary. Because IDF ranks by "
        "*absolute rarity*, growing the corpus continually surfaces rarer cues that displace the "
        "incumbents — the vocabulary chases the tail instead of settling on concepts.\n",
        "\nThis experiment holds stages 0-4 fixed and varies **only stage 5**, to determine which "
        "ranking rule yields a vocabulary that is stable, well-utilised, and downstream-useful.\n",

        "\n## 2. Ranking methods under test\n",
        "\n| Arm | Rule | Intent |\n|-----|------|--------|\n",
        "| `random` | random K from the pool (seeded) | **ablation floor** — ranking removed |\n",
        "| `df` | highest document frequency first | frequency-only extreme |\n",
        "| `idf` | highest `log(N/(1+df))` — rarest first | **current behaviour** |\n",
        "| `df_idf` | `df · log(N/(1+df))` | frequency × distinctiveness balance |\n",
        "| `band` | closeness to a target prevalence `df/N` | corpus-size-invariant |\n",
        "| `cluster` | k-means into K clusters, one representative each | spanning codebook |\n",
        "\n**`random`** selects K cues uniformly at random from the cleaned pool. It removes the "
        "ranking component entirely, giving a genuine ablation floor: any rule that cannot beat "
        "random is not earning its place.\n",
        "\n**`df`** ranks by how many songs contain the cue. Maximally stable (common cues stay "
        "common) but least discriminative: it fills the vocabulary with generic, high-coverage "
        "terms that apply to most songs.\n",
        "\n**`idf`** is the current rule, ranking by rarity. Maximally discriminative per cue, but "
        "corpus-size-dependent and, at the extreme, it rewards noise — a one-off typo has the "
        "highest possible IDF.\n",
        "\n**`df_idf`** multiplies frequency by distinctiveness, so the score peaks at "
        "*mid-frequency* cues: recurrent enough to generalise, rare enough to discriminate. "
        "Mid-frequency cues scale proportionally with the corpus, so this should be markedly "
        "more stable than IDF. **Caveat (found during analysis):** with the default "
        "`max_df_frac=0.3`, this score is monotonic in df across the ENTIRE df-band-filtered "
        "range (its peak sits at df/N ≈ 1/e ≈ 0.368, above the 0.3 cutoff) — so at that default,\n"
        "`df_idf` selects the identical set as `df`. Raise `max_df_frac` past ~0.37 for this arm "
        "to actually diverge from `df`.\n",
        "\n**`band`** scores by distance to a target prevalence, `−|log(df/N) − log(target)|`. "
        "Because it uses the *relative* rate `df/N` rather than absolute counts, the criterion is "
        "invariant to corpus size, directly attacking the churn mechanism.\n",
        "\n**`cluster`** partitions the candidate embeddings into K clusters (MiniBatchKMeans, "
        "df-weighted, seeded) and elects the cue nearest each centroid. The vocabulary becomes a "
        "codebook that *spans* the concept space rather than a leaderboard of rare phrases; "
        "near-duplicates collapse into one cluster, so redundancy is removed by construction.\n",

        "\n## 3. Experimental design\n",
        "\n**Controlled.** For each corpus size, extraction and cleaning (stages 0-4) run **once**, "
        "and all arms rank that *identical* candidate pool — so the arms differ only in stage 5. "
        "Everything else is fixed: method, min_df, dedup threshold, lyrics mode, top_n, K, "
        "cues/song, assignment parameters, eval sample and all seeds.\n",
        f"\n**Measured.** (1) Stability — Jaccard overlap of the vocabulary across corpus sizes. "
        f"(2) Vocabulary health — effective vocab, utilisation, usage entropy, cues used <5x, "
        f"intra-item cue diversity (measured at N={rl}, {args.num_cues} cues/song). "
        f"(3) Retrieval — Level 2 cue-to-song retrieval with an independent encoder.\n",
        "\nLevel 3 reconstruction is deliberately not run for every arm (LLM cost); run it for the "
        "leading arms with `run_compare.py --rank-by <arm> --level3`.\n",

        "\n## 4. Results\n",
        "\n### 4.1a Stability — consecutive-corpus overlap (Jaccard; higher = more stable)\n",
        "| Arm | " + " | ".join(f"{a}→{b}" for a, b in zip(limits, limits[1:])) + " |\n",
        "|-----|" + "----|" * max(len(limits) - 1, 1) + "\n",
    ]
    for a in arms:
        cells = [f"{_jac(vocab_sets[a][x], vocab_sets[a][y]):.3f}" for x, y in zip(limits, limits[1:])]
        L.append(f"| `{a}` | " + " | ".join(cells) + " |\n")

    L += [f"\n### 4.1b Convergence — overlap of each corpus size with the largest (N={final})\n",
          "Rising toward 1.0 across the row = the vocabulary is converging to a stable set.\n\n",
          "| Arm | " + " | ".join(f"N={n}" for n in limits) + " |\n",
          "|-----|" + "----|" * len(limits) + "\n"]
    for a in arms:
        cells = [f"{_jac(vocab_sets[a][n], vocab_sets[a][final]):.3f}" for n in limits]
        L.append(f"| `{a}` | " + " | ".join(cells) + " |\n")

    L += [f"\n### 4.2 Vocabulary health (N={rl}, {args.num_cues} cues/song)\n",
          "| Arm | effective vocab | utilisation | entropy (bits) | cues used <5x | intra-cos ↓ |\n",
          "|-----|-----------------|-------------|----------------|---------------|-------------|\n"]
    for a in arms:
        h = health.get(a)
        if not h:
            continue
        L.append(f"| `{a}` | {h['effective']} / {h['real_slots']} | {h['util']} | "
                 f"{h['entropy']} | {h['rare_lt5']} | {h['intra_cos']} |\n")

    if retrieval:
        L += [f"\n### 4.3 Level 2 — cue-to-song retrieval (N={rl}, independent encoder)\n",
              "| Arm | R@1 | R@5 | R@10 | MRR | median rank ↓ |\n",
              "|-----|-----|-----|------|-----|---------------|\n"]
        for a in arms:
            r = retrieval.get(a)
            if r:
                L.append(f"| `{a}` | {r['r1']} | {r['r5']} | {r['r10']} | {r['mrr']} | {r['median']} |\n")

    L += ["\n### 4.4 Level 3 — reconstruction\n",
          "Not run per-arm here (LLM cost). Run for the leading arms:\n",
          f"\n```\nrun_compare.py --rank-by <arm> --limit {rl} --min-df {args.min_df} "
          f"--dedup-threshold {args.dedup_threshold} --level3\n```\n",
          "\n## 5. Verdict\n",
          "| Axis | Winner | Note |\n|------|--------|------|\n",
          "| Stability | — | highest Jaccard in 4.1 |\n",
          "| Utilisation / health | — | highest utilisation, fewest rare cues, lowest intra-cos |\n",
          "| Retrieval | — | highest MRR in 4.3 |\n",
          "\n**Recommended default:** —\n"]

    cue_io.atomic_write_text(path, "".join(L))
    print(f"\n[rank] report -> {path}")


if __name__ == "__main__":
    main()
