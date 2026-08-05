"""pipeline.py — shared extract -> clean -> assign -> export orchestration.

Used by both run_production.py (single method, no evaluation) and
run_compare.py (multi-method comparison + evaluation). Evaluation itself
(coverage stats aside) lives in cue_eval.py / run_compare.py — this module only
builds the vocabulary and the per-song cue mapping, nothing downstream of that.

Moved out of run_compare.py so a second entry point (run_production.py) doesn't
have to duplicate it. The one behavior change from the old run_compare.py
version: `out_dir` is now an explicit parameter instead of being read off a
mutable module-level RUN_DIR global.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cue_extractors   # noqa: E402
import cue_normalize    # noqa: E402
import cue_assign       # noqa: E402
import cue_export       # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_data_schema"))
from schema import CUE_CANDIDATES_PER_ITEM  # noqa: E402


def finish_from_raw(
    method: str,
    raw: dict[str, list[str]],
    items,
    lyrics_proc: dict[str, str],
    out_dir: str,
    *,
    min_df: int = 5,
    max_df_frac: float = 0.3,
    dedup_threshold: float = 0.92,
    rank_by: str = "idf",
    num_cues: int = CUE_CANDIDATES_PER_ITEM,
    vocab_size: int = 2048,
    embedder: str = cue_normalize.DEFAULT_EMBEDDER,
    assignment_strategy: str = "relevance",
    candidate_k: int = 64,
    export_scores: bool = False,
    block_tokens: Optional[set[str]] = None,
    verbose: bool = True,
):
    """normalize -> assign -> export. Returns (vocab, item2cues, norm_stats, cue_emb).

    dedup_threshold: cosine above which two cues are merged in semantic dedup
                     (higher = fewer merges = larger vocabulary).
    num_cues: ranked candidates stored per song. Production stores 16 and WP-C
              consumes the first 8; experimental values still validate when
              readers are told the matching stored width.
    embedder: sentence-transformers model (or a cue_normalize.EMBEDDER_MODELS key)
              used for BOTH semantic dedup and assignment relevance. Passed to
              both build_vocab_normalized and assign_all here so cues and songs
              always share one vector space — never let the two calls default
              independently.
    """
    norm = cue_normalize.build_vocab_normalized(
        raw, vocab_size=vocab_size, min_df=min_df, max_df_frac=max_df_frac,
        dedup_threshold=dedup_threshold, rank_by=rank_by, embedder=embedder,
        block_tokens=block_tokens, verbose=verbose)
    vocab, cue_emb = norm["vocab"], norm["embeddings"]
    assigned = cue_assign.assign_all(
        items, lyrics_proc, vocab, cue_emb, n_cues=num_cues,
        candidate_k=candidate_k, strategy=assignment_strategy,
        embedder=embedder, verbose=verbose, return_scores=export_scores)
    if export_scores:
        item2cues, score_mapping = assigned
    else:
        item2cues, score_mapping = assigned, None
    cue_export.export_outputs(
        vocab, item2cues, out_dir, score_mapping=score_mapping,
        assignment_metadata={
            "strategy": assignment_strategy,
            "score": "cosine_similarity",
            "ordering": (
                "relevance_desc_then_cue_id_asc"
                if assignment_strategy == "relevance" else "greedy_mmr"),
            "candidate_k": candidate_k,
            "embedder": embedder,
            "fallback": "expand_to_n_then_unk_tail",
        })
    return vocab, item2cues, norm["stats"], cue_emb


def build_vocab_and_assign(
    method: str,
    items,
    lyrics_proc: dict[str, str],
    out_dir: str,
    *,
    force: bool = False,
    top_n: int = 40,
    cache_tag: str = "",
    vocab_items=None,
    min_df: int = 5,
    max_df_frac: float = 0.3,
    dedup_threshold: float = 0.92,
    rank_by: str = "idf",
    num_cues: int = CUE_CANDIDATES_PER_ITEM,
    vocab_size: int = 2048,
    embedder: str = cue_normalize.DEFAULT_EMBEDDER,
    assignment_strategy: str = "relevance",
    candidate_k: int = 64,
    export_scores: bool = False,
    block_tokens: Optional[set[str]] = None,
    verbose: bool = True,
):
    """extract -> normalize -> assign -> export. Returns (vocab, item2cues, norm_stats, cue_emb).

    vocab_items: corpus used for vocabulary building (raw cue extraction + df/idf/dedup).
    Defaults to `items`. Pass a train-only split for held-out evaluation so the
    vocabulary never sees held-out test songs; cues are still assigned to every
    item in `items` (item2cues.json stays the full deliverable either way).

    embedder: see finish_from_raw — the single dedup+assignment embedder choice.
    """
    if verbose:
        print(f"\n########## METHOD: {method} ##########")
    vocab_source = items if vocab_items is None else vocab_items
    raw = cue_extractors.extract_raw_cues(vocab_source, lyrics_proc, method=method, force=force,
                                          top_n=top_n, cache_tag=cache_tag)
    return finish_from_raw(
        method, raw, items, lyrics_proc, out_dir,
        min_df=min_df, max_df_frac=max_df_frac, dedup_threshold=dedup_threshold,
        rank_by=rank_by, num_cues=num_cues, vocab_size=vocab_size, embedder=embedder,
        assignment_strategy=assignment_strategy, candidate_k=candidate_k,
        export_scores=export_scores,
        block_tokens=block_tokens, verbose=verbose)
