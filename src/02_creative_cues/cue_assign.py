"""cue_assign.py — WP-B Phase 3: per-song cue assignment (paper §4.2 step 4).

Assigns CUE_TOKENS=6 cues to each song from a normalized vocabulary, using
semantic relevance + diversity regularization (the practical realization of the
paper's "PMI with diversity regularization that enforces pairwise semantic
distance"):

  relevance(song, cue) = cosine(song_embedding, cue_embedding)
  selection            = greedy MMR over the top-K relevant cues:
                           argmax_c  lambda * rel(c) - (1-lambda) * max_{s in S} cos(c, s)

This reuses the cue embeddings already produced by cue_normalize (no recompute)
and embeds song text once with the SAME backend, so cues and songs share a space.

Output is a {item_id: CueMappingEntry} dict — the exact format cue_mining.export_outputs
and CueMappingEntry.load_mapping() consume, so the new pipeline plugs straight into
the existing schema contract.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_data_schema"))
from schema import CatalogItem, CueMappingEntry, CUE_TOKENS  # noqa: E402

import cue_normalize  # noqa: E402

UNK_CUE_ID = 0


def _song_text(item: CatalogItem, lyrics: str, lyrics_cap: int = 0) -> str:
    # Artist is excluded so cue assignment is driven by song content, not identity.
    # lyrics arrive already preprocessed upstream (cue_lyrics); cap=0 = no re-truncation.
    meta = " ".join(p for p in [item.title, item.genre, item.mood,
                                item.lyric_excerpt, *item.tags] if p)
    if lyrics:
        return f"{meta} {lyrics[:lyrics_cap] if lyrics_cap else lyrics}"
    return meta


def assign_all(
    catalog: list[CatalogItem],
    lyrics_dict: Optional[dict[str, str]],
    vocab: list[str],
    cue_embeddings: np.ndarray,
    n_cues: int = CUE_TOKENS,
    lam: float = 0.7,
    candidate_k: int = 40,
    embed_fn: Optional[Callable[[list[str]], np.ndarray]] = None,
    verbose: bool = True,
) -> dict[str, CueMappingEntry]:
    """Assign n_cues per song. Returns {item_id: CueMappingEntry} (validated).

    vocab[0] == '<unk>'; cue_embeddings[k] aligns with vocab[k+1] (L2-normalized).
    Padding rows (vocab '<pad_*'') have zero embeddings and are skipped.
    """
    lyrics_dict = lyrics_dict or {}

    # valid (non-pad, non-zero) cue indices into cue_embeddings
    cue_norms = np.linalg.norm(cue_embeddings, axis=1)
    valid_mask = np.array([
        (cue_norms[k] > 0) and not vocab[k + 1].startswith("<pad_")
        for k in range(len(cue_embeddings))
    ])
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        # no usable cues — everyone gets <unk>
        return {it.item_id: CueMappingEntry(it.item_id, [UNK_CUE_ID] * n_cues)
                for it in catalog}
    valid_emb = cue_embeddings[valid_idx]              # (V, d), normalized

    # embed all songs once with the same backend used for the vocab
    if embed_fn is None:
        embed_fn, backend = cue_normalize._make_embedder()
        if verbose:
            print(f"[assign] song embedder: {backend}")
    texts = [_song_text(it, lyrics_dict.get(it.item_id, "")) for it in catalog]
    song_emb = embed_fn(texts)                          # (N, d), normalized

    result: dict[str, CueMappingEntry] = {}
    for r, it in enumerate(catalog):
        rel_all = valid_emb @ song_emb[r]               # cosine to every valid cue
        # restrict to the top-K most relevant cues as MMR candidates
        k = min(candidate_k, len(valid_idx))
        cand_local = np.argpartition(rel_all, -k)[-k:]  # indices into valid_idx
        cand_local = cand_local[np.argsort(rel_all[cand_local])[::-1]]

        selected_local: list[int] = []
        for _ in range(min(n_cues, len(cand_local))):
            best_score, best = -1e9, None
            for c in cand_local:
                if c in selected_local:
                    continue
                if selected_local:
                    sims = valid_emb[selected_local] @ valid_emb[c]
                    diversity_pen = float(sims.max())
                else:
                    diversity_pen = 0.0
                score = lam * float(rel_all[c]) - (1 - lam) * diversity_pen
                if score > best_score:
                    best_score, best = score, c
            if best is None:
                break
            selected_local.append(best)

        # map back: valid_idx[local] -> cue_embeddings row -> vocab index (+1)
        cue_ids = [int(valid_idx[c]) + 1 for c in selected_local]
        while len(cue_ids) < n_cues:
            cue_ids.append(UNK_CUE_ID)
        entry = CueMappingEntry(item_id=it.item_id, cue_ids=cue_ids[:n_cues])
        entry.validate()
        result[it.item_id] = entry

        if verbose and (r + 1) % 1000 == 0:
            print(f"[assign] {r + 1}/{len(catalog)}")

    return result
