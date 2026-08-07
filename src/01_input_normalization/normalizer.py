"""01_input_normalization/normalizer.py — WP-A: Context Prefix Construction.

**Owner:** Student 1 (WP-A)

Goal
----
Convert any raw user input into a clean, fixed-length ContextPrefix:

    user_input  →  normalize()  →  ContextPrefix(item_ids=[m1, ..., mK])

The output is consumed by 03_backbone_recommender (diffusion inference)
and 04_synthesis (verbalization style summary).

Input types handled
-------------------
  'song_only'  : list of item_id strings already in the catalog
  'text_only'  : a natural-language query (song title, artist, description)
  'hybrid'     : dict with keys 'text' and 'item_ids'
  'padded'     : fewer than K items — expand by retrieval
  auto-trim    : more than K items — keep most representative K

Interface contract
------------------
  Input  : see normalize() docstring
  Output : ContextPrefix from 00_data_schema/schema.py
           - item_ids : list[str], all valid catalog IDs, length K
           - source   : one of 'song_only', 'text_only', 'hybrid', 'padded'
  Invariant: every item_id in the output must exist in the backbone's
             clhe_token.json (verified by the backbone tokenizer).

Implementation roadmap (see TODO.md)
-------------------------------------
  Week 1 : song_only and text_only baselines
  Week 2 : hybrid, too-few expansion, too-many selection, deduplication
  Week 3 : compare retrieval strategies; export 20+ examples; write report
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '00_data_schema'))
from schema import CatalogItem, ContextPrefix  # noqa: E402

import numpy as np
from typing import Union


# ---------------------------------------------------------------------------
# Catalog loader
# ---------------------------------------------------------------------------

def load_catalog_from_dict(catalog_path: str) -> list[CatalogItem]:
    """Compatibility alias for the shared dict/list catalog loader."""
    return CatalogItem.load_catalog(catalog_path)


# ---------------------------------------------------------------------------
# Input type detection
# ---------------------------------------------------------------------------

def identify_input_type(user_input) -> str:
    """Detect which input mode the user provided.

    Returns one of: 'song_only', 'text_only', 'hybrid', 'unknown'.
    """
    if isinstance(user_input, dict):
        return 'hybrid'
    if isinstance(user_input, str):
        return 'text_only'
    if isinstance(user_input, (list, tuple)) and all(
        isinstance(x, (str, int)) for x in user_input
    ):
        return 'song_only'
    return 'unknown'


# ---------------------------------------------------------------------------
# Song-only path
# ---------------------------------------------------------------------------

def select_items(
    item_ids: list[str],
    catalog_ids: set[str],
    K: int,
    catalog_embs: np.ndarray | None = None,
    catalog_id_list: list[str] | None = None,
) -> list[str]:
    """Filter, deduplicate, and select K items from a candidate list.

    Steps:
      1. Filter out items not in catalog (unknown IDs).
      2. Deduplicate while preserving order.
      3. If len < K: flag as needing expansion (returned as-is; caller pads).
      4. If len > K: select most representative K items.
         - With embeddings + id list: greedy max-min distance coverage.
         - Without embeddings:        take first K (deterministic).

    Parameters
    ----------
    item_ids       : raw item ID list from user.
    catalog_ids    : set of all valid item IDs in the backbone catalog.
    K              : target context length.
    catalog_embs   : optional (N, d) matrix for coverage-based selection.
    catalog_id_list: ordered list of N item IDs matching catalog_embs rows.

    Returns
    -------
    list[str]: filtered and selected item IDs, length <= K.
    """
    seen = set()
    valid = []
    for iid in (str(x) for x in item_ids):
        if iid in catalog_ids and iid not in seen:
            valid.append(iid)
            seen.add(iid)

    if len(valid) <= K:
        return valid  # too-few: caller decides whether to pad

    # Too many: select K most representative.
    if catalog_embs is not None and catalog_id_list is not None:
        return _greedy_coverage_select(valid, catalog_embs, catalog_id_list, K)
    return valid[:K]


def _greedy_coverage_select(
    item_ids: list[str],
    catalog_embs: np.ndarray,
    catalog_id_list: list[str],
    K: int,
) -> list[str]:
    """Select K items that maximally cover the embedding space (greedy max-min).

    Algorithm:
      1. Seed with item_ids[0].
      2. Repeatedly pick the candidate whose minimum cosine distance to all
         already-selected items is largest (max-min greedy).
      3. Repeat until K items selected.

    Uses cosine distance (1 - cosine_similarity) as the distance metric.
    """
    if len(item_ids) <= K:
        return item_ids[:K]

    id_to_row = {iid: i for i, iid in enumerate(catalog_id_list)}
    missing = [iid for iid in item_ids if iid not in id_to_row]
    if missing:
        raise ValueError(f"Candidate IDs missing from embedding row order: {missing[:5]}")
    rows = [id_to_row[iid] for iid in item_ids]

    embs = catalog_embs[rows].astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    embs_n = embs / norms  # (n, d) L2-normalised

    selected = [0]
    remaining = list(range(1, len(rows)))

    while len(selected) < K and remaining:
        sel_embs = embs_n[selected]   # (n_sel, d)
        rem_embs = embs_n[remaining]  # (n_rem, d)
        # cosine similarity of each remaining candidate to every selected item
        sims = rem_embs @ sel_embs.T  # (n_rem, n_sel)
        # min cosine distance to nearest selected item  (1 − max_similarity)
        min_dists = 1.0 - sims.max(axis=1)  # (n_rem,)
        best = int(np.argmax(min_dists))
        selected.append(remaining[best])
        remaining.pop(best)

    return [item_ids[i] for i in selected]


# ---------------------------------------------------------------------------
# Text-only path
# ---------------------------------------------------------------------------

def embed_text_query(text: str, model=None) -> np.ndarray:
    """Embed a text query into the catalog embedding space.

    Parameters
    ----------
    text  : raw user query string.
    model : sentence encoder with .encode(text, normalize_embeddings=True) → ndarray.

    Returns
    -------
    np.ndarray: 1-D L2-normalised embedding vector.
    """
    if model is None:
        raise NotImplementedError(
            "embed_text_query requires an encoder model. "
            "Pass a SentenceTransformer or equivalent as `model`."
        )
    vec = model.encode(text, normalize_embeddings=True)
    return np.array(vec, dtype=np.float32)


def retrieve_by_embedding(
    query_emb: np.ndarray,
    catalog_embs: np.ndarray,
    catalog_ids: list[str],
    K: int,
) -> list[str]:
    """Retrieve top-K catalog items by cosine similarity to query_emb.

    Parameters
    ----------
    query_emb   : (d,) query embedding.
    catalog_embs: (N, d) catalog embedding matrix (L2-normalised rows preferred).
    catalog_ids : ordered list of N item IDs matching catalog_embs rows.
    K           : number of items to retrieve.

    Returns
    -------
    list[str]: top-K item IDs sorted by descending similarity.
    """
    query_emb = np.asarray(query_emb, dtype=np.float32)
    catalog_embs = np.asarray(catalog_embs, dtype=np.float32)
    if query_emb.ndim != 1:
        raise ValueError(f"query_emb must be 1-D, got shape {query_emb.shape}")
    if catalog_embs.ndim != 2:
        raise ValueError(f"catalog_embs must be 2-D, got shape {catalog_embs.shape}")
    if catalog_embs.shape[0] != len(catalog_ids):
        raise ValueError(
            f"catalog_embs has {catalog_embs.shape[0]} rows but catalog_ids has {len(catalog_ids)} entries"
        )
    if catalog_embs.shape[1] != query_emb.shape[0]:
        raise ValueError(
            f"Embedding dimension mismatch: query={query_emb.shape[0]}, catalog={catalog_embs.shape[1]}"
        )
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if not np.isfinite(query_emb).all() or not np.isfinite(catalog_embs).all():
        raise ValueError("Embeddings must contain only finite values")
    if np.linalg.norm(query_emb) <= 1e-12:
        raise ValueError("query_emb must be non-zero")

    q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    C = catalog_embs / (np.linalg.norm(catalog_embs, axis=1, keepdims=True) + 1e-9)
    sims = C @ q
    top_k = np.argsort(sims)[::-1][:K]
    return [catalog_ids[i] for i in top_k]


# ---------------------------------------------------------------------------
# Expansion (too-few songs)
# ---------------------------------------------------------------------------

def expand_to_K(
    item_ids: list[str],
    catalog_embs: np.ndarray,
    catalog_ids: list[str],
    K: int,
) -> list[str]:
    """Expand a too-short item list to length K by retrieving similar songs.

    Strategy:
      1. Compute the centroid of the input items' embeddings.
      2. Retrieve the nearest neighbors to that centroid in catalog_embs.
      3. Exclude items already in item_ids.
      4. Append until length == K.

    Parameters
    ----------
    item_ids    : current (short) list of catalog IDs.
    catalog_embs: (N, d) catalog embedding matrix; rows indexed by catalog_ids order.
    catalog_ids : ordered list of N item IDs matching catalog_embs rows.
    K           : target length.

    Returns
    -------
    list[str]: item_ids padded to length K (or unchanged if already >= K).
    """
    if len(item_ids) >= K:
        return item_ids[:K]

    id_to_row = {iid: i for i, iid in enumerate(catalog_ids)}
    rows = [id_to_row[iid] for iid in item_ids if iid in id_to_row]
    if not rows:
        return item_ids

    # Centroid of input embeddings
    input_embs = catalog_embs[rows].astype(np.float32)
    centroid = input_embs.mean(axis=0)

    # Retrieve K + len(item_ids) candidates to have enough after filtering
    n_needed = K - len(item_ids)
    candidates = retrieve_by_embedding(
        centroid, catalog_embs, catalog_ids, K + len(item_ids)
    )

    included = set(item_ids)
    additions = [iid for iid in candidates if iid not in included][:n_needed]

    return item_ids + additions


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def normalize(
    user_input: Union[str, list, dict],
    catalog_metadata: list[CatalogItem],
    catalog_embs: np.ndarray | None = None,
    K: int = 5,
    text_encoder=None,
) -> ContextPrefix:
    """Convert raw user input into a standardized ContextPrefix.

    Parameters
    ----------
    user_input:
        One of:
          str                — text query ("upbeat indie songs for a road trip")
          str                — single item ID if the string is all-numeric
          list[str | int]    — one or more item IDs
          dict               — {'text': str, 'item_ids': list[str|int]}  (hybrid)
    catalog_metadata:
        List of CatalogItem for the full catalog (from 00_data_schema).
    catalog_embs:
        Optional (N, d) np.ndarray. Required for text-only, expansion, and hybrid.
        Row order must match catalog_metadata insertion order.
        For text_only: pass catalog_text_embs.npy or catalog_clap_embs.npy.
        For hybrid/expansion: any embedding space works (same space used for all ops).
    K:
        Target reference-set length. Must be at least two for next-song generation.
    text_encoder:
        Encoder with .encode(text, normalize_embeddings=True) → np.ndarray.
        Required for text-only and hybrid inputs.

    Returns
    -------
    ContextPrefix with:
        item_ids : list[str], all valid catalog IDs
        source   : 'song_only' | 'text_only' | 'hybrid' | 'padded'
        items    : list[CatalogItem] matching item_ids

    Raises
    ------
    NotImplementedError : if text_encoder is required but not provided.
    ValueError          : if input is empty or contains no resolvable items.
    """
    if K < 2:
        raise ValueError(f"Next-song generation requires K >= 2 references, got {K}")
    if not catalog_metadata:
        raise ValueError("catalog_metadata must not be empty")

    catalog_id_list = [item.item_id for item in catalog_metadata]
    if K > len(catalog_id_list):
        raise ValueError(f"K={K} exceeds catalog size {len(catalog_id_list)}")
    if len(set(catalog_id_list)) != len(catalog_id_list):
        raise ValueError("catalog_metadata contains duplicate item IDs")
    if catalog_embs is not None:
        catalog_embs = np.asarray(catalog_embs, dtype=np.float32)
        if catalog_embs.ndim != 2 or catalog_embs.shape[0] != len(catalog_metadata):
            raise ValueError(
                "catalog_embs must be a 2-D per-item matrix with one row per catalog item; "
                f"got {catalog_embs.shape} for {len(catalog_metadata)} items"
            )

    catalog_id_set  = set(catalog_id_list)
    id_to_item      = {item.item_id: item for item in catalog_metadata}

    input_type = identify_input_type(user_input)

    # --- song-only ---
    if input_type == 'song_only':
        raw_ids = [str(x) for x in user_input]
        selected = select_items(raw_ids, catalog_id_set, K, catalog_embs, catalog_id_list)
        if len(selected) == 0:
            raise ValueError("No valid catalog items found in input.")
        source = 'song_only'
        if len(selected) < K:
            source = 'padded'
            if catalog_embs is not None:
                selected = expand_to_K(selected, catalog_embs, catalog_id_list, K)

    # --- text-only ---
    elif input_type == 'text_only':
        if not user_input.strip():
            raise ValueError("Text-only input must not be blank")
        # Allow a bare numeric string to be treated as a single item ID
        if user_input.strip().isdigit() and user_input.strip() in catalog_id_set:
            selected = [user_input.strip()]
            source = 'padded'
            if catalog_embs is not None:
                selected = expand_to_K(selected, catalog_embs, catalog_id_list, K)
        else:
            if catalog_embs is None:
                raise ValueError("catalog_embs is required for text-only input.")
            query_emb = embed_text_query(user_input, model=text_encoder)
            selected = retrieve_by_embedding(query_emb, catalog_embs, catalog_id_list, K)
            source = 'text_only'

    # --- hybrid ---
    elif input_type == 'hybrid':
        text_query   = user_input.get('text', '')
        raw_song_ids = [str(x) for x in user_input.get('item_ids', [])]

        if not text_query and not raw_song_ids:
            raise ValueError("Hybrid input must contain at least 'text' or 'item_ids'.")
        if catalog_embs is None:
            raise ValueError("catalog_embs is required for hybrid input.")

        # Seed songs are always guaranteed in the output (dedup, preserve order)
        seen_seeds: set[str] = set()
        valid_seeds: list[str] = []
        for iid in (str(x) for x in raw_song_ids):
            if iid in catalog_id_set and iid not in seen_seeds:
                valid_seeds.append(iid)
                seen_seeds.add(iid)

        if len(valid_seeds) >= K:
            # More seeds than K — diversity-select K from seeds only
            selected = _greedy_coverage_select(valid_seeds, catalog_embs, catalog_id_list, K)
        else:
            # Fill exactly the remaining slots with text retrieval
            n_needed = K - len(valid_seeds)
            additions: list[str] = []
            if text_query:
                # Over-retrieve to absorb any seed duplicates in the result list
                candidates = retrieve_by_embedding(
                    embed_text_query(text_query, model=text_encoder),
                    catalog_embs, catalog_id_list,
                    n_needed + len(valid_seeds),
                )
                additions = [iid for iid in candidates if iid not in seen_seeds][:n_needed]
            selected = valid_seeds + additions

            # A hybrid request containing only seed IDs still has a well-defined
            # completion: expand from their embedding centroid.
            if len(selected) < K and selected:
                selected = expand_to_K(selected, catalog_embs, catalog_id_list, K)

        if not selected:
            raise ValueError("No valid catalog items found in hybrid input.")

        source = 'hybrid'

    else:
        raise ValueError(
            f"Unrecognized input type: {type(user_input)}. "
            "Expected str, list[str|int], or dict."
        )

    items = [id_to_item[iid] for iid in selected if iid in id_to_item]
    ctx = ContextPrefix(
        item_ids=selected,
        source=source,
        raw_input=str(user_input),
        items=items,
    )
    ctx.validate()
    if len(ctx.item_ids) != K:
        raise ValueError(
            f"Could not construct a full context prefix: expected {K}, got {len(ctx.item_ids)}")
    return ctx
