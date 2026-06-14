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
) -> list[str]:
    """Filter, deduplicate, and select K items from a candidate list.

    Steps:
      1. Filter out items not in catalog (unknown IDs).
      2. Deduplicate while preserving order.
      3. If len < K: flag as needing expansion (returned as-is; caller pads).
      4. If len > K: select most representative K items.
         - Without embeddings: take first K (deterministic).
         - With embeddings: pick K items maximizing coverage
           (greedy max-min distance; TODO Week 2).

    Parameters
    ----------
    item_ids   : raw item ID list from user.
    catalog_ids: set of all valid item IDs in the backbone catalog.
    K          : target context length.
    catalog_embs: optional (N, d) matrix for coverage-based selection.

    Returns
    -------
    list[str]: filtered and selected item IDs, length <= K.
    """
    # Filter and deduplicate
    seen = set()
    valid = []
    for iid in (str(x) for x in item_ids):
        if iid in catalog_ids and iid not in seen:
            valid.append(iid)
            seen.add(iid)

    if len(valid) <= K:
        return valid  # too-few case: caller handles expansion

    # Too many: select K most representative
    # TODO Week 2: replace with coverage-maximizing greedy selection
    if catalog_embs is not None:
        return _greedy_coverage_select(valid, catalog_embs, K)
    return valid[:K]


def _greedy_coverage_select(
    item_ids: list[str],
    catalog_embs: np.ndarray,
    K: int,
) -> list[str]:
    """Select K items that maximally cover the embedding space (greedy max-min).

    TODO (WP-A Week 2): implement this properly.
    Placeholder: returns first K items.
    """
    # TODO: greedy max-min distance selection
    return item_ids[:K]


# ---------------------------------------------------------------------------
# Text-only path
# ---------------------------------------------------------------------------

def embed_text_query(text: str, model=None) -> np.ndarray:
    """Embed a text query into the catalog embedding space.

    Parameters
    ----------
    text  : raw user query string.
    model : optional sentence encoder (e.g. SentenceTransformer).
            If None, raises NotImplementedError — caller must provide a model.

    Returns
    -------
    np.ndarray: 1-D embedding vector compatible with catalog_embs.

    TODO (WP-A Week 1): integrate a baseline sentence encoder.
    Strategy options:
      - SentenceTransformer('all-MiniLM-L6-v2') as baseline
      - CLAP audio-text encoder for better audio-text alignment
      - Match into CLHE space via learned linear projection (Week 2)
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
    catalog_embs: (N, d) catalog embedding matrix.
    catalog_ids : ordered list of N item IDs matching catalog_embs rows.
    K           : number of items to retrieve.

    Returns
    -------
    list[str]: top-K item IDs sorted by descending similarity.

    TODO (WP-A Week 2): replace numpy cosine with faiss IndexFlatIP
    for catalogs > 100k items.
    """
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

    Strategy: compute centroid of input embeddings, retrieve nearest neighbors,
    exclude already-included items.

    TODO (WP-A Week 2): implement this.
    Placeholder: returns input list unchanged.
    """
    if len(item_ids) >= K:
        return item_ids[:K]
    # TODO: compute centroid → retrieve nearest neighbors → deduplicate → pad
    return item_ids


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
        Optional (N, d) np.ndarray of CLHE embeddings. Required for text-only
        and expansion paths. N must equal len(catalog_metadata).
    K:
        Target context prefix length.
    text_encoder:
        Optional text/sentence encoder with .encode(text) → np.ndarray.
        Required for text-only and hybrid inputs.

    Returns
    -------
    ContextPrefix with:
        item_ids : list[str], all valid catalog IDs
        source   : 'song_only' | 'text_only' | 'hybrid' | 'padded'
        items    : list[CatalogItem] matching item_ids

    Raises
    ------
    NotImplementedError:
        For input types that haven't been implemented yet (Week 1/2).
    ValueError:
        If user_input is empty or contains no resolvable items.
    """
    catalog_id_list = [item.item_id for item in catalog_metadata]
    catalog_id_set  = set(catalog_id_list)
    id_to_item      = {item.item_id: item for item in catalog_metadata}

    input_type = identify_input_type(user_input)

    # --- song-only ---
    if input_type == 'song_only':
        raw_ids = [str(x) for x in user_input]
        selected = select_items(raw_ids, catalog_id_set, K, catalog_embs)
        if len(selected) == 0:
            raise ValueError("No valid catalog items found in input.")
        source = 'song_only'
        if len(selected) < K:
            source = 'padded'
            if catalog_embs is not None:
                selected = expand_to_K(selected, catalog_embs, catalog_id_list, K)

    # --- text-only ---
    elif input_type == 'text_only':
        # Allow a bare numeric string to be treated as a single item ID
        if user_input.strip().isdigit() and user_input.strip() in catalog_id_set:
            selected = [user_input.strip()]
            source = 'song_only'
            if catalog_embs is not None:
                selected = expand_to_K(selected, catalog_embs, catalog_id_list, K)
                source = 'padded'
        else:
            if catalog_embs is None:
                raise ValueError(
                    "catalog_embs is required for text-only input."
                )
            query_emb = embed_text_query(user_input, model=text_encoder)
            selected = retrieve_by_embedding(query_emb, catalog_embs, catalog_id_list, K)
            source = 'text_only'

    # --- hybrid ---
    elif input_type == 'hybrid':
        # TODO (WP-A Week 2): implement hybrid path
        raise NotImplementedError(
            "Hybrid input (text + item_ids) is a WP-A Week 2 deliverable.\n"
            "Implement: combine text-retrieval results with song items, "
            "then select K most representative."
        )

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
    return ctx
