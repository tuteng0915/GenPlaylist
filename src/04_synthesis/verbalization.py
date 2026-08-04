"""04_synthesis/verbalization.py — WP-C: Latent Verbalization Pipeline.

**Owner:** Student 3 (WP-C)

Goal
----
Translate the single DDBC-predicted next-item embedding into attributes and
lyrics for ACE-Step synthesis, conditioned on multiple reference songs:

    GeneratedItem  →  verbalize()  →  {music_attributes, lyric_draft,
                                        neighbors, style_summary}

Pipeline
--------
z_hat_emb  (CLHE vec, dim=64)
  ↓  knn_verbalize()   → nearest catalog songs (neighbors)
μ_C_emb    (playlist centroid)
  ↓  knn_verbalize()   → playlist style context (style_summary)

neighbors + style_summary + σ²_C
  ↓  generate_music_attributes()  →  comma-separated ACE-Step style tags
  ↓  generate_lyrics()            →  ACE-Step markup lyrics

Adapted from VibeMus/assistant.py.

Implementation roadmap
----------------------
  - Replace _call_qwen3() stub with real DashScope SDK call
  - Build faiss IndexFlatIP for kNN over 254k-item catalog
  - Tune σ²_C diversity threshold from training set Q66
  - Cache the reference-centroid style summary across research-time alternatives
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, os.path.dirname(__file__))

from shared.schema import CUE_VOCAB_SIZE, CatalogItem, GeneratedItem

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# LLM client (DashScope / Qwen3)
# ---------------------------------------------------------------------------

def _call_qwen3(prompt: str, system: str = "") -> str:
    """Call OpenAI API for verbalization (replaces DashScope/Qwen)."""
    import os
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://api.openai.com/v1",
    )

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=4096,
    )
    return response.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# kNN catalog lookup
# ---------------------------------------------------------------------------

def knn_verbalize(
    query_emb: np.ndarray,
    catalog_embs: np.ndarray,
    catalog_metadata: list[CatalogItem],
    k: int = 5,
) -> list[CatalogItem]:
    """Return k CatalogItems nearest to query_emb (cosine similarity).

    Parameters
    ----------
    query_emb        : shape (d,) — z_hat_emb or μ_C centroid.
    catalog_embs     : shape (N, d) — CLHE embeddings for all catalog items.
                       N must equal len(catalog_metadata).
    catalog_metadata : N CatalogItems in the same row order as catalog_embs.
    k                : number of neighbors to return.

    Returns
    -------
    list[CatalogItem]: k nearest items, descending similarity.

    TODO (WP-C): replace numpy cosine with faiss IndexFlatIP for large catalogs.
    Normalize catalog_embs once at module load, not per call.
    """
    query_emb = np.asarray(query_emb, dtype=np.float32)
    catalog_embs = np.asarray(catalog_embs, dtype=np.float32)
    if query_emb.ndim != 1 or catalog_embs.ndim != 2:
        raise ValueError(
            f"Expected query [D] and catalog [N,D], got {query_emb.shape}, {catalog_embs.shape}")
    if catalog_embs.shape != (len(catalog_metadata), query_emb.shape[0]):
        raise ValueError(
            f"Catalog alignment mismatch: embeddings={catalog_embs.shape}, "
            f"metadata={len(catalog_metadata)}, query_dim={query_emb.shape[0]}")
    if not 1 <= k <= len(catalog_metadata):
        raise ValueError(f"k must be in 1..{len(catalog_metadata)}, got {k}")
    if not np.isfinite(query_emb).all() or not np.isfinite(catalog_embs).all():
        raise ValueError("kNN embeddings contain NaN or infinity")
    query_norm = np.linalg.norm(query_emb)
    if query_norm <= 1e-12:
        raise ValueError("kNN query embedding must be non-zero")
    q = query_emb / query_norm
    C = catalog_embs / np.maximum(
        np.linalg.norm(catalog_embs, axis=1, keepdims=True), 1e-9)
    sims = C @ q                              # (N,)
    top_k_idx = np.argsort(sims)[::-1][:k]
    return [catalog_metadata[i] for i in top_k_idx]


def _format_neighbor_block(neighbors: list[CatalogItem]) -> str:
    """Format a neighbor list into a readable LLM prompt block."""
    lines = []
    for i, item in enumerate(neighbors, 1):
        lines.append(f"  [{i}] {item.to_prompt_line()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Music attribute generation
# ---------------------------------------------------------------------------

_ATTRIBUTE_SYSTEM = (
    "You are a music style analyst. Given multiple ordered reference songs and a "
    "DDBC-predicted semantic target, describe exactly one original song that would "
    "work as the next track after those references.\n"
    "Output ONLY a comma-separated list of tags, no other text.\n"
    "Required fields: genre, mood, tempo (e.g. '120 BPM'), instrumentation, key (e.g. 'C major'), language.\n"
    "Example: pop, melancholic, 95 BPM, piano and strings, A minor, English"
)


def generate_music_attributes(
    neighbors: list[CatalogItem],
    style_summary: list[CatalogItem],
    sigma_c2: float,
    reference_music: list[CatalogItem] | None = None,
    cue_terms: list[str] | None = None,
    user_instruction: str = "",
    llm_call=None,
) -> str:
    """Generate comma-separated ACE-Step style tags for the next playlist item.

    Parameters
    ----------
    neighbors     : kNN neighbors of z_hat — target semantic position.
    style_summary : kNN neighbors of μ_C  — global playlist style.
    sigma_c2      : playlist dispersion; higher → more thematic latitude.

    Returns
    -------
    str: comma-separated tags, e.g. "indie pop, bittersweet, 108 BPM, guitar, E minor, English"
    """
    nb_block = _format_neighbor_block(neighbors)
    ss_block = _format_neighbor_block(style_summary)
    ref_block = _format_neighbor_block(reference_music or [])

    # Calibration note: threshold 1.0 is a placeholder.
    # TODO: calibrate against Q66 of σ²_C distribution on training set.
    diversity_hint = (
        "The playlist is stylistically diverse — the new song may vary significantly "
        "in mood and instrumentation while still fitting the overall theme."
        if sigma_c2 > 1.0
        else "The playlist is compact — the new song should closely match its style."
    )

    prompt = (
        f"## Ordered reference music\n{ref_block}\n\n"
        f"## Reference-set style anchors (centroid neighbors)\n{ss_block}\n\n"
        f"## Predicted next-song anchors (nearest to the DDBC output)\n{nb_block}\n\n"
        f"## Creative cues\n{', '.join(cue_terms or ['<unk>'])}\n\n"
        f"## User instruction\n{user_instruction or 'No additional instruction.'}\n\n"
        f"## Playlist structure note\n{diversity_hint}\n\n"
        "Generate comma-separated attributes for exactly one next song. "
        "Do not repeat, copy, or name any reference or neighbor song."
    )

    caller = llm_call or _call_qwen3
    return caller(prompt, system=_ATTRIBUTE_SYSTEM).strip()


# ---------------------------------------------------------------------------
# Lyric generation
# ---------------------------------------------------------------------------

_LYRICS_SYSTEM = (
    "You are a professional lyricist. Given multiple ordered reference songs and a "
    "DDBC-predicted semantic target, write original lyrics for exactly one next song.\n"
    "Format rules (ACE-Step markup):\n"
    "  - Start each section on its own line: [verse], [chorus], or [bridge]\n"
    "  - Each sung line on its own line\n"
    "  - Blank line between sections\n"
    "Do NOT copy existing lyrics. Capture their emotional arc and thematic "
    "content while introducing variation."
)


def generate_lyrics(
    neighbors: list[CatalogItem],
    style_summary: list[CatalogItem],
    music_attributes: str,
    sigma_c2: float,
    reference_music: list[CatalogItem] | None = None,
    cue_terms: list[str] | None = None,
    user_instruction: str = "",
    llm_call=None,
) -> str:
    """Generate ACE-Step markup lyrics for the next playlist item.

    Parameters
    ----------
    neighbors        : kNN neighbors of z_hat.
    style_summary    : kNN neighbors of μ_C.
    music_attributes : output of generate_music_attributes().
    sigma_c2         : playlist dispersion.

    Returns
    -------
    str: ACE-Step markup lyric draft, e.g.:
        [verse]
        Staring at the neon rain
        ...
        [chorus]
        ...
    """
    nb_block = _format_neighbor_block(neighbors)
    ss_block = _format_neighbor_block(style_summary)
    ref_block = _format_neighbor_block(reference_music or [])

    diversity_hint = (
        "The playlist is diverse — feel free to explore different imagery and themes."
        if sigma_c2 > 1.0
        else "The playlist is compact — keep the emotional tone and imagery consistent."
    )

    prompt = (
        f"## Music attributes for the new song\n{music_attributes}\n\n"
        f"## Ordered reference music\n{ref_block}\n\n"
        f"## Reference-set style anchors (centroid neighbors)\n{ss_block}\n\n"
        f"## Predicted next-song anchors\n{nb_block}\n\n"
        f"## Creative cues\n{', '.join(cue_terms or ['<unk>'])}\n\n"
        f"## User instruction\n{user_instruction or 'No additional instruction.'}\n\n"
        f"## Playlist structure note\n{diversity_hint}\n\n"
        "Write original lyrics for exactly one next song in ACE-Step markup format."
    )

    caller = llm_call or _call_qwen3
    return caller(prompt, system=_LYRICS_SYSTEM).strip()


# ---------------------------------------------------------------------------
# End-to-end verbalization convenience wrapper
# ---------------------------------------------------------------------------

def verbalize(
    generated: GeneratedItem,
    catalog_embs: np.ndarray,
    catalog_metadata: list[CatalogItem],
    k: int = 5,
    cue_vocab: list[str] | None = None,
    llm_call=None,
) -> dict:
    """Full verbalization: GeneratedItem → attributes + lyrics.

    Parameters
    ----------
    generated        : output from backbone diffusion model.
    catalog_embs     : (N, d) CLHE embedding matrix (d=64 for current backbone).
    catalog_metadata : N CatalogItems in same row order as catalog_embs.
    k                : kNN neighborhood size (paper recommends k=5).

    Returns
    -------
    dict:
        "neighbors"       : list[CatalogItem]  — z_hat kNN
        "style_summary"   : list[CatalogItem]  — μ_C kNN
        "music_attributes": str                — comma-separated tags
        "lyric_draft"     : str                — ACE-Step markup
    """
    generated.validate()
    if cue_vocab is not None:
        if len(cue_vocab) != CUE_VOCAB_SIZE:
            raise ValueError(
                f"cue_vocab must contain {CUE_VOCAB_SIZE} entries, got {len(cue_vocab)}")
        cue_terms = [cue_vocab[cue_id] for cue_id in generated.cue_ids]
    else:
        cue_terms = [f"cue_{cue_id}" for cue_id in generated.cue_ids]
    user_instruction = (
        generated.context_prefix.raw_input.strip()
        if generated.context_prefix is not None else "")

    reference_music = []
    if generated.context_prefix is not None:
        metadata_by_id = {item.item_id: item for item in catalog_metadata}
        missing = [item_id for item_id in generated.context_prefix.item_ids
                   if item_id not in metadata_by_id]
        if missing:
            raise ValueError(f"Reference music missing from catalog metadata: {missing[:5]}")
        reference_music = [metadata_by_id[item_id]
                           for item_id in generated.context_prefix.item_ids]

    neighbors     = knn_verbalize(generated.z_hat_emb, catalog_embs, catalog_metadata, k)
    style_summary = knn_verbalize(generated.mu_c_emb,  catalog_embs, catalog_metadata, k)

    music_attributes = generate_music_attributes(
        neighbors, style_summary, generated.sigma_c2,
        reference_music=reference_music,
        cue_terms=cue_terms, user_instruction=user_instruction, llm_call=llm_call)
    lyric_draft = generate_lyrics(
        neighbors, style_summary, music_attributes, generated.sigma_c2,
        reference_music=reference_music,
        cue_terms=cue_terms, user_instruction=user_instruction, llm_call=llm_call)

    return {
        "neighbors":        neighbors,
        "style_summary":    style_summary,
        "reference_music":  reference_music,
        "music_attributes": music_attributes,
        "lyric_draft":      lyric_draft,
        "cue_terms":        cue_terms,
    }
