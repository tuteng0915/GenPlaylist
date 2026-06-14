"""04_synthesis/verbalization.py — WP-C: Latent Verbalization Pipeline.

**Owner:** Student 3 (WP-C)

Goal
----
Translate a generated CLHE embedding (z_hat) into music_attributes + lyric_draft
for ACE-Step synthesis:

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
  - For S candidate items: share one style_summary, one LLM call per candidate
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '00_data_schema'))
from schema import CatalogItem, GeneratedItem, SynthesisResult  # noqa: E402

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# LLM client (DashScope / Qwen3)
# ---------------------------------------------------------------------------

def _call_qwen3(prompt: str, system: str = "") -> str:
    """Call Qwen3 via DashScope API.

    TODO (WP-C): replace stub with real DashScope call:
        import dashscope
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise EnvironmentError("DASHSCOPE_API_KEY is not set.")
        response = dashscope.Generation.call(
            model="qwen-plus",           # or "qwen3-7b-instruct"
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            api_key=api_key,
        )
        return response.output.text.strip()
    """
    print("[verbalization] _call_qwen3 is a stub — returning placeholder output.")
    return "STUB OUTPUT"


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
    q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    C = catalog_embs / (np.linalg.norm(catalog_embs, axis=1, keepdims=True) + 1e-9)
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
    "You are a music style analyst. Given descriptions of semantically nearby songs "
    "and the global style of a playlist, produce concise music attributes for a new song "
    "that fits into the playlist.\n"
    "Output ONLY a comma-separated list of tags, no other text.\n"
    "Required fields: genre, mood, tempo (e.g. '120 BPM'), instrumentation, key (e.g. 'C major'), language.\n"
    "Example: pop, melancholic, 95 BPM, piano and strings, A minor, English"
)


def generate_music_attributes(
    neighbors: list[CatalogItem],
    style_summary: list[CatalogItem],
    sigma_c2: float,
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

    # Calibration note: threshold 1.0 is a placeholder.
    # TODO: calibrate against Q66 of σ²_C distribution on training set.
    diversity_hint = (
        "The playlist is stylistically diverse — the new song may vary significantly "
        "in mood and instrumentation while still fitting the overall theme."
        if sigma_c2 > 1.0
        else "The playlist is compact — the new song should closely match its style."
    )

    prompt = (
        f"## Playlist style (centroid neighbors)\n{ss_block}\n\n"
        f"## Target position neighbors (nearest to generated embedding)\n{nb_block}\n\n"
        f"## Playlist structure note\n{diversity_hint}\n\n"
        "Generate music attributes (comma-separated tags) for the new song. "
        "Do NOT copy any neighbor song — use them only as style reference."
    )

    return _call_qwen3(prompt, system=_ATTRIBUTE_SYSTEM).strip()


# ---------------------------------------------------------------------------
# Lyric generation
# ---------------------------------------------------------------------------

_LYRICS_SYSTEM = (
    "You are a professional lyricist. Given descriptions of semantically nearby songs "
    "and the overall playlist style, write original lyrics for a new song.\n"
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

    diversity_hint = (
        "The playlist is diverse — feel free to explore different imagery and themes."
        if sigma_c2 > 1.0
        else "The playlist is compact — keep the emotional tone and imagery consistent."
    )

    prompt = (
        f"## Music attributes for the new song\n{music_attributes}\n\n"
        f"## Playlist style (centroid neighbors)\n{ss_block}\n\n"
        f"## Target position neighbors\n{nb_block}\n\n"
        f"## Playlist structure note\n{diversity_hint}\n\n"
        "Write original lyrics in ACE-Step markup format."
    )

    return _call_qwen3(prompt, system=_LYRICS_SYSTEM).strip()


# ---------------------------------------------------------------------------
# End-to-end verbalization convenience wrapper
# ---------------------------------------------------------------------------

def verbalize(
    generated: GeneratedItem,
    catalog_embs: np.ndarray,
    catalog_metadata: list[CatalogItem],
    k: int = 5,
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
    neighbors     = knn_verbalize(generated.z_hat_emb, catalog_embs, catalog_metadata, k)
    style_summary = knn_verbalize(generated.mu_c_emb,  catalog_embs, catalog_metadata, k)

    music_attributes = generate_music_attributes(neighbors, style_summary, generated.sigma_c2)
    lyric_draft      = generate_lyrics(neighbors, style_summary, music_attributes, generated.sigma_c2)

    return {
        "neighbors":        neighbors,
        "style_summary":    style_summary,
        "music_attributes": music_attributes,
        "lyric_draft":      lyric_draft,
    }
