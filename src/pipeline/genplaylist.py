"""pipeline/genplaylist.py — End-to-end GenPlaylist pipeline coordinator.

This module wires all four Work Packages into a single callable:

    ContextPrefix  →  generate()  →  list[SynthesisResult]

It is the only place where cross-WP imports happen.  Each WP can be
developed and tested in isolation; this file is updated as each WP
becomes ready.

Full pipeline (paper §4):

    C = (m1,...,mt)
      │
      ▼  playlist_structure.compute_playlist_structure()   [backbone_recommender]
    (μ_C, σ²_C)
      │
      ▼  backbone_recommender diffusion model              [backbone_recommender]
    z_hat_emb  (next-item CLHE embedding)
      │
      ▼  verbalization.verbalize(GeneratedItem, ...)       [04_synthesis / WP-C]
    music_attributes + lyric_draft
      │
      ▼  synthesis.synthesize()                            [04_synthesis / WP-C]
    audio_path
      │
      ▼  SynthesisResult

Usage
-----
    from pipeline.genplaylist import generate
    ctx = ContextPrefix(item_ids=["42", "17", "83"], source="song_only")
    results = generate(ctx, n_samples=3)
    for r in results:
        print(r.audio_path, r.music_attributes)
"""

from __future__ import annotations

import importlib.util
import os
import sys

_SRC = os.path.dirname(os.path.dirname(__file__))   # .../src/
sys.path.insert(0, os.path.join(_SRC, '00_data_schema'))
from schema import CatalogItem, ContextPrefix, GeneratedItem, SynthesisResult  # noqa: E402

import numpy as np


# ---------------------------------------------------------------------------
# Catalog assets (loaded once at module import; replaced by real paths at runtime)
# ---------------------------------------------------------------------------

_catalog_embs: np.ndarray | None = None       # shape (N, CLHE_EMB_DIM)
_catalog_metadata: list[CatalogItem] | None = None


def _load_catalog(catalog_emb_path: str, catalog_metadata_path: str) -> None:
    """Load CLHE catalog embeddings and CatalogItem list into module-level cache.

    catalog_emb_path     : path to clhe_weight.npy (or a pre-extracted .npy)
    catalog_metadata_path: path to catalog_metadata.json (list of CatalogItem dicts)
    """
    global _catalog_embs, _catalog_metadata
    _catalog_embs = np.load(catalog_emb_path).astype(np.float32)
    _catalog_metadata = CatalogItem.load_catalog(catalog_metadata_path)


# ---------------------------------------------------------------------------
# Module import helper (handles numerically-prefixed directories)
# ---------------------------------------------------------------------------

def _import_from(rel_dir: str, module_name: str):
    """Import a Python module from a numerically-prefixed src/ subdirectory."""
    path = os.path.join(_SRC, rel_dir, module_name + ".py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Backbone inference (WP-D)
# ---------------------------------------------------------------------------

def _run_backbone(
    context_prefix: ContextPrefix,
    n_samples: int,
    catalog_embs: np.ndarray,
    catalog_metadata: list[CatalogItem],
) -> list[GeneratedItem]:
    """Run backbone diffusion model → n_samples GeneratedItem candidates.

    TODO (WP-D Week 2): import backbone_recommender, load checkpoint,
    call diffusion.restore_model_and_semi_ar_sample(), decode tokens
    via MDLMTokenizer._token_to_feature(), compute (μ_C, σ²_C) via
    playlist_structure.compute_playlist_structure().

    Until then: raises NotImplementedError.  Use mock_run.py for testing.
    """
    raise NotImplementedError(
        "backbone inference not yet wired (WP-D Week 2). "
        "Use mock_run.py for end-to-end pipeline testing."
    )


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------

def generate(
    context_prefix: ContextPrefix,
    n_samples: int = 3,
    audio_duration: int = 30,
    k_neighbors: int = 5,
    catalog_embs: np.ndarray | None = None,
    catalog_metadata: list[CatalogItem] | None = None,
) -> list[SynthesisResult]:
    """Generate n_samples next-song candidates for context_prefix.

    Parameters
    ----------
    context_prefix  : standardized playlist context (from WP-A or directly).
    n_samples       : how many independent candidates to draw.
    audio_duration  : clip length in seconds for ACE-Step.
    k_neighbors     : kNN neighborhood size for verbalization.
    catalog_embs    : (N, d) CLHE embedding matrix; falls back to module cache.
    catalog_metadata: list[CatalogItem]; falls back to module cache.

    Returns
    -------
    list[SynthesisResult], one per sample.
    """
    embs = catalog_embs if catalog_embs is not None else _catalog_embs
    meta = catalog_metadata if catalog_metadata is not None else _catalog_metadata

    if embs is None or meta is None:
        raise ValueError(
            "Catalog not loaded. Call _load_catalog() first, or pass "
            "catalog_embs and catalog_metadata directly to generate()."
        )

    context_prefix.validate()

    verbalization_mod = _import_from("04_synthesis", "verbalization")
    synthesis_mod     = _import_from("04_synthesis", "synthesis")
    verbalize  = verbalization_mod.verbalize
    synthesize = synthesis_mod.synthesize

    # Step 1: backbone diffusion → n_samples next-item embeddings
    generated_items = _run_backbone(context_prefix, n_samples, embs, meta)

    # Step 2: verbalize + synthesize each candidate
    results = []
    for item in generated_items:
        verb = verbalize(item, embs, meta, k=k_neighbors)
        audio_path = synthesize(
            music_attributes=verb["music_attributes"],
            lyric_draft=verb["lyric_draft"],
            audio_duration=audio_duration,
        )
        results.append(SynthesisResult(
            audio_path=audio_path,
            music_attributes=verb["music_attributes"],
            lyric_draft=verb["lyric_draft"],
            neighbors=verb["neighbors"],
            style_summary=verb["style_summary"],
            generated_item=item,
        ))

    return results
