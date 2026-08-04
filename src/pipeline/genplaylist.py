"""pipeline/genplaylist.py — End-to-end GenPlaylist pipeline coordinator.

This module wires all four Work Packages into a single callable:

    raw reference input → GenPlaylistPipeline.generate() → SynthesisResult

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
    from pipeline import GenPlaylistPipeline
    pipeline = GenPlaylistPipeline.from_environment()
    result = pipeline.generate(["42", "17", "83"])
    print(result.audio_path, result.music_attributes)
"""

from __future__ import annotations

import importlib.util
import importlib
import functools
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

_SRC = os.path.dirname(os.path.dirname(__file__))   # .../src/
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, '00_data_schema'))
from schema import (  # noqa: E402
    CUE_VOCAB_SIZE,
    CatalogItem,
    ContextPrefix,
    GeneratedItem,
    SynthesisResult,
)
from shared.artifacts import (  # noqa: E402
    CatalogArtifacts,
    build_item_id_to_row,
    load_catalog_artifacts,
    validate_catalog_alignment,
)

import numpy as np


# ---------------------------------------------------------------------------
# Catalog assets (loaded once at module import; replaced by real paths at runtime)
# ---------------------------------------------------------------------------

_catalog_embs: np.ndarray | None = None       # shape (N, CLHE_EMB_DIM)
_catalog_metadata: list[CatalogItem] | None = None
_item_id_to_row: dict[str, int] | None = None


def _load_catalog(
    catalog_emb_path: str,
    catalog_metadata_path: str,
    item_id_to_row_path: str,
) -> None:
    """Load validated per-item CLHE catalog artifacts into the module cache.

    ``catalog_emb_path`` must be an N x 64 per-item matrix. RVQ codebook weights
    are a different artifact and are rejected by the alignment validator.
    """
    global _catalog_embs, _catalog_metadata, _item_id_to_row
    artifacts = load_catalog_artifacts(
        catalog_metadata_path,
        catalog_emb_path,
        item_id_to_row_path,
    )
    _catalog_embs = artifacts.item_embeddings
    _catalog_metadata = artifacts.items
    _item_id_to_row = artifacts.item_id_to_row


# ---------------------------------------------------------------------------
# Module import helper (handles numerically-prefixed directories)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
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
    item_id_to_row: dict[str, int],
) -> list[GeneratedItem]:
    """Run the configured WP-D inference adapter.

    The bundled runtime loads ``GENPLAYLIST_BACKBONE_CKPT`` lazily. Advanced
    deployments may replace it with ``GENPLAYLIST_BACKBONE_RUNNER=module:function``.
    The coordinator remains importable on CPU machines and never falls back to
    fake or random generation.
    """
    runner_path = os.environ.get("GENPLAYLIST_BACKBONE_RUNNER", "")
    if runner_path:
        if ":" not in runner_path:
            raise ValueError(
                "GENPLAYLIST_BACKBONE_RUNNER must use 'module:function' format")
        module_name, function_name = runner_path.split(":", 1)
        runner = getattr(importlib.import_module(module_name), function_name)
    else:
        runtime = _import_from("03_backbone_recommender", "backbone_runtime")
        runner = runtime.run_backbone
    return runner(
        context_prefix, n_samples, catalog_embs, catalog_metadata, item_id_to_row)


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------

def generate(
    context_prefix: ContextPrefix,
    n_samples: int = 3,
    audio_duration: int = 240,
    k_neighbors: int = 5,
    catalog_embs: np.ndarray | None = None,
    catalog_metadata: list[CatalogItem] | None = None,
    item_id_to_row: dict[str, int] | None = None,
    cue_vocab: list[str] | None = None,
    backbone_runner: Callable | None = None,
    llm_call: Callable | None = None,
    synthesizer: Callable | None = None,
    synthesis_output_dir: str | None = None,
) -> list[SynthesisResult]:
    """Research API: draw alternative candidates for the same next-song slot.

    Every candidate is one item; ``n_samples`` never changes the target length.
    Product/demo code should call :func:`generate_next_song`.

    Parameters
    ----------
    context_prefix  : standardized playlist context (from WP-A or directly).
    n_samples       : how many independent candidates to draw.
    audio_duration  : song length in seconds for ACE-Step (maximum 240).
    k_neighbors     : kNN neighborhood size for verbalization.
    catalog_embs    : (N, d) CLHE embedding matrix; falls back to module cache.
    catalog_metadata: list[CatalogItem]; falls back to module cache.

    Returns
    -------
    list[SynthesisResult], one per sample.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if audio_duration <= 0:
        raise ValueError(f"audio_duration must be positive, got {audio_duration}")
    embs = catalog_embs if catalog_embs is not None else _catalog_embs
    meta = catalog_metadata if catalog_metadata is not None else _catalog_metadata
    mapping = item_id_to_row if item_id_to_row is not None else _item_id_to_row

    if embs is None or meta is None:
        raise ValueError(
            "Catalog not loaded. Call _load_catalog() first, or pass "
            "catalog_embs and catalog_metadata directly to generate()."
        )

    context_prefix.validate()
    if len(context_prefix.item_ids) < 2:
        raise ValueError("Next-song generation requires at least two reference music items")
    if mapping is None:
        mapping = build_item_id_to_row(meta)
    validate_catalog_alignment(meta, np.asarray(embs), mapping)
    missing_context = [item_id for item_id in context_prefix.item_ids if item_id not in mapping]
    if missing_context:
        raise ValueError(f"Context item IDs missing from catalog artifacts: {missing_context[:5]}")

    verbalization_mod = _import_from("04_synthesis", "verbalization")
    synthesis_mod     = _import_from("04_synthesis", "synthesis")
    verbalize  = verbalization_mod.verbalize
    synthesize = synthesis_mod.synthesize

    # Step 1: backbone diffusion → n_samples next-item embeddings
    runner = backbone_runner or _run_backbone
    generated_items = runner(context_prefix, n_samples, embs, meta, mapping)
    if len(generated_items) != n_samples:
        raise ValueError(
            f"Backbone returned {len(generated_items)} candidates; expected {n_samples}")
    for sample_idx, item in enumerate(generated_items):
        if not isinstance(item, GeneratedItem):
            raise TypeError(
                f"Backbone candidate {sample_idx} is {type(item).__name__}, not GeneratedItem")
        item.validate()

    # Step 2: verbalize + synthesize each candidate
    results = []
    for item in generated_items:
        verb = verbalize(
            item, embs, meta, k=k_neighbors, cue_vocab=cue_vocab,
            llm_call=llm_call)
        synth_call = synthesizer or synthesize
        synthesis_kwargs = {}
        if synthesis_output_dir is not None:
            synthesis_kwargs["output_dir"] = synthesis_output_dir
        audio_path = synth_call(
            music_attributes=verb["music_attributes"],
            lyric_draft=verb["lyric_draft"],
            audio_duration=audio_duration,
            **synthesis_kwargs,
        )
        result = SynthesisResult(
            audio_path=audio_path,
            music_attributes=verb["music_attributes"],
            lyric_draft=verb["lyric_draft"],
            neighbors=verb["neighbors"],
            style_summary=verb["style_summary"],
            generated_item=item,
        )
        result.validate()
        results.append(result)

    return results


def generate_next_song(
    context_prefix: ContextPrefix,
    **kwargs,
) -> SynthesisResult:
    """Generate exactly one next song from an ordered set of reference music."""
    if "n_samples" in kwargs:
        raise TypeError("generate_next_song() always predicts one item; omit n_samples")
    results = generate(context_prefix, n_samples=1, **kwargs)
    if len(results) != 1:
        raise RuntimeError(f"Expected exactly one next song, got {len(results)}")
    return results[0]


class GenPlaylistPipeline:
    """Reusable WP-A → WP-D → WP-C next-song pipeline.

    Catalog artifacts are loaded and validated once. Heavy DDBC and ACE-Step
    models remain lazy inside their adapters, so constructing this object does
    not require CUDA or a checkpoint to be loaded immediately.
    """

    def __init__(
        self,
        artifacts: CatalogArtifacts,
        *,
        cue_vocab: list[str] | None = None,
        retrieval_embeddings: np.ndarray | None = None,
        text_encoder=None,
        backbone_runner: Callable | None = None,
        llm_call: Callable | None = None,
        synthesizer: Callable | None = None,
        synthesis_output_dir: str | None = None,
    ):
        artifacts.validate()
        if cue_vocab is not None:
            if len(cue_vocab) != CUE_VOCAB_SIZE:
                raise ValueError(
                    f"cue_vocab must contain {CUE_VOCAB_SIZE} entries, "
                    f"got {len(cue_vocab)}")
            if cue_vocab[0] != "<unk>":
                raise ValueError("cue_vocab[0] must be '<unk>'")
        if retrieval_embeddings is None:
            retrieval_embeddings = artifacts.item_embeddings
        retrieval_embeddings = np.asarray(retrieval_embeddings, dtype=np.float32)
        if retrieval_embeddings.ndim != 2 or (
                retrieval_embeddings.shape[0] != len(artifacts.items)):
            raise ValueError(
                "retrieval_embeddings must be [catalog_items, dimension], got "
                f"{retrieval_embeddings.shape}")
        if not np.isfinite(retrieval_embeddings).all():
            raise ValueError("retrieval_embeddings contain NaN or infinity")

        self.artifacts = artifacts
        self.cue_vocab = cue_vocab
        self.retrieval_embeddings = retrieval_embeddings
        self.text_encoder = text_encoder
        self.backbone_runner = backbone_runner
        self.llm_call = llm_call
        self.synthesizer = synthesizer
        self.synthesis_output_dir = synthesis_output_dir

    @classmethod
    def from_files(
        cls,
        *,
        catalog_metadata_path: str | Path,
        catalog_embeddings_path: str | Path,
        item_id_to_row_path: str | Path,
        cue_vocab_path: str | Path | None = None,
        retrieval_embeddings_path: str | Path | None = None,
        **kwargs,
    ) -> "GenPlaylistPipeline":
        """Load and validate all lightweight startup artifacts from disk."""
        artifacts = load_catalog_artifacts(
            catalog_metadata_path, catalog_embeddings_path, item_id_to_row_path)
        cue_vocab = None
        if cue_vocab_path is not None:
            with Path(cue_vocab_path).open("r", encoding="utf-8") as handle:
                cue_vocab = json.load(handle)
            if not isinstance(cue_vocab, list):
                raise ValueError("cue_vocab.json must contain a JSON list")
        retrieval_embeddings = None
        if retrieval_embeddings_path is not None:
            retrieval_embeddings = np.load(
                retrieval_embeddings_path, allow_pickle=False)
        return cls(
            artifacts,
            cue_vocab=cue_vocab,
            retrieval_embeddings=retrieval_embeddings,
            **kwargs,
        )

    @classmethod
    def from_environment(cls, **kwargs) -> "GenPlaylistPipeline":
        """Construct from the repository defaults plus ``GENPLAYLIST_*`` paths."""
        repo_root = Path(__file__).resolve().parents[2]
        dataset_dir = Path(os.environ.get(
            "GENPLAYLIST_DATASET_DIR", repo_root / "data" / "dataset"))
        cue_default = (
            repo_root / "src" / "02_creative_cues" / "outputs" /
            "production" / "latest" / "cue_vocab.json")
        return cls.from_files(
            catalog_metadata_path=os.environ.get(
                "GENPLAYLIST_CATALOG_METADATA",
                dataset_dir / "catalog_metadata.json"),
            catalog_embeddings_path=os.environ.get(
                "GENPLAYLIST_CATALOG_EMBEDDINGS",
                dataset_dir / "catalog_item_embeddings.npy"),
            item_id_to_row_path=os.environ.get(
                "GENPLAYLIST_ITEM_ID_TO_ROW",
                dataset_dir / "item_id_to_row.json"),
            cue_vocab_path=os.environ.get(
                "GENPLAYLIST_CUE_VOCAB", cue_default),
            **kwargs,
        )

    def normalize_references(
        self,
        user_input,
        *,
        reference_count: int | None = None,
        user_instruction: str = "",
        text_encoder=None,
    ) -> ContextPrefix:
        """Run WP-A and return an ordered, catalog-grounded reference set."""
        if isinstance(user_input, ContextPrefix):
            context = user_input.validate()
            metadata_by_id = {
                item.item_id: item for item in self.artifacts.items}
            missing = [item_id for item_id in context.item_ids
                       if item_id not in metadata_by_id]
            if missing:
                raise ValueError(
                    f"Reference IDs missing from catalog: {missing[:5]}")
            context.items = [metadata_by_id[item_id]
                             for item_id in context.item_ids]
        else:
            if reference_count is None:
                if isinstance(user_input, (list, tuple)):
                    reference_count = len(user_input)
                elif isinstance(user_input, dict) and user_input.get("item_ids"):
                    reference_count = max(2, len(user_input["item_ids"]))
                else:
                    reference_count = 5
            normalizer = _import_from("01_input_normalization", "normalizer")
            context = normalizer.normalize(
                user_input,
                self.artifacts.items,
                self.retrieval_embeddings,
                K=reference_count,
                text_encoder=text_encoder or self.text_encoder,
            )
        # A reference query and a creative instruction are different signals.
        # Only the explicit instruction is forwarded into WP-C prompts.
        context.raw_input = user_instruction.strip()
        return context.validate()

    def generate(
        self,
        user_input,
        *,
        reference_count: int | None = None,
        user_instruction: str = "",
        text_encoder=None,
        audio_duration: int = 30,
        k_neighbors: int = 5,
        backbone_runner: Callable | None = None,
        llm_call: Callable | None = None,
        synthesizer: Callable | None = None,
        synthesis_output_dir: str | None = None,
    ) -> SynthesisResult:
        """Run WP-A → DDBC full-mask inference → WP-C → ACE-Step."""
        context = self.normalize_references(
            user_input,
            reference_count=reference_count,
            user_instruction=user_instruction,
            text_encoder=text_encoder,
        )
        return generate_next_song(
            context,
            audio_duration=audio_duration,
            k_neighbors=k_neighbors,
            catalog_embs=self.artifacts.item_embeddings,
            catalog_metadata=self.artifacts.items,
            item_id_to_row=self.artifacts.item_id_to_row,
            cue_vocab=self.cue_vocab,
            backbone_runner=backbone_runner or self.backbone_runner,
            llm_call=llm_call or self.llm_call,
            synthesizer=synthesizer or self.synthesizer,
            synthesis_output_dir=(
                synthesis_output_dir
                if synthesis_output_dir is not None
                else self.synthesis_output_dir),
        )

    def preflight(self) -> dict:
        """Return a lightweight, serializable startup summary after validation."""
        self.artifacts.validate()
        return {
            "catalog_items": len(self.artifacts.items),
            "catalog_embedding_shape": list(
                self.artifacts.item_embeddings.shape),
            "retrieval_embedding_shape": list(
                self.retrieval_embeddings.shape),
            "cue_vocab_loaded": self.cue_vocab is not None,
            "backbone": "injected" if self.backbone_runner else "checkpoint_runtime",
            "llm": "injected" if self.llm_call else "openai_runtime",
            "synthesizer": "injected" if self.synthesizer else "ace_step_runtime",
            "task": "multiple_references_to_one_next_song",
            "sampler": "ddbc_full_mask_completion",
        }
