#!/usr/bin/env python3
"""Extract CLHE/RVQ artifacts embedded in an official DDBC checkpoint.

The official Spotify checkpoints serialize their legacy ``MDLMTokenizer`` in
``hyper_parameters.tokenizer``.  That object contains the full CLHE item table,
the three RVQ codebooks, and the global semantic-token mapping.  This utility
selects the current catalog rows and writes the explicit GenPlaylist artifacts
without retraining CLHE or guessing a sparse row mapping.

Only trusted checkpoints should be supplied: loading a Lightning checkpoint
requires Python pickle deserialization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_ROOT))

from prepare_server_artifacts import (  # noqa: E402
    atomic_json,
    atomic_npy,
    validate_semantic_tokens,
)
from shared.artifacts import build_item_id_to_row, sha256_file  # noqa: E402
from shared.schema import CLHE_EMB_DIM, SCHEMA_VERSION, CatalogItem, TOKEN_LAYOUT  # noqa: E402


def build_artifact_arrays(
    embedded_tokenizer,
    items: list[CatalogItem],
    *,
    confirm_dense_item_ids: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]], dict[str, int]]:
    """Validate and select artifacts from a serialized DDBC tokenizer."""
    features = np.asarray(embedded_tokenizer.feature)
    codebook = np.asarray(embedded_tokenizer.weight)
    raw_tokens = {
        str(item_id): [int(token) for token in tokens]
        for item_id, tokens in embedded_tokenizer.token.items()
    }

    if features.ndim != 2 or features.shape[1] != CLHE_EMB_DIM:
        raise ValueError(
            f"Embedded CLHE table must be [N,{CLHE_EMB_DIM}], got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Embedded CLHE table contains NaN or infinity")
    expected_codebook = (
        TOKEN_LAYOUT.rq_n_codebooks * TOKEN_LAYOUT.rq_codebook_size,
        CLHE_EMB_DIM,
    )
    if codebook.shape != expected_codebook or not np.isfinite(codebook).all():
        raise ValueError(
            f"Embedded RVQ codebook must be finite {expected_codebook}, got {codebook.shape}")

    # In the official Spotify tokenizer, the full item IDs are the dense source
    # row IDs ("0" ... str(N-1)). Require both an exhaustive proof and an
    # explicit CLI acknowledgement before using that relationship.
    dense_ids = {str(index) for index in range(len(features))}
    if set(raw_tokens) != dense_ids:
        missing = sorted(dense_ids - set(raw_tokens))[:10]
        extra = sorted(set(raw_tokens) - dense_ids)[:10]
        raise ValueError(
            "Checkpoint token IDs are not an exhaustive dense feature-row mapping; "
            f"missing={missing}, extra={extra}")
    if not confirm_dense_item_ids:
        raise ValueError(
            "Pass --confirm-dense-item-ids after verifying that the official DDBC "
            "tokenizer uses numeric item ID i for CLHE feature row i")

    item_ids = [item.item_id for item in items]
    if any(not item_id.isdigit() for item_id in item_ids):
        raise ValueError("Dense checkpoint extraction requires numeric catalog item IDs")
    source_rows = [int(item_id) for item_id in item_ids]
    if source_rows and (min(source_rows) < 0 or max(source_rows) >= len(features)):
        raise ValueError(
            f"Catalog rows {min(source_rows)}..{max(source_rows)} fall outside "
            f"the checkpoint CLHE table with {len(features)} rows")

    selected = np.asarray(features[source_rows], dtype=np.float32)
    if selected.shape != (len(items), CLHE_EMB_DIM) or not np.isfinite(selected).all():
        raise ValueError(f"Selected catalog embeddings are invalid: {selected.shape}")
    semantic_tokens = validate_semantic_tokens(raw_tokens, item_ids)
    row_mapping = build_item_id_to_row(items)
    return selected, codebook.astype(np.float32), semantic_tokens, row_mapping


def load_embedded_tokenizer(checkpoint_path: Path, legacy_module_dir: Path):
    """Load the trusted Lightning checkpoint and return its serialized tokenizer."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not (legacy_module_dir / "tokenizer.py").is_file():
        raise FileNotFoundError(
            f"Legacy tokenizer module not found under {legacy_module_dir}")
    sys.path.insert(0, str(legacy_module_dir))
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Checkpoint extraction requires PyTorch") from exc
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        embedded = checkpoint["hyper_parameters"]["tokenizer"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Checkpoint does not contain hyper_parameters.tokenizer") from exc
    for field in ("feature", "weight", "token"):
        if not hasattr(embedded, field):
            raise ValueError(f"Embedded tokenizer is missing {field!r}")
    return embedded


def write_artifacts(
    output_dir: Path,
    selected: np.ndarray,
    codebook: np.ndarray,
    semantic_tokens: dict[str, list[int]],
    row_mapping: dict[str, int],
    *,
    source_manifest: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "catalog_item_embeddings.npy": output_dir / "catalog_item_embeddings.npy",
        "rvq_codebook_weights.npy": output_dir / "rvq_codebook_weights.npy",
        "semantic_tokens.json": output_dir / "semantic_tokens.json",
        "item_id_to_row.json": output_dir / "item_id_to_row.json",
    }
    atomic_npy(paths["catalog_item_embeddings.npy"], selected)
    atomic_npy(paths["rvq_codebook_weights.npy"], codebook)
    atomic_json(paths["semantic_tokens.json"], semantic_tokens)
    atomic_json(paths["item_id_to_row.json"], row_mapping)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection_mode": "checkpoint-validated-dense-item-ids",
        "catalog_items": len(row_mapping),
        "source": source_manifest,
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    manifest_path = output_dir / "wpd_artifact_manifest.json"
    atomic_json(manifest_path, manifest)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--catalog", type=Path,
        default=REPO_ROOT / "data" / "dataset" / "catalog_metadata.json")
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "data" / "dataset")
    parser.add_argument(
        "--legacy-module-dir", type=Path,
        default=REPO_ROOT / "src" / "03_backbone_recommender")
    parser.add_argument(
        "--confirm-dense-item-ids", action="store_true",
        help="explicitly acknowledge the validated official Spotify ID-to-row contract")
    args = parser.parse_args()

    items = CatalogItem.load_catalog(str(args.catalog))
    embedded = load_embedded_tokenizer(args.checkpoint, args.legacy_module_dir)
    selected, codebook, semantic_tokens, row_mapping = build_artifact_arrays(
        embedded, items, confirm_dense_item_ids=args.confirm_dense_item_ids)
    manifest_path = write_artifacts(
        args.output_dir,
        selected,
        codebook,
        semantic_tokens,
        row_mapping,
        source_manifest={
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_clhe_shape": list(np.asarray(embedded.feature).shape),
            "checkpoint_codebook_shape": list(np.asarray(embedded.weight).shape),
            "checkpoint_semantic_items": len(embedded.token),
            "legacy_special_tokens": {
                "bos": int(embedded.bos_token),
                "boi": int(embedded.boi_token),
                "eos": int(embedded.eos_token),
            },
        },
    )
    print(f"[extract] catalog embeddings : {args.output_dir / 'catalog_item_embeddings.npy'} {selected.shape}")
    print(f"[extract] RVQ codebook       : {args.output_dir / 'rvq_codebook_weights.npy'} {codebook.shape}")
    print(f"[extract] semantic tokens    : {len(semantic_tokens)} items")
    print(f"[extract] manifest           : {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
