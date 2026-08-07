#!/usr/bin/env python3
"""Convert legacy server-side CLHE/RVQ files into GenPlaylist-v1 artifacts.

The common legacy layout is::

    clhe.pt             full per-item CLHE table (often indexed by numeric ID)
    clhe_weight.npy     merged 3 x 256 RVQ codebook
    clhe_token.json     item ID -> [z0, z1, z2, conflict] global token IDs

This script never guesses that sparse numeric IDs are rows unless
``--legacy-dense-numeric-ids`` is explicitly supplied. Prefer a source mapping
when one exists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from shared.artifacts import build_item_id_to_row, sha256_file  # noqa: E402
from shared.schema import CLHE_EMB_DIM, SCHEMA_VERSION, CatalogItem, TOKEN_LAYOUT  # noqa: E402


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=path.parent, suffix=".npy")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise


def load_embedding_source(path: Path):
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            f"Reading {path.name} requires PyTorch; run this in the server training environment") from exc
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only was introduced
        value = torch.load(path, map_location="cpu")
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        # Some exports wrap the tensor under a descriptive key.
        for key in ("embeddings", "features", "weight"):
            candidate = value.get(key)
            if hasattr(candidate, "detach"):
                return candidate.detach().cpu().numpy()
            if isinstance(candidate, np.ndarray):
                return candidate
    raise ValueError(
        f"Unsupported embedding object in {path}; expected Tensor, ndarray, or a dict "
        "containing embeddings/features/weight")


def validate_semantic_tokens(raw: dict, item_ids: list[str]) -> dict[str, list[int]]:
    missing = [item_id for item_id in item_ids if item_id not in raw]
    if missing:
        raise ValueError(
            f"Semantic-token source is missing {len(missing)} catalog IDs: {missing[:10]}")
    output = {}
    for item_id in item_ids:
        tokens = [int(value) for value in raw[item_id]]
        if len(tokens) != TOKEN_LAYOUT.rq_n_codebooks + 1:
            raise ValueError(f"Item {item_id} has {len(tokens)} semantic tokens, expected 4")
        for level, token in enumerate(tokens[:TOKEN_LAYOUT.rq_n_codebooks]):
            start = TOKEN_LAYOUT.rvq_token(level, 0)
            if not start <= token < start + TOKEN_LAYOUT.rq_codebook_size:
                raise ValueError(f"Item {item_id} has invalid RVQ level-{level} token {token}")
        conflict_start = TOKEN_LAYOUT.conflict_token(0)
        conflict_raw = tokens[-1] - conflict_start
        if not 0 <= conflict_raw < TOKEN_LAYOUT.conflict_vocab_size:
            raise ValueError(
                f"Item {item_id} has conflict token {tokens[-1]} (raw={conflict_raw}); "
                f"GenPlaylist-v1 currently allows 0..{TOKEN_LAYOUT.conflict_vocab_size - 1}. "
                "Do not truncate it—inspect the server artifact and update the shared schema if needed.")
        output[item_id] = tokens
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument(
        "--catalog", type=Path,
        default=REPO_ROOT / "data" / "dataset" / "catalog_metadata.json")
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "data" / "dataset")
    parser.add_argument("--source-embeddings", type=Path)
    parser.add_argument("--source-codebook", type=Path)
    parser.add_argument("--source-semantic-tokens", type=Path)
    parser.add_argument("--source-item-id-to-row", type=Path)
    parser.add_argument(
        "--legacy-dense-numeric-ids", action="store_true",
        help="Explicitly assert that row int(item_id) in the legacy full table is correct")
    args = parser.parse_args()

    source_embeddings = args.source_embeddings or args.legacy_dir / "clhe.pt"
    source_codebook = args.source_codebook or args.legacy_dir / "clhe_weight.npy"
    source_tokens = args.source_semantic_tokens or args.legacy_dir / "clhe_token.json"
    for path in (args.catalog, source_embeddings, source_codebook, source_tokens):
        if not path.is_file():
            raise FileNotFoundError(path)

    items = CatalogItem.load_catalog(str(args.catalog))
    item_ids = [item.item_id for item in items]
    output_mapping = build_item_id_to_row(items)
    source_matrix = np.asarray(load_embedding_source(source_embeddings))
    if source_matrix.ndim != 2 or source_matrix.shape[1] != CLHE_EMB_DIM:
        raise ValueError(
            f"Per-item source must be [N,{CLHE_EMB_DIM}], got {source_matrix.shape}. "
            "A (768,64) file is probably the RVQ codebook, not item embeddings.")

    if args.source_item_id_to_row:
        source_mapping = {
            str(key): int(value)
            for key, value in json.loads(
                args.source_item_id_to_row.read_text(encoding="utf-8")).items()
        }
        missing = [item_id for item_id in item_ids if item_id not in source_mapping]
        if missing:
            raise ValueError(f"Source mapping is missing catalog IDs: {missing[:10]}")
        source_rows = [source_mapping[item_id] for item_id in item_ids]
        selection_mode = "explicit-source-mapping"
    elif args.legacy_dense_numeric_ids:
        if any(not item_id.isdigit() for item_id in item_ids):
            raise ValueError("Legacy numeric-row mode requires every catalog ID to be numeric")
        source_rows = [int(item_id) for item_id in item_ids]
        selection_mode = "explicit-legacy-int-item-id"
    else:
        raise ValueError(
            "Supply --source-item-id-to-row, or explicitly confirm the old dense table "
            "with --legacy-dense-numeric-ids")
    if min(source_rows) < 0 or max(source_rows) >= len(source_matrix):
        raise ValueError(
            f"Requested source rows {min(source_rows)}..{max(source_rows)} outside "
            f"embedding table with {len(source_matrix)} rows")
    selected = np.asarray(source_matrix[source_rows], dtype=np.float32)
    if selected.shape != (len(items), CLHE_EMB_DIM) or not np.isfinite(selected).all():
        raise ValueError(f"Selected embeddings are invalid: {selected.shape}")

    codebook = np.load(source_codebook, allow_pickle=False).astype(np.float32)
    expected_codebook = (
        TOKEN_LAYOUT.rq_n_codebooks * TOKEN_LAYOUT.rq_codebook_size, CLHE_EMB_DIM)
    if codebook.shape != expected_codebook or not np.isfinite(codebook).all():
        raise ValueError(
            f"RVQ codebook must be finite {expected_codebook}, got {codebook.shape}")
    raw_tokens = json.loads(source_tokens.read_text(encoding="utf-8"))
    semantic_tokens = validate_semantic_tokens(raw_tokens, item_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedding_out = args.output_dir / "catalog_item_embeddings.npy"
    codebook_out = args.output_dir / "rvq_codebook_weights.npy"
    tokens_out = args.output_dir / "semantic_tokens.json"
    mapping_out = args.output_dir / "item_id_to_row.json"
    atomic_npy(embedding_out, selected)
    atomic_npy(codebook_out, codebook)
    atomic_json(tokens_out, semantic_tokens)
    atomic_json(mapping_out, output_mapping)

    outputs = [embedding_out, codebook_out, tokens_out, mapping_out]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection_mode": selection_mode,
        "catalog_items": len(items),
        "source": {
            "embeddings": str(source_embeddings),
            "embeddings_shape": list(source_matrix.shape),
            "codebook": str(source_codebook),
            "semantic_tokens": str(source_tokens),
        },
        "outputs": {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in outputs
        },
    }
    manifest_out = args.output_dir / "wpd_artifact_manifest.json"
    atomic_json(manifest_out, manifest)

    print(f"[prepare] catalog embeddings : {embedding_out} {selected.shape}")
    print(f"[prepare] RVQ codebook       : {codebook_out} {codebook.shape}")
    print(f"[prepare] semantic tokens    : {tokens_out} ({len(semantic_tokens)} items)")
    print(f"[prepare] row mapping        : {mapping_out}")
    print(f"[prepare] manifest           : {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
