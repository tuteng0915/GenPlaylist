"""Versioned catalog artifact loading and alignment checks.

This module deliberately distinguishes per-item embeddings from RVQ codebook
weights. Passing a 768-row codebook where an N-row catalog matrix is expected
now fails at startup instead of silently grounding songs against the wrong rows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schema import CLHE_EMB_DIM, RQ_CODEBOOK_SIZE, RQ_N_CODEBOOKS, SCHEMA_VERSION, CatalogItem


@dataclass(frozen=True)
class CatalogArtifacts:
    items: list[CatalogItem]
    item_embeddings: np.ndarray
    item_id_to_row: dict[str, int]
    schema_version: str = SCHEMA_VERSION

    def validate(self, embedding_dim: int = CLHE_EMB_DIM) -> "CatalogArtifacts":
        validate_catalog_alignment(
            self.items,
            self.item_embeddings,
            self.item_id_to_row,
            embedding_dim=embedding_dim,
        )
        return self


def build_item_id_to_row(items: list[CatalogItem]) -> dict[str, int]:
    """Build a deterministic row mapping from catalog order."""
    mapping: dict[str, int] = {}
    for row, item in enumerate(items):
        item_id = str(item.item_id)
        if item_id in mapping:
            raise ValueError(f"Duplicate catalog item ID: {item_id}")
        mapping[item_id] = row
    return mapping


def load_item_id_to_row(path: str | Path) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("item_id_to_row must be a JSON object")
    mapping = {str(item_id): int(row) for item_id, row in raw.items()}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("item_id_to_row contains duplicate row indices")
    return mapping


def validate_catalog_alignment(
    items: list[CatalogItem],
    item_embeddings: np.ndarray,
    item_id_to_row: dict[str, int],
    *,
    embedding_dim: int = CLHE_EMB_DIM,
) -> None:
    if item_embeddings.ndim != 2:
        raise ValueError(f"Catalog embeddings must be 2-D, got {item_embeddings.shape}")
    expected_shape = (len(items), embedding_dim)
    if item_embeddings.shape != expected_shape:
        codebook_rows = RQ_N_CODEBOOKS * RQ_CODEBOOK_SIZE
        hint = ""
        if item_embeddings.shape[0] == codebook_rows:
            hint = " This looks like RVQ codebook weights, not per-item embeddings."
        raise ValueError(
            f"Catalog embedding shape mismatch: expected {expected_shape}, "
            f"got {item_embeddings.shape}.{hint}"
        )
    if not np.isfinite(item_embeddings).all():
        raise ValueError("Catalog embeddings contain NaN or infinity")
    if len(item_id_to_row) != len(items):
        raise ValueError(
            f"item_id_to_row has {len(item_id_to_row)} entries for {len(items)} catalog items"
        )

    expected_ids = [item.item_id for item in items]
    if set(item_id_to_row) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(item_id_to_row))[:5]
        extra = sorted(set(item_id_to_row) - set(expected_ids))[:5]
        raise ValueError(f"item_id_to_row ID mismatch; missing={missing}, extra={extra}")

    rows = sorted(item_id_to_row.values())
    if rows != list(range(len(items))):
        raise ValueError("item_id_to_row rows must be a contiguous permutation of 0..N-1")

    for item in items:
        expected_row = item_id_to_row[item.item_id]
        if item.feature_index not in (-1, expected_row):
            raise ValueError(
                f"Catalog item {item.item_id} has feature_index={item.feature_index}, "
                f"but mapping says {expected_row}"
            )
        item.feature_index = expected_row


def load_catalog_artifacts(
    catalog_metadata_path: str | Path,
    item_embeddings_path: str | Path,
    item_id_to_row_path: str | Path,
) -> CatalogArtifacts:
    items = CatalogItem.load_catalog(str(catalog_metadata_path))
    item_embeddings = np.load(item_embeddings_path, allow_pickle=False).astype(np.float32)
    item_id_to_row = load_item_id_to_row(item_id_to_row_path)
    return CatalogArtifacts(items, item_embeddings, item_id_to_row).validate()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
