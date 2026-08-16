#!/usr/bin/env python3
"""Evaluate generated audio against frozen histories in MERT space."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from shared.artifacts import sha256_file  # noqa: E402
from evaluate_mert_proxy import _bootstrap_mean_interval  # noqa: E402
from extract_end_to_end_mert import EXPECTED_EXAMPLES, SYSTEMS, _slug  # noqa: E402


def _load_test_sequences(prepared_dir: Path) -> list[list[str]]:
    from datasets import load_from_disk

    values = [[str(item) for item in row] for row in
              load_from_disk(str(prepared_dir / "raw_dataset"))["test"]["item_seq"]]
    if len(values) != EXPECTED_EXAMPLES or any(len(row) != 20 for row in values):
        raise ValueError("Frozen end-to-end histories must have shape [941, 20]")
    return values


def _diversity(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine distance without materializing an NxN matrix."""
    count = len(embeddings)
    if count < 2:
        raise ValueError("Diversity needs at least two embeddings")
    cosine_sum = float(np.square(embeddings.sum(axis=0)).sum() - count)
    mean_cosine = cosine_sum / (count * (count - 1))
    return 1.0 - mean_cosine


def _maximum_catalog_similarity(
    generated: np.ndarray, catalog: np.ndarray, batch_size: int = 64,
) -> np.ndarray:
    values = []
    for start in range(0, len(generated), batch_size):
        values.append((generated[start:start + batch_size] @ catalog.T).max(axis=1))
    return np.concatenate(values).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--generated-mert-dir", type=Path, required=True)
    parser.add_argument("--catalog-mert-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepared_dir = args.prepared_dir.expanduser().resolve()
    generated_dir = args.generated_mert_dir.expanduser().resolve()
    catalog_dir = args.catalog_mert_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    sequences = _load_test_sequences(prepared_dir)

    catalog_manifest_path = catalog_dir / "mert_manifest.json"
    catalog_manifest = json.loads(catalog_manifest_path.read_text(encoding="utf-8"))
    catalog_path = catalog_dir / "catalog_mert_embeddings_l2.npy"
    ids_path = catalog_dir / "catalog_item_ids.json"
    if catalog_manifest["outputs"][catalog_path.name] != sha256_file(catalog_path):
        raise ValueError("Catalog MERT embedding hash mismatch")
    if catalog_manifest["outputs"][ids_path.name] != sha256_file(ids_path):
        raise ValueError("Catalog MERT ID hash mismatch")
    catalog = np.load(catalog_path, allow_pickle=False).astype(np.float32)
    item_ids = [str(item) for item in json.loads(ids_path.read_text(encoding="utf-8"))]
    item_to_row = {item: row for row, item in enumerate(item_ids)}
    reference_rows = np.asarray([
        [item_to_row[item] for item in row[:15]] for row in sequences], dtype=np.int64)
    next_rows = np.asarray([item_to_row[row[15]] for row in sequences], dtype=np.int64)

    results = {}
    artifacts = {}
    for system in SYSTEMS:
        slug = _slug(system)
        embedding_path = generated_dir / f"{slug}_mert_embeddings_l2.npy"
        manifest_path = generated_dir / f"{slug}_mert_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model_revision") != catalog_manifest.get("model_revision"):
            raise ValueError(f"MERT model revision differs for {system}")
        if manifest.get("output_sha256") != sha256_file(embedding_path):
            raise ValueError(f"Generated MERT hash mismatch for {system}")
        generated = np.load(embedding_path, allow_pickle=False).astype(np.float32)
        if generated.shape != (EXPECTED_EXAMPLES, catalog.shape[1]):
            raise ValueError(f"Generated MERT shape differs for {system}: {generated.shape}")
        if not np.allclose(np.linalg.norm(generated, axis=1), 1.0, atol=1e-4):
            raise ValueError(f"Generated MERT embeddings are not normalized for {system}")
        history_fit = np.einsum(
            "bd,bkd->bk", generated, catalog[reference_rows]).mean(axis=1)
        next_similarity = np.einsum("bd,bd->b", generated, catalog[next_rows])
        reference_max = np.einsum(
            "bd,bkd->bk", generated, catalog[reference_rows]).max(axis=1)
        catalog_max = _maximum_catalog_similarity(generated, catalog)
        per_history = {
            "history_fit": history_fit,
            "next_similarity": next_similarity,
            "reference_max_similarity": reference_max,
            "catalog_max_similarity": catalog_max,
        }
        results[system] = {
            "metrics": {
                name: float(values.mean()) for name, values in per_history.items()
            } | {"cross_history_diversity": _diversity(generated)},
            "confidence_intervals_95": {
                name: _bootstrap_mean_interval(
                    values, samples=args.bootstrap_samples, seed=args.bootstrap_seed)
                for name, values in per_history.items()
            },
        }
        artifacts[system] = {
            "manifest_sha256": sha256_file(manifest_path),
            "embedding_sha256": sha256_file(embedding_path),
        }

    payload = {
        "result_schema": "genplaylist-end-to-end-mert-eval-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "examples": EXPECTED_EXAMPLES,
        "reference_items": 15,
        "target_item": 16,
        "catalog_items": len(catalog),
        "catalog_mert_manifest_sha256": sha256_file(catalog_manifest_path),
        "generated_artifacts": artifacts,
        "systems": results,
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.bootstrap_seed},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
