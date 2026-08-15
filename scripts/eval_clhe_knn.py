#!/usr/bin/env python3
"""Evaluate the deterministic CLHE mean-history nearest-neighbor baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WP_ROOT = REPO_ROOT / "src" / "03_backbone_recommender"
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(WP_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from many_to_many_metrics import calculate_many_to_many_metrics  # noqa: E402
from shared.artifacts import sha256_file  # noqa: E402


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(
        np.linalg.norm(values, axis=-1, keepdims=True), np.finfo(np.float32).eps)


def _retrieve_topk(
    reference_rows: np.ndarray,
    catalog_embeddings_l2: np.ndarray,
    *,
    topk: int,
) -> np.ndarray:
    reference_rows = np.asarray(reference_rows, dtype=np.int64)
    catalog_embeddings_l2 = np.asarray(catalog_embeddings_l2, dtype=np.float32)
    if reference_rows.ndim != 2 or catalog_embeddings_l2.ndim != 2:
        raise ValueError("Reference rows and catalog embeddings must both be matrices")
    if topk <= 0 or topk > len(catalog_embeddings_l2) - reference_rows.shape[1]:
        raise ValueError("Invalid top-k after excluding visible reference items")
    query = _l2_normalize(catalog_embeddings_l2[reference_rows].mean(axis=1))
    scores = query @ catalog_embeddings_l2.T
    for index, rows in enumerate(reference_rows):
        scores[index, rows] = -np.inf
    # Stable full sorting gives a deterministic catalog-row tie break.
    return np.argsort(-scores, axis=1, kind="stable")[:, :topk]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepared_dir = args.prepared_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    vectors = prepared_dir / "vectors"
    reference_rows = np.load(vectors / "eval_reference_rows.npy", allow_pickle=False)
    target_rows = np.load(vectors / "eval_target_rows.npy", allow_pickle=False)
    catalog_embeddings_l2 = np.load(
        vectors / "catalog_embeddings_l2.npy", allow_pickle=False).astype(np.float32)
    item_ids = [
        str(item_id) for item_id in json.loads(
            (vectors / "catalog_item_ids.json").read_text(encoding="utf-8"))]
    if reference_rows.shape != (941, 15) or target_rows.shape != (941, 5):
        raise ValueError(
            f"Frozen evaluation row shapes drifted: {reference_rows.shape}, {target_rows.shape}")
    if len(item_ids) != len(catalog_embeddings_l2):
        raise ValueError("Catalog IDs and CLHE embedding rows differ")

    prediction_rows = _retrieve_topk(
        reference_rows, catalog_embeddings_l2, topk=5)
    id_array = np.asarray(item_ids, dtype=object)
    prediction_ids = id_array[prediction_rows]
    target_ids = id_array[target_rows]
    block = calculate_many_to_many_metrics(
        catalog_embeddings_l2[prediction_rows], catalog_embeddings_l2[target_rows],
        prediction_ids, target_ids)
    metrics = {name: float(values.mean()) for name, values in block.items()}
    metrics.update({
        "coverage_at_5": len(set(prediction_ids.reshape(-1).tolist())) / len(item_ids),
        "unique_predicted_items": len(set(prediction_ids.reshape(-1).tolist())),
    })
    manifest_path = prepared_dir / "prepared_manifest.json"
    payload = {
        "result_schema": "genplaylist-baseline-predictions-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "method": "CLHE-kNN",
        "prepared_data": {
            "path": str(prepared_dir),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "evaluation": {
            "test_examples": len(reference_rows),
            "reference_items": 15,
            "generated_items": 5,
            "catalog_items": len(item_ids),
            "visible_items_excluded": True,
            "query": "L2-normalized mean of fifteen CLHE catalog embeddings",
            "ranking": "exact cosine top-five with stable catalog-row tie break",
        },
        "metrics_clhe_diagnostic": metrics,
        "predictions": {
            "item_ids": prediction_ids.tolist(),
            "target_item_ids": target_ids.tolist(),
            "shape": [len(reference_rows), 5],
        },
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
