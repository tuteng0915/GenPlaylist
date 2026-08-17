#!/usr/bin/env python3
"""Compute frozen 15-to-5 proxy metrics from saved catalog predictions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
WP_ROOT = REPO_ROOT / "src" / "03_backbone_recommender"
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(WP_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from many_to_many_metrics import calculate_many_to_many_metrics  # noqa: E402
from shared.artifacts import sha256_file  # noqa: E402


EXPECTED_ITEMS = 5


def _bootstrap_mean_interval(
    values: np.ndarray, *, samples: int, seed: int,
) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("Bootstrap values must be a non-empty vector")
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        stop = min(start + 256, samples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return [float(value) for value in np.percentile(means, [2.5, 97.5])]


def _calculate_metrics(
    prediction_ids: np.ndarray,
    target_ids: np.ndarray,
    item_to_row: dict[str, int],
    embeddings_l2: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    prediction_ids = np.asarray(prediction_ids, dtype=str)
    target_ids = np.asarray(target_ids, dtype=str)
    if prediction_ids.ndim != 2 or prediction_ids.shape[1] != EXPECTED_ITEMS:
        raise ValueError(
            f"Predictions must have shape [examples, 5], got {prediction_ids.shape}")
    if target_ids.shape != prediction_ids.shape:
        raise ValueError(
            f"Prediction/target ID shapes differ: {prediction_ids.shape}, {target_ids.shape}")
    unknown = sorted(
        (set(prediction_ids.reshape(-1)) | set(target_ids.reshape(-1))) - set(item_to_row))
    if unknown:
        raise ValueError(f"Predictions contain IDs outside the MERT catalog: {unknown[:10]}")

    prediction_rows = np.asarray(
        [[item_to_row[item_id] for item_id in row] for row in prediction_ids],
        dtype=np.int64)
    target_rows = np.asarray(
        [[item_to_row[item_id] for item_id in row] for row in target_ids],
        dtype=np.int64)
    prediction_features = embeddings_l2[prediction_rows]
    target_features = embeddings_l2[target_rows]
    block = calculate_many_to_many_metrics(
        prediction_features, target_features, prediction_ids, target_ids)
    per_history = {
        "n1_mert": np.einsum(
            "bd,bd->b", prediction_features[:, 0], target_features[:, 0]),
        "recall_at_5": block["m2m_recall"],
        "m2m_mert": block["m2m_cosine"],
    }
    unique_predictions = len(set(prediction_ids.reshape(-1).tolist()))
    metrics = {
        name: float(values.mean()) for name, values in per_history.items()
    }
    metrics.update({
        "coverage_at_5": unique_predictions / len(item_to_row),
        "unique_predicted_items": unique_predictions,
        "catalog_items": len(item_to_row),
    })
    return metrics, per_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-result", type=Path, required=True)
    parser.add_argument("--mert-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prediction_path = args.prediction_result.expanduser().resolve()
    mert_dir = args.mert_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    prediction_payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    predictions = prediction_payload.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError(
            "Prediction result has no saved catalog IDs; rerun with result schema v3/v2")
    prediction_ids = np.asarray(predictions.get("item_ids"), dtype=str)
    target_ids = np.asarray(predictions.get("target_item_ids"), dtype=str)

    manifest_path = mert_dir / "mert_manifest.json"
    embedding_path = mert_dir / "catalog_mert_embeddings_l2.npy"
    ids_path = mert_dir / "catalog_item_ids.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete_catalog", False):
        raise ValueError("MERT artifact is a smoke-test subset, not the complete catalog")
    if manifest["outputs"][embedding_path.name] != sha256_file(embedding_path):
        raise ValueError("MERT embedding hash differs from its manifest")
    if manifest["outputs"][ids_path.name] != sha256_file(ids_path):
        raise ValueError("MERT item-ID hash differs from its manifest")
    item_ids = [str(item_id) for item_id in json.loads(ids_path.read_text("utf-8"))]
    embeddings_l2 = np.load(embedding_path, allow_pickle=False).astype(np.float32)
    if embeddings_l2.shape != tuple(manifest["shape"]):
        raise ValueError(f"MERT embedding shape differs from manifest: {embeddings_l2.shape}")
    if len(item_ids) != len(embeddings_l2) or len(set(item_ids)) != len(item_ids):
        raise ValueError("MERT catalog IDs are not unique and aligned with embedding rows")
    if not np.isfinite(embeddings_l2).all() or not np.allclose(
            np.linalg.norm(embeddings_l2, axis=1), 1.0, atol=1e-4):
        raise ValueError("MERT embeddings are non-finite or not L2-normalized")
    item_to_row = {item_id: row for row, item_id in enumerate(item_ids)}

    metrics, per_history = _calculate_metrics(
        prediction_ids, target_ids, item_to_row, embeddings_l2)
    confidence_intervals = {
        name: _bootstrap_mean_interval(
            values, samples=args.bootstrap_samples, seed=args.bootstrap_seed)
        for name, values in per_history.items()
    }
    declared_examples = prediction_payload.get("evaluation", {}).get("test_examples")
    official = (
        prediction_ids.ndim == 2
        and prediction_ids.shape[1] == EXPECTED_ITEMS
        and declared_examples == prediction_ids.shape[0]
    )
    payload = {
        "result_schema": "genplaylist-mert-proxy-eval-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_evaluation": official,
        "prediction_result": {
            "path": str(prediction_path),
            "sha256": sha256_file(prediction_path),
            "result_schema": prediction_payload.get("result_schema"),
        },
        "mert_artifact": {
            "path": str(mert_dir),
            "manifest_sha256": sha256_file(manifest_path),
            "model_name": manifest["model_name"],
            "model_revision": manifest["model_revision"],
        },
        "evaluation": {
            "examples": int(prediction_ids.shape[0]),
            "reference_items": 15,
            "generated_items": EXPECTED_ITEMS,
            "target_items": EXPECTED_ITEMS,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "metrics": metrics,
        "confidence_intervals_95": confidence_intervals,
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
