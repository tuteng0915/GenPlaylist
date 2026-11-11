#!/usr/bin/env python3
"""Recompute paired catalog metrics from frozen, aligned prediction files.

Intervals condition on fitted checkpoints and use history-level resampling;
they do not estimate training-seed variance. Both files must use the same
ordered histories, independently verified against the supplied prepared data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_mert_proxy import _bootstrap_mean_interval, _calculate_metrics
from shared.artifacts import sha256_file


def load_catalog(directory):
    manifest_path = directory / "mert_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete_catalog"):
        raise ValueError("Require complete-catalog MERT features")
    files = [directory / "catalog_mert_embeddings_l2.npy", directory / "catalog_item_ids.json"]
    for path in files:
        if sha256_file(path) != manifest["outputs"][path.name]:
            raise ValueError(f"MERT hash mismatch: {path}")
    matrix = np.load(files[0], allow_pickle=False)
    ids = list(map(str, json.loads(files[1].read_text())))
    if matrix.ndim != 2 or len(ids) != len(matrix) or len(set(ids)) != len(ids):
        raise ValueError("Invalid MERT catalog alignment")
    if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4):
        raise ValueError("MERT features must be finite and normalized")
    return {item: i for i, item in enumerate(ids)}, matrix, sha256_file(manifest_path)


def score_predictions(path, expected_targets, mapping, matrix):
    data = json.loads(path.read_text())
    predictions = data["predictions"]
    targets = np.asarray(predictions["target_item_ids"], dtype=str)
    if not np.array_equal(targets, expected_targets):
        raise ValueError(f"Target/order mismatch against prepared histories: {path}")
    ids = np.asarray(predictions["item_ids"], dtype=str)
    # Bound memory for expanded Music4All: avoid materializing N x 5 x 768 twice.
    chunks = []
    for start in range(0, len(ids), 128):
        _, values = _calculate_metrics(ids[start:start+128], targets[start:start+128], mapping, matrix)
        chunks.append(values)
    per_history = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    metrics = {key: float(values.mean()) for key, values in per_history.items()}
    metrics["coverage_at_5"] = len(set(ids.ravel())) / len(mapping)
    return metrics, per_history


def paired_summary(baseline, variant, samples, seed):
    if baseline.keys() != variant.keys():
        raise ValueError("Metric keys differ")
    result = {}
    for key in baseline:
        left, right = baseline[key], variant[key]
        if left.shape != right.shape or left.ndim != 1:
            raise ValueError("Paired histories must have equal one-dimensional shape")
        difference = right.astype(np.float64) - left.astype(np.float64)
        result[key] = {"difference": float(difference.mean()),
                       "ci95": _bootstrap_mean_interval(difference, samples=samples, seed=seed)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--mert-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    from datasets import load_from_disk
    dataset = load_from_disk(str(args.prepared_dir / "raw_dataset"))["test"]
    histories = [[str(item) for item in row] for row in dataset["item_seq"]]
    if not histories or any(len(row) != 20 for row in histories):
        raise ValueError("Expected nonempty 15-reference / 5-target histories")
    expected = np.asarray([row[15:] for row in histories])
    mapping, matrix, manifest_hash = load_catalog(args.mert_dir)
    base_metrics, baseline = score_predictions(args.baseline, expected, mapping, matrix)
    variant_metrics, variant = score_predictions(args.variant, expected, mapping, matrix)
    payload = {
        "schema": "genplaylist-paired-proxy-v1",
        "examples": len(histories), "difference_direction": "variant minus baseline",
        "resampling_unit": "history", "samples": args.samples, "seed": args.seed,
        "limitation": "Conditional on fitted checkpoints; no training-seed uncertainty. Histories sharing a source user/playlist may be dependent.",
        "baseline": {"path": str(args.baseline.resolve()), "sha256": sha256_file(args.baseline), "metrics": base_metrics},
        "variant": {"path": str(args.variant.resolve()), "sha256": sha256_file(args.variant), "metrics": variant_metrics},
        "mert_manifest_sha256": manifest_hash,
        "prepared_manifest_sha256": sha256_file(args.prepared_dir / "prepared_manifest.json"),
        "paired": paired_summary(baseline, variant, args.samples, args.seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    np.savez_compressed(args.output.with_suffix(".per_history.npz"),
                        **{f"baseline_{k}": v for k, v in baseline.items()},
                        **{f"variant_{k}": v for k, v in variant.items()})
    print(json.dumps(payload["paired"]))


if __name__ == "__main__":
    main()
