#!/usr/bin/env python3
"""Compare one GenPlaylist audio ablation with the frozen default run."""

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
from evaluate_end_to_end_mert import (  # noqa: E402
    _diversity,
    _load_test_sequences,
    _maximum_catalog_similarity,
)
from evaluate_mert_proxy import _bootstrap_mean_interval  # noqa: E402
from extract_end_to_end_mert import EXPECTED_EXAMPLES, _slug  # noqa: E402


SYSTEM = "GenPlaylist"
SLUG = _slug(SYSTEM)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_catalog(
    prepared_dir: Path, catalog_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    manifest_path = catalog_dir / "mert_manifest.json"
    embeddings_path = catalog_dir / "catalog_mert_embeddings_l2.npy"
    ids_path = catalog_dir / "catalog_item_ids.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path in (embeddings_path, ids_path):
        if manifest["outputs"][path.name] != sha256_file(path):
            raise ValueError(f"Catalog artifact hash mismatch: {path}")
    embeddings = np.load(embeddings_path, allow_pickle=False).astype(np.float32)
    item_ids = [str(item) for item in json.loads(ids_path.read_text(encoding="utf-8"))]
    if len(item_ids) != len(embeddings):
        raise ValueError("Catalog IDs and MERT embeddings differ in length")
    item_to_row = {item: row for row, item in enumerate(item_ids)}
    sequences = _load_test_sequences(prepared_dir)
    reference_rows = np.asarray(
        [[item_to_row[item] for item in row[:15]] for row in sequences],
        dtype=np.int64,
    )
    next_rows = np.asarray(
        [item_to_row[row[15]] for row in sequences], dtype=np.int64)
    return embeddings, reference_rows, next_rows, manifest_path


def _load_mert(
    metrics_dir: Path, catalog_manifest: dict, expected_width: int,
) -> tuple[np.ndarray, dict[str, str]]:
    embeddings_path = metrics_dir / f"{SLUG}_mert_embeddings_l2.npy"
    manifest_path = metrics_dir / f"{SLUG}_mert_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_revision") != catalog_manifest.get("model_revision"):
        raise ValueError(f"MERT model revision differs in {metrics_dir}")
    if manifest.get("output_sha256") != sha256_file(embeddings_path):
        raise ValueError(f"Generated MERT hash mismatch in {metrics_dir}")
    embeddings = np.load(embeddings_path, allow_pickle=False).astype(np.float32)
    if embeddings.shape != (EXPECTED_EXAMPLES, expected_width):
        raise ValueError(f"Generated MERT shape differs: {embeddings.shape}")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4):
        raise ValueError(f"Generated MERT is not normalized in {metrics_dir}")
    return embeddings, {
        "manifest_sha256": sha256_file(manifest_path),
        "embeddings_sha256": sha256_file(embeddings_path),
    }


def _mert_scores(
    generated: np.ndarray,
    catalog: np.ndarray,
    reference_rows: np.ndarray,
    next_rows: np.ndarray,
) -> dict[str, np.ndarray | float]:
    similarities = np.einsum("bd,bkd->bk", generated, catalog[reference_rows])
    return {
        "history_fit": similarities.mean(axis=1),
        "next_similarity": np.einsum("bd,bd->b", generated, catalog[next_rows]),
        "reference_max_similarity": similarities.max(axis=1),
        "catalog_max_similarity": _maximum_catalog_similarity(generated, catalog),
        "cross_history_diversity": _diversity(generated),
    }


def _load_clap(metrics_dir: Path) -> tuple[np.ndarray, dict[str, str]]:
    result_path = metrics_dir / f"{SLUG}_clap.json"
    audio_path = metrics_dir / f"{SLUG}_clap_audio_l2.npy"
    text_path = metrics_dir / f"{SLUG}_clap_attribute_l2.npy"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    for path in (audio_path, text_path):
        if result["outputs"][path.name] != sha256_file(path):
            raise ValueError(f"CLAP artifact hash mismatch: {path}")
    audio = np.load(audio_path, allow_pickle=False).astype(np.float32)
    text = np.load(text_path, allow_pickle=False).astype(np.float32)
    if audio.shape != text.shape or audio.shape[0] != EXPECTED_EXAMPLES:
        raise ValueError(f"CLAP embedding shape differs in {metrics_dir}")
    scores = np.einsum("bd,bd->b", audio, text)
    if not np.isclose(scores.mean(), result["clap_a"], atol=1e-7):
        raise ValueError(f"CLAP-A score cannot be reproduced in {metrics_dir}")
    return scores, {
        "result_sha256": sha256_file(result_path),
        "audio_embeddings_sha256": sha256_file(audio_path),
        "text_embeddings_sha256": sha256_file(text_path),
    }


def _paired(
    ablation: np.ndarray,
    baseline: np.ndarray,
    samples: int,
    seed: int,
) -> dict:
    difference = np.asarray(ablation) - np.asarray(baseline)
    return {
        "mean": float(difference.mean()),
        "confidence_interval_95": _bootstrap_mean_interval(
            difference, samples=samples, seed=seed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--catalog-mert-dir", type=Path, required=True)
    parser.add_argument("--baseline-metrics-dir", type=Path, required=True)
    parser.add_argument("--ablation-metrics-dir", type=Path, required=True)
    parser.add_argument("--ablation-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    prepared_dir = args.prepared_dir.expanduser().resolve()
    catalog_dir = args.catalog_mert_dir.expanduser().resolve()
    baseline_dir = args.baseline_metrics_dir.expanduser().resolve()
    ablation_dir = args.ablation_metrics_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    catalog, reference_rows, next_rows, catalog_manifest_path = _load_catalog(
        prepared_dir, catalog_dir)
    catalog_manifest = json.loads(catalog_manifest_path.read_text(encoding="utf-8"))
    mert_embeddings = {}
    artifacts = {"mert": {}, "clap": {}, "fad": {}}
    clap_scores = {}
    labels_and_dirs = (("default", baseline_dir), (args.ablation_name, ablation_dir))
    for label, directory in labels_and_dirs:
        mert_embeddings[label], artifacts["mert"][label] = _load_mert(
            directory, catalog_manifest, catalog.shape[1])
        clap_scores[label], artifacts["clap"][label] = _load_clap(directory)

    mert = {
        label: _mert_scores(
            mert_embeddings[label], catalog, reference_rows, next_rows)
        for label, _ in labels_and_dirs
    }
    systems = {}
    for label, directory in labels_and_dirs:
        fad_path = directory / "fad.json"
        fad = json.loads(fad_path.read_text(encoding="utf-8"))
        artifacts["fad"][label] = sha256_file(fad_path)
        systems[label] = {
            "fad": float(fad["fad"][SYSTEM]),
            "mert": {
                name: float(values) if np.isscalar(values) else float(values.mean())
                for name, values in mert[label].items()
            },
            "clap_a": float(clap_scores[label].mean()),
        }

    array_mert_metrics = tuple(
        name for name, values in mert["default"].items() if not np.isscalar(values))
    paired_mert = {
        name: _paired(
            np.asarray(mert[args.ablation_name][name]),
            np.asarray(mert["default"][name]),
            args.bootstrap_samples,
            args.bootstrap_seed,
        )
        for name in array_mert_metrics
    }
    payload = {
        "result_schema": "genplaylist-end-to-end-ablation-comparison-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "examples": EXPECTED_EXAMPLES,
        "reference_items": 15,
        "target_item": 16,
        "comparison": f"{args.ablation_name} minus default",
        "systems": systems,
        "paired_differences": {
            "mert": paired_mert,
            "cross_history_diversity": {
                "difference": float(
                    mert[args.ablation_name]["cross_history_diversity"]
                    - mert["default"]["cross_history_diversity"])
            },
            "clap_a": _paired(
                clap_scores[args.ablation_name],
                clap_scores["default"],
                args.bootstrap_samples,
                args.bootstrap_seed,
            ),
            "fad": {
                "difference": systems[args.ablation_name]["fad"]
                - systems["default"]["fad"],
                "confidence_interval_95": None,
            },
        },
        "artifacts": {
            "catalog_mert_manifest_sha256": sha256_file(catalog_manifest_path),
            **artifacts,
        },
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "unit": "listening-history context",
        },
    }
    _atomic_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
