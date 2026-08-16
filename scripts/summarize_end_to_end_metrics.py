#!/usr/bin/env python3
"""Combine frozen MERT, VGGish-FAD, and CLAP-A audio results."""

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


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    metrics_dir = args.metrics_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    mert_path = metrics_dir / "mert-evaluation.json"
    fad_path = metrics_dir / "fad.json"
    mert = json.loads(mert_path.read_text(encoding="utf-8"))
    fad = json.loads(fad_path.read_text(encoding="utf-8"))

    system_results = {}
    clap_scores = {}
    artifacts = {
        "mert_evaluation_sha256": sha256_file(mert_path),
        "fad_evaluation_sha256": sha256_file(fad_path),
        "clap": {},
    }
    for system in SYSTEMS:
        slug = _slug(system)
        clap_path = metrics_dir / f"{slug}_clap.json"
        audio_path = metrics_dir / f"{slug}_clap_audio_l2.npy"
        text_path = metrics_dir / f"{slug}_clap_attribute_l2.npy"
        clap = json.loads(clap_path.read_text(encoding="utf-8"))
        if clap["outputs"][audio_path.name] != sha256_file(audio_path):
            raise ValueError(f"CLAP audio embedding hash mismatch for {system}")
        if clap["outputs"][text_path.name] != sha256_file(text_path):
            raise ValueError(f"CLAP text embedding hash mismatch for {system}")
        audio = np.load(audio_path, allow_pickle=False).astype(np.float32)
        text = np.load(text_path, allow_pickle=False).astype(np.float32)
        if audio.shape != text.shape or audio.shape[0] != EXPECTED_EXAMPLES:
            raise ValueError(f"CLAP embedding shape differs for {system}")
        clap_scores[system] = np.einsum("bd,bd->b", audio, text)
        if not np.isclose(clap_scores[system].mean(), clap["clap_a"], atol=1e-7):
            raise ValueError(f"CLAP-A score cannot be reproduced for {system}")
        system_results[system] = {
            "fad": float(fad["fad"][system]),
            "history_fit": float(mert["systems"][system]["metrics"]["history_fit"]),
            "history_fit_confidence_interval_95": mert["systems"][system][
                "confidence_intervals_95"]["history_fit"],
            "clap_a": float(clap["clap_a"]),
            "clap_a_confidence_interval_95": clap["confidence_interval_95"],
        }
        artifacts["clap"][system] = sha256_file(clap_path)

    paired_clap = {}
    for left, right in (
        ("GenPlaylist", "ACE-Step-Direct"),
        ("GenPlaylist", "DDBC-SFT"),
        ("DDBC-SFT", "ACE-Step-Direct"),
    ):
        difference = clap_scores[left] - clap_scores[right]
        paired_clap[f"{left} minus {right}"] = {
            "mean": float(difference.mean()),
            "confidence_interval_95": _bootstrap_mean_interval(
                difference,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            ),
        }

    payload = {
        "result_schema": "genplaylist-end-to-end-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "examples": EXPECTED_EXAMPLES,
        "systems": system_results,
        "paired_clap_a_differences": paired_clap,
        "paired_mert_differences": mert["paired_differences"],
        "artifacts": artifacts,
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
