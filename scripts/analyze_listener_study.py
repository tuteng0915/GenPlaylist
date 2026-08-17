#!/usr/bin/env python3
"""Analyze the frozen blinded GenPlaylist-versus-real listener study."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from shared.artifacts import sha256_file  # noqa: E402
from evaluate_mert_proxy import _bootstrap_mean_interval  # noqa: E402


DIMENSIONS = ("fit", "quality", "novelty")
SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parse_bool(value: object) -> bool:
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}")


def _load_raw_rows(path: Path) -> tuple[list[dict[str, object]], str]:
    if path.suffix.casefold() not in SQLITE_SUFFIXES:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle)), "csv"

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        columns = (
            "session_id", "participant_hash", "case_id", "song_a_is_generated",
            "fit_a", "fit_b", "quality_a", "quality_b", "novelty_a", "novelty_b",
            "preference", "listening_freq", "musical_training", "playback_confirmed",
            "notes", "submitted_utc",
        )
        rows = connection.execute(
            f"SELECT {', '.join(columns)} FROM responses ORDER BY submitted_utc, session_id"
        ).fetchall()
        return [dict(row) for row in rows], "sqlite"
    finally:
        connection.close()


def _rating(row: dict[str, object], name: str) -> float:
    value = float(row[name])
    if not np.isfinite(value) or not 1 <= value <= 5:
        raise ValueError(f"Rating {name} must be in [1, 5], got {value}")
    return value


def _decode_row(row: dict[str, object], participant_column: str) -> dict:
    participant = str(row.get(participant_column, "")).strip()
    if not participant:
        raise ValueError(f"Missing participant identifier in {participant_column}")
    generated_is_a = _parse_bool(row["song_a_is_generated"])
    values = {"participant_id": participant}
    for dimension in DIMENSIONS:
        side_a = _rating(row, f"{dimension}_a")
        side_b = _rating(row, f"{dimension}_b")
        values[f"generated_{dimension}"] = side_a if generated_is_a else side_b
        values[f"real_{dimension}"] = side_b if generated_is_a else side_a

    preference = str(row["preference"]).strip().casefold()
    if preference == "no preference":
        preference_score = 0.5
        preference_label = "tie"
    elif preference in {"song a", "a"}:
        preference_score = 1.0 if generated_is_a else 0.0
        preference_label = "generated" if generated_is_a else "real"
    elif preference in {"song b", "b"}:
        preference_score = 0.0 if generated_is_a else 1.0
        preference_label = "real" if generated_is_a else "generated"
    else:
        raise ValueError(f"Unknown preference value {row['preference']!r}")
    values["generated_preference_score"] = preference_score
    values["preference_label"] = preference_label
    values["musical_training"] = str(row.get("musical_training", "")).strip() or "Unknown"
    return values


def _participant_means(rows: list[dict]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["participant_id"], []).append(row)
    names = [
        f"{system}_{dimension}"
        for system in ("generated", "real")
        for dimension in DIMENSIONS
    ] + ["generated_preference_score"]
    return {
        name: np.asarray([
            np.mean([row[name] for row in participant_rows])
            for participant_rows in grouped.values()
        ], dtype=np.float32)
        for name in names
    }


def _summarize(rows: list[dict], samples: int, seed: int) -> dict:
    if not rows:
        raise ValueError("Listener-study analysis needs at least one response")
    values = _participant_means(rows)
    participants = len(set(row["participant_id"] for row in rows))
    systems = {}
    for system in ("generated", "real"):
        systems[system] = {
            dimension: {
                "mean": float(values[f"{system}_{dimension}"].mean()),
                "confidence_interval_95": _bootstrap_mean_interval(
                    values[f"{system}_{dimension}"], samples=samples, seed=seed),
            }
            for dimension in DIMENSIONS
        }
    paired = {}
    for dimension in DIMENSIONS:
        difference = values[f"generated_{dimension}"] - values[f"real_{dimension}"]
        paired[dimension] = {
            "mean": float(difference.mean()),
            "confidence_interval_95": _bootstrap_mean_interval(
                difference, samples=samples, seed=seed),
        }
    preference = values["generated_preference_score"]
    preference_counts = {
        label: sum(row["preference_label"] == label for row in rows)
        for label in ("generated", "real", "tie")
    }
    return {
        "participants": participants,
        "responses": len(rows),
        "systems": systems,
        "paired_generated_minus_real": paired,
        "preference": {
            "generated_share_ties_half": float(preference.mean()),
            "confidence_interval_95": _bootstrap_mean_interval(
                preference, samples=samples, seed=seed),
            "response_counts": preference_counts,
            "tie_rule": "a no-preference response contributes 0.5 to each system",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--participant-column", default="session_id")
    parser.add_argument("--exclude-participant", action="append", default=[])
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples <= 0 or args.bootstrap_seed < 0:
        raise ValueError("Bootstrap samples must be positive and seed nonnegative")
    responses_path = args.responses.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    raw_rows, source_type = _load_raw_rows(responses_path)
    rows = [_decode_row(row, args.participant_column) for row in raw_rows]
    exclusions = {str(value) for value in args.exclude_participant}
    included = [row for row in rows if row["participant_id"] not in exclusions]
    unknown_exclusions = exclusions - {row["participant_id"] for row in rows}
    if unknown_exclusions:
        raise ValueError(f"Excluded participants are absent: {sorted(unknown_exclusions)}")

    overall = _summarize(
        included, samples=args.bootstrap_samples, seed=args.bootstrap_seed)
    strata = {}
    for label in sorted(set(row["musical_training"] for row in included)):
        selected = [row for row in included if row["musical_training"] == label]
        strata[label] = _summarize(
            selected, samples=args.bootstrap_samples, seed=args.bootstrap_seed)
    payload = {
        "result_schema": "genplaylist-listener-study-analysis-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design": "blinded paired A/B: GenPlaylist versus immediate real next track",
        "input": {
            "path": str(responses_path),
            "sha256": sha256_file(responses_path),
            "source_type": source_type,
            "raw_responses": len(raw_rows),
            "participant_column": args.participant_column,
        },
        "exclusions": sorted(exclusions),
        "overall": overall,
        "by_musical_training": strata,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "unit": "participant",
        },
    }
    _atomic_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
