#!/usr/bin/env python3
"""Audit frozen Qwen verbalizations before audio evaluation or release."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from shared.artifacts import sha256_file  # noqa: E402


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _attribute_field_count(value: str) -> int:
    return sum(bool(field.strip()) for field in value.split(","))


def _audit_system(
    root: Path,
    system: str,
    examples: int,
    expected_cues: int,
    expected_attribute_fields: int,
) -> dict:
    digest = hashlib.sha256()
    duplicate_cues = 0
    attribute_field_histogram: dict[int, int] = {}
    unique_ratios = []
    for index in range(examples):
        path = root / system / f"{index:04d}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("system") != system or record.get("example_index") != index:
            raise ValueError(f"Record identity mismatch: {path}")
        if len(record.get("reference_item_ids", [])) != 15:
            raise ValueError(f"Expected 15 references: {path}")
        cue_ids = record.get("cue_ids")
        cue_terms = record.get("cue_terms")
        if not isinstance(cue_ids, list) or not isinstance(cue_terms, list):
            raise ValueError(f"Cue arrays are missing: {path}")
        if len(cue_ids) != expected_cues or len(cue_terms) != expected_cues:
            raise ValueError(f"Expected {expected_cues} cues: {path}")
        if any(not str(term).strip() for term in cue_terms):
            raise ValueError(f"Empty cue term: {path}")
        normalized_cues = [str(term).strip().casefold() for term in cue_terms]
        unique_count = len(set(normalized_cues))
        duplicate_cues += int(unique_count < expected_cues)
        unique_ratios.append(unique_count / expected_cues)

        attributes = str(record.get("music_attributes", "")).strip()
        lyrics = str(record.get("lyric_draft", "")).strip()
        if not attributes or not lyrics:
            raise ValueError(f"Empty attributes or lyrics: {path}")
        lowered = lyrics.casefold()
        if "[verse]" not in lowered or "[chorus]" not in lowered:
            raise ValueError(f"Lyrics lack verse/chorus markers: {path}")
        fields = _attribute_field_count(attributes)
        attribute_field_histogram[fields] = attribute_field_histogram.get(fields, 0) + 1
        digest.update(sha256_file(path).encode("ascii"))

    exact_attribute_fields = attribute_field_histogram.get(
        expected_attribute_fields, 0)
    return {
        "examples": examples,
        "records_fingerprint": digest.hexdigest(),
        "all_nonempty": True,
        "all_have_verse_and_chorus": True,
        "expected_cues_per_record": expected_cues,
        "records_with_repeated_cue_terms": duplicate_cues,
        "mean_cue_unique_ratio": sum(unique_ratios) / len(unique_ratios),
        "expected_attribute_fields": expected_attribute_fields,
        "records_with_exact_attribute_fields": exact_attribute_fields,
        "records_with_other_attribute_field_count": examples - exact_attribute_fields,
        "attribute_field_count_histogram": {
            str(key): value for key, value in sorted(attribute_field_histogram.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbalization-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system", action="append")
    parser.add_argument("--expected-examples", type=int, default=941)
    parser.add_argument("--expected-cues", type=int, default=8)
    parser.add_argument("--expected-attribute-fields", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(
        args.expected_examples,
        args.expected_cues,
        args.expected_attribute_fields,
    ) <= 0:
        raise ValueError("Expected counts must be positive")
    root = args.verbalization_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest_path = root / "verbalization_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    systems = tuple(args.system or manifest.get("systems", ()))
    if not systems:
        raise ValueError("No systems were selected for audit")
    unexpected = set(systems) - set(manifest.get("systems", ()))
    if unexpected:
        raise ValueError(f"Systems absent from verbalization manifest: {sorted(unexpected)}")
    results = {
        system: _audit_system(
            root,
            system,
            args.expected_examples,
            args.expected_cues,
            args.expected_attribute_fields,
        )
        for system in systems
    }
    payload = {
        "result_schema": "genplaylist-end-to-end-verbalization-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verbalization_manifest_sha256": sha256_file(manifest_path),
        "model_name": manifest.get("model_name"),
        "model_revision": manifest.get("model_revision"),
        "systems": results,
    }
    _atomic_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
