#!/usr/bin/env python3
"""Materialize a WP-C dataset root from frozen Music4All sequence splits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


SEQUENCE_SCHEMA = "genplaylist-music4all-sequences-v1"
DATASET_SCHEMA = "genplaylist-music4all-dataset-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_verified(source: Path, destination: Path, expected: str | None = None) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    actual = _sha256(source)
    if expected is not None and actual != expected:
        raise ValueError(f"Source hash mismatch for {source}: {actual} != {expected}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied = _sha256(destination)
    if copied != actual:
        raise ValueError(f"Copied file hash mismatch for {destination}")
    return {"size_bytes": destination.stat().st_size, "sha256": copied}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sequence_dir = args.sequence_dir.expanduser().resolve()
    catalog_dir = args.catalog_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite dataset root: {output_dir}")
    sequence_manifest_path = sequence_dir / "sequence_manifest.json"
    sequence_manifest = json.loads(sequence_manifest_path.read_text(encoding="utf-8"))
    if sequence_manifest.get("result_schema") != SEQUENCE_SCHEMA:
        raise ValueError("Unsupported Music4All sequence manifest")
    counts = sequence_manifest["scan"]["output_windows_by_split"]
    expected_counts = {"train": int(counts["train"]), "test": int(counts["test"])}
    if min(expected_counts.values()) <= 0:
        raise ValueError(f"Invalid sequence counts: {expected_counts}")
    catalog_metadata_path = catalog_dir / "catalog_metadata.json"
    catalog_metadata = json.loads(catalog_metadata_path.read_text(encoding="utf-8"))
    if not isinstance(catalog_metadata, (dict, list)) or not catalog_metadata:
        raise ValueError("Catalog metadata must be a non-empty object or list")
    catalog_items = len(catalog_metadata)
    accepted_items = int(sequence_manifest["configuration"]["accepted_mapping_items"])
    if accepted_items != catalog_items:
        raise ValueError(
            "Sequence mapping and materialized catalog differ: "
            f"mapping={accepted_items}, catalog={catalog_items}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        outputs = {}
        frozen_split_entries = sequence_manifest["outputs"]
        for split in ("train", "val", "test"):
            source = sequence_dir / "splits" / f"{split}.txt"
            outputs[f"splits/{split}.txt"] = _copy_verified(
                source,
                temporary / "splits" / f"{split}.txt",
                frozen_split_entries[split]["sha256"],
            )
        for name in (
            "catalog_metadata.json", "catalog.json", "complete_ids.txt", "stats.json",
        ):
            source = catalog_dir / name
            if source.is_file():
                outputs[name] = _copy_verified(source, temporary / name)

        dataset_card = {
            "result_schema": DATASET_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "Music4All-Onion v2 chronological listening histories",
            "frozen": True,
            "allow_repeated_items": True,
            "seed": sequence_manifest["configuration"]["seed"],
            "wp_c_split_counts": expected_counts,
            "source_user_split": "80/20 user-disjoint train/test; validation is empty",
            "eligible_users": sequence_manifest["scan"]["eligible_users_by_split"],
            "catalog": {
                "genplaylist_items": catalog_items,
                "accepted_music4all_items": accepted_items,
            },
            "sequence_protocol": sequence_manifest["protocol"],
            "sequence_configuration": sequence_manifest["configuration"],
            "sequence_manifest": {
                "path": str(sequence_manifest_path),
                "sha256": _sha256(sequence_manifest_path),
            },
        }
        card_path = temporary / "dataset_card.json"
        card_path.write_text(
            json.dumps(dataset_card, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        outputs["dataset_card.json"] = {
            "size_bytes": card_path.stat().st_size,
            "sha256": _sha256(card_path),
        }
        materialization = {
            "result_schema": DATASET_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_sequence_manifest_sha256": _sha256(sequence_manifest_path),
            "outputs": outputs,
        }
        materialization_path = temporary / "materialization_manifest.json"
        materialization_path.write_text(
            json.dumps(materialization, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "output_dir": str(output_dir),
        "wp_c_split_counts": expected_counts,
        "manifest_sha256": _sha256(output_dir / "materialization_manifest.json"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
