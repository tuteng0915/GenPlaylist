#!/usr/bin/env python3
"""Dependency-free test for Music4All WP-C dataset materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


SCRIPT = Path(__file__).with_name("materialize_music4all_dataset.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_materialization() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sequence = root / "sequence"
        catalog = root / "catalog"
        (sequence / "splits").mkdir(parents=True)
        catalog.mkdir()
        split_entries = {}
        for split, text in (
            ("train", "train," + ",".join(str(i) for i in range(20)) + "\n"),
            ("val", ""),
            ("test", "test," + ",".join(str(i) for i in range(20)) + "\n"),
        ):
            path = sequence / "splits" / f"{split}.txt"
            path.write_text(text, encoding="utf-8")
            split_entries[split] = {"sha256": _sha256(path)}
        (catalog / "catalog_metadata.json").write_text(
            json.dumps({"0": {"item_id": "0"}, "1": {"item_id": "1"}}) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "result_schema": "genplaylist-music4all-sequences-v1",
            "configuration": {
                "seed": 42, "accepted_mapping_items": 2,
                "train_user_cap": 16, "max_skipped_events": 5,
                "min_unique_references": 8, "min_unique_targets": 3,
            },
            "protocol": {"window": "15 references plus 5 targets"},
            "scan": {
                "output_windows_by_split": {"train": 1, "test": 1},
                "eligible_users_by_split": {"train": 1, "test": 1},
            },
            "outputs": split_entries,
        }
        (sequence / "sequence_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        output = root / "dataset"
        subprocess.run([
            sys.executable, str(SCRIPT),
            "--sequence-dir", str(sequence),
            "--catalog-dir", str(catalog),
            "--output-dir", str(output),
        ], check=True, capture_output=True, text=True)
        card = json.loads((output / "dataset_card.json").read_text(encoding="utf-8"))
        assert card["wp_c_split_counts"] == {"train": 1, "test": 1}
        assert card["catalog"] == {
            "genplaylist_items": 2, "accepted_music4all_items": 2,
        }
        assert (output / "splits" / "val.txt").read_text() == ""


if __name__ == "__main__":
    test_materialization()
    print("Music4All dataset materialization tests passed")
