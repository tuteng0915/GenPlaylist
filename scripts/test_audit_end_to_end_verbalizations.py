#!/usr/bin/env python3
"""Unit tests for the end-to-end verbalization audit."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile


SCRIPT = Path(__file__).with_name("audit_end_to_end_verbalizations.py")
SPEC = importlib.util.spec_from_file_location("audit_end_to_end_verbalizations", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_audit_counts_duplicates_and_attribute_fields() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        system = "GenPlaylist"
        (root / system).mkdir()
        records = [
            {
                "system": system,
                "example_index": 0,
                "reference_item_ids": [str(i) for i in range(15)],
                "cue_ids": [1, 1],
                "cue_terms": ["rain", "Rain"],
                "music_attributes": "pop, warm",
                "lyric_draft": "[Verse] line\n[Chorus] line",
            },
            {
                "system": system,
                "example_index": 1,
                "reference_item_ids": [str(i) for i in range(15)],
                "cue_ids": [1, 2],
                "cue_terms": ["rain", "sun"],
                "music_attributes": "rock",
                "lyric_draft": "[verse] line\n[chorus] line",
            },
        ]
        for index, record in enumerate(records):
            (root / system / f"{index:04d}.json").write_text(
                json.dumps(record), encoding="utf-8")
        result = MODULE._audit_system(root, system, 2, 2, 2)
        assert result["records_with_repeated_cue_terms"] == 1
        assert result["mean_cue_unique_ratio"] == 0.75
        assert result["records_with_exact_attribute_fields"] == 1
        assert result["attribute_field_count_histogram"] == {"1": 1, "2": 1}


if __name__ == "__main__":
    test_audit_counts_duplicates_and_attribute_fields()
    print("end-to-end verbalization audit tests passed")
