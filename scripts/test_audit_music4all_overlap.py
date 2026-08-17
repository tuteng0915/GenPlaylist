#!/usr/bin/env python3
"""Dependency-free tests for Music4All overlap auditing."""

from __future__ import annotations

import bz2
import importlib.util
from pathlib import Path
import tempfile


SCRIPT = Path(__file__).with_name("audit_music4all_overlap.py")
SPEC = importlib.util.spec_from_file_location("audit_music4all_overlap", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_version_normalization_is_separate_from_strict_matching() -> None:
    assert MODULE._normalize("Song - 2009 Remaster") != MODULE._normalize("Song")
    assert MODULE._relaxed_title("Song - 2009 Remaster") == MODULE._normalize("Song")
    assert MODULE._normalize("Booker T. & the M.G.'s") == MODULE._normalize(
        "Booker T and the MGs")


def test_only_one_to_one_keys_are_accepted() -> None:
    current = {
        "1": {"artist": "Artist", "title": "Unique"},
        "2": {"artist": "Artist", "title": "Duplicate"},
    }
    music4all = [
        {"id": "a", "artist": "Artist", "song": "Unique"},
        {"id": "b", "artist": "Artist", "song": "Duplicate"},
        {"id": "c", "artist": "Artist", "song": "Duplicate"},
    ]
    assert MODULE._unique_key_matches(current, music4all, relaxed=False) == {"a": "1"}


def test_interaction_stats_distinguish_filtered_and_contiguous_windows() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "events.tsv.bz2"
        rows = ["user_id\ttrack_id\ttimestamp\n"]
        for index in range(10):
            rows.append(f"u\tm\t2026-01-01 00:00:{index:02d}\n")
        rows.append("u\tx\t2026-01-01 00:00:10\n")
        for index in range(11, 22):
            rows.append(f"u\tm\t2026-01-01 00:00:{index:02d}\n")
        with bz2.open(path, "wt", encoding="utf-8") as handle:
            handle.writelines(rows)
        result = MODULE._interaction_stats(path, {"m"})
        assert result["mapped_events"] == 21
        assert result["filtered_subsequence_length20_windows"] == 2
        assert result["contiguous_supported_length20_windows"] == 0
        assert result["rows_grouped_by_user"] is True
        assert result["timestamp_order"] == "ascending"


if __name__ == "__main__":
    test_version_normalization_is_separate_from_strict_matching()
    test_only_one_to_one_keys_are_accepted()
    test_interaction_stats_distinguish_filtered_and_contiguous_windows()
    print("Music4All overlap audit tests passed")
