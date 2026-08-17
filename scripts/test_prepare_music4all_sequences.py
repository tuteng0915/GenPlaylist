#!/usr/bin/env python3
"""Dependency-free tests for chronological Music4All sequence preparation."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile


SCRIPT = Path(__file__).with_name("prepare_music4all_sequences.py")
SPEC = importlib.util.spec_from_file_location("prepare_music4all_sequences", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_mapping_defaults_to_strict() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "mapping.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "music4all_id", "genplaylist_item_id", "match_type",
            ])
            writer.writeheader()
            writer.writerow({
                "music4all_id": "a", "genplaylist_item_id": "1",
                "match_type": "strict",
            })
            writer.writerow({
                "music4all_id": "b", "genplaylist_item_id": "2",
                "match_type": "relaxed-version",
            })
        assert MODULE._load_mapping(path, False) == {"a": "1"}
        assert MODULE._load_mapping(path, True) == {"a": "1", "b": "2"}


def test_sorted_scan_breaks_runs_and_retains_repeats() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sorted_path = root / "sorted.tsv"
        rows = []
        # This user has two supported fragments separated by one unsupported event.
        for index in range(10):
            rows.append(f"broken\t2026-01-01 00:00:{index:02d}\t{index + 1}\t1\n")
        rows.append("broken\t2026-01-01 00:00:10\t11\t\n")
        for index in range(11, 22):
            rows.append(f"broken\t2026-01-01 00:00:{index:02d}\t{index + 1}\t1\n")
        # This user has 21 adjacent supported events, hence two windows.  Repeats
        # are intentionally preserved and the cap deterministically keeps one.
        for index in range(21):
            item = "2" if index == 20 else "1"
            rows.append(f"eligible\t2026-01-02 00:00:{index:02d}\t{100 + index}\t{item}\n")
        sorted_path.write_text("".join(rows), encoding="utf-8")
        result = MODULE._scan_sorted_events(
            sorted_path,
            root / "train.txt",
            root / "test.txt",
            seed=42,
            test_fraction=0.0,
            train_user_cap=1,
            max_skipped_events=0,
            min_unique_references=1,
            min_unique_targets=1,
            progress_every=0,
        )
        assert result["eligible_windows_by_split"] == {"train": 2, "test": 0}
        assert result["eligible_users_by_split"] == {"train": 1, "test": 0}
        assert result["output_windows_by_split"] == {"train": 1, "test": 0}
        assert len((root / "train.txt").read_text().splitlines()) == 1
        repeats = result["repeat_statistics_over_all_eligible_windows"]
        assert repeats["windows_with_any_repeated_item"] == 2
        assert repeats["windows_with_target_reference_overlap"] == 2


def test_gap_and_diversity_constraints() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sorted_path = root / "sorted.tsv"
        rows = []
        source_row = 0
        for index in range(20):
            source_row += 1
            item = str(index + 1)
            rows.append(
                f"u\t2026-01-01 00:00:{source_row:02d}\t{source_row}\t{item}\n"
            )
            if index == 9:
                source_row += 1
                rows.append(
                    f"u\t2026-01-01 00:00:{source_row:02d}\t{source_row}\t\n"
                )
        sorted_path.write_text("".join(rows), encoding="utf-8")
        accepted = MODULE._scan_sorted_events(
            sorted_path, root / "train.txt", root / "test.txt",
            seed=42, test_fraction=0.0, train_user_cap=16,
            max_skipped_events=1, min_unique_references=8,
            min_unique_targets=3, progress_every=0,
        )
        assert accepted["output_windows_by_split"]["train"] == 1
        rejected = MODULE._scan_sorted_events(
            sorted_path, root / "train2.txt", root / "test2.txt",
            seed=42, test_fraction=0.0, train_user_cap=16,
            max_skipped_events=0, min_unique_references=8,
            min_unique_targets=3, progress_every=0,
        )
        assert rejected["output_windows_by_split"]["train"] == 0


def test_user_split_is_stable() -> None:
    first = MODULE._user_split("user", 42, 0.2)
    assert first == MODULE._user_split("user", 42, 0.2)
    assert MODULE._pseudonym("user", 42) == MODULE._pseudonym("user", 42)
    assert "user" not in MODULE._pseudonym("user", 42)


if __name__ == "__main__":
    test_mapping_defaults_to_strict()
    test_sorted_scan_breaks_runs_and_retains_repeats()
    test_gap_and_diversity_constraints()
    test_user_split_is_stable()
    print("Music4All sequence preparation tests passed")
