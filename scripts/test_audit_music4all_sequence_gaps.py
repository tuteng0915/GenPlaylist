#!/usr/bin/env python3
"""Dependency-free tests for the Music4All skipped-event audit."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile


SCRIPT = Path(__file__).with_name("audit_music4all_sequence_gaps.py")
SPEC = importlib.util.spec_from_file_location("audit_music4all_sequence_gaps", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_gap_threshold_changes_eligibility() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sorted.tsv"
        rows = []
        source_row = 0
        for index in range(20):
            source_row += 1
            rows.append(
                f"u\t2026-01-01 00:00:{source_row:02d}\t{source_row}\t{index}\n"
            )
            if index == 9:
                source_row += 1
                rows.append(
                    f"u\t2026-01-01 00:00:{source_row:02d}\t{source_row}\t\n"
                )
        path.write_text("".join(rows), encoding="utf-8")
        result = MODULE._audit_sorted(
            path, (0, 1, None), seed=42, test_fraction=0.0,
            progress_every=0,
        )
        assert result["thresholds"]["0"]["windows"] == 0
        assert result["thresholds"]["1"]["windows"] == 1
        assert result["thresholds"]["all"]["windows"] == 1
        assert result["thresholds"]["1"]["fractions"]["all_20_items_unique"] == 1.0


def test_threshold_parser() -> None:
    assert MODULE._parse_thresholds("0, 5, all") == (0, 5, None)


def test_diversity_filter_is_reported_separately() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sorted.tsv"
        path.write_text("".join(
            f"u\t2026-01-01 00:00:{index:02d}\t{index}\t{index % 2}\n"
            for index in range(20)
        ), encoding="utf-8")
        result = MODULE._audit_sorted(
            path, (0,), seed=42, test_fraction=0.0, progress_every=0,
            min_unique_references=8, min_unique_targets=3,
        )
        current = result["thresholds"]["0"]
        assert current["windows"] == 1
        assert current["qualifying_windows"] == 0
        assert current["qualifying_train_rows_by_cap"]["16"] == 0


def test_train_target_support_is_sequence_based() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "sorted.tsv"
        support_path = root / "support.csv"
        path.write_text("".join(
            f"u\t2026-01-01 00:00:{index:02d}\t{index}\t{index}\n"
            for index in range(20)
        ), encoding="utf-8")
        result = MODULE._audit_sorted(
            path, (5,), seed=42, test_fraction=0.0, progress_every=0,
            min_unique_references=8, min_unique_targets=3,
            item_support_threshold=5, item_support_output=support_path,
        )
        with support_path.open(newline="", encoding="utf-8") as handle:
            rows = {row["item_id"]: row for row in csv.DictReader(handle)}
        assert rows["14"]["train_target_users"] == "0"
        assert rows["15"]["train_target_occurrences"] == "1"
        assert rows["15"]["train_target_windows"] == "1"
        assert rows["15"]["train_target_users"] == "1"
        support = result["sequence_target_support"]
        assert support["observed_mapped_items"] == 20
        assert support["items_with_positive_train_target_user_support"] == 5


if __name__ == "__main__":
    test_gap_threshold_changes_eligibility()
    test_threshold_parser()
    test_diversity_filter_is_reported_separately()
    test_train_target_support_is_sequence_based()
    print("Music4All sequence gap audit tests passed")
