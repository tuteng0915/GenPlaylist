#!/usr/bin/env python3
"""Dependency-free tests for the Music4All skipped-event audit."""

from __future__ import annotations

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


if __name__ == "__main__":
    test_gap_threshold_changes_eligibility()
    test_threshold_parser()
    print("Music4All sequence gap audit tests passed")
