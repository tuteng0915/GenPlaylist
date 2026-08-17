#!/usr/bin/env python3
"""Unit tests for blinded listener-study analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sqlite3
import tempfile

import numpy as np


SCRIPT = Path(__file__).with_name("analyze_listener_study.py")
SPEC = importlib.util.spec_from_file_location("analyze_listener_study", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(generated_is_a: bool, participant: str, preference: str) -> dict[str, str]:
    return {
        "session_id": participant,
        "song_a_is_generated": str(generated_is_a),
        "fit_a": "5",
        "fit_b": "2",
        "quality_a": "4",
        "quality_b": "3",
        "novelty_a": "3",
        "novelty_b": "1",
        "preference": preference,
        "musical_training": "No",
    }


def test_unblinding_respects_side_assignment() -> None:
    first = MODULE._decode_row(_row(True, "p1", "Song A"), "session_id")
    second = MODULE._decode_row(_row(False, "p2", "Song A"), "session_id")
    assert first["generated_fit"] == 5 and first["real_fit"] == 2
    assert first["preference_label"] == "generated"
    assert second["generated_fit"] == 2 and second["real_fit"] == 5
    assert second["preference_label"] == "real"


def test_participant_is_bootstrap_unit() -> None:
    rows = [
        MODULE._decode_row(_row(True, "p1", "Song A"), "session_id"),
        MODULE._decode_row(_row(True, "p1", "No preference"), "session_id"),
        MODULE._decode_row(_row(False, "p2", "Song B"), "session_id"),
    ]
    result = MODULE._summarize(rows, samples=100, seed=7)
    assert result["participants"] == 2 and result["responses"] == 3
    assert np.isclose(result["preference"]["generated_share_ties_half"], 0.875)
    assert result["preference"]["response_counts"] == {
        "generated": 2, "real": 0, "tie": 1}


def test_sqlite_collection_database_loads_without_csv_export() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "responses.sqlite3"
        connection = sqlite3.connect(path)
        connection.execute(
            """CREATE TABLE responses (
                session_id TEXT, participant_hash TEXT, case_id TEXT,
                song_a_is_generated INTEGER, fit_a INTEGER, fit_b INTEGER,
                quality_a INTEGER, quality_b INTEGER, novelty_a INTEGER,
                novelty_b INTEGER, preference TEXT, listening_freq TEXT,
                musical_training TEXT, playback_confirmed INTEGER,
                notes TEXT, submitted_utc TEXT)"""
        )
        connection.execute(
            "INSERT INTO responses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "session-1", "participant-1", "case-1", 1, 5, 2, 4, 3, 3, 1,
                "Song A", "Daily", "No", 1, "", "2026-08-17T00:00:00+00:00",
            ),
        )
        connection.commit()
        connection.close()
        rows, source_type = MODULE._load_raw_rows(path)
        assert source_type == "sqlite" and len(rows) == 1
        decoded = MODULE._decode_row(rows[0], "participant_hash")
        assert decoded["participant_id"] == "participant-1"
        assert decoded["generated_fit"] == 5


if __name__ == "__main__":
    test_unblinding_respects_side_assignment()
    test_participant_is_bootstrap_unit()
    test_sqlite_collection_database_loads_without_csv_export()
    print("listener-study analysis tests passed")
