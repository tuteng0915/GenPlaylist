#!/usr/bin/env python3
"""Tests for transactional frozen listener-study collection."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile


SCRIPT = Path(__file__).with_name("run_frozen_listener_study.py")
SPEC = importlib.util.spec_from_file_location("run_frozen_listener_study", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _first(values):
    return list(values)[0]


def test_assignments_are_balanced_and_duplicates_resume() -> None:
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "responses.sqlite3"
        connection = MODULE._connect(database)
        MODULE._initialize_database(connection)
        case_ids = ("case-a", "case-b", "case-c")
        assignments = [
            MODULE._assign_participant(
                connection, f"participant-{index}", case_ids, chooser=_first)
            for index in range(12)
        ]
        counts = {case_id: 0 for case_id in case_ids}
        sides = {case_id: {0: 0, 1: 0} for case_id in case_ids}
        for assignment in assignments:
            counts[assignment["case_id"]] += 1
            sides[assignment["case_id"]][
                assignment["display_song_a_is_generated"]] += 1
        assert set(counts.values()) == {4}
        assert all(values[0] == values[1] == 2 for values in sides.values())
        resumed = MODULE._assign_participant(
            connection, "participant-0", case_ids, chooser=_first)
        assert resumed["session_id"] == assignments[0]["session_id"]
        connection.close()


def test_submission_is_atomic_and_cannot_repeat() -> None:
    with tempfile.TemporaryDirectory() as directory:
        connection = MODULE._connect(Path(directory) / "responses.sqlite3")
        MODULE._initialize_database(connection)
        assignment = MODULE._assign_participant(
            connection, "participant", ("case-a",), chooser=_first)
        ratings = {
            "fit_a": 5, "fit_b": 3,
            "quality_a": 4, "quality_b": 2,
            "novelty_a": 3, "novelty_b": 1,
        }
        MODULE._submit_response(
            connection, assignment["session_id"], ratings,
            "Song A", "Daily", "No", True, "")
        assert connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 1
        try:
            MODULE._submit_response(
                connection, assignment["session_id"], ratings,
                "Song A", "Daily", "No", True, "")
        except MODULE.AlreadySubmittedError:
            pass
        else:
            raise AssertionError("Duplicate submission was accepted")
        try:
            MODULE._assign_participant(
                connection, "participant", ("case-a",), chooser=_first)
        except MODULE.AlreadySubmittedError:
            pass
        else:
            raise AssertionError("Submitted participant was reassigned")
        connection.close()


def test_participant_codes_are_hashed() -> None:
    key = b"k" * 32
    assert MODULE._participant_hash("worker-123", key) == MODULE._participant_hash(
        " worker-123 ", key)
    assert "worker-123" not in MODULE._participant_hash("worker-123", key)


if __name__ == "__main__":
    test_assignments_are_balanced_and_duplicates_resume()
    test_submission_is_atomic_and_cannot_repeat()
    test_participant_codes_are_hashed()
    print("frozen listener-study service tests passed")
