#!/usr/bin/env python3
"""Unit tests for blinded listener-study analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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


if __name__ == "__main__":
    test_unblinding_respects_side_assignment()
    test_participant_is_bootstrap_unit()
    print("listener-study analysis tests passed")
