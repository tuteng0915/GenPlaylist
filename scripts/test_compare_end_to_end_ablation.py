#!/usr/bin/env python3
"""Unit tests for paired end-to-end ablation comparison helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("compare_end_to_end_ablation.py")
SPEC = importlib.util.spec_from_file_location("compare_end_to_end_ablation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_mert_scores() -> None:
    catalog = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    generated = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    references = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    next_rows = np.asarray([1, 0], dtype=np.int64)
    scores = MODULE._mert_scores(generated, catalog, references, next_rows)
    np.testing.assert_allclose(scores["history_fit"], [0.5, 0.5])
    np.testing.assert_allclose(scores["next_similarity"], [0.0, 0.0])
    np.testing.assert_allclose(scores["reference_max_similarity"], [1.0, 1.0])
    np.testing.assert_allclose(scores["catalog_max_similarity"], [1.0, 1.0])
    assert np.isclose(scores["cross_history_diversity"], 1.0)


def test_paired_direction() -> None:
    result = MODULE._paired(
        np.asarray([3.0, 5.0]), np.asarray([1.0, 2.0]), samples=100, seed=7)
    assert np.isclose(result["mean"], 2.5)
    assert len(result["confidence_interval_95"]) == 2


if __name__ == "__main__":
    test_mert_scores()
    test_paired_direction()
    print("end-to-end ablation comparison tests passed")
