"""Dependency-light tests for primary playlist-proxy metrics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("evaluate_mert_proxy.py")
SPEC = importlib.util.spec_from_file_location("evaluate_mert_proxy", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_permuted_block_has_perfect_m2m_and_recall_but_ordered_n1():
    embeddings = np.eye(5, dtype=np.float32)
    item_to_row = {str(index): index for index in range(5)}
    targets = np.asarray([["0", "1", "2", "3", "4"]])
    predictions = np.asarray([["4", "3", "2", "1", "0"]])
    metrics, per_history = MODULE._calculate_metrics(
        predictions, targets, item_to_row, embeddings)
    assert metrics["n1_mert"] == 0.0
    assert metrics["recall_at_5"] == 1.0
    assert metrics["m2m_mert"] == 1.0
    assert metrics["coverage_at_5"] == 1.0
    assert per_history["m2m_mert"].shape == (1,)


def test_duplicate_predictions_receive_multiset_recall_credit():
    embeddings = np.eye(5, dtype=np.float32)
    item_to_row = {str(index): index for index in range(5)}
    targets = np.asarray([["0", "1", "2", "3", "4"]])
    predictions = np.asarray([["0", "0", "0", "0", "0"]])
    metrics, _ = MODULE._calculate_metrics(
        predictions, targets, item_to_row, embeddings)
    assert np.isclose(metrics["recall_at_5"], 0.2)
    assert np.isclose(metrics["coverage_at_5"], 0.2)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
