"""Dependency-light tests for the exact CLHE-kNN baseline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("eval_clhe_knn.py")
SPEC = importlib.util.spec_from_file_location("eval_clhe_knn", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_visible_rows_are_excluded_and_ties_are_stable():
    catalog = np.asarray([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [-1.0, 0.0],
    ], dtype=np.float32)
    catalog = MODULE._l2_normalize(catalog)
    rows = MODULE._retrieve_topk(
        np.asarray([[0, 1]], dtype=np.int64), catalog, topk=2)
    assert rows.tolist() == [[2, 3]]


if __name__ == "__main__":
    test_visible_rows_are_excluded_and_ties_are_stable()
    print("  PASS  test_visible_rows_are_excluded_and_ties_are_stable")
