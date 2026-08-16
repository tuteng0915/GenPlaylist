"""Dependency-light tests for end-to-end plan conversion."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("run_end_to_end_verbalization.py")
SPEC = importlib.util.spec_from_file_location("run_end_to_end_verbalization", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_semantic_plan_reconstructs_three_codebooks():
    tokens = [
        MODULE.TOKEN_LAYOUT.rvq_token(0, 2),
        MODULE.TOKEN_LAYOUT.rvq_token(1, 3),
        MODULE.TOKEN_LAYOUT.rvq_token(2, 4),
        MODULE.TOKEN_LAYOUT.conflict_token(1),
    ]
    weights = np.arange(3 * 256 * 2, dtype=np.float32).reshape(3 * 256, 2)
    codes, conflict, reconstructed = MODULE._decode_semantic_plan(tokens, weights)
    assert codes == (2, 3, 4)
    assert conflict == 1
    expected = weights[[2, 256 + 3, 512 + 4]].sum(axis=0)
    assert np.array_equal(reconstructed, expected)


def test_reference_structure_is_mean_and_mean_squared_distance():
    embeddings = np.asarray([
        [1.0, 0.0],
        [3.0, 0.0],
        [0.0, 8.0],
    ], dtype=np.float32)
    mu, sigma = MODULE._reference_structure(
        ["a", "b"], {"a": 0, "b": 1, "c": 2}, embeddings)
    assert np.array_equal(mu, np.asarray([2.0, 0.0], dtype=np.float32))
    assert np.isclose(sigma, 1.0)


if __name__ == "__main__":
    test_semantic_plan_reconstructs_three_codebooks()
    test_reference_structure_is_mean_and_mean_squared_distance()
    print("  PASS  end-to-end verbalization plan tests")
