"""Dependency-light tests for deterministic MERT preprocessing helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("extract_mert_embeddings.py")
SPEC = importlib.util.spec_from_file_location("extract_mert_embeddings", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_center_crop_bounds_are_deterministic():
    assert MODULE._center_crop_bounds(100, 30) == (35, 65)
    assert MODULE._center_crop_bounds(101, 30) == (35, 65)
    assert MODULE._center_crop_bounds(12, 30) == (0, 12)


def test_masked_mean_excludes_padding():
    values = np.asarray([[[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]]])
    mask = np.asarray([[True, True, False]])
    np.testing.assert_allclose(
        MODULE._masked_mean_numpy(values, mask), np.asarray([[2.0, 4.0]]))


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
