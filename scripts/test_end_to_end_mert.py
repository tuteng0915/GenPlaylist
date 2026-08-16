#!/usr/bin/env python3
"""Dependency-light regression checks for end-to-end MERT metrics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("evaluate_end_to_end_mert.py")
SPEC = importlib.util.spec_from_file_location("evaluate_end_to_end_mert", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def main() -> int:
    values = np.eye(3, dtype=np.float32)
    assert np.isclose(MODULE._diversity(values), 1.0)
    catalog = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    generated = np.asarray([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    maxima = MODULE._maximum_catalog_similarity(generated, catalog, batch_size=1)
    assert np.allclose(maxima, [1.0, 0.8])
    print("end-to-end MERT metric tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
