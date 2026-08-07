"""Dependency-light metric test; real CLAP/FAD runs are explicit integrations."""

from __future__ import annotations

import numpy as np

from eval_matrix import _clap_oas


def test_clap_oas_identity():
    embeddings = np.eye(3, dtype=np.float32)
    assert np.isclose(_clap_oas(embeddings, embeddings), 1.0)


def test_clap_oas_permutation_invariant():
    embeddings = np.eye(3, dtype=np.float32)
    assert np.isclose(_clap_oas(embeddings[[2, 0, 1]], embeddings), 1.0)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
