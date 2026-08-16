#!/usr/bin/env python3
"""Dependency-light CLAP-A regression checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("evaluate_end_to_end_clap.py")
SPEC = importlib.util.spec_from_file_location("evaluate_end_to_end_clap", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def main() -> int:
    values = MODULE._normalize(np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))
    assert np.allclose(values, [[0.6, 0.8], [0.0, 1.0]])
    try:
        MODULE._normalize(np.zeros((1, 2), dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("Zero CLAP embedding was accepted")

    class Projection:
        def modules(self):
            return [type("Layer", (), {"out_features": 256})(),
                    type("Layer", (), {"out_features": 512})()]

    assert MODULE._projection_output_dim(Projection()) == 512
    print("end-to-end CLAP-A tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
