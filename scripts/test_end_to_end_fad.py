#!/usr/bin/env python3
"""Regression checks for FAD audio-only staging."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile


SCRIPT = Path(__file__).with_name("evaluate_end_to_end_fad.py")
SPEC = importlib.util.spec_from_file_location("evaluate_end_to_end_fad", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source.mp3"
        source.write_bytes(b"test")
        staging = root / "staging"
        MODULE._ensure_link(staging, "0000.mp3", source)
        MODULE._ensure_link(staging, "0000.mp3", source)
        assert (staging / "0000.mp3").resolve() == source.resolve()
        other = root / "other.mp3"
        other.write_bytes(b"other")
        try:
            MODULE._ensure_link(staging, "0000.mp3", other)
        except ValueError:
            pass
        else:
            raise AssertionError("A conflicting staging link was accepted")
    print("end-to-end FAD staging tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
