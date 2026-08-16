#!/usr/bin/env python3
"""Syntax/import smoke test for the end-to-end metric summarizer."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("summarize_end_to_end_metrics.py")
SPEC = importlib.util.spec_from_file_location("summarize_end_to_end_metrics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


if __name__ == "__main__":
    assert MODULE.EXPECTED_EXAMPLES == 941
    assert tuple(MODULE.SYSTEMS) == ("ACE-Step-Direct", "DDBC-SFT", "GenPlaylist")
    print("end-to-end summary import test passed")
