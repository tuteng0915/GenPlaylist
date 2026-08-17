#!/usr/bin/env python3
"""Dependency-free checks for the Music4All-Onion downloader."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("download_music4all_onion.py")
SPEC = importlib.util.spec_from_file_location("download_music4all_onion", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_ranges_are_contiguous_and_complete() -> None:
    ranges = MODULE._ranges(101, 8)
    assert ranges[0][0] == 0 and ranges[-1][1] == 100
    assert sum(end - start + 1 for start, end in ranges) == 101
    assert all(left[1] + 1 == right[0] for left, right in zip(ranges, ranges[1:]))


def test_more_workers_than_bytes_is_safe() -> None:
    assert MODULE._ranges(3, 8) == [(0, 0), (1, 1), (2, 2)]


if __name__ == "__main__":
    test_ranges_are_contiguous_and_complete()
    test_more_workers_than_bytes_is_safe()
    print("Music4All-Onion downloader tests passed")
