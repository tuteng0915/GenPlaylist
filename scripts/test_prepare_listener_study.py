#!/usr/bin/env python3
"""Unit tests for frozen listener-study selection and blinding."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("prepare_listener_study.py")
SPEC = importlib.util.spec_from_file_location("prepare_listener_study", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_selection_is_deterministic_and_unique() -> None:
    first = MODULE._select_indices(941, 25, 42)
    second = MODULE._select_indices(941, 25, 42)
    assert first == second
    assert first == sorted(first)
    assert len(first) == len(set(first)) == 25


def test_side_assignment_is_balanced() -> None:
    for count in (1, 2, 25, 26):
        values = MODULE._generated_on_side_a(count, 42)
        assert len(values) == count
        assert abs(sum(values) - (count - sum(values))) <= 1
        assert values == MODULE._generated_on_side_a(count, 42)


def test_case_ids_are_opaque_and_stable() -> None:
    value = MODULE._opaque_case_id(17, 42)
    assert value == MODULE._opaque_case_id(17, 42)
    assert value != MODULE._opaque_case_id(18, 42)
    assert value.startswith("case-") and "17" not in value


if __name__ == "__main__":
    test_selection_is_deterministic_and_unique()
    test_side_assignment_is_balanced()
    test_case_ids_are_opaque_and_stable()
    print("listener-study preparation tests passed")
