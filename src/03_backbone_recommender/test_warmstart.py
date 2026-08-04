"""Dependency-light tests for semantic DDBC vocabulary remapping."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("warmstart", HERE / "warmstart.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from shared.schema import TOKEN_LAYOUT  # noqa: E402


def test_token_mapping_preserves_shared_semantics_and_remaps_specials():
    mapping = module.build_token_row_mapping(
        source_vocab_size=1028, source_boi=1025, source_eos=1026)
    mapped = dict(mapping)
    assert mapped[0] == 0
    assert mapped[768] == 768
    assert mapped[842] == 842
    assert mapped[1025] == TOKEN_LAYOUT.boi_token
    assert mapped[1026] == TOKEN_LAYOUT.eos_token
    assert mapped[1027] == TOKEN_LAYOUT.mask_token
    assert not any(
        TOKEN_LAYOUT.cue_token_start <= target < TOKEN_LAYOUT.mask_token
        for _, target in mapping)


def test_remap_retains_new_cue_initialization():
    source = np.arange(1028 * 2, dtype=np.float32).reshape(1028, 2)
    target = np.full((TOKEN_LAYOUT.runtime_vocab_size, 2), -7.0, dtype=np.float32)
    mapping = module.build_token_row_mapping(
        source_vocab_size=1028, source_boi=1025, source_eos=1026)
    remapped = module.remap_vocab_rows(source, target, mapping)
    assert np.array_equal(remapped[842], source[842])
    assert np.array_equal(remapped[TOKEN_LAYOUT.boi_token], source[1025])
    assert np.array_equal(remapped[TOKEN_LAYOUT.eos_token], source[1026])
    assert np.array_equal(remapped[TOKEN_LAYOUT.mask_token], source[1027])
    assert np.all(remapped[TOKEN_LAYOUT.cue_token_start:TOKEN_LAYOUT.mask_token] == -7.0)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
