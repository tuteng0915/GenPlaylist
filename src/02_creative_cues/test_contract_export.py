"""Dependency-light WP-B production contract checks."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import config
import cue_export
from schema import CUE_VOCAB_SIZE, CueMappingEntry


def test_default_is_eight_cues():
    assert config.DEFAULT.num_cues == 8
    assert config.PRESETS["research-18-cues"].num_cues == 18


def test_manifest_marks_production_compatibility():
    vocab = ["<unk>"] + [f"cue-{index}" for index in range(1, CUE_VOCAB_SIZE)]
    mapping = {"18996": CueMappingEntry("18996", list(range(8)))}
    with tempfile.TemporaryDirectory() as temp_dir:
        cue_export.export_outputs(vocab, mapping, temp_dir)
        manifest = json.loads(
            (Path(temp_dir) / "cue_manifest.json").read_text(encoding="utf-8"))
        assert manifest["wp_d_compatible"] is True
        assert manifest["cues_per_item"] == 8
        assert manifest["token_layout"]["tokens_per_item"] == 13


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
