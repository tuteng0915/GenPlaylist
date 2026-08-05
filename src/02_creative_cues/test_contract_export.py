"""Dependency-light WP-B production contract checks."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import config
import cue_export
from schema import CUE_CANDIDATES_PER_ITEM, CUE_TOKENS, CUE_VOCAB_SIZE, CueMappingEntry


def test_default_stores_sixteen_and_activates_eight_cues():
    assert config.DEFAULT.num_cues == CUE_CANDIDATES_PER_ITEM == 16
    assert config.DEFAULT.active_cues == CUE_TOKENS == 8
    assert config.DEFAULT.assignment_strategy == "relevance"
    assert config.DEFAULT.candidate_k == 64
    assert config.PRESETS["research-18-cues"].num_cues == 18


def test_manifest_marks_production_compatibility():
    vocab = ["<unk>"] + [f"cue-{index}" for index in range(1, CUE_VOCAB_SIZE)]
    mapping = {
        "18996": CueMappingEntry(
            "18996", list(range(CUE_CANDIDATES_PER_ITEM)))
    }
    scores = {
        "18996": [1.0 - index / 100 for index in range(CUE_CANDIDATES_PER_ITEM)]
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        cue_export.export_outputs(
            vocab, mapping, temp_dir, score_mapping=scores,
            assignment_metadata={"strategy": "relevance"})
        manifest = json.loads(
            (Path(temp_dir) / "cue_manifest.json").read_text(encoding="utf-8"))
        assert manifest["wp_d_compatible"] is True
        assert manifest["stored_cues_per_item"] == 16
        assert manifest["default_active_cues"] == 8
        assert manifest["token_layout"]["tokens_per_item"] == 13
        assert len(manifest["cue_vocab_sha256"]) == 64
        assert len(manifest["item2cues_sha256"]) == 64
        assert (Path(temp_dir) / "item2cue_scores.json").is_file()
        assert (Path(temp_dir) / "item_cues.tsv").is_file()


def test_health_metrics_separate_active_and_stored_cues():
    vocab = ["<unk>", "one", "two", "three", "four"]
    mapping = {
        "a": CueMappingEntry("a", [1, 2, 3, 0]),
        "b": CueMappingEntry("b", [2, 1, 4, 0]),
    }
    active = cue_export.compute_coverage_stats(mapping, vocab, cue_limit=2)
    stored = cue_export.compute_coverage_stats(mapping, vocab)

    assert active["slots_per_item"] == [2]
    assert active["coverage_rate"] == 1.0
    assert active["unk_rate"] == 0.0
    assert active["vocab_coverage"] == 0.5
    assert stored["slots_per_item"] == [4]
    assert stored["coverage_rate"] == 0.0
    assert stored["unk_rate"] == 0.25

    with tempfile.TemporaryDirectory() as temp_dir:
        cue_export.write_report(
            stored, temp_dir, active_stats=active, stored_stats=stored)
        report = (Path(temp_dir) / "cue_report.md").read_text(encoding="utf-8")
        assert "Active@8" in report
        assert "Stored@16" in report


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
