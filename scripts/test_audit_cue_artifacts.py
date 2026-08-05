"""Checks WP-B coverage accounting for the frozen evaluation windows."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
import json


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "audit_cue_artifacts", HERE / "audit_cue_artifacts.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_frozen_window_coverage_counts_references_and_targets():
    mapping = {str(index): [1] * 16 for index in range(20)}
    with tempfile.TemporaryDirectory() as temp_dir:
        split = Path(temp_dir) / "test.txt"
        split.write_text(
            "playlist," + ",".join(str(index) for index in range(20)) + "\n",
            encoding="utf-8",
        )
        report = module._audit_test_windows(mapping, split)
    assert report["eligible_playlists"] == 1
    assert report["reference_slots"] == 15
    assert report["target_slots"] == 5
    assert report["missing_item_count"] == 0


def test_audit_separates_active_eight_from_stored_sixteen():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        vocab = ["<unk>"] + [f"cue-{index}" for index in range(1, 18)]
        mapping = {
            "a": list(range(1, 17)),
            "b": list(range(1, 9)) + [17] * 8,
        }
        (root / "cue_vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        (root / "item2cues.json").write_text(json.dumps(mapping), encoding="utf-8")
        report = module.audit(root)

    assert report["active_8_cue_health"]["slots_per_item"] == [8]
    assert report["active_8_cue_health"]["distinct_non_unk_assigned"] == 8
    assert report["stored_16_cue_health"]["slots_per_item"] == [16]
    assert report["stored_16_cue_health"]["distinct_non_unk_assigned"] == 17


if __name__ == "__main__":
    test_frozen_window_coverage_counts_references_and_targets()
    print("  PASS  test_frozen_window_coverage_counts_references_and_targets")
    test_audit_separates_active_eight_from_stored_sixteen()
    print("  PASS  test_audit_separates_active_eight_from_stored_sixteen")
