"""Checks WP-B coverage accounting for the frozen evaluation windows."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


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


if __name__ == "__main__":
    test_frozen_window_coverage_counts_references_and_targets()
    print("  PASS  test_frozen_window_coverage_counts_references_and_targets")
