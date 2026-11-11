"""Tests of paired inference and rejection of misaligned prediction rows."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from compare_proxy_predictions import paired_summary, score_predictions


def test_pairing_retains_shared_noise():
    baseline = {"metric": np.arange(50, dtype=float)}
    variant = {"metric": baseline["metric"] + 0.25}
    result = paired_summary(baseline, variant, 1000, 42)["metric"]
    assert result == {"difference": 0.25, "ci95": [0.25, 0.25]}


def test_reordered_targets_rejected_and_duplicates_not_overcredited():
    targets = np.asarray([["0", "1", "2", "3", "4"], ["1", "2", "3", "4", "5"]])
    payload = {"predictions": {"target_item_ids": targets.tolist(), "item_ids": [["0"] * 5, ["1"] * 5]}}
    with TemporaryDirectory() as directory:
        path = Path(directory) / "predictions.json"
        path.write_text(json.dumps(payload))
        metrics, _ = score_predictions(path, targets, {str(i): i for i in range(6)}, np.eye(6, dtype=np.float32))
        assert np.isclose(metrics["recall_at_5"], 0.2)
        try:
            score_predictions(path, targets[::-1], {str(i): i for i in range(6)}, np.eye(6, dtype=np.float32))
        except ValueError as error:
            assert "Target/order mismatch" in str(error)
        else:
            raise AssertionError("Reordered histories accepted")


if __name__ == "__main__":
    test_pairing_retains_shared_noise()
    test_reordered_targets_rejected_and_duplicates_not_overcredited()
    print("PASS paired comparison and history alignment")
