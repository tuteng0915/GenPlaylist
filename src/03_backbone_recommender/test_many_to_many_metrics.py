"""Tests for fixed-set Hungarian matching metrics."""

from __future__ import annotations

import numpy as np

from many_to_many_metrics import (
    calculate_cue_multiset_metrics,
    calculate_many_to_many_metrics,
)


def test_permuted_five_by_five_predictions_match_perfectly():
    targets = np.eye(5, dtype=np.float32)[None, :, :]
    predictions = targets[:, [3, 1, 4, 0, 2], :]
    target_ids = np.asarray([["a", "b", "c", "d", "e"]], dtype=object)
    prediction_ids = target_ids[:, [3, 1, 4, 0, 2]]
    metrics = calculate_many_to_many_metrics(
        predictions, targets, prediction_ids, target_ids)

    assert np.allclose(metrics["m2m_cosine"], 1.0)
    assert np.allclose(metrics["m2m_exact_matches"], 5.0)
    assert np.allclose(metrics["m2m_recall"], 1.0)
    assert np.allclose(metrics["m2m_unique_ratio"], 1.0)


def test_duplicate_samples_receive_only_multiset_credit():
    targets = np.eye(5, dtype=np.float32)[None, :, :]
    predictions = np.repeat(targets[:, :1, :], 5, axis=1)
    target_ids = np.asarray([["a", "b", "c", "d", "e"]], dtype=object)
    prediction_ids = np.asarray([["a", "a", "a", "a", "a"]], dtype=object)
    metrics = calculate_many_to_many_metrics(
        predictions, targets, prediction_ids, target_ids)

    assert np.allclose(metrics["m2m_exact_matches"], 1.0)
    assert np.allclose(metrics["m2m_recall"], 0.2)
    assert np.allclose(metrics["m2m_precision"], 0.2)
    assert np.allclose(metrics["m2m_unique_ratio"], 0.2)


def test_cue_metrics_are_order_free_and_multiset_aware():
    targets = np.asarray([[[1, 2], [2, 3]]])
    permuted = np.asarray([[[3, 2], [2, 1]]])
    perfect = calculate_cue_multiset_metrics(permuted, targets)
    assert np.allclose(perfect["m2m_cue_recall"], 1.0)
    assert np.allclose(perfect["m2m_cue_precision"], 1.0)

    duplicated = np.asarray([[[1, 1], [1, 1]]])
    partial = calculate_cue_multiset_metrics(duplicated, targets)
    assert np.allclose(partial["m2m_cue_recall"], 0.25)
    assert np.allclose(partial["m2m_cue_unique_ratio"], 0.25)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
