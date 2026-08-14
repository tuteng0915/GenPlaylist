"""Order-free metrics for five jointly generated songs against five targets."""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


def calculate_many_to_many_metrics(
    prediction_features: np.ndarray,
    target_features: np.ndarray,
    prediction_ids: np.ndarray,
    target_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate per-example Hungarian semantic and multiset item metrics.

    Both feature tensors are ``[batch, items, embedding_dim]``.  The two item
    axes must have equal length (five in the frozen protocol).  Semantic cosine
    similarity is maximized with a one-to-one Hungarian assignment.  Exact item
    scores use multiset intersection so duplicate samples receive no extra
    credit while remaining well-defined.
    """
    predictions = np.asarray(prediction_features, dtype=np.float32)
    targets = np.asarray(target_features, dtype=np.float32)
    pred_ids = np.asarray(prediction_ids, dtype=object)
    truth_ids = np.asarray(target_ids, dtype=object)
    if predictions.ndim != 3 or targets.ndim != 3:
        raise ValueError("Feature tensors must have shape [batch, items, dim]")
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction/target feature shapes differ: {predictions.shape} vs {targets.shape}")
    if pred_ids.shape != predictions.shape[:2] or truth_ids.shape != targets.shape[:2]:
        raise ValueError("ID tensors must match the feature batch and item dimensions")

    batch_size, item_count, _ = predictions.shape
    if item_count == 0:
        raise ValueError("Many-to-many evaluation requires at least one item")
    pred_norm = _l2_normalize(predictions)
    target_norm = _l2_normalize(targets)
    output = {
        "m2m_cosine": [],
        "m2m_exact_matches": [],
        "m2m_recall": [],
        "m2m_precision": [],
        "m2m_f1": [],
        "m2m_hit": [],
        "m2m_unique_ratio": [],
    }
    for batch_index in range(batch_size):
        similarities = pred_norm[batch_index] @ target_norm[batch_index].T
        rows, columns = linear_sum_assignment(-similarities)
        output["m2m_cosine"].append(float(similarities[rows, columns].mean()))

        pred_counter = Counter(pred_ids[batch_index].tolist())
        truth_counter = Counter(truth_ids[batch_index].tolist())
        matches = sum(
            min(count, truth_counter[item_id])
            for item_id, count in pred_counter.items()
        )
        recall = matches / max(sum(truth_counter.values()), 1)
        precision = matches / max(sum(pred_counter.values()), 1)
        f1 = 2 * recall * precision / max(recall + precision, np.finfo(np.float32).eps)
        output["m2m_exact_matches"].append(float(matches))
        output["m2m_recall"].append(float(recall))
        output["m2m_precision"].append(float(precision))
        output["m2m_f1"].append(float(f1))
        output["m2m_hit"].append(float(matches > 0))
        output["m2m_unique_ratio"].append(len(pred_counter) / item_count)

    return {
        name: np.asarray(values, dtype=np.float32)
        for name, values in output.items()
    }


def calculate_cue_multiset_metrics(
    prediction_cues: np.ndarray,
    target_cues: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compare generated and target cue multisets across each five-item bundle."""
    predictions = np.asarray(prediction_cues)
    targets = np.asarray(target_cues)
    if predictions.ndim != 3 or targets.ndim != 3:
        raise ValueError("Cue tensors must have shape [batch, items, cues]")
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction/target cue shapes differ: {predictions.shape} vs {targets.shape}")

    output = {
        "m2m_cue_recall": [],
        "m2m_cue_precision": [],
        "m2m_cue_f1": [],
        "m2m_cue_unique_ratio": [],
    }
    for batch_index in range(predictions.shape[0]):
        predicted = predictions[batch_index].reshape(-1).tolist()
        truth = targets[batch_index].reshape(-1).tolist()
        pred_counter = Counter(predicted)
        truth_counter = Counter(truth)
        matches = sum(
            min(count, truth_counter[cue])
            for cue, count in pred_counter.items()
        )
        recall = matches / max(len(truth), 1)
        precision = matches / max(len(predicted), 1)
        f1 = 2 * recall * precision / max(
            recall + precision, np.finfo(np.float32).eps)
        output["m2m_cue_recall"].append(recall)
        output["m2m_cue_precision"].append(precision)
        output["m2m_cue_f1"].append(f1)
        output["m2m_cue_unique_ratio"].append(
            len(pred_counter) / max(len(predicted), 1))

    return {
        name: np.asarray(values, dtype=np.float32)
        for name, values in output.items()
    }
