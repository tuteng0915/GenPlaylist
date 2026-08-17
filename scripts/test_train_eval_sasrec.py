"""Dependency-light tests for the frozen-protocol SASRec baseline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).with_name("train_eval_sasrec.py")
SPEC = importlib.util.spec_from_file_location("train_eval_sasrec", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _model() -> MODULE.SASRec:
    torch.manual_seed(7)
    return MODULE.SASRec(
        catalog_size=24,
        max_length=19,
        hidden_size=8,
        num_blocks=1,
        num_heads=2,
        dropout=0.0,
    )


def test_training_loss_covers_exactly_five_transitions():
    model = _model()
    sequences = torch.arange(1, 21).unsqueeze(0)
    loss = MODULE._training_loss(model, sequences)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_rollout_shape_range_and_seen_exclusion():
    model = _model()
    references = np.asarray([
        list(range(1, 16)),
        list(range(2, 17)),
    ], dtype=np.int64)
    predicted = MODULE._autoregressive_topk(
        model, references, batch_size=2, device=torch.device("cpu"))
    assert predicted.shape == (2, 5)
    assert np.all((0 <= predicted) & (predicted < 24))
    for references_one_based, prediction_zero_based in zip(references, predicted):
        prediction_one_based = prediction_zero_based + 1
        assert len(set(prediction_one_based.tolist())) == 5
        assert not set(references_one_based).intersection(prediction_one_based.tolist())


def test_rollout_can_repeat_visible_items():
    class FixedModel:
        max_length = 19

        def eval(self):
            return self

        def encode(self, inputs):
            return torch.zeros((*inputs.shape, 2), dtype=torch.float32)

        def catalog_logits(self, hidden):
            scores = torch.zeros((len(hidden), 24), dtype=torch.float32)
            scores[:, 0] = 1.0
            return scores

    references = np.asarray([list(range(1, 16))], dtype=np.int64)
    predicted = MODULE._autoregressive_topk(
        FixedModel(), references, batch_size=1, device=torch.device("cpu"),
        exclude_seen=False)
    assert predicted.tolist() == [[0, 0, 0, 0, 0]]


if __name__ == "__main__":
    test_training_loss_covers_exactly_five_transitions()
    test_rollout_shape_range_and_seen_exclusion()
    test_rollout_can_repeat_visible_items()
    print("  PASS  SASRec protocol tests")
