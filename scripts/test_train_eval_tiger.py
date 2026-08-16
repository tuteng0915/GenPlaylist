"""Tests for TIGER semantic-ID batching, constraints, and scheduling."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).with_name("train_eval_tiger.py")
SPEC = importlib.util.spec_from_file_location("train_eval_tiger", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _tokens() -> np.ndarray:
    return np.asarray([
        [2, 10, 20, 30],
        [2, 10, 20, 31],
        [2, 11, 21, 30],
        [3, 12, 22, 30],
        [3, 12, 23, 30],
        [4, 13, 24, 30],
    ], dtype=np.int64)


def test_trie_excludes_complete_items_without_blocking_siblings():
    trie = MODULE.SemanticTrie(_tokens())
    assert trie.allowed((), {0}) == [2, 3, 4]
    assert trie.allowed((2,), {0}) == [10, 11]
    assert trie.allowed((2, 10, 20), {0}) == [31]
    assert trie.allowed((2, 10, 20, 31), {0}) == [MODULE.EOS_TOKEN]


def test_batch_uses_only_suffix_next_item_transitions():
    rows = torch.tensor([list(range(20)), list(range(19, -1, -1))])
    semantic = torch.arange(40, 120).reshape(20, 4)
    generator = torch.Generator().manual_seed(3)
    inputs, attention, labels = MODULE._make_training_batch(
        rows,
        semantic,
        batch_size=8,
        generator=generator,
        device=torch.device("cpu"),
    )
    assert inputs.shape == (8, 76)
    assert attention.shape == inputs.shape
    assert labels.shape == (8, 5)
    assert set(attention.sum(dim=1).tolist()).issubset({60, 64, 68, 72, 76})
    assert torch.all(labels[:, -1].eq(MODULE.EOS_TOKEN))


def test_inverse_square_root_schedule():
    assert MODULE._learning_rate(1, peak=0.01, constant_steps=10_000) == 0.01
    assert MODULE._learning_rate(10_000, peak=0.01, constant_steps=10_000) == 0.01
    assert np.isclose(
        MODULE._learning_rate(40_000, peak=0.01, constant_steps=10_000), 0.005)


def test_constrained_generation_returns_a_valid_unseen_row():
    class Args:
        d_model = 32
        d_kv = 8
        d_ff = 64
        num_layers = 1
        num_heads = 2
        dropout = 0.0

    tokens = _tokens()
    trie = MODULE.SemanticTrie(tokens)
    model = MODULE._build_model(Args(), int(tokens.max()) + 1)
    decoded = MODULE._generate_next_rows(
        model,
        np.asarray([[0, 1]], dtype=np.int64),
        tokens,
        trie,
        beam_size=3,
        batch_size=1,
        device=torch.device("cpu"),
    )
    assert decoded.shape == (1,)
    assert int(decoded[0]) in {2, 3, 4, 5}


if __name__ == "__main__":
    test_trie_excludes_complete_items_without_blocking_siblings()
    test_batch_uses_only_suffix_next_item_transitions()
    test_inverse_square_root_schedule()
    test_constrained_generation_returns_a_valid_unseen_row()
    print("  PASS  TIGER protocol tests")
