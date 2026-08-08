"""Checks joint-target training metrics and completion extraction."""

from __future__ import annotations

import torch

from diffusion import Diffusion


class FakeTokenizer:
    n_digit = 3
    tokens_per_item = 13
    bos_token = 101


class AttrDict(dict):
    __getattr__ = dict.__getitem__


def make_weight_config():
    return AttrDict(training=AttrDict(layer_loss_weights=AttrDict(
        enabled=True,
        normalize=True,
        rvq_weights=[2.0, 1.5, 1.0],
        conflict_weight=0.5,
        cue_weight=1.0,
        warmup=AttrDict(
            enabled=True,
            start_step=1000,
            end_step=5000,
            initial_rvq_weights=[2.0, 1.5, 1.0],
            initial_conflict_weight=0.5,
            initial_cue_weight=0.1,
        ),
    )))


def test_layer_nll_excludes_reference_items():
    model = object.__new__(Diffusion)
    model.tokenizer = FakeTokenizer()

    # BOS + three complete items + EOS. Only the third item's payload is target.
    seq_len = 1 + 3 * FakeTokenizer.tokens_per_item + 1
    x0 = torch.zeros((1, seq_len), dtype=torch.long)
    losses = torch.full((1, seq_len), 100.0)
    target_mask = torch.zeros((1, seq_len), dtype=torch.bool)
    target_start = 1 + 2 * FakeTokenizer.tokens_per_item
    target_mask[:, target_start + 1:target_start + FakeTokenizer.tokens_per_item] = True

    losses[:, target_start + 1] = 1.0
    losses[:, target_start + 2] = 2.0
    losses[:, target_start + 3] = 3.0
    losses[:, target_start + 4] = 4.0
    losses[:, target_start + 5:target_start + 13] = 5.0

    stats = model._compute_layer_nll_stats(x0, losses, target_mask=target_mask)
    assert torch.isclose(stats["layer_nll/d0"], torch.tensor(1.0))
    assert torch.isclose(stats["layer_nll/d1"], torch.tensor(2.0))
    assert torch.isclose(stats["layer_nll/d2"], torch.tensor(3.0))
    assert torch.isclose(stats["layer_nll/conflict"], torch.tensor(4.0))
    assert torch.isclose(stats["layer_nll/cues"], torch.tensor(5.0))


def test_extract_joint_five_item_completion():
    model = object.__new__(Diffusion)
    model.tokenizer = FakeTokenizer()

    context_width = 1 + 15 * FakeTokenizer.tokens_per_item
    full_width = context_width + 5 * FakeTokenizer.tokens_per_item + 1
    completed = torch.arange(full_width, dtype=torch.long).unsqueeze(0)
    completion_mask = torch.zeros_like(completed, dtype=torch.bool)
    for item_index in range(5):
        block_start = context_width + item_index * FakeTokenizer.tokens_per_item
        completion_mask[:, block_start + 1:block_start + FakeTokenizer.tokens_per_item] = True

    extracted = model.extract_item_completion(
        completed, completion_mask, num_items=5)

    assert extracted.shape == (1, 2 + 5 * FakeTokenizer.tokens_per_item)
    assert extracted[0, 0].item() == FakeTokenizer.bos_token
    assert torch.equal(extracted[0, 1:], completed[0, context_width:])


def test_cue_weight_warmup_and_normalization():
    model = object.__new__(Diffusion)
    model.tokenizer = FakeTokenizer()
    model.config = make_weight_config()

    rvq, conflict, cue = model._current_layer_loss_weights(global_step=0)
    assert rvq == [2.0, 1.5, 1.0]
    assert conflict == 0.5
    assert cue == 0.1

    _, _, midpoint_cue = model._current_layer_loss_weights(global_step=3000)
    assert abs(midpoint_cue - 0.55) < 1e-8
    _, _, final_cue = model._current_layer_loss_weights(global_step=5000)
    assert final_cue == 1.0

    sequence = torch.zeros((1, 15), dtype=torch.long)
    target_mask = torch.zeros_like(sequence, dtype=torch.bool)
    target_mask[:, 2:14] = True
    raw_weights = model._compute_position_weights(sequence, global_step=0)
    assert torch.equal(
        raw_weights[0, 2:6], torch.tensor([2.0, 1.5, 1.0, 0.5]))
    assert torch.allclose(raw_weights[0, 6:14], torch.full((8,), 0.1))
    normalized = model._normalize_active_position_weights(
        raw_weights, target_mask)
    assert torch.isclose(normalized[target_mask].mean(), torch.tensor(1.0))


if __name__ == "__main__":
    test_layer_nll_excludes_reference_items()
    test_extract_joint_five_item_completion()
    test_cue_weight_warmup_and_normalization()
    print("  PASS  test_layer_nll_excludes_reference_items")
    print("  PASS  test_extract_joint_five_item_completion")
    print("  PASS  test_cue_weight_warmup_and_normalization")
