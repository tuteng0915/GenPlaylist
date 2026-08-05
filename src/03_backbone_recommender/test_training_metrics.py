"""Checks that per-token-type training metrics only score the next item."""

from __future__ import annotations

import torch

from diffusion import Diffusion


class FakeTokenizer:
    n_digit = 3
    tokens_per_item = 13


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


if __name__ == "__main__":
    test_layer_nll_excludes_reference_items()
    print("  PASS  test_layer_nll_excludes_reference_items")
