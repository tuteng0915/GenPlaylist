"""Frozen cross-WP next-song training and evaluation protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class NextSongProtocol:
    """Single source of truth for the GenPlaylist-v1 experiment split."""

    min_reference_items: int = 2
    train_total_items: int = 16
    eval_total_items: int = 20
    eval_reference_items: int = 15
    eval_target_items: int = 5
    eval_num_samples: int = 5

    @property
    def train_reference_items(self) -> int:
        return self.train_total_items - 1

    def model_token_length(self, tokens_per_item: int) -> int:
        return 2 + self.train_total_items * int(tokens_per_item)

    def validate(self) -> "NextSongProtocol":
        if self.min_reference_items < 2:
            raise ValueError("Next-song generation requires at least two references")
        if self.train_total_items != self.train_reference_items + 1:
            raise ValueError("Training must reserve exactly one next-item target")
        if self.eval_total_items != self.eval_reference_items + self.eval_target_items:
            raise ValueError("Evaluation references and targets must sum to eval_total_items")
        if self.eval_num_samples != self.eval_target_items:
            raise ValueError("Evaluation must draw one sample per ground-truth item")
        return self

    def validate_config(self, config: Mapping) -> "NextSongProtocol":
        """Reject Hydra/dict overrides that drift from the frozen contract."""
        configured_seq_len = int(config.get("seq_len", self.train_total_items))
        if configured_seq_len != self.train_total_items:
            raise ValueError(
                f"Frozen seq_len is {self.train_total_items}, got {configured_seq_len}")
        protocol = config.get("protocol", {})
        expected = {
            "min_reference_items": self.min_reference_items,
            "eval_total_items": self.eval_total_items,
            "eval_reference_items": self.eval_reference_items,
            "eval_target_items": self.eval_target_items,
            "eval_num_samples": self.eval_num_samples,
        }
        for name, value in expected.items():
            configured = int(protocol.get(name, value))
            if configured != value:
                raise ValueError(
                    f"Frozen protocol.{name} is {value}, got {configured}")
        return self

    def split_evaluation_items(
        self, item_ids: Sequence[str],
    ) -> tuple[list[str], list[str]]:
        """Return the deterministic first-20 15-reference/5-target window."""
        ids = [str(item_id) for item_id in item_ids]
        if len(ids) < self.eval_total_items:
            raise ValueError(
                f"Evaluation requires at least {self.eval_total_items} items, got {len(ids)}")
        window = ids[:self.eval_total_items]
        return (
            window[:self.eval_reference_items],
            window[self.eval_reference_items:],
        )


FROZEN_NEXT_SONG_PROTOCOL = NextSongProtocol().validate()
