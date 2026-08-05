"""Stochastic settings for the official post-training WP-C evaluation.

This is deliberately separate from ``shared.protocol``: sampling settings do
not define prepared-data bytes and therefore must not invalidate that cache.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class OfficialEvaluationProtocol:
    sampling_steps: int = 256
    seed: int = 1
    use_ema: bool = True

    def as_dict(self) -> dict:
        return asdict(self)

    def validate_config(
        self, config: Mapping, *, allow_override: bool = False,
    ) -> "OfficialEvaluationProtocol":
        sampling = config.get("sampling", {})
        evaluation = config.get("eval", {})
        actual = {
            "seed": int(config.get("seed", self.seed)),
            "sampling.steps": int(sampling.get("steps", self.sampling_steps)),
            "eval.disable_ema": bool(evaluation.get("disable_ema", not self.use_ema)),
        }
        expected = {
            "seed": self.seed,
            "sampling.steps": self.sampling_steps,
            "eval.disable_ema": not self.use_ema,
        }
        drift = {
            name: (expected[name], value)
            for name, value in actual.items() if value != expected[name]
        }
        if drift and not allow_override:
            rendered = ", ".join(
                f"{name}: expected {wanted}, got {found}"
                for name, (wanted, found) in drift.items())
            raise ValueError(
                "Official GenPlaylist evaluation settings drifted: " + rendered)
        return self


OFFICIAL_EVALUATION_PROTOCOL = OfficialEvaluationProtocol()
