"""Dependency-light checks for the frozen cross-WP protocol."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shared.protocol import FROZEN_NEXT_SONG_PROTOCOL


def test_frozen_lengths_and_model_width():
    protocol = FROZEN_NEXT_SONG_PROTOCOL
    assert protocol.train_total_items == 16
    assert protocol.train_reference_items == 15
    assert protocol.eval_total_items == 20
    assert protocol.eval_reference_items == 15
    assert protocol.eval_target_items == protocol.eval_num_samples == 5
    assert protocol.eval_sampling_steps == 256
    assert protocol.eval_seed == 1
    assert protocol.eval_use_ema is True
    assert protocol.model_token_length(13) == 210


def test_evaluation_split_is_first_fifteen_then_five():
    references, targets = FROZEN_NEXT_SONG_PROTOCOL.split_evaluation_items(
        [str(index) for index in range(25)])
    assert references == [str(index) for index in range(15)]
    assert targets == [str(index) for index in range(15, 20)]


def test_config_drift_is_rejected():
    try:
        FROZEN_NEXT_SONG_PROTOCOL.validate_config({"seq_len": 30})
    except ValueError as exc:
        assert "Frozen seq_len is 16" in str(exc)
    else:
        raise AssertionError("Expected a drifting training length to fail")


def test_official_stochastic_eval_settings_are_frozen():
    protocol = FROZEN_NEXT_SONG_PROTOCOL
    protocol.validate_evaluation_config({
        "seed": 1,
        "sampling": {"steps": 256},
        "eval": {"disable_ema": False},
    })
    try:
        protocol.validate_evaluation_config({
            "seed": 1,
            "sampling": {"steps": 25},
            "eval": {"disable_ema": False},
        })
    except ValueError as exc:
        assert "sampling.steps" in str(exc)
    else:
        raise AssertionError("Expected official sampling-step drift to be rejected")

    protocol.validate_evaluation_config({
        "seed": 99,
        "sampling": {"steps": 1},
        "eval": {"disable_ema": True},
    }, allow_override=True)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
