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
    assert protocol.train_total_items == 20
    assert protocol.train_reference_items == 15
    assert protocol.train_target_items == 5
    assert protocol.eval_total_items == 20
    assert protocol.eval_reference_items == 15
    assert protocol.eval_target_items == protocol.eval_generated_items == 5
    assert protocol.eval_num_samples == 1
    assert protocol.model_token_length(13) == 262


def test_evaluation_split_is_first_fifteen_then_five():
    references, targets = FROZEN_NEXT_SONG_PROTOCOL.split_evaluation_items(
        [str(index) for index in range(25)])
    assert references == [str(index) for index in range(15)]
    assert targets == [str(index) for index in range(15, 20)]


def test_config_drift_is_rejected():
    try:
        FROZEN_NEXT_SONG_PROTOCOL.validate_config({"seq_len": 30})
    except ValueError as exc:
        assert "Frozen seq_len is 20" in str(exc)
    else:
        raise AssertionError("Expected a drifting training length to fail")




if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
