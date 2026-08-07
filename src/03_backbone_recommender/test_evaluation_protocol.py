"""Dependency-light checks for official stochastic evaluation settings."""

from evaluation_protocol import OFFICIAL_EVALUATION_PROTOCOL


def test_official_stochastic_eval_settings_are_frozen():
    protocol = OFFICIAL_EVALUATION_PROTOCOL
    assert protocol.as_dict() == {
        "sampling_steps": 256,
        "seed": 1,
        "use_ema": True,
    }
    protocol.validate_config({
        "seed": 1,
        "sampling": {"steps": 256},
        "eval": {"disable_ema": False},
    })
    try:
        protocol.validate_config({
            "seed": 1,
            "sampling": {"steps": 25},
            "eval": {"disable_ema": False},
        })
    except ValueError as exc:
        assert "sampling.steps" in str(exc)
    else:
        raise AssertionError("Expected official sampling-step drift to be rejected")

    protocol.validate_config({
        "seed": 99,
        "sampling": {"steps": 1},
        "eval": {"disable_ema": True},
    }, allow_override=True)


if __name__ == "__main__":
    test_official_stochastic_eval_settings_are_frozen()
    print("  PASS  test_official_stochastic_eval_settings_are_frozen")
