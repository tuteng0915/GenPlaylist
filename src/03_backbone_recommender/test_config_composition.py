"""Regression checks for standalone Hydra configuration composition."""

from __future__ import annotations

from config_composition import compose_wp_c_config


def test_default_groups_are_composed():
    config = compose_wp_c_config()
    assert config.model.name == "small"
    assert config.model.length == 262
    assert config.data.train == "spotify"
    assert config.sampling.structure_conditioning is False


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
