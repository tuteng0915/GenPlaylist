"""Hydra configuration composition shared by standalone WP-C utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


WP_ROOT = Path(__file__).resolve().parent


def _device_count() -> int:
    """Return a usable device divisor without importing torch at module import."""
    try:
        import torch
    except ImportError:
        return 1
    return max(torch.cuda.device_count(), 1)


def compose_wp_c_config(overrides: Iterable[str] = ()):
    """Compose the complete WP-C config, including all Hydra default groups.

    Loading ``configs/config.yaml`` with ``OmegaConf.load`` is insufficient:
    Hydra's ``defaults`` entries are not expanded, so groups such as ``model``
    and ``data`` are absent. Standalone preparation and validation scripts use
    this helper to match the configuration seen by the training entry point.
    """
    import hydra
    from omegaconf import OmegaConf

    resolvers = {
        "cwd": lambda: os.getcwd(),
        "device_count": _device_count,
        "eval": eval,
        "div_up": lambda x, y: (x + y - 1) // y,
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)

    with hydra.initialize_config_dir(
            version_base=None, config_dir=str(WP_ROOT / "configs")):
        return hydra.compose(config_name="config", overrides=list(overrides))
