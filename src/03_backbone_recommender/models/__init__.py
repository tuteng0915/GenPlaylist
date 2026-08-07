"""Backbone modules, with optional CUDA-heavy implementations loaded lazily."""

from __future__ import annotations

import importlib

from . import dit
from . import ema


def __getattr__(name: str):
  if name in {"dimamba", "autoregressive"}:
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
  raise AttributeError(name)
