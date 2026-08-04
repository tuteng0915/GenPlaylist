"""Stable import path for the GenPlaylist cross-WP data contract.

The definitions currently remain in ``00_data_schema/schema.py`` so legacy
entry points that import ``schema`` keep working. New code should import from
``shared.schema``; the legacy module can become a thin compatibility shim after
all work packages have migrated.
"""

from __future__ import annotations

import sys
from pathlib import Path

_LEGACY_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "00_data_schema"
if str(_LEGACY_SCHEMA_DIR) not in sys.path:
    sys.path.insert(0, str(_LEGACY_SCHEMA_DIR))

from schema import (  # noqa: E402,F401
    BOI_TOKEN,
    CLHE_EMB_DIM,
    CONFLICT_OFFSET,
    CONFLICT_TOKEN_START,
    CONFLICT_VOCAB_SIZE,
    CUE_TOKEN_START,
    CUE_CANDIDATES_PER_ITEM,
    CUE_TOKENS,
    CUE_VOCAB_SIZE,
    EOS_TOKEN,
    MASK_TOKEN,
    RQ_N_CODEBOOKS,
    RQ_CODEBOOK_SIZE,
    RUNTIME_VOCAB_SIZE,
    SCHEMA_VERSION,
    TOKEN_LAYOUT,
    TOKEN_OFFSET,
    TOKENS_PER_ITEM,
    VOCAB_SIZE,
    CatalogItem,
    ContextPrefix,
    CueMappingEntry,
    GeneratedItem,
    SynthesisResult,
    TokenLayout,
)

__all__ = [
    "BOI_TOKEN",
    "CLHE_EMB_DIM",
    "CONFLICT_OFFSET",
    "CONFLICT_TOKEN_START",
    "CONFLICT_VOCAB_SIZE",
    "CUE_TOKEN_START",
    "CUE_CANDIDATES_PER_ITEM",
    "CUE_TOKENS",
    "CUE_VOCAB_SIZE",
    "EOS_TOKEN",
    "MASK_TOKEN",
    "RQ_N_CODEBOOKS",
    "RQ_CODEBOOK_SIZE",
    "RUNTIME_VOCAB_SIZE",
    "SCHEMA_VERSION",
    "TOKEN_LAYOUT",
    "TOKEN_OFFSET",
    "TOKENS_PER_ITEM",
    "VOCAB_SIZE",
    "CatalogItem",
    "ContextPrefix",
    "CueMappingEntry",
    "GeneratedItem",
    "SynthesisResult",
    "TokenLayout",
]
