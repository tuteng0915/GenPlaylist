"""data_loading.py — shared catalog/lyrics loading for every WP-B entry point.

Every run script (run_production.py, run_compare.py, sweeps/*) used to define
its own near-identical load_data(). Collapsed here so there's one place that
knows where the repo's data lives and how lyrics get preprocessed.

Path note: this module lives at <repo_root>/src/02_creative_cues/data_loading.py,
two directories below the repo root — REPO_ROOT is computed relative to THIS
file, not the caller's __file__, so callers get the right path regardless of
where they themselves live (top-level, sweeps/, or legacy/).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(Path(__file__).resolve().parent))               # 02_creative_cues/ (siblings)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_data_schema"))
from schema import CatalogItem  # noqa: E402
import cue_lyrics  # noqa: E402
import cue_clients  # noqa: E402  (side effect: loads repo-root .env before the env lookups below)

# Overridable so a different machine (e.g. a remote server with data mounted
# elsewhere) can point elsewhere without touching code — set in the repo-root
# .env or the environment. Unset = same paths as before.
CATALOG_PATH = Path(os.environ.get(
    "CUE_CATALOG_PATH", str(REPO_ROOT / "data" / "dataset" / "catalog_metadata.json")))
LYRICS_DIR = Path(os.environ.get(
    "CUE_LYRICS_DIR", str(REPO_ROOT / "data" / "lyrics" / "spotify")))

_CATALOG_CACHE: Optional[list[tuple[str, dict]]] = None


def _raw_catalog() -> list[tuple[str, dict]]:
    """Load + cache catalog_metadata.json's raw (item_id, entry) pairs, in file order."""
    global _CATALOG_CACHE
    if _CATALOG_CACHE is None:
        with open(CATALOG_PATH, encoding="utf-8") as f:
            _CATALOG_CACHE = list(json.load(f).items())
    return _CATALOG_CACHE


def load_catalog_and_lyrics(
    limit: Optional[int] = None,
) -> tuple[list[CatalogItem], dict[str, str]]:
    """Load the first `limit` catalog items (None = all) + their raw lyrics text.

    Returns (items, lyrics_raw) where lyrics_raw is {item_id: raw lyric string},
    present only for items that have a lyrics file on disk.
    """
    entries = _raw_catalog()[:limit] if limit else _raw_catalog()
    items = [
        CatalogItem(**{k: v for k, v in entry.items() if k in CatalogItem.__dataclass_fields__})
        for _iid, entry in entries
    ]
    lyrics_raw: dict[str, str] = {}
    for it in items:
        path = LYRICS_DIR / f"{it.item_id}.txt"
        if path.is_file():
            lyrics_raw[it.item_id] = path.read_text(encoding="utf-8", errors="ignore")
    return items, lyrics_raw


def build_lyrics_proc(
    lyrics_raw: dict[str, str], lyrics_mode: str, lyrics_cap: int
) -> dict[str, str]:
    """Preprocess raw lyrics per `lyrics_mode` (cap/full/dedup/summarize). See cue_lyrics.MODES."""
    return {
        iid: cue_lyrics.preprocess_lyrics(text, lyrics_mode, lyrics_cap)
        for iid, text in lyrics_raw.items()
    }
