"""config.py — named presets for run_production.py.

Production runs shouldn't need a wall of CLI flags — pick a preset by name and
everything else is fixed here. Add a new preset (or tweak DEFAULT) when a
setting changes, instead of adding a new flag.

The DEFAULT preset is the cross-WP production contract.  Experimental settings
remain available as explicitly named presets, but must never silently become
the artifacts consumed by WP-C. The manifest key ``wp_d_compatible`` retains
its legacy name for artifact backward compatibility.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_data_schema"))
from schema import CUE_CANDIDATES_PER_ITEM, CUE_TOKENS, CUE_VOCAB_SIZE  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cue_normalize  # noqa: E402


@dataclass(frozen=True)
class ProductionConfig:
    """Fixed settings for a production vocab+mapping build. No evaluation fields
    here on purpose — run_production.py never scores anything, so held-out
    splits, eval-sample sizes, and Level 1-3 settings don't apply.
    """

    method: str = "llm"                 # tfidf | yake | keybert | llm
    limit: int | None = None            # None = full catalog

    # Step 1 — extraction
    top_n: int = 100                    # raw candidate cues kept per song before cleaning

    # Step 0 — lyrics preprocessing
    lyrics_mode: str = "dedup"          # cap | full | dedup | summarize
    lyrics_cap: int = 2000              # char cap for cap/dedup modes

    # Step 2 — cleaning
    min_df_frac: float = cue_normalize.MIN_DF_FRAC_DEFAULT
    # ^ min songs a cue must appear in, as a fraction of the corpus actually being
    # built (resolve_min_df(min_df_frac, len(items)), floored at MIN_DF_FLOOR=2) —
    # not a fixed absolute count, so smoke tests (--limit) and future catalog growth
    # both get a proportionally appropriate floor instead of reusing the full-catalog
    # value. Validated in sweeps/sweep_min_df.py; see resolve_min_df's docstring.
    max_df_frac: float = 0.3            # drop cues in more than this fraction of songs
    dedup_threshold: float = 0.92       # cosine above which near-duplicate cues merge
    rank_by: str = "idf"                # stage-5 vocabulary selection rule
    vocab_size: int = CUE_VOCAB_SIZE    # 2048; do not change without 00_data_schema sign-off

    # Step 3 — assignment. Store a ranked master list; WP-C consumes its first
    # CUE_TOKENS entries so cue-count ablations never rebuild the mapping.
    num_cues: int = CUE_CANDIDATES_PER_ITEM
    active_cues: int = CUE_TOKENS
    assignment_strategy: str = "relevance"
    candidate_k: int = 64
    embedder: str = "minilm"

    force: bool = False                 # bypass extraction/cleaning caches


DEFAULT = ProductionConfig()

# Named alternates, derived from DEFAULT so they only need to state the delta.
# Add new presets here as decisions change, rather than new CLI flags.
PRESETS: dict[str, ProductionConfig] = {
    "default": DEFAULT,
    "tfidf": replace(DEFAULT, method="tfidf"),
    # Paper/ablation compatibility only. It does not match the frozen 16-row
    # master-table contract and is therefore marked non-production.
    "research-18-cues": replace(DEFAULT, num_cues=18),
}


def get_preset(name: str) -> ProductionConfig:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown config preset '{name}'. Available: {', '.join(sorted(PRESETS))}"
        ) from None
