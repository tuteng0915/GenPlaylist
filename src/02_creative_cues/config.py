"""config.py — named presets for run_production.py.

Production runs shouldn't need a wall of CLI flags — pick a preset by name and
everything else is fixed here. Add a new preset (or tweak DEFAULT) when a
setting changes, instead of adding a new flag.

The DEFAULT preset reflects what the rank-by / min-df / cue-count comparisons
in this project concluded (see METHOD_WRITEUP.md and the run reports under
outputs/runs/): `idf` ranking held its own against `cluster` and `band` on
vocab health and reconstruction quality, `llm` extraction produced the most
diverse and coherent cues of the four methods, and 18 cues/song is the budget
those comparisons were run and validated at.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_data_schema"))
from schema import CUE_VOCAB_SIZE  # noqa: E402


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
    min_df: int = 5                     # min songs a cue must appear in
    max_df_frac: float = 0.3            # drop cues in more than this fraction of songs
    dedup_threshold: float = 0.92       # cosine above which near-duplicate cues merge
    rank_by: str = "idf"                # stage-5 vocabulary selection rule
    vocab_size: int = CUE_VOCAB_SIZE    # 2048; do not change without 00_data_schema sign-off

    # Step 3 — assignment
    num_cues: int = 18                  # cues assigned per song (schema default is 6 — see
                                         # METHOD_WRITEUP.md's non-default-cue-count note; this
                                         # preset intentionally uses the validated 18-cue budget)

    force: bool = False                 # bypass extraction/cleaning caches


DEFAULT = ProductionConfig()

# Named alternates, derived from DEFAULT so they only need to state the delta.
# Add new presets here as decisions change, rather than new CLI flags.
PRESETS: dict[str, ProductionConfig] = {
    "default": DEFAULT,
    "tfidf": replace(DEFAULT, method="tfidf"),
    "schema-default-cues": replace(DEFAULT, num_cues=6),
}


def get_preset(name: str) -> ProductionConfig:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown config preset '{name}'. Available: {', '.join(sorted(PRESETS))}"
        ) from None
