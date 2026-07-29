"""Test verbalization pipeline with mock GeneratedItem.

Run from project root:
    python src/test_verbalization.py
"""

import sys
import os
#sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, '/home/wjzhang/tt_workspace/model/GenPlaylist/src/04_synthesis')
sys.path.insert(0, '/home/wjzhang/tt_workspace/model/GenPlaylist/src/00_data_schema')
import numpy as np
from schema import (
    CatalogItem, GeneratedItem, ContextPrefix,
    CLHE_EMB_DIM
)
from verbalization import verbalize

# ---------------------------------------------------------------------------
# 1. Mock catalog (5 songs)
# ---------------------------------------------------------------------------

catalog_metadata = [
    CatalogItem(
        item_id="0",
        title="Slow Dancing in the Dark",
        artist="Joji",
        album="BALLADS 1",
        genre="indie pop",
        mood="melancholic",
        tempo=76.0,
        key="D minor",
        language="en",
        lyric_excerpt="I don't want to be alone when I'm fallin' in love"
    ),
    CatalogItem(
        item_id="1",
        title="Falling",
        artist="Harry Styles",
        album="Fine Line",
        genre="pop",
        mood="emotional",
        tempo=95.0,
        key="F major",
        language="en",
        lyric_excerpt="I'm in my bed and you're not here"
    ),
    CatalogItem(
        item_id="2",
        title="Blinding Lights",
        artist="The Weeknd",
        album="After Hours",
        genre="synth-pop",
        mood="energetic",
        tempo=171.0,
        key="F minor",
        language="en",
        lyric_excerpt="I said ooh I'm drowning in the night"
    ),
    CatalogItem(
        item_id="3",
        title="drivers license",
        artist="Olivia Rodrigo",
        album="SOUR",
        genre="pop",
        mood="sad",
        tempo=72.0,
        key="A major",
        language="en",
        lyric_excerpt="I got my driver's license last week"
    ),
    CatalogItem(
        item_id="4",
        title="Levitating",
        artist="Dua Lipa",
        album="Future Nostalgia",
        genre="dance pop",
        mood="upbeat",
        tempo=103.0,
        key="B major",
        language="en",
        lyric_excerpt="I got you moonlight, you're my starlight"
    ),
]

np.random.seed(42)
catalog_embs = np.random.randn(len(catalog_metadata), CLHE_EMB_DIM).astype(np.float32)

# ---------------------------------------------------------------------------
# 2. Mock GeneratedItem
# ---------------------------------------------------------------------------

mock_generated = GeneratedItem(
    rvq_codes=(42, 17, 203),
    conflict_code=5,
    z_hat_emb=np.random.randn(CLHE_EMB_DIM).astype(np.float32),
    mu_c_emb=np.random.randn(CLHE_EMB_DIM).astype(np.float32),
    sigma_c2=0.3,
    cue_ids=[],
    sample_idx=0,
    context_prefix=ContextPrefix(
        item_ids=["0", "1", "2"],
        source="song_only",
        raw_input="test playlist"
    )
)

mock_generated.validate()
print("GeneratedItem validation passed.")

# ---------------------------------------------------------------------------
# 3. Run verbalization
# ---------------------------------------------------------------------------

print("\nRunning verbalization (2 Qwen API calls)...\n")

result = verbalize(
    generated=mock_generated,
    catalog_embs=catalog_embs,
    catalog_metadata=catalog_metadata,
    k=3
)

# ---------------------------------------------------------------------------
# 4. Print results
# ---------------------------------------------------------------------------

print("=" * 60)
print("VERBALIZATION RESULT")
print("=" * 60)

print("\n--- Nearest Neighbors (z_hat position) ---")
for n in result["neighbors"]:
    print(f"  {n.to_prompt_line()}")

print("\n--- Style Summary (playlist centroid) ---")
for n in result["style_summary"]:
    print(f"  {n.to_prompt_line()}")

print("\n--- Music Attributes ---")
print(result["music_attributes"])

print("\n--- Lyric Draft ---")
print(result["lyric_draft"])

print("\n" + "=" * 60)
print("Test passed!")
print("=" * 60)
