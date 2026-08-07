"""Test synthesis pipeline using verbalization output.

Run from project root (with vibemus conda env active):
    python src/test_synthesis.py

This runs the full verbalization -> synthesis chain on mock data.
Produces one MP3 file in outputs/test/.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_data_schema"))
import numpy as np
from schema import (
    CatalogItem, GeneratedItem, ContextPrefix,
    CLHE_EMB_DIM
)
from verbalization import verbalize
from synthesis import synthesize

# ---------------------------------------------------------------------------
# 1. Same mock catalog as test_verbalization.py
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
catalog_embs = np.random.randn(
    len(catalog_metadata), CLHE_EMB_DIM
).astype(np.float32)

mock_generated = GeneratedItem(
    rvq_codes=(42, 17, 203),
    conflict_code=5,
    z_hat_emb=np.random.randn(CLHE_EMB_DIM).astype(np.float32),
    mu_c_emb=np.random.randn(CLHE_EMB_DIM).astype(np.float32),
    sigma_c2=0.3,
    cue_ids=list(range(8)),
    sample_idx=0,
    context_prefix=ContextPrefix(
        item_ids=["0", "1", "2"],
        source="song_only",
        raw_input="test playlist"
    )
)

# ---------------------------------------------------------------------------
# 2. Verbalize
# ---------------------------------------------------------------------------

print("Step 1: Verbalization (2 Qwen API calls)...")
def fake_llm(prompt, system=""):
    if "style analyst" in system:
        return "indie pop, melancholic, 95 BPM, guitar, D minor, English"
    return "[verse]\nNeon over quiet streets\n\n[chorus]\nWe find our way"

verb_result = verbalize(
    generated=mock_generated,
    catalog_embs=catalog_embs,
    catalog_metadata=catalog_metadata,
    k=3,
    llm_call=fake_llm,
)

print(f"Music attributes: {verb_result['music_attributes']}")
print(f"Lyric draft (first 3 lines):")
for line in verb_result['lyric_draft'].split('\n')[:3]:
    print(f"  {line}")

# ---------------------------------------------------------------------------
# 3. Synthesize
# ---------------------------------------------------------------------------

print("\nStep 2: Synthesis (ACE-Step, ~30 seconds of audio)...")
print("This will take 1-3 minutes on GPU...")

audio_path = synthesize(
    music_attributes=verb_result['music_attributes'],
    lyric_draft=verb_result['lyric_draft'],
    audio_duration=30.0,
    output_dir="outputs/test",
    filename="test_synthesis_output"
)

print(f"\nGenerated audio saved to: {audio_path}")
print(f"File size: {os.path.getsize(audio_path) / 1024 / 1024:.1f} MB")
print("\nFull pipeline test passed!")
