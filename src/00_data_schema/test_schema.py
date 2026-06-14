"""00_data_schema/test_schema.py — Smoke tests for all schema dataclasses.

Tests are grounded in the CURRENT backbone config:
  rq_n_codebooks=3, rq_codebook_size=256, no creative cues.

Run from repo root:
    python src/00_data_schema/test_schema.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))  # .../src/00_data_schema/

import numpy as np
from schema import (
    RQ_N_CODEBOOKS, RQ_CODEBOOK_SIZE, CUE_VOCAB_SIZE, CLHE_EMB_DIM,
    CatalogItem, ContextPrefix, CueMappingEntry, GeneratedItem, SynthesisResult,
)


def test_catalog_item():
    item = CatalogItem(
        item_id="42",
        feature_index=41,      # 0-indexed row in clhe.pt
        title="Neon Rain",
        artist="The Static",
        genre="indie pop",
        mood="melancholic",
        tempo=95.0,
        key="A minor",
        language="en",
        lyric_excerpt="Staring at the neon rain",
        tags=["guitar", "reverb"],
    )
    line = item.to_prompt_line()
    assert "Neon Rain" in line
    assert "A minor" in line
    assert "95 BPM" in line
    print(f"  to_prompt_line: {line}")


def test_catalog_item_minimal():
    item = CatalogItem(item_id="1", title="Unknown Track")
    line = item.to_prompt_line()
    assert "Unknown Track" in line


def test_catalog_item_from_metadata_string():
    # Spotify backbone metadata.json format: "'Title' by Artist in album'Album'"
    item = CatalogItem.from_metadata_string(
        "0", "'Hegira' by Within The Ruins in album'Phenomena'"
    )
    assert item.item_id == "0"
    assert item.feature_index == 0      # auto-assigned from item_id
    assert item.title  == "Hegira"
    assert item.artist == "Within The Ruins"
    assert item.album  == "Phenomena"

    # Fallback for unparseable string
    item2 = CatalogItem.from_metadata_string("5", "some weird format string")
    assert item2.item_id == "5"
    assert item2.title   != ""          # should at least have something


def test_context_prefix_valid():
    ctx = ContextPrefix(item_ids=["1", "2", "3"], source="song_only")
    ctx.validate()

    ctx_hybrid = ContextPrefix(
        item_ids=["1", "2"],
        source="hybrid",
        items=[CatalogItem(item_id="1"), CatalogItem(item_id="2")],
    )
    ctx_hybrid.validate()


def test_context_prefix_invalid():
    try:
        ContextPrefix(item_ids=[]).validate()
        assert False, "empty item_ids should fail"
    except AssertionError:
        pass

    try:
        ContextPrefix(
            item_ids=["1", "2"],
            items=[CatalogItem(item_id="1")],  # length mismatch
        ).validate()
        assert False
    except AssertionError:
        pass


def test_cue_mapping_entry_valid():
    entry = CueMappingEntry(item_id="99", cue_ids=[0, 100, 200, 300, 400, 500])
    entry.validate()


def test_cue_mapping_entry_invalid():
    try:
        CueMappingEntry(item_id="1", cue_ids=[0, 1, 2]).validate()  # too few
        assert False
    except AssertionError:
        pass
    try:
        CueMappingEntry(item_id="1", cue_ids=[0, 1, 2, 3, 4, CUE_VOCAB_SIZE]).validate()  # out of range
        assert False
    except AssertionError:
        pass


def test_generated_item_current_backbone():
    """Current backbone: 3 RVQ codes, codebook size 256, no cue_ids."""
    d = CLHE_EMB_DIM  # 64, confirmed from clhe_weight.npy shape (768, 64)
    item = GeneratedItem(
        rvq_codes=(0, 128, 255),   # 3 codes, each in [0, 256)
        conflict_code=5,
        z_hat_emb=np.random.randn(d).astype(np.float32),
        mu_c_emb=np.random.randn(d).astype(np.float32),
        sigma_c2=0.42,
        cue_ids=[],                # empty = not yet extended
        sample_idx=0,
    )
    item.validate()
    assert item.cue_ids == []


def test_generated_item_with_cues():
    """Future state: 3 RVQ + 6 creative cues (after WP-B + tokenizer update)."""
    d = CLHE_EMB_DIM
    item = GeneratedItem(
        rvq_codes=(0, 128, 255),
        conflict_code=5,
        z_hat_emb=np.random.randn(d).astype(np.float32),
        mu_c_emb=np.random.randn(d).astype(np.float32),
        sigma_c2=0.8,
        cue_ids=[10, 20, 30, 40, 50, 60],
        sample_idx=1,
    )
    item.validate()


def test_generated_item_invalid_codes():
    d = CLHE_EMB_DIM
    base = dict(
        conflict_code=5,
        z_hat_emb=np.random.randn(d).astype(np.float32),
        mu_c_emb=np.random.randn(d).astype(np.float32),
        sigma_c2=0.5,
    )
    # Wrong number of codes
    try:
        GeneratedItem(rvq_codes=(0, 1), **base).validate()  # only 2 instead of 3
        assert False
    except AssertionError:
        pass
    # Out-of-range code
    try:
        GeneratedItem(rvq_codes=(0, 128, RQ_CODEBOOK_SIZE), **base).validate()
        assert False
    except AssertionError:
        pass
    # Negative sigma_c2
    try:
        GeneratedItem(rvq_codes=(0, 1, 2), **{**base, "sigma_c2": -1.0}).validate()
        assert False
    except AssertionError:
        pass
    # Dimension mismatch: mu_c_emb has wrong shape
    try:
        GeneratedItem(
            rvq_codes=(0, 1, 2),
            **{**base, "mu_c_emb": np.zeros(128)}  # wrong dim (expected 64)
        ).validate()
        assert False
    except AssertionError:
        pass


def test_synthesis_result_no_audio():
    result = SynthesisResult(
        audio_path="/nonexistent/path.wav",
        music_attributes="pop, 120 BPM",
        lyric_draft="[verse]\nHello world",
    )
    try:
        result.validate()
        assert False, "should fail — audio file does not exist"
    except AssertionError:
        pass


if __name__ == "__main__":
    tests = [
        test_catalog_item,
        test_catalog_item_minimal,
        test_catalog_item_from_metadata_string,
        test_context_prefix_valid,
        test_context_prefix_invalid,
        test_cue_mapping_entry_valid,
        test_cue_mapping_entry_invalid,
        test_generated_item_current_backbone,
        test_generated_item_with_cues,
        test_generated_item_invalid_codes,
        test_synthesis_result_no_audio,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
