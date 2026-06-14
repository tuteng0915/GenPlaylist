"""00_data_schema/schema.py — Data format contracts shared across all Work Packages.

Every WP reads from and writes to these formats.
Changes here must be coordinated with all module owners.

Pipeline flow
-------------
raw user input
  ─[WP-A]──► ContextPrefix          (01_input_normalization → 03_backbone_recommender)
lyrics / metadata
  ─[WP-B]──► CueMappingEntry        (02_creative_cues → 03_backbone_recommender tokenizer)
CLHE embedding space
  ─[WP-D]──► CatalogItem            (shared catalog metadata format used by all WPs)
diffusion model output
  ─[WP-D]──► GeneratedItem          (03_backbone_recommender → 04_synthesis)
ACE-Step output
  ─[WP-C]──► SynthesisResult        (04_synthesis → pipeline / evaluation / demo)

Backbone constants (verified against Spotify dataset)
------------------------------------------------------
  Dataset path  : 03_backbone_recommender/datasets/spotify/
  rq_n_codebooks  = 3          (config.yaml)
  rq_codebook_size = 256       (config.yaml)
  CLHE_EMB_DIM    = 64         (clhe_weight.npy shape: (768, 64))
  tokens_per_item = 5          BOI(1) + z0 + z1 + z2 + z_conf
  Token offsets   :
    z0    ∈ [1,   256]   offset = 256*0 + 1
    z1    ∈ [257, 512]   offset = 256*1 + 1
    z2    ∈ [513, 768]   offset = 256*2 + 1
    z_conf∈ [769, 1024]  offset = 256*3     (conflict digit)
    BOI   = 1025         (rq_n_codebooks+1)*rq_codebook_size + 1
    EOS   = 1026
  vocab_size (training) = 1027
  vocab_size (runtime)  = 1028  (+1 MASK token added in diffusion.py)
  Item IDs     : str(int), 0-indexed, matching metadata.json / token.json keys
  feature_index: int(item_id) — direct row index into clhe_weight.npy merged codebook

GenPlaylist extension (TODO — not yet in backbone)
---------------------------------------------------
  rq_n_codebooks  = 4    (paper target)
  rq_codebook_size = 128
  CUE_TOKENS      = 6    (WP-B deliverable)
  tokens_per_item = 12   BOI + z0..z3 + z_conf + c0..c5
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Backbone constants  (do NOT change without updating backbone config)
# ---------------------------------------------------------------------------

RQ_N_CODEBOOKS    = 3
RQ_CODEBOOK_SIZE  = 256
CLHE_EMB_DIM      = 64    # verified: clhe_weight.npy shape = (768, 64)
TOKENS_PER_ITEM   = 5     # BOI + RQ_N_CODEBOOKS codes + 1 conflict digit
BOI_TOKEN         = (RQ_N_CODEBOOKS + 1) * RQ_CODEBOOK_SIZE + 1   # = 1025
EOS_TOKEN         = BOI_TOKEN + 1                                   # = 1026
VOCAB_SIZE        = EOS_TOKEN + 1                                   # = 1027 (training)

# Token offset per RVQ level  (token_id = raw_code + TOKEN_OFFSET[level])
TOKEN_OFFSET = [RQ_CODEBOOK_SIZE * level + 1 for level in range(RQ_N_CODEBOOKS)]
# z_conf offset = RQ_N_CODEBOOKS * RQ_CODEBOOK_SIZE  (= 768)
CONFLICT_OFFSET = RQ_N_CODEBOOKS * RQ_CODEBOOK_SIZE   # = 768

# GenPlaylist extension (TODO)
CUE_TOKENS   = 0     # TODO: 6 after WP-B delivers item2cues.json
CUE_VOCAB_SIZE = 2048


# ---------------------------------------------------------------------------
# CatalogItem
# ---------------------------------------------------------------------------

@dataclass
class CatalogItem:
    """Metadata for a single catalog song.

    Source: datasets/{dataset}/metadata.json + item_info.json (if available).

    Item ID convention
    ------------------
    item_id = str(int), 0-indexed.
    metadata.json uses these as string keys: '0', '1', '2', ...
    feature_index = int(item_id) — direct row index into clhe_weight.npy.

    Raw token_ids for a given item can be read from clhe_token.json[item_id]:
        [z0_token, z1_token, z2_token, z_conf_token]
    Raw codes (before offset) are in clhe_sid.npy[int(item_id)]:
        (z0_raw, z1_raw, z2_raw, conflict_digit)

    Fields
    ------
    item_id         : string item ID ('0', '1', ...).
    feature_index   : int(item_id) — row index in clhe_weight.npy.
    title           : song title.
    artist          : artist name.
    album           : album name (available in Spotify metadata string).
    genre           : primary genre (not in Spotify; may come from audio tags).
    mood            : mood label (not in Spotify; may come from audio tags).
    tempo           : BPM (not in raw dataset; None by default).
    key             : musical key (not in raw dataset; None by default).
    language        : ISO 639-1 code; None if not available.
    lyric_excerpt   : short lyric snippet for verbalization prompts.
    audio_path      : path to raw audio; None if not locally available.
    tags            : free-form tag list.
    """
    item_id: str
    feature_index: int = -1
    title: str = ""
    artist: str = ""
    album: str = ""
    genre: str = ""
    mood: str = ""
    tempo: Optional[float] = None
    key: Optional[str] = None
    language: Optional[str] = None
    lyric_excerpt: str = ""
    audio_path: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Auto-assign feature_index from item_id if not set
        if self.feature_index == -1 and self.item_id.isdigit():
            self.feature_index = int(self.item_id)

    def to_prompt_line(self) -> str:
        """One-line natural-language description for LLM prompts."""
        parts = []
        if self.title:
            parts.append(f'"{self.title}"')
        if self.artist:
            parts.append(f"by {self.artist}")
        if self.album:
            parts.append(f"({self.album})")
        attrs = ", ".join(filter(None, [
            self.genre, self.mood,
            f"{self.tempo:.0f} BPM" if self.tempo else None,
            self.key, self.language,
        ]))
        if attrs:
            parts.append(f"| {attrs}")
        if self.lyric_excerpt:
            parts.append(f'| "{self.lyric_excerpt}"')
        return " ".join(parts) if parts else f"item_{self.item_id}"

    @staticmethod
    def from_metadata_string(item_id: str, meta_str: str) -> "CatalogItem":
        """Parse a Spotify metadata.json value string into a CatalogItem.

        Spotify format (from AbstractDataset._process_meta):
            "'Title' by Artist in album'Album'"
        or:
            "'track_name' by artist_name in album'album_name'"
        """
        item = CatalogItem(item_id=item_id)
        # Pattern: 'Title' by Artist in album'Album'
        m = re.match(r"'(.+?)'\s+by\s+(.+?)\s+in\s+album'(.+?)'$", meta_str)
        if m:
            item.title  = m.group(1)
            item.artist = m.group(2)
            item.album  = m.group(3)
        else:
            # Fallback: use full string as title
            item.title = meta_str.strip("'")
        return item

    @staticmethod
    def load_catalog(catalog_metadata_path: str) -> list["CatalogItem"]:
        """Load a catalog_metadata.json file into a list of CatalogItem.

        Expected format: list of dicts with at minimum "item_id".
        Missing fields default to empty string / None.
        feature_index is auto-assigned via __post_init__ if absent.
        """
        with open(catalog_metadata_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [
            CatalogItem(**{k: v for k, v in entry.items()
                           if k in CatalogItem.__dataclass_fields__})
            for entry in raw
        ]

    @staticmethod
    def load_from_backbone_metadata(metadata_json_path: str) -> list["CatalogItem"]:
        """Load directly from backbone metadata.json (Spotify format).

        metadata.json: {"0": "'Title' by Artist in album'Album'", ...}
        """
        with open(metadata_json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        items = []
        for item_id, meta_str in sorted(raw.items(), key=lambda x: int(x[0])):
            items.append(CatalogItem.from_metadata_string(item_id, meta_str))
        return items


# ---------------------------------------------------------------------------
# ContextPrefix  (WP-A → WP-D)
# ---------------------------------------------------------------------------

@dataclass
class ContextPrefix:
    """Standardized playlist context prefix produced by input_normalization.

    WP-A writes this; backbone_recommender and pipeline read it.

    How backbone_recommender uses this
    ------------------------------------
    item_ids → MDLMTokenizer.tokenize_function() → looks up token[str(item_id)]
    in clhe_token.json.  IDs must be '0'-indexed strings matching that file.

    Fields
    ------
    item_ids  : ordered list of string item IDs ('0', '1', ...), length K.
    source    : 'song_only' | 'text_only' | 'hybrid' | 'padded' | 'unknown'.
    raw_input : original user input string (for logging / debugging).
    items     : optional CatalogItem list; used by WP-C verbalization for
                style_summary prompt (μ_C neighbors description).
    """
    item_ids: list[str]
    source: str = "unknown"
    raw_input: str = ""
    items: list[CatalogItem] = field(default_factory=list)

    def validate(self):
        assert len(self.item_ids) > 0, "ContextPrefix must contain at least one item."
        assert self.source in ("song_only", "text_only", "hybrid", "padded", "unknown"), \
            f"Unknown source: {self.source}"
        if self.items:
            assert len(self.items) == len(self.item_ids), \
                "items length must match item_ids length."


# ---------------------------------------------------------------------------
# CueMappingEntry  (WP-B → WP-D)
# ---------------------------------------------------------------------------

@dataclass
class CueMappingEntry:
    """Song-to-cue mapping produced by creative_cues (WP-B).

    Output file: 02_creative_cues/outputs/item2cues.json
    Format     : {"0": [c0, c1, c2, c3, c4, c5], "1": [...], ...}

    Note: the current backbone does NOT use creative cues (CUE_TOKENS=0).
    The tokenizer will be extended once WP-B delivers this file.

    Fields
    ------
    item_id  : string item ID matching clhe_token.json keys.
    cue_ids  : exactly 6 cue vocab indices, each in [0, CUE_VOCAB_SIZE).
               Index 0 = 'unknown' fallback.
    """
    item_id: str
    cue_ids: list[int]   # TODO: required 6 items (WP-B deliverable)

    def validate(self):
        assert len(self.cue_ids) == 6, \
            f"Expected 6 cues for item '{self.item_id}', got {len(self.cue_ids)}"
        assert all(0 <= c < CUE_VOCAB_SIZE for c in self.cue_ids), \
            f"Cue ID out of [0, {CUE_VOCAB_SIZE}) for item '{self.item_id}': {self.cue_ids}"

    @staticmethod
    def load_mapping(item2cues_path: str) -> dict[str, "CueMappingEntry"]:
        """Load item2cues.json → {item_id: CueMappingEntry}.  Validates all entries."""
        with open(item2cues_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        out = {}
        for item_id, cue_ids in raw.items():
            e = CueMappingEntry(item_id=str(item_id), cue_ids=cue_ids)
            e.validate()
            out[str(item_id)] = e
        return out


# ---------------------------------------------------------------------------
# GeneratedItem  (WP-D → WP-C)
# ---------------------------------------------------------------------------

@dataclass
class GeneratedItem:
    """Next-item candidate from the backbone diffusion model.

    One instance per sampling run (sample_idx 0..S-1).

    How backbone produces this (current state)
    -------------------------------------------
    1. diffusion.restore_model_and_semi_ar_sample() returns token sequences.
    2. Parse per-item: each item = TOKENS_PER_ITEM=5 positions
       [BOI, z0_token, z1_token, z2_token, z_conf_token].
    3. Raw codes: z_raw[i] = z_token[i] - TOKEN_OFFSET[i]     (i = 0,1,2)
                  conf_raw = z_conf_token - CONFLICT_OFFSET
    4. z_hat_emb = tokenizer._token_to_feature([z0_t, z1_t, z2_t, z_conf_t])
       = sum of clhe_weight[z_raw[i] + i*256] for i in 0,1,2   (dim=64)
    5. mu_c_emb, sigma_c2 from playlist_structure.compute_playlist_structure().

    Fields
    ------
    rvq_codes     : raw RVQ codes per level, length RQ_N_CODEBOOKS=3.
                    Each in [0, RQ_CODEBOOK_SIZE=256).
    conflict_code : raw conflict digit, in [0, RQ_CODEBOOK_SIZE=256).
                    Encodes how many items share the same (z0,z1,z2) prefix.
    z_hat_emb     : CLHE embedding decoded from rvq_codes, shape (CLHE_EMB_DIM,)=(64,).
    mu_c_emb      : playlist centroid, shape (64,). Same for all S samples.
    sigma_c2      : playlist dispersion σ²_C ≥ 0. Same for all S samples.
    cue_ids       : 6 creative cue indices (WP-B TODO). Empty = not yet extended.
    sample_idx    : which of the S samples this is (0-indexed).
    context_prefix: the ContextPrefix used as input.
    """
    rvq_codes: tuple           # length = RQ_N_CODEBOOKS = 3; each in [0, 256)
    conflict_code: int         # in [0, 256)
    z_hat_emb: np.ndarray      # shape (64,)
    mu_c_emb: np.ndarray       # shape (64,)
    sigma_c2: float
    cue_ids: list[int] = field(default_factory=list)  # TODO: 6 items after WP-B
    sample_idx: int = 0
    context_prefix: Optional[ContextPrefix] = None

    def validate(self):
        assert len(self.rvq_codes) == RQ_N_CODEBOOKS, \
            f"Expected {RQ_N_CODEBOOKS} RVQ codes, got {len(self.rvq_codes)}"
        assert all(0 <= c < RQ_CODEBOOK_SIZE for c in self.rvq_codes), \
            f"RVQ code out of [0, {RQ_CODEBOOK_SIZE}): {self.rvq_codes}"
        assert 0 <= self.conflict_code < RQ_CODEBOOK_SIZE, \
            f"conflict_code out of [0, {RQ_CODEBOOK_SIZE}): {self.conflict_code}"
        assert self.z_hat_emb.shape == (CLHE_EMB_DIM,), \
            f"z_hat_emb must be shape ({CLHE_EMB_DIM},), got {self.z_hat_emb.shape}"
        assert self.mu_c_emb.shape == (CLHE_EMB_DIM,), \
            f"mu_c_emb must be shape ({CLHE_EMB_DIM},), got {self.mu_c_emb.shape}"
        assert self.sigma_c2 >= 0.0, f"sigma_c2 must be ≥ 0, got {self.sigma_c2}"
        if self.cue_ids:   # validate only when present
            assert len(self.cue_ids) == 6, \
                f"cue_ids must have 6 entries when set, got {len(self.cue_ids)}"
            assert all(0 <= c < CUE_VOCAB_SIZE for c in self.cue_ids), \
                f"cue_id out of [0, {CUE_VOCAB_SIZE}): {self.cue_ids}"


# ---------------------------------------------------------------------------
# SynthesisResult  (WP-C → Demo / Evaluation)
# ---------------------------------------------------------------------------

@dataclass
class SynthesisResult:
    """Music synthesis result from 04_synthesis.

    04_synthesis writes this; demo UI, evaluation, and pipeline read it.

    Fields
    ------
    audio_path      : absolute path to generated .wav file.
    music_attributes: comma-separated ACE-Step style tags.
    lyric_draft     : ACE-Step markup lyrics ([verse]/[chorus]/[bridge]).
    neighbors       : kNN neighbors of z_hat_emb (verbalization source).
    style_summary   : kNN neighbors of mu_c_emb (playlist-level style).
    generated_item  : the GeneratedItem that produced this result.
    """
    audio_path: str
    music_attributes: str
    lyric_draft: str
    neighbors: list[CatalogItem] = field(default_factory=list)
    style_summary: list[CatalogItem] = field(default_factory=list)
    generated_item: Optional[GeneratedItem] = None

    def validate(self):
        assert os.path.isfile(self.audio_path), \
            f"audio_path not found: {self.audio_path}"
        assert self.music_attributes.strip(), "music_attributes must not be empty."
        assert self.lyric_draft.strip(), "lyric_draft must not be empty."
