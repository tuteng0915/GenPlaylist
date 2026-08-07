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
  ─[shared]► CatalogItem            (shared catalog metadata format used by all WPs)
diffusion model output
  ─[WP-C]──► GeneratedItem          (03_backbone_recommender → 04_synthesis)
ACE-Step output
  ─[WP-D]──► SynthesisResult        (04_synthesis → pipeline / evaluation / demo)

Frozen GenPlaylist v1 contract
------------------------------
  Item IDs are opaque strings. Matrix access always goes through item_id_to_row.
  catalog_item_embeddings.npy is N x 64 (one row per song).
  rvq_codebook_weights.npy is 768 x 64 (three 256-entry codebooks).

  Per-item token sequence (stride 13):
    [BOI, z0, z1, z2, z_conf, c0, c1, c2, c3, c4, c5, c6, c7]

  Token IDs:
    BOS       0
    z0        1..256
    z1        257..512
    z2        513..768
    z_conf    769..842       (74 observed conflict values)
    BOI       843
    EOS       844
    cues      845..2892      (2048 entries, cue id 0 is <unk>)
    MASK      2893           (diffusion runtime only)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Frozen GenPlaylist v1 token and embedding contract
# ---------------------------------------------------------------------------

SCHEMA_VERSION     = "genplaylist-v1"
RQ_N_CODEBOOKS    = 3
RQ_CODEBOOK_SIZE  = 256
CLHE_EMB_DIM      = 64
CONFLICT_VOCAB_SIZE = 74
CUE_TOKENS        = 8
CUE_CANDIDATES_PER_ITEM = 16
CUE_VOCAB_SIZE    = 2048
TOKENS_PER_ITEM   = 1 + RQ_N_CODEBOOKS + 1 + CUE_TOKENS  # = 13

# RVQ token_id = raw_code + TOKEN_OFFSET[level], raw_code is 0-based.
TOKEN_OFFSET = [RQ_CODEBOOK_SIZE * level + 1 for level in range(RQ_N_CODEBOOKS)]
CONFLICT_TOKEN_START = 1 + RQ_N_CODEBOOKS * RQ_CODEBOOK_SIZE       # 769
CONFLICT_OFFSET = CONFLICT_TOKEN_START  # token_id = raw 0-based conflict + offset
BOI_TOKEN = CONFLICT_TOKEN_START + CONFLICT_VOCAB_SIZE             # 843
EOS_TOKEN = BOI_TOKEN + 1                                          # 844
CUE_TOKEN_START = EOS_TOKEN + 1                                    # 845
MASK_TOKEN = CUE_TOKEN_START + CUE_VOCAB_SIZE                      # 2893
VOCAB_SIZE = MASK_TOKEN                 # tokenizer ids 0..2892, excludes MASK
RUNTIME_VOCAB_SIZE = MASK_TOKEN + 1     # diffusion ids 0..2893


@dataclass(frozen=True)
class TokenLayout:
    """Single source of truth for GenPlaylist v1 token offsets."""

    schema_version: str = SCHEMA_VERSION
    rq_n_codebooks: int = RQ_N_CODEBOOKS
    rq_codebook_size: int = RQ_CODEBOOK_SIZE
    conflict_vocab_size: int = CONFLICT_VOCAB_SIZE
    cue_tokens: int = CUE_TOKENS
    cue_vocab_size: int = CUE_VOCAB_SIZE

    @property
    def tokens_per_item(self) -> int:
        return 1 + self.rq_n_codebooks + 1 + self.cue_tokens

    @property
    def conflict_token_start(self) -> int:
        return 1 + self.rq_n_codebooks * self.rq_codebook_size

    @property
    def boi_token(self) -> int:
        return self.conflict_token_start + self.conflict_vocab_size

    @property
    def eos_token(self) -> int:
        return self.boi_token + 1

    @property
    def cue_token_start(self) -> int:
        return self.eos_token + 1

    @property
    def mask_token(self) -> int:
        return self.cue_token_start + self.cue_vocab_size

    @property
    def vocab_size(self) -> int:
        """Tokenizer vocabulary size, excluding the diffusion MASK token."""
        return self.mask_token

    @property
    def runtime_vocab_size(self) -> int:
        return self.mask_token + 1

    def rvq_token(self, level: int, raw_code: int) -> int:
        if not 0 <= level < self.rq_n_codebooks:
            raise ValueError(f"RVQ level out of range: {level}")
        if not 0 <= raw_code < self.rq_codebook_size:
            raise ValueError(f"RVQ code out of range: {raw_code}")
        return 1 + level * self.rq_codebook_size + raw_code

    def conflict_token(self, raw_code: int) -> int:
        if not 0 <= raw_code < self.conflict_vocab_size:
            raise ValueError(f"Conflict code out of range: {raw_code}")
        return self.conflict_token_start + raw_code

    def cue_token(self, cue_id: int) -> int:
        if not 0 <= cue_id < self.cue_vocab_size:
            raise ValueError(f"Cue id out of range: {cue_id}")
        return self.cue_token_start + cue_id


TOKEN_LAYOUT = TokenLayout()


# ---------------------------------------------------------------------------
# CatalogItem
# ---------------------------------------------------------------------------

@dataclass
class CatalogItem:
    """Metadata for a single catalog song.

    Source: datasets/{dataset}/metadata.json + item_info.json (if available).

    Item ID convention
    ------------------
    ``item_id`` is an opaque string and may be sparse (for example ``"18996"``).
    ``feature_index`` is assigned from the versioned ``item_id_to_row.json``
    artifact. It must never be inferred with ``int(item_id)``.

    Raw token_ids for a given item can be read from clhe_token.json[item_id]:
        [z0_token, z1_token, z2_token, z_conf_token]
    Raw codes are stored in the semantic-token artifact and joined by item ID;
    they are never indexed with ``int(item_id)``.

    Fields
    ------
    item_id         : opaque string item ID.
    feature_index   : row in catalog_item_embeddings.npy; -1 until mapped.
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
        self.item_id = str(self.item_id)
        if not self.item_id:
            raise ValueError("CatalogItem.item_id must not be empty")

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

        Supports both formats used in this repository:
          - ``[{"item_id": "...", ...}, ...]``
          - ``{"item_id": {metadata fields...}, ...}``

        Dict keys are treated as authoritative item IDs. Sparse numeric IDs are
        preserved and are not interpreted as embedding row numbers.
        """
        with open(catalog_metadata_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            entries = []
            for item_id, entry in raw.items():
                if isinstance(entry, str):
                    entries.append(CatalogItem.from_metadata_string(str(item_id), entry))
                    continue
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"Catalog entry '{item_id}' must be an object or metadata string"
                    )
                values = dict(entry)
                values["item_id"] = str(item_id)
                entries.append(CatalogItem(**{
                    k: v for k, v in values.items()
                    if k in CatalogItem.__dataclass_fields__
                }))
            return entries

        if isinstance(raw, list):
            entries = []
            for index, entry in enumerate(raw):
                if not isinstance(entry, dict):
                    raise ValueError(f"Catalog list entry {index} must be an object")
                if "item_id" not in entry:
                    raise ValueError(f"Catalog list entry {index} has no item_id")
                entries.append(CatalogItem(**{
                    k: v for k, v in entry.items()
                    if k in CatalogItem.__dataclass_fields__
                }))
            return entries

        raise ValueError("Catalog metadata must be a JSON object or list")

    @staticmethod
    def load_from_backbone_metadata(metadata_json_path: str) -> list["CatalogItem"]:
        """Load directly from backbone metadata.json (Spotify format).

        metadata.json: {"0": "'Title' by Artist in album'Album'", ...}
        """
        with open(metadata_json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        items = []
        for item_id, meta_str in raw.items():
            items.append(CatalogItem.from_metadata_string(item_id, meta_str))
        return items


# ---------------------------------------------------------------------------
# ContextPrefix  (WP-A → WP-C)
# ---------------------------------------------------------------------------

@dataclass
class ContextPrefix:
    """Standardized playlist context prefix produced by input_normalization.

    WP-A writes this; backbone_recommender and pipeline read it.

    How backbone_recommender uses this
    ------------------------------------
    item_ids → MDLMTokenizer.tokenize_function() → looks up token[str(item_id)]
    in the token artifact. Embedding access separately uses item_id_to_row.json.

    Fields
    ------
    item_ids  : ordered list of opaque string item IDs, length K.
    source    : 'song_only' | 'text_only' | 'hybrid' | 'padded' | 'unknown'.
    raw_input : original user input string (for logging / debugging).
    items     : optional CatalogItem list; used by WP-D verbalization for
                style_summary prompt (μ_C neighbors description).
    """
    item_ids: list[str]
    source: str = "unknown"
    raw_input: str = ""
    items: list[CatalogItem] = field(default_factory=list)

    def validate(self):
        if not self.item_ids:
            raise ValueError("ContextPrefix must contain at least one item")
        self.item_ids = [str(item_id) for item_id in self.item_ids]
        if any(not item_id for item_id in self.item_ids):
            raise ValueError("ContextPrefix contains an empty item ID")
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("ContextPrefix item_ids must be unique")
        valid_sources = {"song_only", "text_only", "hybrid", "padded", "unknown"}
        if self.source not in valid_sources:
            raise ValueError(f"Unknown source: {self.source}")
        if self.items and len(self.items) != len(self.item_ids):
            raise ValueError("items length must match item_ids length")
        if self.items:
            actual_ids = [item.item_id for item in self.items]
            if actual_ids != self.item_ids:
                raise ValueError("items must have the same IDs and order as item_ids")
        return self


# ---------------------------------------------------------------------------
# CueMappingEntry  (WP-B → WP-C)
# ---------------------------------------------------------------------------

@dataclass
class CueMappingEntry:
    """Song-to-cue mapping produced by creative_cues (WP-B).

    Output file: 02_creative_cues/outputs/item2cues.json
    Format     : {"0": [c0, c1, ..., c15], "1": [...], ...}

    Fields
    ------
    item_id  : string item ID matching clhe_token.json keys.
    cue_ids  : exactly CUE_CANDIDATES_PER_ITEM=16 ranked cue candidates, each in
               [0, CUE_VOCAB_SIZE). Downstream models select a prefix; the
               GenPlaylist-v1 default consumes CUE_TOKENS=8.
               Index 0 = 'unknown' fallback.
    """
    item_id: str
    cue_ids: list[int]

    def validate(
        self,
        n_cues: int = CUE_CANDIDATES_PER_ITEM,
        vocab_size: int = CUE_VOCAB_SIZE,
    ):
        """n_cues: expected stored cue count. Defaults to the 16-candidate
        cross-WP artifact contract;
        pass the actual count for item2cues.json files produced with a non-default
        --num-cues (run_compare.py) so validation checks against the right length.

        vocab_size: upper bound for cue_ids. Defaults to the CUE_VOCAB_SIZE=2048 WP-C
        contract; pass the actual value for files produced with a non-default
        --vocab-size so validation checks against the right bound instead of falsely
        rejecting in-range indices from a larger vocabulary."""
        if len(self.cue_ids) != n_cues:
            raise ValueError(
                f"Expected {n_cues} cues for item '{self.item_id}', got {len(self.cue_ids)}"
            )
        if not all(isinstance(c, int) and 0 <= c < vocab_size for c in self.cue_ids):
            raise ValueError(
                f"Cue ID out of [0, {vocab_size}) for item '{self.item_id}': {self.cue_ids}"
            )
        return self

    @staticmethod
    def load_mapping(item2cues_path: str, n_cues: int = CUE_CANDIDATES_PER_ITEM,
                     vocab_size: int = CUE_VOCAB_SIZE) -> dict[str, "CueMappingEntry"]:
        """Load item2cues.json → {item_id: CueMappingEntry}.  Validates all entries.

        n_cues: expected stored cue count per item (default 16). Models may use
        only a ranked prefix, currently CUE_TOKENS=8.
        Pass the value used to produce the file if it came from a run_compare.py
        --num-cues run that overrode the default.
        vocab_size: expected vocab size (default CUE_VOCAB_SIZE=2048). Pass the value
        used to produce the file if it came from a --vocab-size run that overrode
        the default, so in-range indices from a larger/smaller vocab validate correctly.
        """
        with open(item2cues_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        out = {}
        for item_id, cue_ids in raw.items():
            e = CueMappingEntry(item_id=str(item_id), cue_ids=cue_ids)
            e.validate(n_cues=n_cues, vocab_size=vocab_size)
            out[str(item_id)] = e
        return out


# ---------------------------------------------------------------------------
# GeneratedItem  (WP-C → WP-D)
# ---------------------------------------------------------------------------

@dataclass
class GeneratedItem:
    """Next-item candidate from the backbone diffusion model.

    One instance per sampling run (sample_idx 0..S-1).

    How backbone produces this (current state)
    -------------------------------------------
    1. Append ``[BOI, MASK x 12, EOS]`` to the reference sequence and run
       diffusion.restore_model_and_sample_next_item() as full-mask completion.
    2. Parse per-item: each item = TOKENS_PER_ITEM=13 positions
       [BOI, z0_token, z1_token, z2_token, z_conf_token, c0, ..., c7].
    3. Raw codes: z_raw[i] = z_token[i] - TOKEN_OFFSET[i]     (i = 0,1,2)
                  conf_raw = z_conf_token - CONFLICT_OFFSET
    4. z_hat_emb = tokenizer._token_to_feature([z0_t, z1_t, z2_t, z_conf_t])
       = sum of clhe_weight[z_raw[i] + i*256] for i in 0,1,2   (dim=64)
    5. mu_c_emb, sigma_c2 from playlist_structure.compute_playlist_structure().

    Fields
    ------
    rvq_codes     : raw RVQ codes per level, length RQ_N_CODEBOOKS=3.
                    Each in [0, RQ_CODEBOOK_SIZE=256).
    conflict_code : raw conflict digit, in [0, CONFLICT_VOCAB_SIZE=74).
                    Encodes how many items share the same (z0,z1,z2) prefix.
    z_hat_emb     : CLHE embedding decoded from rvq_codes, shape (CLHE_EMB_DIM,)=(64,).
    mu_c_emb      : playlist centroid, shape (64,). Same for all S samples.
    sigma_c2      : playlist dispersion σ²_C ≥ 0. Same for all S samples.
    cue_ids       : exactly CUE_TOKENS=8 creative cue indices.
    sample_idx    : which of the S samples this is (0-indexed).
    context_prefix: the ContextPrefix used as input.
    """
    rvq_codes: tuple           # length = RQ_N_CODEBOOKS = 3; each in [0, 256)
    conflict_code: int         # in [0, 256)
    z_hat_emb: np.ndarray      # shape (64,)
    mu_c_emb: np.ndarray       # shape (64,)
    sigma_c2: float
    cue_ids: list[int] = field(default_factory=list)
    sample_idx: int = 0
    context_prefix: Optional[ContextPrefix] = None

    def validate(self, allow_missing_cues: bool = False):
        if len(self.rvq_codes) != RQ_N_CODEBOOKS:
            raise ValueError(f"Expected {RQ_N_CODEBOOKS} RVQ codes, got {len(self.rvq_codes)}")
        if not all(isinstance(c, int) and 0 <= c < RQ_CODEBOOK_SIZE for c in self.rvq_codes):
            raise ValueError(f"RVQ code out of [0, {RQ_CODEBOOK_SIZE}): {self.rvq_codes}")
        if not isinstance(self.conflict_code, int) or not 0 <= self.conflict_code < CONFLICT_VOCAB_SIZE:
            raise ValueError(
                f"conflict_code out of [0, {CONFLICT_VOCAB_SIZE}): {self.conflict_code}"
            )
        if not isinstance(self.z_hat_emb, np.ndarray) or self.z_hat_emb.shape != (CLHE_EMB_DIM,):
            shape = getattr(self.z_hat_emb, "shape", None)
            raise ValueError(f"z_hat_emb must be shape ({CLHE_EMB_DIM},), got {shape}")
        if not isinstance(self.mu_c_emb, np.ndarray) or self.mu_c_emb.shape != (CLHE_EMB_DIM,):
            shape = getattr(self.mu_c_emb, "shape", None)
            raise ValueError(f"mu_c_emb must be shape ({CLHE_EMB_DIM},), got {shape}")
        if not np.isfinite(self.z_hat_emb).all() or not np.isfinite(self.mu_c_emb).all():
            raise ValueError("GeneratedItem embeddings must contain only finite values")
        if not np.isfinite(self.sigma_c2) or self.sigma_c2 < 0.0:
            raise ValueError(f"sigma_c2 must be finite and ≥ 0, got {self.sigma_c2}")
        if not self.cue_ids and allow_missing_cues:
            return self
        CueMappingEntry(item_id="<generated>", cue_ids=self.cue_ids).validate(
            n_cues=CUE_TOKENS)
        return self


# ---------------------------------------------------------------------------
# SynthesisResult  (WP-D → Demo / Evaluation)
# ---------------------------------------------------------------------------

@dataclass
class SynthesisResult:
    """Music synthesis result from 04_synthesis.

    04_synthesis writes this; demo UI, evaluation, and pipeline read it.

    Fields
    ------
    audio_path      : absolute path to a generated audio file.
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
        if not os.path.isfile(self.audio_path):
            raise ValueError(f"audio_path not found: {self.audio_path}")
        if not self.music_attributes.strip():
            raise ValueError("music_attributes must not be empty")
        if not self.lyric_draft.strip():
            raise ValueError("lyric_draft must not be empty")
        if self.generated_item is not None:
            self.generated_item.validate()
        return self
