# GenPlaylist — Architecture & Design Document

> **Status**: Implementation contract aligned with the frozen training and evaluation protocol.
> **Base**: DDBC-Seq codebase (copied into `GenPlaylist_Code/`)

---

## 1. Overview

GenPlaylist extends DDBC's discrete diffusion framework toward **reference-based personalized music generation**: given a set of reference songs expressing a user's musical preference, generate a new personalized song (not from a fixed catalog) whose semantic position and textual generation intent align with that preference, via a frozen pretrained music generator.

The full pipeline:

```
Ordered references C = (m_1, ..., m_t), t ≥ 2
  │
  ├─ CLHE encode ──────────────────────────────────────────────────────┐
  │   E(m) ∈ R^64  (CLHE backbone, frozen)                             │
  │                                                                     │
  ├─ RVQ discretize ────────────────────────────────────────────────────┤
  │   E(m) → z(m) = (z1, z2, z3, z_conf)  [L=3, K=256, 1-indexed]     │
  │   + store 16 ranked cues; activate first 8 per item (WP-B)         │
  │                                                                     │
  ├─ History-conditioned masked diffusion ──────────────────────────────┤
  │   conditioning: 15 visible reference-item token blocks (fixed)     │
  │   target input: [BOI, MASK×12, EOS]; jointly denoise MASK payload │
  │   output: next-item token sequence [z1, z2, z3, z_conf, c1..c8]   │
  │                                                                     │
  └─ Prompt construction + synthesis ───────────────────────────────────┘
      decode RVQ → Ê(m); kNN lookup in catalog CLHE space
      creative cues + neighbor metadata → LLM prompt assembly
      LLM (Qwen3) → music attributes + lyric draft
      ACE-Step (frozen) → personalized audio
```

---

## 2. Item Representation

### 2.1 Embedding

Each item `m` is represented by the **CLHE embedding**:

```
E(m) ∈ R^64   (CLHE backbone, frozen weights in clhe_weight.npy)
```

CLHE fuses audio features and collaborative-filtering signals, providing a shared music embedding space sensitive to both acoustic and semantic properties.
Verbalization does not use a separate text encoder — instead, the generated latent `Ê(m)` is grounded via kNN retrieval in catalog CLHE space, using the retrieved neighbors' metadata (title, artist, genre, mood, lyric_excerpt) as the textual proxy.

### 2.2 Token Sequence per Item

After RVQ quantization + creative cue assignment, each item occupies **13 token slots** in the sequence:

```
[BOI, z1, z2, z3, z_conf, c1, c2, c3, c4, c5, c6, c7, c8]
  ↑                  ↑               ↑
BOI token     conflict digit   8 creative cue tokens
```

Stride `k = 13`. RVQ codes are **1-indexed** (z1 ∈ [1,256], z2 ∈ [257,512], z3 ∈ [513,768]); `z_conf` is a separate conflict-avoidance digit (74 observed values, range 769–842).

Full sequence structure:

```
[BOS,  BOI, z1..z3, z_conf, c1..c8,  BOI, z1..z3, z_conf, c1..c8, ..., EOS]
  ↑    ←──── item 1 (k=13) ─────────→  ←──── item 2 ──────────────→      ↑
pos 0                                                                   pos L-1
```

### 2.3 Vocabulary Layout

| Range | Token type | Count |
|---|---|---|
| 0 | BOS | 1 |
| 1 – 768 | RVQ codes (L=3 × K=256, 1-indexed) | 768 |
| 769 – 842 | conflict digit z_conf | 74 |
| 843 | BOI | 1 |
| 844 | EOS | 1 |
| 845 – 2892 | Creative cues | 2048 |
| 2893 | MASK (diffusion) | 1 |
| **Total** | | **2894** |

Note: CLHE codes are 1-indexed; embedding reconstruction uses `weight[code - 1]` for each of the three levels.

---

## 3. History Conditioning

The fifteen reference item blocks remain visible throughout training and reverse
diffusion, providing the conditioning history for all five target blocks.
- The default model does not inject the cached history mean or dispersion;
  `sampling.structure_conditioning=false` is covered by a configuration test.
- Prepared mean/dispersion arrays are retained only for diagnostics and legacy
  checkpoint compatibility, not as default backbone inputs.
- Every eligible chronological segment becomes a 20-song rolling window with
  exactly 15 references and five continuation targets.
- Training samples require all 20 songs; stride is one song.

Inference uses the same full sequence layout as training. It appends five target
slots whose 60 payload positions are all MASK, then jointly reverse-denoises
the complete continuation. The legacy DDBC next-block/semi-AR loop is not used.

---

## 4. Creative Cues

### 4.1 Motivation

Beyond RVQ codes (which capture acoustic/semantic structure), each generated item carries **8 creative cue tokens** — discrete tokens representing lyrical imagery, themes, motifs, and cultural references. These serve as:

1. Part of the jointly-generated token sequence (diffusion generates them alongside RVQ codes)
2. A lightweight verbalization signal (cue tokens → human-readable imagery words)

### 4.2 Vocabulary construction (offline, one-time)

**Step 1 — Lyric scraping**

```
item_info.json (track_name, artist_name)
    ↓ language detection (fasttext)
    ↙              ↘
English          Chinese
Genius API       NetEase Cloud Music API
(lyricsgenius)   (pyncm)
    ↓
lyrics.json  {item_id: "raw lyrics text"}
```

Items without lyrics fall back to metadata (title + artist + genre tags).

**Step 2 — Raw cue extraction (Qwen3-7B, multilingual)**

Prompt:
```
You are a music analysis expert. Extract 8-10 key imagery words or phrases from the following lyrics.
Requirements: prefer concrete nouns/phrases (e.g. "train platform" over "longing"),
              include scenes, objects, characters, and cultural references,
              English or original language is fine, preserve the source language.
Lyrics: {lyrics}
Output: comma-separated list, no explanation.
```

Output: `cues_raw.json` — `{item_id: ["train platform", "old photo", "broken phone", ...]}`

**Step 3 — Vocabulary filtering to 2048**

```
1. Normalize: traditional→simplified Chinese, English lemmatize, lowercase
2. Count df(cue) across all items
3. Filter: 5 ≤ df ≤ 0.3 × N  (suppress hapax & popular)
4. Embed with multilingual sentence-BERT
5. Semantic dedup: merge pairs with cosine > 0.92
6. Sort by IDF descending → take top-2048
```

Output: `creative_cues_vocab.json` — `{cue_text: cue_id}` (2048 entries)

**Step 4 — Per-item assignment (16 stored / 8 active)**

```python
for each item:
    raw_cues → nearest vocab entry (sentence-BERT cosine)
    → sort by PMI(cue, item)
    → greedy diverse selection (pairwise distance > threshold)
    → top-16 relevance-ranked cue IDs; WP-C consumes the first 8
```

Output: `item2cues.json` — `{item_id: [cue_id_1, ..., cue_id_16]}`

### 4.3 Position-type mask

For each item, let `offset = (pos - 1) % 13`. The production tokenizer and
diffusion sampler use this table:

| offset | Token type | Legal range |
|---|---|---|
| 0 | BOI | 843 |
| 1 | z1 | 1–256 |
| 2 | z2 | 257–512 |
| 3 | z3 | 513–768 |
| 4 | conflict | 769–842 |
| 5–12 | creative cues c1–c8 | 845–2892 |

Position 0 is BOS, the final position is EOS, and MASK is never a legal clean
prediction.

---

## 5. Verbalization (Post-generation)

After the diffusion generates new token sequences:

```
generated tokens [z1, z2, z3, z_conf, c1..c8]
    ↓
decode RVQ → Ê(m) = weight[z1-1] + weight[z2-1] + weight[z3-1]  ∈ R^64
    │
    ├─ kNN in catalog CLHE space (faiss IndexFlatIP)
    │   → top-k neighbors → metadata (title, artist, genre, mood, tempo, key, lyric_excerpt)
    │
    ├─ creative cue tokens c1..c8 → cue vocabulary lookup → imagery words
    │
    └─ LLM (Qwen3) prompt assembly:
         neighbor metadata + cue words + ordered-reference summary
         → music attributes (genre, mood, tempo, key, instrumentation, language)
         → lyric draft with [verse]/[chorus]/[bridge] section markers
                    ↓
              ACE-Step (frozen) → personalized audio waveform
```

---

## 6. Evaluation

WP-C evaluation is frozen independently of the WP-D audio demo:

1. Keep test playlists with at least 20 songs and retain the first 20.
2. Use songs 1–15 as the unchanged reference context and songs 16–20 as the
   five ground-truth future songs.
3. Jointly full-mask-sample five continuation items once; never sequentially
   feed a generated item back into the context.
4. Retrieve predictions against the full 5,119-song catalog and solve the 5x5
   prediction/target cosine matrix with Hungarian assignment.

| WP-C metric | Description |
|---|---|
| Matched CLHE cosine ↑ | Mean cosine under optimal one-to-one 5x5 assignment |
| Exact recall / precision / F1 ↑ | Multiset catalog-item overlap; duplicates receive no extra credit |
| Any hit ↑ | Whether at least one of the five predictions matches |
| Unique ratio ↑ | Unique retrieved predictions divided by five |

WP-D synthesis/audio evaluation remains a separate later stage:

| Metric | Description |
|---|---|
| FAD ↓ | Fréchet Audio Distance vs. held-out real music |
| MERT History Fit ↑ | Mean similarity between generated audio and the 15 references |
| CLAP-A ↑ | Cosine similarity between generated audio and its music-attribute condition |
| Next-song similarity ↑ | Diagnostic similarity to the compatible held-out successor |
| Human eval | History fit / quality / novelty / preference (5-point scales) |

---

## 7. Files to Create / Modify

| File | Change |
|---|---|
| `models/dit.py` | Condition on visible history tokens |
| `diffusion.py` | Full-mask continuation; generation not restricted to catalog items |
| `tokenizer.py` | Vocab = 2894; main stride k=13; configurable 0/4/8/16 cues; update legal-position mask |
| `dataset.py` | Build rolling 15-to-5 windows; cache optional history diagnostics |
| `evaluator.py` | Report frozen proxy metrics and separate MERT/FAD/CLAP audio metrics |
| `configs/config.yaml` | vocab_size=2894, rq_n_codebooks=3, rq_codebook_size=256, active_cue_tokens=8 |
| `verbalization.py` | kNN via faiss; LLM prompt assembly from cues + neighbor metadata; Qwen3 API call |
| `synthesis.py` | ACE-Step frozen pipeline wrapper; style_ref_audio_path support |

---

## 8. Resolved Decisions

| Decision | Status |
|---|---|
| Dataset: Spotify MPD v2 subset (6,585 playlists / 5,119 songs) | ✓ |
| Embedding: CLHE backbone, frozen (clhe_weight.npy, 768×64) | ✓ |
| RVQ: L=3 codebooks, K=256 entries, 1-indexed codes | ✓ |
| Conflict digit z_conf: 74 observed values (range 769–842) | ✓ |
| Main token stride: k=13 per item; 0/4/16-cue ablations retrain | ✓ |
| Vocab size: 2894 (incl. MASK token) | ✓ |
| Creative cues vocab size: 2048 | ✓ |
| Stored cue candidates per item: 16, active prefix: 8 | ✓ |
| Training: rolling 15 references → five targets (20 total) | ✓ |
| Evaluation: first 20, one joint five-item completion | ✓ |
| Preference conditioning through visible history tokens | ✓ |
| Verbalization: kNN in CLHE space + Qwen3 LLM (no T5) | ✓ |
| Synthesis: ACE-Step (frozen) | ✓ |
