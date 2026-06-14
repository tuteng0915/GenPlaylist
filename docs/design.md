# GenPlaylist — Architecture & Design Document

> **Status**: Pre-implementation design. Decisions confirmed in discussion; code not yet written.
> **Base**: DDBC-Seq codebase (copied into `GenPlaylist_Code/`)

---

## 1. Overview

GenPlaylist extends DDBC's discrete diffusion framework from bundle retrieval to **playlist latent expansion**: given a seed set of songs, generate *new* songs (not from a fixed catalog) that continue the playlist while respecting its semantic structure.

The full pipeline:

```
Seed set C = {m_1, ..., m_k}
  │
  ├─ CLHE encode ──────────────────────────────────────────────────────┐
  │   audio_feat + CF_feat → E_clhe(m) ∈ R^d_c                        │
  │   text (lyrics/tags) via T5-small → E_t5(m) ∈ R^d_t               │
  │   concat → E(m) = [E_clhe(m) | E_t5(m)] ∈ R^(d_c + d_t)          │
  │                                                                     │
  ├─ Compute playlist structure ────────────────────────────────────────┤
  │   μ_C  = mean{ E(m) }                                              │
  │   σ²_C = mean{ ||E(m) - μ_C||² }                                  │
  │                                                                     │
  ├─ RVQ discretize ────────────────────────────────────────────────────┤
  │   E(m) → z(m) = (d0, d1, d2, d3, conflict)   [4-level, 128 codes] │
  │   + assign 6 creative cue tokens per item                          │
  │                                                                     │
  ├─ Dispersion-conditioned masked diffusion ───────────────────────────┤
  │   conditioning: seed tokens (fixed) + μ_C, σ²_C (injected)        │
  │   output: n new item token sequences                               │
  │                                                                     │
  └─ Verbalization ─────────────────────────────────────────────────────┘
      generated tokens → decode RVQ → Ê(m)
      Ê_t5 component → nearest neighbor in T5 space
        → full T5 hidden states → T5 decoder → lyric imagery draft
      → LLM refine → ACE-Step synthesis
```

---

## 2. Item Representation

### 2.1 Embedding

Each item `m` is represented by a **concatenated embedding**:

```
E(m) = [ E_clhe(m) | E_t5(m) ]
```

- `E_clhe(m)`: CLHE encoder output — fuses audio features + CF features
- `E_t5(m)`: T5-small encoder mean-pool over lyrics/tags tokens

The split is preserved at all times so the T5 component can be used independently for verbalization. The two components are **never mixed** (no linear projection across the boundary).

### 2.2 Token Sequence per Item

After RVQ quantization + creative cue assignment, each item occupies **12 token slots** in the sequence:

```
[BOI, d0, d1, d2, d3, conflict, c1, c2, c3, c4, c5, c6]
  ↑                                ↑
BOI token                    6 creative cue tokens
```

Stride `k = 12` (was 6 in DDBC-Seq with 4 codebooks).

Full sequence structure:

```
[BOS,  BOI, d0..d3, conflict, c1..c6,  BOI, d0..d3, conflict, c1..c6, ..., EOS]
  ↑    ←──── item 1 (k=12) ────────→   ←──── item 2 ────────────────→      ↑
pos 0                                                                    pos L-1
```

### 2.3 Vocabulary Layout

| Range | Token type | Count |
|---|---|---|
| 0 | BOS | 1 |
| 1 – 128 | d0 (RVQ level 0) | 128 |
| 129 – 256 | d1 (RVQ level 1) | 128 |
| 257 – 384 | d2 (RVQ level 2) | 128 |
| 385 – 512 | d3 (RVQ level 3) | 128 |
| 513 – 640 | conflict digit | 128 |
| 641 | BOI | 1 |
| 642 | EOS | 1 |
| 643 – 2690 | Creative cues | 2048 |
| **Total** | | **2691** |

Formula cross-check (n_digit=4, codebook_size=128):
- BOI = (n_digit + 1) × codebook_size + 1 = 5 × 128 + 1 = **641** ✓
- EOS = BOI + 1 = **642** ✓
- Creative cues start = EOS + 1 = **643** ✓

---

## 3. Dispersion Conditioning

### 3.1 What is injected

For each playlist / training sequence, two scalars characterize the seed set `C`:

```
μ_C  = (1/|C|) Σ E(m)          centroid vector ∈ R^(d_c + d_t)
σ²_C = (1/|C|) Σ ||E(m) - μ_C||²   scalar dispersion
```

### 3.2 Injection mechanism

Both are projected into the model's hidden space and injected as **additional conditioning signals** (not CFG prefix tokens). Concretely, in `models/dit.py`:

```python
# Existing sigma conditioning
c = F.silu(self.sigma_map(sigma))          # [B, hidden]

# New dispersion conditioning (added alongside sigma)
disp_c = F.silu(self.disp_map(sigma_c))    # σ²_C: scalar → [B, hidden]
cent_c = F.silu(self.cent_map(mu_c))       # μ_C: vector → [B, hidden]

c = c + disp_c + cent_c                    # fused conditioning
```

`disp_map`: `nn.Linear(1, hidden_size)`
`cent_map`: `nn.Linear(d_emb, hidden_size)`

This allows a single model to produce tight continuations (low σ²_C) or diverse ones (high σ²_C) without separate models.

### 3.3 Training

- Seed tokens are **never masked** (fixed context), identical to DDBC.
- σ²_C and μ_C are computed from the seed items in each training batch.
- Passed through `_forward_pass_diffusion` → `forward` → DIT.

---

## 4. Creative Cues

### 4.1 Motivation

Beyond RVQ codes (which capture acoustic/semantic structure), each generated item carries **6 creative cue tokens** — discrete tokens representing lyrical imagery, themes, motifs, and cultural references. These serve as:

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

**Step 4 — Per-item assignment (K=6)**

```python
for each item:
    raw_cues → nearest vocab entry (sentence-BERT cosine)
    → sort by PMI(cue, item)
    → greedy diverse selection (pairwise distance > threshold)
    → top-6 cue IDs
```

Output: `item_cues.json` — `{item_id: [cue_id_1, ..., cue_id_6]}`

### 4.3 Illegal mask update

`_apply_illegal_mask` needs to handle cue positions. With stride k=12:

| `pos % 12` | Token type | Legal range |
|---|---|---|
| 1 | BOI | [641, 641] |
| 2 | d0 | [1, 128] |
| 3 | d1 | [129, 256] |
| 4 | d2 | [257, 384] |
| 5 | d3 | [385, 512] |
| 0 | conflict | [513, 640] |
| 6–11 | creative cues c1–c6 | [643, 2690] |

---

## 5. Verbalization (Post-generation)

After the diffusion generates new token sequences:

```
generated tokens
    ↓
decode RVQ → Ê(m) = [Ê_clhe | Ê_t5]
    │
    ├─ Ê_t5 → nearest neighbor in T5 embedding space
    │          → retrieve full T5 hidden state sequence
    │          → T5 decoder → lyric imagery draft
    │
    ├─ creative cue tokens → cue vocabulary lookup → imagery words (direct)
    │
    └─ LLM (Qwen3) refine: imagery draft + cue words → lyrics + music attributes
                    ↓
              ACE-Step synthesis → audio
```

The T5 component is used for **retrieval**, not direct decoding from a compressed vector — the decoder receives the full token-level hidden states of the nearest neighbor, preserving decodability.

---

## 6. Evaluation

Retrieval metrics (Recall, Precision, Hit) are **dropped**. Generated music cannot overlap the catalog by design.

| Metric | Description |
|---|---|
| FAD ↓ | Fréchet Audio Distance vs. held-out real music |
| CLAP Score ↑ | Cosine similarity between generated audio CLAP embedding and lyric/attribute text |
| Dispersion Match Δσ² ↓ | How well generated set matches seed playlist's σ²_C |
| Centroid Distance CD ↓ | Distance between generated centroid and seed centroid |
| Human eval | Coherence / Music Quality / Overall Satisfaction (5-point Likert) |

---

## 7. Files to Create / Modify

| File | Change |
|---|---|
| `models/dit.py` | Add `disp_map`, `cent_map`; inject σ²_C, μ_C into conditioning |
| `diffusion.py` | Pass μ_C, σ²_C through full call chain; update generation to not restrict to catalog items |
| `tokenizer.py` | Extend vocab to 2691; stride k=12; load `item_cues.json`; update token assembly |
| `dataset.py` | Compute μ_C, σ²_C per batch; add to batch dict; new playlist data loader |
| `evaluator.py` | Replace retrieval metrics with FAD, CLAP Score, Δσ², CD |
| `configs/config.yaml` | Add dispersion conditioning params, creative cues params |
| `models/t5_encoder.py` | **New**: T5-small encoder wrapper (mean-pool output + stored hidden states) |
| `models/clhe.py` | **New**: CLHE fusion module wrapper |
| `scripts/scrape_lyrics.py` | **New**: Genius + NetEase scraper |
| `scripts/build_creative_cues.py` | **New**: Full NLP pipeline (extract → filter → vocab → assign) |

---

## 8. Open Questions / Deferred Decisions

| Question | Status |
|---|---|
| Actual playlist dataset (not Spotify MPD) | User will provide |
| Audio feature extraction method (MFCC / CLAP / other) | TBD |
| CF features availability | TBD |
| RVQ training: from scratch vs. Faiss (current approach) | Keep Faiss for now |
| T5 model variant: `t5-small` vs `t5-base` | TBD |
| n_codebooks=4, codebook_size=128 confirmed | ✓ |
| Creative cues vocab size = 2048 confirmed | ✓ |
| K=6 cues per item confirmed | ✓ |
| Dispersion conditioning via additive projection confirmed | ✓ |
