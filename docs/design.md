# GenPlaylist — Architecture & Design Document

> **Status**: Pre-implementation design. Decisions confirmed in discussion; code not yet written.
> **Base**: DDBC-Seq codebase (copied into `GenPlaylist_Code/`)

---

## 1. Overview

GenPlaylist extends DDBC's discrete diffusion framework toward **reference-based personalized music generation**: given a set of reference songs expressing a user's musical preference, generate a new personalized song (not from a fixed catalog) whose semantic position and textual generation intent align with that preference, via a frozen pretrained music generator.

The full pipeline:

```
Reference set C = {m_1, ..., m_k}
  │
  ├─ CLHE encode ──────────────────────────────────────────────────────┐
  │   E(m) ∈ R^64  (CLHE backbone, frozen)                             │
  │                                                                     │
  ├─ Preference structure ──────────────────────────────────────────────┤
  │   μ_C  = mean{ E(m) }                                              │
  │   σ²_C = mean{ ||E(m) - μ_C||² }                                  │
  │                                                                     │
  ├─ RVQ discretize ────────────────────────────────────────────────────┤
  │   E(m) → z(m) = (z1, z2, z3, z_conf)  [L=3, K=256, 1-indexed]     │
  │   + assign 6 creative cue tokens per item  (WP-B)                  │
  │                                                                     │
  ├─ Dispersion-conditioned masked diffusion ───────────────────────────┤
  │   conditioning: seed tokens (fixed) + μ_C, σ²_C (injected)        │
  │   output: next-item token sequence [z1, z2, z3, z_conf, c1..c6]   │
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

After RVQ quantization + creative cue assignment, each item occupies **11 token slots** in the sequence:

```
[BOI, z1, z2, z3, z_conf, c1, c2, c3, c4, c5, c6]
  ↑                  ↑         ↑
BOI token     conflict digit   6 creative cue tokens
```

Stride `k = 11`. RVQ codes are **1-indexed** (z1 ∈ [1,256], z2 ∈ [257,512], z3 ∈ [513,768]); `z_conf` is a separate conflict-avoidance digit (74 observed values, range 769–842).

Full sequence structure:

```
[BOS,  BOI, z1..z3, z_conf, c1..c6,  BOI, z1..z3, z_conf, c1..c6, ..., EOS]
  ↑    ←──── item 1 (k=11) ─────────→  ←──── item 2 ──────────────→      ↑
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
generated tokens [z1, z2, z3, z_conf, c1..c6]
    ↓
decode RVQ → Ê(m) = weight[z1-1] + weight[z2-1] + weight[z3-1]  ∈ R^64
    │
    ├─ kNN in catalog CLHE space (faiss IndexFlatIP)
    │   → top-k neighbors → metadata (title, artist, genre, mood, tempo, key, lyric_excerpt)
    │
    ├─ creative cue tokens c1..c6 → cue vocabulary lookup → imagery words
    │
    └─ LLM (Qwen3) prompt assembly:
         neighbor metadata + cue words + playlist style summary (from μ_C kNN)
         → music attributes (genre, mood, tempo, key, instrumentation, language)
         → lyric draft with [verse]/[chorus]/[bridge] section markers
                    ↓
              ACE-Step (frozen) → personalized audio waveform
```

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
| `diffusion.py` | Pass μ_C, σ²_C through full call chain; generation not restricted to catalog items |
| `tokenizer.py` | Vocab = 2894; stride k=11; L=3/K=256 RVQ (1-indexed); load `item2cues.json`; update illegal-mask table |
| `dataset.py` | Compute μ_C, σ²_C per batch; filter: original length 30–90, freq≥10, post-filter 10–60 |
| `evaluator.py` | Replace retrieval metrics with FAD, CLAP-Sim, Δσ², CD, MERT/CLAP/ImageBind semantic sim |
| `configs/config.yaml` | vocab_size=2894, rq_n_codebooks=3, rq_codebook_size=256, stride=11, dispersion_cond=true |
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
| Token stride: k=11 per item | ✓ |
| Vocab size: 2894 (incl. MASK token) | ✓ |
| Creative cues vocab size: 2048 | ✓ |
| K=6 cues per item | ✓ |
| Dispersion conditioning via additive SiLU projection | ✓ |
| Verbalization: kNN in CLHE space + Qwen3 LLM (no T5) | ✓ |
| Synthesis: ACE-Step (frozen) | ✓ |
| Avg σ²_C (v2 training set): 0.282 (Q33=0.255, Q66=0.310) | ✓ |
