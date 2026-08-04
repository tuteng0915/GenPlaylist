# GenPlaylist: Reference-Conditioned Next-Song Generation with DDBC

**Authors:** Anonymous | **Venue:** Under review | **Year:** 2025
**Local PDF:** [main.pdf](../main.pdf)

---

## TL;DR

GenPlaylist bridges next-item recommendation and music synthesis. Given multiple ordered reference tracks, it uses DDBC to predict exactly one next-item latent while conditioning on the references' semantic centroid and dispersion. An LLM verbalizes that latent into attributes and lyrics, and ACE-Step synthesizes one original next song. The output is generated music rather than a retrieved catalog item.

---

## Problem & Motivation

- Standard playlist continuation is retrieval: pick existing songs from a catalog. This limits novelty to what already exists.
- A playlist is a *structured semantic set* with characteristic **compactness** (e.g., covers of one song) and **diversity** (e.g., a genre mix). A single compatibility model ignores this playlist-level structure.
- Music embedding spaces (CLHE/MERT/CLAP) and audio generation latent spaces (EnCodec) are fundamentally incompatible — you can't simply route a recommendation embedding directly to an audio model.
- Text (lyrics + tags) is the universal bridge: recommendation models are grounded in catalog metadata; audio generation models (MusicGen, ACE-Step) condition on natural language. Lyrics carry richer musical structure than free-form descriptions.

---

## Method

Four-stage pipeline:

```
Ordered references C = (m1, ..., mt), t ≥ 2
  → [§4.1] CLHE encode → compute μ_C, σ²_C
  → [§4.2] RVQ discretize → token matrix Z^(0) ∈ N^(|C|×L)
  → [§4.3] DDBC masked diffusion → one next-item latent ẑ_(t+1)
  → [§4.4] Latent verbalization (top-k lookup + LLM) → lyrics L, attributes A
  → [§4.5] ACE-Step synthesis → one next-song audio track
```

**Key concepts:**

**Semantic centroid & dispersion** (§3): For reference sequence C, compute
- μ_C = mean of CLHE embeddings
- σ²_C = mean squared distance from centroid

A valid next song should be compatible with the reference structure while remaining distinct from every reference.

**RVQ** (§4.2): Each continuous embedding E(m) ∈ ℝ^d is discretized into an L-level code tuple z(m) = (z_{m,1}, …, z_{m,L}) via residual vector quantization. Codebooks are trained with the encoder. This is identical to the DDBC tokenization.

**Dispersion-conditioned masked discrete diffusion** (§4.3): The DDBC absorbing-mask process corrupts only the single next-item payload. At inference the system appends `[BOI, MASK×12, EOS]` and jointly denoises those twelve positions; it does not autoregressively append multiple blocks. The bidirectional DiT conditions on projected σ²_C and μ_C through AdaLN, while reference tokens remain fixed.

**Latent verbalization** (§4.4): Find top-k catalog neighbors of the single predicted latent by cosine similarity in CLHE space. The prompt includes the actual ordered references, target-latent neighbors, creative cues, and a reference-centroid style summary.

**Lyric & attribute generation** (§4.5): The LLM produces one attribute set A = {genre, mood, tempo, instrumentation, key, language} and one lyric draft L following ACE-Step markup. ACE-Step then synthesizes the next-song audio.

---

## Key Figures

### Figure 1 — Full Pipeline Overview
![Fig 3](fig_3-3.png)
**What it shows:** End-to-end flow from multiple references + optional instruction through DDBC next-item prediction, latent verbalization, and one-song synthesis.
**Key insight:** The research question is whether a DDBC next-item latent can serve as a useful semantic plan for generating the next song rather than retrieving it.

### Figure — Dispersion Conditioning & Evaluation Setup
![Fig 4](fig_4-4.png)
**What it shows:** Problem formulation, RVQ equations, diffusion forward/reverse process, and evaluation metrics (FAD, CLAP Score, human evaluation axes).
**Key insight:** σ²_C is injected as a conditioning signal so the same model can produce tight continuations for compact playlists and scattered ones for diverse playlists.

### Figure — Results Tables
![Fig 6](fig_6-6.png)
**What it shows:** Table 2 (main results on NetEase + Spotify MPD), Table 3 (ablation), Table 4 (dispersion match by compactness bin), Table 5 (semantic similarity to GT continuation).
**Key insight:** All numerical cells are redacted ("–") in this draft — the paper is still placeholder for camera-ready results. The table structure reveals what they plan to demonstrate: FAD↓, CLAP↑, human Coherence/Quality/Overall.

---

## Key Results

All result cells are redacted in this draft. Planned comparison groups:

| Category | Methods |
|---|---|
| Retrieval-based | Pop, SASRec, BGCN, **DDBC** |
| Continuous diffusion | DiffRec, DMSR |
| Generative (no DDBC next-item plan) | MusicGen-Text, ACE-Step-LLM |
| GenPlaylist ablations | w/o disp., w/o verbal., **Full** |

Evaluation: FAD↓ (audio quality), CLAP↑ (lyric/attribute adherence), human Coherence/Quality/Overall↑. Also: Dispersion Match Δσ²↓ and Centroid Distance CD↓ (structural metrics, ablation only).

---

## Strengths

- Reuses DDBC's discrete diffusion machinery for a clear next-item generation task
- Dispersion conditioning explicitly models playlist-level semantic structure — one model handles both compact and diverse playlists
- Text as the bridge between incompatible embedding spaces is well-motivated and avoids the paired-data problem
- Evaluation design avoids the GT-matching trap (generated music can't overlap catalog by construction)

## Weaknesses / Limitations

- Results are fully redacted — can't assess whether the approach actually works
- CLHE encoder is proprietary to NetEase; reproducibility on other datasets depends on a substitute encoder
- Lyrics → audio quality ultimately bottlenecked by ACE-Step and Qwen3, not the diffusion module
- No ablation on the number of RVQ levels L or codebook size
- σ²_C can be noisy when only two or three reference songs are supplied

## Open Questions

- How sensitive is generation quality to the number and order of reference songs?
- Can the verbalization step be skipped with a direct latent→audio decoder (bridging the embedding incompatibility)?
- How does novelty (distance from references) trade off against next-song compatibility?
- Is CLHE replaceable with CLAP or MERT for open-source reproducibility?

---

## One-line Takeaway

> GenPlaylist uses DDBC to plan one next-item latent from multiple references, then turns that plan into one original song through verbalization and audio synthesis.
