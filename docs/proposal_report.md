# GenPlaylist: Structure-Aware Playlist Generation via Latent Expansion

**Authors:** Anonymous | **Venue:** Under review | **Year:** 2025
**Local PDF:** [main.pdf](../main.pdf)

---

## TL;DR

GenPlaylist bridges playlist recommendation and music synthesis by treating playlist continuation as latent space expansion rather than catalog retrieval. Given a seed set of songs, it models the playlist's semantic centroid and dispersion, expands the latent representation via dispersion-conditioned masked discrete diffusion (adapted from DDBC), verbalizes the generated embeddings into lyrics and style attributes via an LLM, and synthesizes new audio with ACE-Step. The result is a system that generates genuinely new music that fits the playlist's structure rather than selecting from an existing catalog.

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
Seed set C
  → [§4.1] CLHE encode → compute μ_C, σ²_C
  → [§4.2] RVQ discretize → token matrix Z^(0) ∈ N^(|C|×L)
  → [§4.3] Dispersion-conditioned masked diffusion → Ẑ_Y (n new latent codes)
  → [§4.4] Latent verbalization (top-k neighbor lookup + LLM) → lyrics L_j, attributes A_j
  → [§4.5] ACE-Step synthesis → n new audio tracks
```

**Key concepts:**

**Semantic centroid & dispersion** (§3): For seed set C, compute
- μ_C = mean of CLHE embeddings
- σ²_C = mean squared distance from centroid

A valid continuation should scatter around μ_C at scale σ²_C (Cohesion) while each piece differing from all seeds (Novelty).

**RVQ** (§4.2): Each continuous embedding E(m) ∈ ℝ^d is discretized into an L-level code tuple z(m) = (z_{m,1}, …, z_{m,L}) via residual vector quantization. Codebooks are trained with the encoder. This is identical to the DDBC tokenization.

**Dispersion-conditioned masked discrete diffusion** (§4.3): Absorbing-mask Markov chain; at each step t, target tokens are masked with probability β_t. The reverse denoiser (bidirectional Transformer) conditions on σ²_C and μ_C projected into embedding space and **prepended as context tokens** before the target slots. Seed tokens are never masked (fixed context). Training objective: NELBO = -E[log p_θ(z^(0)_{j,ℓ} | Z^(t), t, μ_C, σ²_C)].

**Latent verbalization** (§4.4): Find top-k nearest catalog neighbors of each generated Ê(m_j) by cosine similarity in CLHE space. Read off their metadata as a vocabulary. Playlist-level lookup on μ_C produces a shared style summary prepended to all slots.

**Lyric & attribute generation** (§4.5): LLM (Qwen3) produces music attributes A_j = {genre, mood, tempo, instrumentation, key, language} and a lyric draft L_j following ACE-Step markup (verse/chorus/bridge markers). ACE-Step synthesizes audio conditioning on attributes, lyrics, and the nearest neighbor's audio as reference.

---

## Key Figures

### Figure 1 — Full Pipeline Overview
![Fig 3](fig_3-3.png)
**What it shows:** End-to-end flow from seed songs + text instruction through diffusion-based playlist expansion, latent-to-text decoding (Route C preferred), LLM lyric/tag generation, to final audio synthesis.
**Key insight:** The "main research contribution" (left half) is the diffusion expansion module. The right half (verbalization + synthesis) is engineering/demo infrastructure. The paper's novelty claim is squarely in the latent expansion stage.

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
| Generative (no latent expansion) | MusicGen-Text, ACE-Step-LLM |
| GenPlaylist ablations | w/o disp., w/o verbal., **Full** |

Evaluation: FAD↓ (audio quality), CLAP↑ (lyric/attribute adherence), human Coherence/Quality/Overall↑. Also: Dispersion Match Δσ²↓ and Centroid Distance CD↓ (structural metrics, ablation only).

---

## Strengths

- Elegant reuse of DDBC's discrete diffusion machinery for a genuinely new task (latent expansion vs. catalog retrieval)
- Dispersion conditioning explicitly models playlist-level semantic structure — one model handles both compact and diverse playlists
- Text as the bridge between incompatible embedding spaces is well-motivated and avoids the paired-data problem
- Evaluation design avoids the GT-matching trap (generated music can't overlap catalog by construction)

## Weaknesses / Limitations

- Results are fully redacted — can't assess whether the approach actually works
- CLHE encoder is proprietary to NetEase; reproducibility on other datasets depends on a substitute encoder
- Lyrics → audio quality ultimately bottlenecked by ACE-Step and Qwen3, not the diffusion module
- No ablation on the number of RVQ levels L or codebook size
- σ²_C from a small seed set (|p|/2 songs) may be a noisy dispersion estimate

## Open Questions

- How sensitive is generation quality to seed set size? The paper uses first ⌊|p|/2⌋ as seeds.
- Can the verbalization step be skipped with a direct latent→audio decoder (bridging the embedding incompatibility)?
- How does Novelty (cosine distance from seeds) trade off against Cohesion in practice?
- Is CLHE replaceable with CLAP or MERT for open-source reproducibility?

---

## One-line Takeaway

> GenPlaylist repurposes DDBC's discrete diffusion from bundle retrieval to playlist *generation* by conditioning on per-playlist semantic dispersion and routing generated latents through text to a music synthesis model.
