# 03_backbone_recommender — TODO (WP-D)

**Owner:** Mentor
**Goal:** Train and validate the dispersion-conditioned masked discrete diffusion model.

**Input:** `ContextPrefix` from WP-A · `item2cues.json` from WP-B
**Output:** `GeneratedItem` (z_hat_emb, cue_ids, μ_C, σ²_C) → passed to `04_synthesis`

---

## dataset.py
- [ ] Add NetEase Cloud Music playlist format (field mapping: playlist_id, track_list, metadata)
- [ ] Unified filter: sequence length ≥ 5; split into context prefix + first held-out next item
- [ ] Fill Table 1 dataset statistics: `#Playlists`, `#Songs`, `Avg. len`, `Avg. σ²_C`

---

## tokenizer.py
- [ ] Update RVQ params: `rq_n_codebooks=4`, `rq_codebook_size=128`
- [ ] Update vocabulary layout → **vocab_size = 2691**
  - BOS: 1 | RVQ L0–L3: 4×128=512 | Conflict: 128 | BOI: 1 | EOS: 1 | Creative Cues: 2048
- [ ] Update `boi_token = 641`, `eos_token = 642`
- [ ] Implement **12-token item stride**: `[BOI, z1, z2, z3, z4, z_conf, c1, c2, c3, c4, c5, c6]`
- [ ] Load `02_creative_cues/outputs/item2cues.json`; map cue IDs to vocab offset (+643)
- [ ] Fallback: if `item2cues.json` absent, fill cue positions with 0 (also used for `w/o cues` ablation)
- [ ] Implement `make_type_mask(seq_len, stride=12)` — per-position legal token range for inference

---

## playlist_structure.py
- [ ] Switch embedding source from faiss weight matrix to real CLHE encoder output
      once CLHE embeddings are available
- [ ] Call `compute_playlist_structure()` inside `tokenizer.tokenize_function()`;
      attach `mu_c` (tensor `[d]`) and `sigma_c2` (float) to each batch
- [ ] Compute Q33/Q66 over training set; save to `{dataset_dir}/dispersion_tiers.json`
- [ ] Use `get_dispersion_tier()` to label test samples as compact/medium/diverse (Table 3)

---

## models/dit.py
- [ ] Add `DispersionEmbedder`:
  - `W_mu: nn.Linear(d_emb, d_hidden, bias=False)` + SiLU
  - `W_sigma: nn.Linear(1, d_hidden, bias=False)` + SiLU
- [ ] Change conditioning signal in `DIT.forward`:
  `c = SiLU(W_τ·τ) + SiLU(W_μ·μ_C) + SiLU(W_{σ²}·σ²_C)`
  replacing current CFG embedding path
- [ ] Update `configs/config.yaml`: add `dispersion_cond: true`, `centroid_dim: 512`

---

## diffusion.py
- [ ] **Forward corruption**: exclude seed token positions from absorbing mask
      (only the target next-item slot is masked)
- [ ] **Inference**: in `semi_ar_sample`, ensure seed positions are never unmasked
- [ ] **Batch conditioning**: extract `mu_c [B, d]` and `sigma_c2 [B, 1]` from batch;
      pass to `backbone.forward()`
- [ ] **Type mask**: apply `make_type_mask()` at inference to filter illegal token logits
- [ ] Add `generate_mode` to `main.py`: load checkpoint, run inference,
      return `GeneratedItem` list for `pipeline/genplaylist.py`

---

## evaluator.py
- [ ] Integrate **MERT** (acoustic): `AutoModel.from_pretrained("m-a-p/MERT-v1-95M")`
- [ ] Integrate **CLAP** (audio-language encoder)
- [ ] Integrate **ImageBind** (cross-modal encoder)
- [ ] Implement `compute_semantic_sim(gen_audio, gt_audio, encoder)` → cosine similarity
- [ ] Compute GT upper bound: intra-set similarity of ground-truth playlists
- [ ] Implement `compute_dispersion_match(seed_embs, gen_emb)` → Δσ²
      (encode generated audio through CLHE first)
- [ ] Implement `compute_centroid_distance(seed_embs, gen_emb)` → CD (for ablation table)
- [ ] Integrate **FAD** (Fréchet Audio Distance); build genre/style reference sets
- [ ] Implement **Condition CLAP**: `cos(CLAP(audio), CLAP(attributes_text + lyrics))`
- [ ] Partition test results by compact/medium/diverse tier → Table 3

---

## Ablation configs
- [ ] `w/o disp.` — set `dispersion_cond: false` in config; c = SiLU(W_τ·τ) only
- [ ] `w/o cues` — tokenizer fallback path (cue positions = 0); stride still 12
- [ ] `w/o verbal.` — skip kNN lookup in verbalization; LLM receives title + seed metadata only
- [ ] `w/o diffusion` (mean imputation) — use μ_C nearest-neighbor directly as ẑ_{t+1}

---

## Baselines
- [ ] **Pop**: global popularity rank; return top-1 embedding
- [ ] **DDBC**: current code in decode-to-catalog mode (direct comparison baseline)
- [ ] **DreamRec**: generate embedding → kNN retrieve from catalog
- [ ] **ACE-Step-LLM**: skip diffusion; LLM + ACE-Step only (ablates the diffusion module)
- [ ] **MusicGen-Text**: generate from playlist title only
