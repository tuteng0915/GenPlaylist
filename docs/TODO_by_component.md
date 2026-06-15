# GenPlaylist_Code — Todo List by Component

Each section corresponds to one source file or folder; items link directly to what needs changing.

---

## 1. `dataset.py` + `configs/data/`

Data loading layer. Currently only handles the Spotify MPD format.

- [ ] Dataset already filtered (v2 subset in `data/playlists/mpd_subset/`): original length 30–90, freq≥10, post-filter 10–60, 8-round convergence → 6,585 playlists / 5,119 songs
- [ ] Split each playlist into first ⌊|p|/2⌋ songs as reference context + first held-out next item as proxy target
- [x] Table 1 dataset statistics: 6,585 playlists / 5,119 songs / avg len 28.7 / avg σ²_C 0.282

**External dependency**: normalized context prefix format (item_id list) from WP-A

---

## 2. `tokenizer.py`

Joint RVQ + Creative Cue tokenization. Currently pure RVQ (3–4 levels, K=256).

### 2a. RVQ parameters
- [ ] Use L=3, K=256 (1-indexed codes from CLHE backbone); vocab structure → **vocab_size = 2894**
  - BOS: 1
  - RVQ codes (1-indexed, global): 768  (z1 ∈ [1,256], z2 ∈ [257,512], z3 ∈ [513,768])
  - Conflict digit z_conf: 74  (observed range 769–842)
  - BOI: 1 / EOS: 1
  - Creative Cues: 2048
  - MASK token: 1
- [ ] Update `boi_token`, `eos_token`, `vocab_size` accordingly
- [ ] Embedding reconstruction: `E(m) = weight[z1-1] + weight[z2-1] + weight[z3-1]`

### 2b. Joint RVQ + Creative Cue stride (11 tokens per item)
- [ ] Modify `_tokenize_once`: item stride = `[BOI, z1, z2, z3, z_conf, c1, c2, c3, c4, c5, c6]` (11 tokens)
- [ ] Load `item2cues.json` from WP-B; map cue IDs to vocabulary offset
- [ ] Fallback: when WP-B output is not ready, fill cue positions with 0 (also the `w/o cues` ablation path)

### 2c. Position-type illegal masking
- [ ] Implement `make_type_mask(seq_len, stride=11)` → valid token range per position
- [ ] Apply in `diffusion.py` inference loop to filter illegal tokens from logits

**External dependency**: WP-B outputs `item2cues.json` and `cue_vocab.json` (2048 entries)

---

## 3. `playlist_structure.py` ← **new file (draft created)**

Computes μ_C and σ²_C as conditioning signals for the diffusion model.

- [ ] Switch embedding source in `compute_playlist_structure()` from faiss weight matrix to
  real CLHE encoder output (replace once CLHE embeddings are ready)
- [ ] Call from `tokenizer.tokenize_function()` during collation; attach `mu_c` and `sigma_c2` to batch
- [ ] Compute σ²_C over full training set; call `compute_dispersion_tiers()` to get Q33/Q66;
  save to `{dataset_dir}/dispersion_tiers.json`
- [ ] Use `get_dispersion_tier()` at test time to label each sample compact/medium/diverse
  (for Table 3 stratified analysis)

---

## 4. `models/dit.py`

DIT backbone + AdaLN conditioning. Currently `cond_dim` only accepts CFG context embedding.

- [ ] Add `DispersionEmbedder` module:
  ```python
  W_mu    : nn.Linear(d_emb, d_hidden, bias=False)
  W_sigma : nn.Linear(1,     d_hidden, bias=False)
  ```
  Both use SiLU activation, corresponding to paper Eq. (5)
- [ ] Modify `DIT.__init__`: set `cond_dim` to `d_hidden`
  (τ + μ_C + σ²_C projected and summed; `adaLN_modulation` input dim unchanged)
- [ ] Modify conditioning construction in `DIT.forward`:
  ```
  c = SiLU(W_τ·τ) + SiLU(W_μ·μ_C) + SiLU(W_{σ²}·σ²_C)
  ```
  replacing the original CFG embedding path
- [ ] Add to `configs/config.yaml`:
  ```yaml
  dispersion_cond: true
  centroid_dim: 512    # CLHE embedding dim, TBD
  ```

---

## 5. `diffusion.py`

Masked discrete diffusion training + inference. Currently masking is applied uniformly across the full sequence.

- [ ] **Forward corruption**: exclude seed token positions from the absorbing mask
  (seed tokens remain visible throughout denoising; only the target next-item slot is masked)
- [ ] **Reverse denoising**: ensure seed position logits are excluded from unmask updates
  in `semi_ar_sample` / `_denoiser_step`
- [ ] **Batch conditioning**: extract `mu_c` (tensor `[B, d]`) and `sigma_c2` (tensor `[B, 1]`)
  from batch; pass into `backbone.forward()` conditioning path
- [ ] **Position-type illegal masking** (with tokenizer.py §2c):
  call `make_type_mask()` at inference to filter logits

---

## 6. `verbalization.py` ← **new file (draft created)**

Adapted from VibeMus/assistant.py LLM tag/lyric generation logic; chat-loop removed, replaced with functional interface.

- [ ] Implement real DashScope/Qwen3 call: remove stub in `_call_qwen3()`,
  wire up `dashscope.Generation.call(model="qwen3-7b-instruct", ...)`
- [ ] Replace numpy cosine in `knn_verbalize()` with faiss `IndexFlatIP`
  (numpy is a bottleneck when catalog > 100k items)
- [ ] Build faiss index once at startup and cache (catalog embeddings are static)
- [ ] Load `catalog_metadata.json` (title/artist/genre/mood/tempo/key/language/lyric_excerpt)
- [ ] Calibrate the diversity threshold `sigma_c2 > 1.0` against training-set Q66
- [ ] For S candidates: call `verbalize()` independently per candidate; share one `style_summary` (μ_C neighbors)

---

## 7. `synthesis.py` ← **new file (draft created)**

Adapted from VibeMus/pipeline.py and VibeMus/tools.py.

- [ ] Confirm ACE-Step install path and test import
- [ ] Enable `torch_compile=True` (requires torch ≥ 2.3)
- [ ] Implement `style_ref_audio_path` support: when ref is present, switch to ACE-Step
  `task='edit'` path (ref = kNN nearest-neighbor audio from verbalization)
- [ ] Synthesize S candidates in parallel (independent pipe calls)
- [ ] Wire `output_dir` to `configs/config.yaml` instead of hardcoding

---

## 8. `evaluator.py`

Currently only supports catalog-match metrics (Recall/Precision/Jaccard/OAS). Generative metrics need to be added.

- [ ] Integrate **MERT** (acoustic encoder):
  `from transformers import AutoModel; model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M")`
- [ ] Integrate **CLAP** (audio-language encoder)
- [ ] Integrate **ImageBind** (cross-modal encoder)
- [ ] Implement `compute_semantic_sim(gen_audio_path, gt_audio_path, encoder)` → cosine similarity
- [ ] Implement GT upper bound: intra-set similarity within each playlist
- [ ] Implement **Dispersion Match** Δσ²:
  pass generated audio through CLHE → embedding → compute new σ²_C with seed →
  `|σ²_{C ∪ {ẑ}} - σ²_C|`
- [ ] Implement **Centroid Distance** CD (for ablation table)
- [ ] Integrate **FAD** (Fréchet Audio Distance); build genre/style reference set
- [ ] Implement **Condition CLAP**: `cos(CLAP(generated_audio), CLAP(attributes_text + lyrics))`
- [ ] Stratify Δσ² by compact/medium/diverse tiers → Table 3

---

## 9. `configs/config.yaml`

- [ ] Update `vocab_size: 2692` (2691 + 1 MASK token)
- [ ] Update `rq_n_codebooks: 4`, `rq_codebook_size: 128`, `stride: 12`
- [ ] Add `dispersion_cond: true`, `centroid_dim: 512`
- [ ] Add `verbalization.k: 5`, `verbalization.sigma_threshold: TBD`
- [ ] Add `synthesis.audio_duration: 30`, `synthesis.output_dir: outputs/`
- [ ] Add ablation flags: `ablation.no_dispersion`, `ablation.no_cues`,
  `ablation.no_verbalization`, `ablation.no_diffusion`
- [ ] Update `scripts/train_spotify.sh`

---

## File Structure (after changes)

```
GenPlaylist_Code/
├── dataset.py               # data loading
├── tokenizer.py             # RVQ + Creative Cue, 12-token stride     ← major changes
├── playlist_structure.py    # μ_C, σ²_C computation                   ← new ✓
├── models/
│   └── dit.py               # AdaLN dispersion conditioning           ← modified
├── diffusion.py             # seed masking + batch conditioning        ← modified
├── verbalization.py         # kNN lookup + Qwen3 prompting            ← new ✓ (from VibeMus)
├── synthesis.py             # ACE-Step wrapper                        ← new ✓ (from VibeMus)
├── evaluator.py             # MERT/CLAP/IB/FAD/Δσ²                   ← major changes
├── main.py                  # add generate_mode entry point           ← minor changes
└── configs/config.yaml      # new parameters                          ← modified
```

---

## Inter-module Interface Contract

```
WP-B  item2cues.json           →  tokenizer.py    (§2b)
WP-A  context prefix C         →  dataset.py      (§1)
playlist_structure.py μ_C/σ²   →  diffusion.py    (§5, batch conditioning)
diffusion.py  z_hat_emb         →  verbalization.py (§6)
verbalization.py attrs+lyrics   →  synthesis.py    (§7)
synthesis.py  audio_path        →  evaluator.py    (§8, semantic sim / FAD)
synthesis.py  audio_path        →  WP-C Demo API
```
