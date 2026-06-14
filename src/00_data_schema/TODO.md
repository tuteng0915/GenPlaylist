# 00_data_schema — TODO

**Owner:** Mentor (WP-D)
All other WPs import from `schema.py` — coordinate before making changes.

---

## Status

- [x] `CatalogItem` — per-song metadata; `feature_index = int(item_id)`; `from_metadata_string()` parses Spotify format; `to_prompt_line()` for LLM prompts
- [x] `ContextPrefix` — WP-A → WP-D; string item IDs matching `clhe_token.json` keys; `validate()`
- [x] `CueMappingEntry` — WP-B → WP-D; 6 cues per item; `load_mapping()` + `validate()`
- [x] `GeneratedItem` — WP-D → WP-C; 3 RVQ codes + conflict digit + 64-dim embeddings; `validate()`
- [x] `SynthesisResult` — WP-C → Demo/Eval; `validate()` checks audio file exists on disk
- [x] `test_schema.py` — 11/11 tests passing; all constants grounded in actual backbone data

---

## Confirmed from backbone (Spotify dataset)

| Parameter | Value | Source |
|-----------|-------|--------|
| `rq_n_codebooks` | 3 | `config.yaml` |
| `rq_codebook_size` | 256 | `config.yaml` |
| `CLHE_EMB_DIM` | 64 | `clhe_weight.npy` shape (768, 64) |
| tokens per item | 5 | BOI + z0 + z1 + z2 + z_conf |
| `boi_token` | 1025 | `(3+1)*256 + 1` |
| `eos_token` | 1026 | `boi_token + 1` |
| `vocab_size` (training) | 1027 | |
| `vocab_size` (runtime) | 1028 | +1 MASK token in `diffusion.py` |
| Item ID type | `str` | keys in `metadata.json`, `clhe_token.json` |
| `feature_index` | `int(item_id)` | row index in `clhe_weight.npy` merged codebook |

Token offset per RVQ level:
- z0 ∈ [1, 256] — offset = 1
- z1 ∈ [257, 512] — offset = 257
- z2 ∈ [513, 768] — offset = 513
- z_conf ∈ [769, 1024] — offset = 769

---

## Pending

### build_catalog_metadata.py

- [ ] Write `build_catalog_metadata.py` that reads `03_backbone_recommender/datasets/spotify/metadata.json` and produces `outputs/catalog_metadata.json`:
  - Parse each entry via `CatalogItem.from_metadata_string(item_id, meta_str)`
  - Set `feature_index = int(item_id)`
  - Output format: JSON list of dicts matching `CatalogItem.__dataclass_fields__`
  - Script should print: N items written, N parse failures (entries that didn't match the regex)
- [ ] Run the script on the full Spotify catalog (N=254155); inspect 20 random entries for correctness
- [ ] Check that `CatalogItem.load_catalog("outputs/catalog_metadata.json")` round-trips cleanly

### Known issues in backbone (do NOT modify backbone to fix)

- `evaluator.py:26–33` hardcodes `/home/sjj/wenhao/DISCO/datasets/` — fails on any other machine.
  **Fix at runtime:** patch `config.evaluator.dataset_root` before calling `main.py`.

### GenPlaylist extension checkpoints (update as WPs deliver)

- [ ] WP-B delivers `item2cues.json` → run `CueMappingEntry.load_mapping()` on full Spotify catalog; confirm 100% coverage (every item_id in `clhe_token.json` has an entry)
- [ ] After tokenizer update to 12-token stride: change `RQ_N_CODEBOOKS=4`, `RQ_CODEBOOK_SIZE=128`, `CUE_TOKENS=6`; re-run `test_schema.py` to catch breakage early
- [ ] WP-C first `.wav` output → run `SynthesisResult.validate()` on a real result to confirm `audio_path` is a valid file
