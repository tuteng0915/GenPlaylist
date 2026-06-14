# pipeline — TODO

**Owner:** Mentor
**Goal:** Wire all four Work Packages into a single end-to-end callable.

```
ContextPrefix → generate() → list[SynthesisResult]
```

The pipeline is intentionally stub-heavy; `mock_run.py` verifies the full chain with mock data.
Each item below becomes unblocked once the referenced WP delivers its output.

---

## genplaylist.py — wiring (in delivery order)

### After WP-B delivers item2cues.json

- [ ] Pass `item2cues_path` to the backbone tokenizer config at pipeline startup:
  - Add a `config` parameter to `generate()` or read from an env var `GENPLAYLIST_ITEM2CUES`
  - Verify `CueMappingEntry.load_mapping(item2cues_path)` succeeds without assertion errors
  - Print coverage warning if any item_id in the current ContextPrefix is missing from item2cues

### After WP-A delivers normalizer.py

- [ ] Replace direct `ContextPrefix(item_ids=[...])` construction in callers with:
  ```python
  from normalizer import normalize
  ctx = normalize(user_input, catalog_metadata, catalog_embs, K=5)
  ```
- [ ] Confirm `ctx.validate()` passes for all 5 input types (song-only, text-only, hybrid, padded, too-many)

### After WP-D delivers backbone inference

- [ ] Implement `_run_backbone()` in `genplaylist.py`:
  - Import `03_backbone_recommender/diffusion.py` via `_import_from()`
  - Load checkpoint path from env var `GENPLAYLIST_BACKBONE_CKPT`
  - Call `diffusion.restore_model_and_semi_ar_sample()` with context token sequence
  - Parse output tokens → `GeneratedItem` (decode RVQ codes + conflict digit, compute z_hat via `tokenizer._token_to_feature()`)
  - Call `playlist_structure.compute_playlist_structure()` to get `mu_c` and `sigma_c2`
  - Return `list[GeneratedItem]` of length `n_samples`
- [ ] Add a smoke test: run `_run_backbone()` on 3 context prefixes from `examples/`; verify all returned items pass `GeneratedItem.validate()`

### After WP-C delivers verbalization

- [ ] Confirm `_call_qwen3()` is no longer a stub:
  - Set `DASHSCOPE_API_KEY` in env; run `generate()` on one context; check `music_attributes` is non-empty
  - Check `lyric_draft` contains at least one `[verse]` or `[chorus]` label
- [ ] For S=3 samples: verify that each sample gets its own `neighbors` but shares the same `style_summary`

### After WP-C delivers synthesis

- [ ] Confirm ACE-Step is installed and `synthesize()` writes a real `.wav` file:
  - Run `SynthesisResult.validate()` on each result; it checks `os.path.isfile(audio_path)`
  - Play one generated clip manually; confirm it is audible music

---

## Catalog asset loading

- [ ] Add a `startup(config_path)` function that reads paths from a config YAML and calls `_load_catalog()`:
  ```yaml
  catalog_emb_path:      datasets/spotify/clhe_weight.npy
  catalog_metadata_path: 00_data_schema/outputs/catalog_metadata.json
  ```
- [ ] Ensure `_catalog_embs` and `_catalog_metadata` are never `None` when `generate()` is called; raise a clear `RuntimeError` with setup instructions if they are
- [ ] Load once at module import (or first call); do not reload per `generate()` call

---

## Catalog metadata build

- [ ] Write `00_data_schema/build_catalog_metadata.py`:
  - Input: `03_backbone_recommender/datasets/spotify/metadata.json`
  - Parse each entry via `CatalogItem.from_metadata_string()`
  - Output: `00_data_schema/outputs/catalog_metadata.json` (list of CatalogItem dicts)
  - Include `feature_index = int(item_id)` for each entry
- [ ] Run on Spotify dataset (N=254155); confirm all entries parse without error
- [ ] Run on NetEase dataset if available; note any field differences

---

## Evaluation hooks

- [ ] After `generate()`, optionally run semantic similarity metrics:
  - Accept a `gt_item_id: str | None` parameter in `generate()`
  - If provided, compute cosine similarity between each `z_hat_emb` and `catalog_embs[int(gt_item_id)]`
  - Log result as `{"sigma_c2": ..., "cosine_to_gt": ..., "sample_idx": ...}` to `eval_log.jsonl`
- [ ] Log `sigma_c2` tier (compact / medium / diverse) per call; track distribution across test set
- [ ] Provide a `run_eval.py` script that:
  - Loads `03_backbone_recommender/datasets/spotify/test.txt`
  - For each playlist in test split: uses last item as GT, feeds prefix to `generate()`
  - Collects cosine-to-GT and `sigma_c2` per sample; reports mean ± std
