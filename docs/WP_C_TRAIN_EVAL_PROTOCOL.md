# Frozen joint 15-to-5 training and evaluation protocol

This is the authoritative GenPlaylist-v3 experimental contract shared by
WP-A, WP-B, and WP-C. Machine-readable data-shape values are defined once in
`src/shared/protocol.py`; official stochastic evaluation settings are isolated
in `src/03_backbone_recommender/evaluation_protocol.py` so they do not alter
prepared-data fingerprints. WP-C rejects silent overrides of either contract.

| Setting | Frozen value |
|---|---:|
| Training references | exactly 15 |
| Training targets | exactly 5 |
| Training window | 20 songs, rolling stride 1 |
| Test window | first 20 songs |
| Test references | songs 1–15 |
| Test ground truth | songs 16–20 |
| Joint completion draws | 1 draw containing 5 songs |
| Active cues per song | first 8 of 16 stored |
| Evaluation sources | original val + test, exposed only as `test` |
| Official reverse-diffusion steps | 256 |
| Official evaluation seed | 1 |
| Evaluation weights | EMA |

WP-D synthesis/demo is explicitly excluded from this change. Its production UI
continues to generate and present one next song; no WP-D demo source is modified.

## Training

- The backbone remains DDBC and is fine-tuned for joint five-item completion.
- Each training row contains exactly 20 songs: 15 chronological references and
  the next five songs as targets.
- Every 20-song rolling window is emitted with stride one. On the frozen Spotify
  split, 3,804 eligible playlists produce 57,331 training windows and 286,655
  supervised target items.
- Adjacent-swap augmentation is disabled because it changes the continuation task.
- Each song still occupies 13 tokens: BOI, three RVQ tokens, one conflict token,
  and the first eight of its 16 stored ranked cue IDs.
- The maximum model sequence length is 262 tokens: `BOS + 20 * 13 + EOS`.
- Only the 60 payload positions belonging to songs 16–20 are corrupted and
  scored. All 15 reference items and all BOI/EOS boundaries remain fixed.

### Loss-weight ablation

The official baseline keeps `training.layer_loss_weights.enabled=false`, so all
12 payload positions per target item have equal weight. The optional curriculum
is an explicitly named ablation, not part of the baseline:

- RVQ weights are `[2.0, 1.5, 1.0]` and conflict weight is `0.5`.
- Cue weight is held at `0.1` through step 1,000, then linearly increased to
  `1.0` by step 5,000.
- Active position weights are normalized to mean one. Changing the curriculum
  therefore does not change the overall loss scale or effective learning rate.
- Training logs the effective weights plus unweighted NLL for `d0`, `d1`, `d2`,
  conflict, and cues. Evaluation additionally reports order-free cue multiset
  recall, precision, F1, and uniqueness across the five-item continuation.

Run the baseline with the default `train_spotify.sh`. Run the ablation with
`GENPLAYLIST_LAYER_LOSS_CURRICULUM=true`; the run name records `uniform` or
`rvq-cue-warmup` automatically. Both runs must warm-start the same official DDBC
checkpoint and otherwise use identical seeds, data, and schedules.

## Unified test evaluation

- Concatenate the original `val.txt` and `test.txt` sources in that order and
  expose them only as the `test` split. There is no separate `valid` split.
- Keep only playlists with at least 20 songs and deterministically retain their
  first 20 chronological songs. This gives 473 + 468 = 941 test examples.
- Songs 1–15 are the shared reference context.  Songs 16–20 are the five
  ground-truth future songs.
- Append five `[BOI, MASK×12]` blocks and jointly denoise all 60 payload
  positions in one full-MASK pass. No generated song is fed back as context.
- Retrieve each generated RVQ prediction against the complete 5,119-song catalog,
  compute a 5x5 cosine-similarity matrix between retrieved and ground-truth CLHE
  representations, then use Hungarian assignment for the optimal one-to-one,
  order-free match.
- Report optimal matched cosine, multiset exact matches, recall, precision, F1,
  any-hit rate, and prediction unique ratio.  Duplicate generations cannot earn
  repeated credit for a single ground-truth item.
- Use 256 reverse-diffusion steps, seed 1, and EMA weights for the official run.
  Shorter samplers are smoke tests or explicit speed ablations, not the headline result.
- Backbone training has no validation loader and does not inspect the unified
  test set. Checkpoints are saved every 500 steps plus `last.ckpt`; final metrics
  are computed only after training on the unified 941-example test set.

## Cross-WP responsibilities

- **WP-A:** its retrieval evaluation uses the same first-20, 15-reference,
  5-target split. The previous configurable 50/50 split is retired.
- **WP-B:** every reference and target item must resolve to the frozen 16-cue
  ranked table; WP-C consumes only positions 1–8. Cue mining never reads whether
  an item is a reference or target, preventing split-role leakage.
- **WP-C:** expands rolling 20-song training windows, enforces the 262-token
  maximum, performs one joint five-item full-MASK completion, full-catalog
  retrieval, and 5x5 matching metrics.
- **WP-D demo:** unchanged until a separate demo decision is made.

## Reproducibility rules

- Preserve playlist order and take the first 20; do not random-crop test rows.
- Preserve source order: eligible original-validation rows first, then eligible
  original-test rows.
- Do not sequentially feed any generated item back into the 15-song context.
- Do not use adjacent-swap augmentation.
- Do not change one of the frozen numbers independently. A new setting requires
  a new named protocol/version rather than silently editing GenPlaylist-v3.

## Server run note

Training still requires the server-side Spotify DDBC checkpoint and frozen
semantic/RVQ artifacts.  Use `scripts/train_spotify.sh` in the `music` conda
environment after setting `GENPLAYLIST_DATA_ROOT` and, for warm-start training,
`GENPLAYLIST_WARMSTART_CKPT` when their default paths do not apply.  Split and
catalog metadata may live under `GENPLAYLIST_DATA_ROOT` while generated
`item_id_to_row.json`, semantic IDs, catalog embeddings, and RVQ weights live
under `GENPLAYLIST_ARTIFACT_ROOT`; `scripts/train_spotify.sh` passes all four
artifact paths explicitly.

Prepare all checkpoint-independent data once before training:

```bash
conda run -n music python scripts/prepare_wp_c_data.py \
  --data-dir /home/wjzhang/tt_workspace/data/data/dataset \
  --artifact-dir /home/wjzhang/tt_workspace/model/GenPlaylist/data/dataset \
  --output-dir /home/wjzhang/tt_workspace/data/data/processed/genplaylist-v3-20item-joint-15to5

conda run -n music python scripts/validate_wp_c_prepared_data.py \
  --data-dir /home/wjzhang/tt_workspace/data/data/dataset \
  --artifact-dir /home/wjzhang/tt_workspace/model/GenPlaylist/data/dataset \
  --prepared-dir /home/wjzhang/tt_workspace/data/data/processed/genplaylist-v3-20item-joint-15to5
```

The versioned output contains raw and tokenized Arrow `DatasetDict`s, normalized
catalog CLHE and RVQ-reconstruction matrices, semantic/cue matrices, full legal
type masks, and every checkpoint-independent 15->5 test tensor. Its manifest
pins source, preparation-code, and output hashes. The validator rechecks every
generated file, representative fresh tokenizations, batch shapes/masks, catalog alignment,
and every vector identity. `train_spotify.sh` loads this cache by default and
fails rather than silently recomputing if it is absent or stale.

Run the official post-training evaluation with the final checkpoint:

```bash
cd /home/wjzhang/tt_workspace/model/GenPlaylist
export GENPLAYLIST_DATA_ROOT=/home/wjzhang/tt_workspace/data/data/dataset
export GENPLAYLIST_ARTIFACT_ROOT=/home/wjzhang/tt_workspace/model/GenPlaylist/data/dataset
export GENPLAYLIST_PREPARED_DATA_ROOT=/home/wjzhang/tt_workspace/data/data/processed/genplaylist-v3-20item-joint-15to5
export GENPLAYLIST_EVAL_CKPT=/path/to/new/joint-15to5/checkpoint.ckpt
conda run -n music bash src/03_backbone_recommender/scripts/eval_spotify.sh
```

The runner evaluates all 941 rows and atomically writes a JSON containing the
metrics, checkpoint and prepared-manifest hashes, git commit, seed, sampler,
sampling-step count, catalog size, and protocol values under
`src/03_backbone_recommender/outputs/evaluation/`.
For a deliberately non-official smoke run, set both a shorter
`GENPLAYLIST_EVAL_SAMPLING_STEPS` and
`GENPLAYLIST_EVAL_ALLOW_PROTOCOL_OVERRIDE=true`; the JSON is then marked
`official_protocol=false`.

WP-A uses the same unified test windows. Its canonical CLHE baseline is:

```bash
conda run -n music python src/01_input_normalization/build_recall_eval.py \
  --data-dir /home/wjzhang/tt_workspace/data/data/dataset \
  --artifact-dir /home/wjzhang/tt_workspace/model/GenPlaylist/data/dataset
```

Every evaluated encoder must cover exactly the same 5,119 catalog IDs. The
reported MRR is the full-catalog reciprocal rank, not a Top-50 truncation.
