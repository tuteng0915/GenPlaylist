# Frozen next-song training and evaluation protocol

This is the authoritative GenPlaylist-v2 experimental contract shared by
WP-A, WP-B, and WP-C. Machine-readable values are defined once in
`src/shared/protocol.py`; WP-C rejects Hydra overrides that drift from them.

| Setting | Frozen value |
|---|---:|
| Minimum training references | 2 |
| Maximum training references | 15 |
| Training items including target | 16 |
| Test window | first 20 songs |
| Test references | songs 1–15 |
| Test ground truth | songs 16–20 |
| Independent next-one samples | 5 |
| Active cues per song | first 8 of 16 stored |
| Evaluation sources | original val + test, exposed only as `test` |

WP-D synthesis/demo is explicitly excluded from this change. Its production UI
continues to generate and present one next song; no WP-D demo source is modified.

## Training

- The backbone remains DDBC and is fine-tuned for next-one-item completion.
- Each training row contains at most 16 songs: up to 15 chronological reference
  songs and exactly one next-song target.
- Every usable chronological prefix is emitted.  On the frozen Spotify split,
  5,269 playlists therefore become 140,433 training examples rather than 5,269.
- Contexts start at two reference songs.  Long histories use the most recent 15
  references.  Adjacent-swap augmentation is disabled because it changes the
  next-song task.
- Each song still occupies 13 tokens: BOI, three RVQ tokens, one conflict token,
  and the first eight of its 16 stored ranked cue IDs.
- The maximum model sequence length is 210 tokens: `BOS + 16 * 13 + EOS`.

## Unified test evaluation

- Concatenate the original `val.txt` and `test.txt` sources in that order and
  expose them only as the `test` split. There is no separate `valid` split.
- Keep only playlists with at least 20 songs and deterministically retain their
  first 20 chronological songs. This gives 473 + 468 = 941 test examples.
- Songs 1–15 are the shared reference context.  Songs 16–20 are the five
  ground-truth future songs.
- Run the next-one full-MASK sampler independently five times from the unchanged
  15-song context.  Generated songs are not autoregressively fed back.
- Retrieve each generated RVQ prediction against the complete 5,119-song catalog,
  compute a 5x5 cosine-similarity matrix between retrieved and ground-truth CLHE
  representations, then use Hungarian assignment for the optimal one-to-one,
  order-free match.
- Report optimal matched cosine, multiset exact matches, recall, precision, F1,
  any-hit rate, and prediction unique ratio.  Duplicate generations cannot earn
  repeated credit for a single ground-truth item.
- Backbone training has no validation loader and does not inspect the unified
  test set. Checkpoints are saved every 500 steps plus `last.ckpt`; final metrics
  are computed only after training on the unified 941-example test set.

## Cross-WP responsibilities

- **WP-A:** its retrieval evaluation uses the same first-20, 15-reference,
  5-target split. The previous configurable 50/50 split is retired.
- **WP-B:** every reference and target item must resolve to the frozen 16-cue
  ranked table; WP-C consumes only positions 1–8. Cue mining never reads whether
  an item is a reference or target, preventing split-role leakage.
- **WP-C:** expands training prefixes, enforces the 210-token maximum, performs
  five independent full-mask next-one draws, full-catalog retrieval, and 5x5
  matching metrics.
- **WP-D demo:** unchanged until a separate demo decision is made.

## Reproducibility rules

- Preserve playlist order and take the first 20; do not random-crop test rows.
- Preserve source order: eligible original-validation rows first, then eligible
  original-test rows.
- Do not feed any sampled prediction back into the 15-song context.
- Do not use adjacent-swap augmentation.
- Do not change one of the frozen numbers independently. A new setting requires
  a new named protocol/version rather than silently editing GenPlaylist-v2.

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
  --output-dir /home/wjzhang/tt_workspace/data/data/processed/genplaylist-v2-16item-unified-test-15to5

conda run -n music python scripts/validate_wp_c_prepared_data.py \
  --data-dir /home/wjzhang/tt_workspace/data/data/dataset \
  --artifact-dir /home/wjzhang/tt_workspace/model/GenPlaylist/data/dataset \
  --prepared-dir /home/wjzhang/tt_workspace/data/data/processed/genplaylist-v2-16item-unified-test-15to5
```

The versioned output contains raw and tokenized Arrow `DatasetDict`s, normalized
catalog CLHE and RVQ-reconstruction matrices, semantic/cue matrices, full legal
type masks, and every checkpoint-independent 15->5 test tensor. Its manifest
pins source, preparation-code, and output hashes. The validator rechecks every
generated file, representative fresh tokenizations, batch shapes/masks, catalog alignment,
and every vector identity. `train_spotify.sh` loads this cache by default and
fails rather than silently recomputing if it is absent or stale.
