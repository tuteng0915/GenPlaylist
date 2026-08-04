# WP-C frozen train/evaluation protocol

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

## Evaluation

- Keep only test playlists with at least 20 songs and deterministically retain
  their first 20 chronological songs.  This gives 468 examples in the current
  frozen test split.
- Songs 1–15 are the shared reference context.  Songs 16–20 are the five
  ground-truth future songs.
- Run the next-one full-MASK sampler independently five times from the unchanged
  15-song context.  Generated songs are not autoregressively fed back.
- Compute a 5x5 cosine-similarity matrix between generated and ground-truth CLHE
  representations, then use Hungarian assignment for the optimal one-to-one,
  order-free match.
- Report optimal matched cosine, multiset exact matches, recall, precision, F1,
  any-hit rate, and prediction unique ratio.  Duplicate generations cannot earn
  repeated credit for a single ground-truth item.

## Server run note

Training still requires the server-side Spotify DDBC checkpoint and frozen
semantic/RVQ artifacts.  Use `scripts/train_spotify.sh` in the `music` conda
environment after setting `GENPLAYLIST_DATA_ROOT` and, for warm-start training,
`GENPLAYLIST_WARMSTART_CKPT` when their default paths do not apply.  Split and
catalog metadata may live under `GENPLAYLIST_DATA_ROOT` while generated
`item_id_to_row.json`, semantic IDs, catalog embeddings, and RVQ weights live
under `GENPLAYLIST_ARTIFACT_ROOT`; `scripts/train_spotify.sh` passes all four
artifact paths explicitly.
