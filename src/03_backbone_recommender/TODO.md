# WP-C — status and remaining TODO

## Implemented

- [x] Read `data/dataset/splits/{train,val,test}.txt` and preserve sparse IDs as strings.
- [x] Expand playlists into rolling 15-reference -> 5-target training windows.
- [x] Add a configurable-cue tokenizer (8 cues in the main 13-token layout).
- [x] Emit context embeddings, attention masks, and target masks.
- [x] Decode tokens to a validated `GeneratedItem` and 64-D reconstruction.
- [x] Condition DiT on visible history tokens.
- [x] Corrupt all five targets only and normalize loss by target-token count.
- [x] Resolve codebook paths portably and provide a PyTorch attention fallback.
- [x] Select the new tokenizer in the Spotify training config and pad variable-length batches.
- [x] Exclude padded positions from attention keys with an explicit sequence mask.
- [x] Apply the per-position legal-token mask in the shared model forward path.
- [x] Enforce the shared GenPlaylist-v4 protocol constants at data/tokenizer startup.
- [x] Require 15 training references while retaining a one-item WP-D wrapper.
- [x] Freeze test rows to 20 songs: 15 references plus five future targets.
- [x] Draw one joint five-item full-MASK completion from the same context.
- [x] Score predictions and targets with order-free 5x5 Hungarian matching.
- [x] Replace production next-block sampling with explicit full-mask completion.
- [x] Add semantic warm-start loading for the official 1,028-token DDBC Spotify checkpoint.
- [x] Use the normalized RVQ/cue curriculum in Full; retain uniform weighting as
      the controlled ablation.
- [x] Report generated-cue multiset recall/F1/uniqueness during joint 5x5 evaluation.
- [x] Add extraction of the checkpoint's embedded CLHE/RVQ artifacts for the 5,119-item catalog.

## Remaining / training blockers

- [x] Run the verified checkpoint extractor and generate the final 16-stored/8-active cue artifacts.
- [x] Thread structure conditions through full-mask completion sampling.
- [ ] Train a new checkpoint; old checkpoints are structurally incompatible.
- [x] Implement a lazy checkpoint-backed `run_backbone` adapter.
- [ ] Validate the adapter against the newly trained server checkpoint.
- [ ] Run retrieval, dispersion-tier, semantic-audio, and ablation evaluations.
