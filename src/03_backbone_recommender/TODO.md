# WP-D — status and remaining TODO

## Implemented

- [x] Read `data/dataset/splits/{train,val,test}.txt` and preserve sparse IDs as strings.
- [x] Keep playlists shorter than `seq_len`; augmentation retains originals.
- [x] Add the 13-token tokenizer with strict semantic/cue artifact checks.
- [x] Emit context embeddings, `mu_c`, `sigma_c2`, attention mask, and target mask.
- [x] Decode tokens to a validated `GeneratedItem` and 64-D reconstruction.
- [x] Add `mu_c`/`sigma_c2` conditioning to DiT.
- [x] Corrupt targets only and normalize loss by target-token count across the batch.
- [x] Resolve codebook paths portably and provide a PyTorch attention fallback.
- [x] Select the new tokenizer in the Spotify training config and pad variable-length batches.
- [x] Exclude padded positions from attention keys with an explicit sequence mask.
- [x] Apply the per-position legal-token mask in the shared model forward path.
- [x] Train with all preceding songs as references and the last song as the only target.
- [x] Require at least two references and expose a fixed one-item production sampler.
- [x] Make recommendation evaluation reject multi-item targets.
- [x] Replace production next-block sampling with explicit full-mask completion.

## Remaining / training blockers

- [ ] Generate per-item CLHE, RVQ codebook, semantic-token, and eight-cue artifacts.
- [x] Thread structure conditions through full-mask completion sampling.
- [ ] Train a new checkpoint; old checkpoints are structurally incompatible.
- [x] Implement a lazy checkpoint-backed `run_backbone` adapter.
- [ ] Validate the adapter against the newly trained server checkpoint.
- [ ] Run retrieval, dispersion-tier, semantic-audio, and ablation evaluations.
