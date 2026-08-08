# GenPlaylist — TODO by Work Package

## Frozen task definition

Input is an ordered set of 15 reference tracks. DDBC jointly predicts five
continuation token groups in one full-MASK sampling run. Training and evaluation
both use 15 references followed by five targets. WP-D remains unchanged and may
consume one selected latent plan until its demo contract is revised separately.

## WP-A — Reference input construction

- [x] Normalize song-only, text-only, and hybrid input into ordered catalog IDs.
- [x] Deduplicate, validate, trim, and retrieve/pad to fixed `K`.
- [x] Preserve opaque string IDs and explicit embedding row mappings.
- [x] Enforce deployment `K >= 2`; the pipeline rejects fewer than two distinct references.
- [x] Replace the old 50/50 retrieval evaluation split with first-20, 15->5.
- [ ] Evaluate sensitivity to reference count and order.

## WP-B — Creative cues

- [x] Freeze 16 ranked cue IDs per item and consume the first eight in WP-C.
- [x] Export and validate `cue_vocab.json`, `item2cues.json`, and a manifest.
- [x] Reserve larger cue sets for explicit ablations only.
- [x] Build and hash the final production cue artifacts on all 5,119 items.
- [x] Verify full coverage and zero unknown padding on the frozen test windows.

## WP-C — DDBC backbone and joint continuation prediction

- [x] Freeze the 13-token item layout and legal position masks.
- [x] Expand every eligible playlist into rolling 20-song windows with stride one.
- [x] Require exactly 15 references plus five targets in training data.
- [x] Keep reference tokens fixed; corrupt and score all five target payloads.
- [x] Condition DDBC/DiT on reference centroid `mu_c` and dispersion `sigma_c2`.
- [x] Add full-mask completion that appends five `[BOI, MASK×12]` blocks.
- [x] Keep the legacy next-block/semi-AR sampler out of the production path.
- [x] Freeze evaluation to first 20 songs: 15 references and five targets.
- [x] Draw one joint five-item completion without sequential feedback.
- [x] Score the 5x5 sets with full-catalog retrieval and Hungarian matching.
- [x] Generate/convert the server-side CLHE, RVQ, semantic-token, and cue artifacts.
- [ ] Train the new joint-15to5 checkpoint; the earlier next-one SFT checkpoint is only a baseline.
- [ ] Run checkpoint preflight and one real joint five-item inference on the server.
- [ ] Run the frozen joint 15->5 evaluation with the newly trained checkpoint.
- [x] Add a versioned official evaluator that freezes EMA, seed 1, 256 steps,
  full-catalog retrieval, and atomic result metadata.
- [x] Add an opt-in, loss-scale-normalized RVQ/cue curriculum and effective-weight logs.
- [x] Add order-free generated-cue multiset metrics for the five-item continuation.
- [ ] Compare the uniform-loss baseline with the RVQ/cue curriculum under an otherwise identical schedule.

## WP-D — Verbalization, synthesis, and study UI

The demo behavior is intentionally frozen for now; this protocol update does
not authorize WP-D demo code or UI changes.

- [x] Prompt on the actual ordered reference music, not only centroid neighbors.
- [x] Include the DDBC-predicted target neighbors and eight decoded creative cues.
- [x] Ask the LLM for exactly one original next song and forbid copying references.
- [x] Keep the demo on the singular `generate_next_song()` API.
- [x] Present the study as generated-vs-real next-song comparison.
- [ ] Add structured LLM output validation, retry/timeout, and failure logging.
- [ ] Calibrate the dispersion wording threshold from server training statistics.
- [ ] Install/pin ACE-Step and listen-check real synthesized next songs.
- [ ] Add study consent/privacy text and validate multi-user deployment.

## End-to-end pipeline and evaluation

- [x] Expose `generate_next_song(ContextPrefix) -> SynthesisResult`.
- [x] Keep an explicit research API for alternative samples of the same next slot.
- [x] Validate catalog alignment, sparse IDs, generated token types, and output schema.
- [ ] Version all server artifact paths/hashes and the newly trained checkpoint.
- [ ] Run WP-C latent evaluation as one joint five-item completion against five future songs.
- [ ] Compute corpus-level FAD/CLAP plus human next-song fit, quality, and novelty.
- [ ] Run the full backbone -> verbalization -> ACE-Step path and inspect WAV output.

Server-only steps and expected artifact locations are documented in
[`SERVER_MIGRATION.md`](SERVER_MIGRATION.md).
