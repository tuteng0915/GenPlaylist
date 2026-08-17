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

- [x] Freeze the 8-cue main layout and support separately trained 0/4/16-cue layouts.
- [x] Expand every eligible playlist into rolling 20-song windows with stride one.
- [x] Require exactly 15 references plus five targets in training data.
- [x] Keep reference tokens fixed; corrupt and score all five target payloads.
- [x] Condition DiT on the visible reference-token history.
- [x] Add full-mask completion that appends five `[BOI, MASK×12]` blocks.
- [x] Keep the legacy next-block/semi-AR sampler out of the production path.
- [x] Freeze evaluation to first 20 songs: 15 references and five targets.
- [x] Draw one joint five-item completion without sequential feedback.
- [x] Score the 5x5 sets with full-catalog retrieval and Hungarian matching.
- [x] Generate/convert the server-side CLHE, RVQ, semantic-token, and cue artifacts.
- [x] Train the new joint-15to5 checkpoint; the earlier next-one SFT checkpoint is only a baseline.
- [x] Run checkpoint preflight and one real joint five-item inference on the server.
- [x] Run the frozen joint 15->5 evaluation with the newly trained checkpoint.
- [x] Add a versioned official evaluator that freezes EMA, seed 1, 256 steps,
  full-catalog retrieval, and atomic result metadata.
- [x] Use the loss-scale-normalized RVQ/cue curriculum in Full and retain
      uniform weighting as an ablation; log effective weights.
- [x] Add order-free generated-cue multiset metrics for the five-item continuation.
- [x] Compare the uniform-loss baseline with the RVQ/cue curriculum under an otherwise identical schedule.

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
- [x] Provide a separate consent-gated, pseudonymous, transactional listener-study
      service and validate it in the server environment; keep the WP-D demo unchanged.

## End-to-end pipeline and evaluation

- [x] Expose `generate_next_song(ContextPrefix) -> SynthesisResult`.
- [x] Keep an explicit research API for alternative samples of the same next slot.
- [x] Validate catalog alignment, sparse IDs, generated token types, and output schema.
- [x] Version server artifact paths/hashes and the newly trained checkpoint.
- [x] Run WP-C latent evaluation as one joint five-item completion against five future songs.
- [x] Compute corpus-level MERT History Fit, VGGish FAD, and CLAP-A.
- [ ] Run the frozen WP-D listener study for history fit, quality, novelty, and preference.
- [x] Freeze current-protocol listener-study case preparation and participant-level analysis.
- [x] Run the full backbone -> verbalization -> ACE-Step path and validate generated audio.

## Music4All-Onion sequential revision

- [x] Select and freeze Music4All-Onion as the primary sequential-listening dataset.
- [x] Download and checksum the 252,984,396-event timestamp table.
- [x] Audit conservative overlap with the frozen 5,119-track MPD catalog.
- [ ] Manually audit the 155 version-normalized matches; keep strict matches as primary meanwhile.
- [ ] Stable-sort events within user and build strict contiguous 15->5 windows.
- [ ] Freeze a user-disjoint split and per-user training-window cap.
- [ ] Update proxy evaluators to retain repeated listens and restrict retrieval to the mapped catalog.
- [ ] Retrain DDBC-SFT and GenPlaylist on the sequential windows and rerun baselines.
- [ ] Regenerate the Music4All end-to-end audio suite and automatic metrics.
- [ ] Update the paper so Music4All is primary and MPD is the playlist-proxy comparison.

Server-only steps and expected artifact locations are documented in
[`SERVER_MIGRATION.md`](SERVER_MIGRATION.md).
