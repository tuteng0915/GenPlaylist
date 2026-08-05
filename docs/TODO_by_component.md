# GenPlaylist — TODO by Work Package

## Frozen task definition

Input is an ordered set of at least two reference tracks. DDBC predicts exactly
one next-item token group per sampling run. Training windows contain at most
15 references plus one target. Evaluation fixes the first 20 test songs as
15 references plus five ground-truth songs and independently samples the same
next-one slot five times. WP-D turns one selected latent plan into one new song;
the WP-D demo is not changed by the offline evaluation protocol.

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

## WP-C — DDBC backbone and next-item prediction

- [x] Freeze the 13-token item layout and legal position masks.
- [x] Expand every chronological prefix into at most 15 references plus one target.
- [x] Require at least two references plus one target in training data.
- [x] Keep reference tokens fixed; corrupt and score only the next-item payload.
- [x] Condition DDBC/DiT on reference centroid `mu_c` and dispersion `sigma_c2`.
- [x] Add full-mask completion that appends `[BOI, MASK×12, EOS]` and emits one item.
- [x] Keep the legacy next-block/semi-AR sampler out of the production path.
- [x] Freeze evaluation to first 20 songs: 15 references and five targets.
- [x] Draw five independent next-one samples without feeding predictions back.
- [x] Score the 5x5 sets with full-catalog retrieval and Hungarian matching.
- [ ] Generate/convert the server-side CLHE, RVQ, semantic-token, and cue artifacts.
- [ ] Train a new checkpoint; old multi-item or old-vocabulary checkpoints are incompatible.
- [ ] Run checkpoint preflight and one real next-item inference on the server.
- [ ] Run the frozen 15->5 evaluation with the newly trained checkpoint.

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
- [ ] Run WP-C latent evaluation as five samples against five future songs.
- [ ] Compute corpus-level FAD/CLAP plus human next-song fit, quality, and novelty.
- [ ] Run the full backbone -> verbalization -> ACE-Step path and inspect WAV output.

Server-only steps and expected artifact locations are documented in
[`SERVER_MIGRATION.md`](SERVER_MIGRATION.md).
