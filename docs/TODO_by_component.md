# GenPlaylist — TODO by Work Package

## Frozen task definition

Input is an ordered set of at least two reference tracks. DDBC predicts exactly
one next-item token group. WP-C turns that single latent plan into exactly one
new song. `n_samples` may be used only for research-time stochastic alternatives;
it never means predicting multiple playlist positions.

## WP-A — Reference input construction

- [x] Normalize song-only, text-only, and hybrid input into ordered catalog IDs.
- [x] Deduplicate, validate, trim, and retrieve/pad to fixed `K`.
- [x] Preserve opaque string IDs and explicit embedding row mappings.
- [x] Enforce deployment `K >= 2`; the pipeline rejects fewer than two distinct references.
- [ ] Evaluate sensitivity to reference count and order.

## WP-B — Creative cues

- [x] Freeze the production interface at eight cue IDs per catalog item.
- [x] Export and validate `cue_vocab.json`, `item2cues.json`, and a manifest.
- [x] Reserve larger cue sets for explicit ablations only.
- [ ] Build the final production cue artifacts on the server dataset.
- [ ] Report cue coverage and unknown-cue rate on the next-song test split.

## WP-D — DDBC backbone and next-item prediction

- [x] Freeze the 13-token item layout and legal position masks.
- [x] Train on all preceding items as references and the final item as the only target.
- [x] Require at least two references plus one target in training data.
- [x] Keep reference tokens fixed; corrupt and score only the next-item payload.
- [x] Condition DDBC/DiT on reference centroid `mu_c` and dispersion `sigma_c2`.
- [x] Add full-mask completion that appends `[BOI, MASK×12, EOS]` and emits one item.
- [x] Keep the legacy next-block/semi-AR sampler out of the production path.
- [x] Make `rec_eval` reject labels containing more than one target item.
- [ ] Generate/convert the server-side CLHE, RVQ, semantic-token, and cue artifacts.
- [ ] Train a new checkpoint; old multi-item or old-vocabulary checkpoints are incompatible.
- [ ] Run checkpoint preflight and one real next-item inference on the server.
- [ ] Evaluate semantic similarity and ranking against the held-out immediate next song.

## WP-C — Verbalization, synthesis, and study UI

- [x] Prompt on the actual ordered reference music, not only centroid neighbors.
- [x] Include the DDBC-predicted target neighbors and eight decoded creative cues.
- [x] Ask the LLM for exactly one original next song and forbid copying references.
- [x] Make the demo call the singular `generate_next_song()` API.
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
- [ ] Evaluate one generated and one held-out next song per reference sequence.
- [ ] Compute corpus-level FAD/CLAP plus human next-song fit, quality, and novelty.
- [ ] Run the full backbone -> verbalization -> ACE-Step path and inspect WAV output.

Server-only steps and expected artifact locations are documented in
[`SERVER_MIGRATION.md`](SERVER_MIGRATION.md).
