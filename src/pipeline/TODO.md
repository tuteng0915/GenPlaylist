# Pipeline — status and remaining TODO

## Implemented

- [x] Load metadata, per-item embeddings, and row mapping as one validated bundle.
- [x] Validate context IDs and every generated/synthesis boundary.
- [x] Pass WP-B cue vocabulary terms through WP-C into WP-D verbalization.
- [x] Support injected backbone, LLM, and synthesizer functions for tests.
- [x] Remove the random demo fallback; missing runtime now fails clearly.
- [x] Provide a bundled checkpoint-backed runner, with an optional external override.
- [x] Expose a singular `generate_next_song()` production API.
- [x] Enforce at least two references and exactly one predicted item.
- [x] Add a reusable WP-A → WP-C → WP-D pipeline object with one-time artifact loading.
- [x] Cache dynamically imported WP modules so heavyweight model singletons persist.
- [x] Add a server CLI with lightweight preflight and full-generation modes.

## Remaining / blocked by WP-C checkpoint

- [ ] Run the bundled checkpoint adapter against the newly trained server checkpoint.
- [ ] Add a versioned startup config with every artifact path and hash.
- [ ] Run one real backbone → verbalization → ACE-Step request and validate the WAV.
- [ ] Add evaluation logs keyed by run/artifact/checkpoint versions.
