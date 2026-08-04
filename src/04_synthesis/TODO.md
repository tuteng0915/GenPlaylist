# WP-C — status and remaining TODO

## Implemented

- [x] Validate 64-D kNN inputs and catalog alignment.
- [x] Include all eight decoded creative cues in attribute and lyric prompts.
- [x] Support an injected LLM for deterministic tests.
- [x] Load ACE-Step lazily and configure path/device through environment variables.
- [x] Remove random embeddings and candidates from the demo.
- [x] Use repository-relative paths and per-session Gradio study state.
- [x] Use UUID sessions and an exclusive file lock for CSV writes on Linux/macOS.
- [x] Prompt with the actual ordered references and one DDBC-predicted next-song target.
- [x] Make the demo produce exactly one next song through the singular pipeline API.

## Remaining / external runtime

- [ ] Install and pin ACE-Step; listen-check a real 30-second synthesis.
- [ ] Add LLM retry, timeout, structured-output validation, and failure logging.
- [ ] Replace NumPy kNN with a cached normalized matrix or FAISS.
- [ ] Load WP-D dispersion thresholds instead of the temporary `1.0` threshold.
- [ ] Add consent/privacy text and use a transactional store for multi-host deployment.
- [ ] Run multi-user and real-audio validation before data collection.
