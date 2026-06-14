# WP-C — Synthesis & Demo

**Owner:** Student 3
**Goal:** Verbalize generated embeddings → music attributes + lyrics → audio; build demo + user study.

**Read first:** `00_data_schema/schema.py` → `GeneratedItem`, `SynthesisResult`, `CatalogItem`
**Interface stubs:** `verbalization.py`, `synthesis.py`, `app.py`
**Write results to:** `outputs/`

---

## Tasks

### verbalization.py
- [ ] **LLM** — replace `_call_qwen3()` stub with a real DashScope API call (set `DASHSCOPE_API_KEY` in `.env`); add retry + 30 s timeout
- [ ] **kNN index** — build a `faiss.IndexFlatIP` over all catalog CLHE embeddings at module load; use it in `knn_verbalize()` instead of the numpy loop
- [ ] **Diversity threshold** — compute Q33/Q66 of `σ²_C` on training playlists; save to `outputs/dispersion_tiers.json`; replace the hardcoded threshold

### synthesis.py
- [ ] **ACE-Step** — install and wire `ACEStepPipeline`; test a 30 s generation; record peak VRAM and generation time
- [ ] **Style reference** — pass the nearest-neighbor audio path as `style_ref_audio_path` when available; switch to `task="edit"`

### app.py
- [ ] **UI skeleton** — Gradio `Blocks` layout: input panel + 3 candidate tabs (audio player, attributes, lyrics, neighbor list)
- [ ] **Cold-start** — load `outputs/example_results.json` so the demo is usable before the live pipeline is ready
- [ ] **Live pipeline** — wire "Generate" button end-to-end
- [ ] **User study mode** — toggle that hides method info; Likert sliders (Coherence / Music Quality / Overall); write session logs to `outputs/evaluation_logs/`

---

## Metrics

| Metric | Target |
|--------|--------|
| `SynthesisResult.validate()` pass rate | 100% |
| `music_attributes` has ≥ 4 comma-separated fields | 100% of calls |
| `lyric_draft` contains `[verse]`, `[chorus]`, or `[bridge]` | 100% of calls |
| LLM latency (single call) | < 10 s |
| faiss kNN query time (k=10) | < 50 ms |
| ACE-Step generation time (30 s audio) | record observed |
| User study sessions collected | ≥ 10 |
| Mean Coherence Likert score | report value |

---

## Result files

| File | Status |
|------|--------|
| `outputs/example_results.json` | placeholder (empty list) |
| `outputs/dispersion_tiers.json` | placeholder (empty dict) |
| `outputs/audio/` | placeholder directory |
| `outputs/evaluation_logs/` | placeholder directory |
| `outputs/report.md` | placeholder (template) |
