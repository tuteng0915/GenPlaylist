# WP-C — Prompt Construction, Verbalization, and Interactive Evaluation

**Owner:** Student 3
**Goal:** Convert creative cues and reference metadata into explicit generation prompts for the frozen ACE-Step generator; build demo and user study to evaluate the full system.

**Read first:** `00_data_schema/schema.py` → `GeneratedItem`, `SynthesisResult`, `CatalogItem`
**Interface stubs:** `verbalization.py`, `synthesis.py`, `app.py`
**Write results to:** `outputs/`

---

## Tasks

### Prompt Construction (verbalization.py)
- [ ] **kNN retrieval** — build a `faiss.IndexFlatIP` over all catalog CLHE embeddings; retrieve k=5 nearest neighbors for `ẑ_{t+1}` (next-item slot) and for `μ_C` (playlist centroid)
- [ ] **Prompt assembly** — combine creative cues (from WP-B `item2cues.json`), neighbor metadata (title/artist/genre/mood/tempo/key/lyric_excerpt from `catalog_metadata.json`), and playlist style summary into LLM input
- [ ] **LLM call** — replace `_call_qwen3()` stub with a real DashScope API call (set `DASHSCOPE_API_KEY` in `.env`); add retry + 30 s timeout; generate music attributes + lyric draft with `[verse]`/`[chorus]`/`[bridge]` markers
- [ ] **Diversity threshold** — load Q33/Q66 of `σ²_C` from `outputs/dispersion_tiers.json` (produced by WP-D); use to give wider thematic latitude for diverse reference sets

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
