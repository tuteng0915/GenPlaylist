# WP-B — Creative Cue Mining

**Owner:** Student 2
**Goal:** Build a 2048-entry cue vocabulary; assign 6 cue tokens per catalog song.

**Read first:** `00_data_schema/schema.py` → `CueMappingEntry`, `CUE_VOCAB_SIZE`, `CUE_TOKENS`
**Interface stub:** `cue_mining.py` (unimplemented paths raise `NotImplementedError`)
**Write results to:** `outputs/`

---

## Tasks

- [ ] **Audit** — for each item, collect all available text (title, artist, lyrics, genre, mood); measure missing rates per field
- [ ] **Vocab extraction** — extract candidate cue phrases using TF-IDF as baseline; optionally improve with YAKE or KeyBERT; keep top-2047 after frequency/IDF filtering; prepend `"<unk>"` at index 0 → exactly 2048 entries; save to `outputs/cue_vocab.json`
- [ ] **Cue assignment** — for each song, score all vocab entries by PMI; pick the 6 most relevant and mutually diverse cues; pad with `0` if fewer than 6 qualify; save to `outputs/item2cues.json`
- [ ] **Validation** — every entry must pass `CueMappingEntry.validate()`; no item_id from `clhe_token.json` may be missing
- [ ] **Stats** — run `compute_coverage_stats()`; save to `outputs/coverage_stats.json`
- [ ] **Report** — fill in `outputs/cue_report.md`

---

## Metrics

| Metric | Target |
|--------|--------|
| `len(cue_vocab)` | exactly 2048 |
| `cue_vocab[0]` | `"<unk>"` |
| `validate()` pass rate on full catalog | 100% |
| No item_id missing from `item2cues.json` | 0 missing |
| Items with ≥ 1 non-`<unk>` cue | ≥ 40% |
| UNK rate (fraction of all cue slots) | ≤ 60% |
| Vocab entries used by ≥ 1 item | ≥ 50% of 2048 |
| Avg within-item pairwise cosine of 6 cues | < 0.7 |

---

## Result files

| File | Status |
|------|--------|
| `outputs/cue_vocab.json` | placeholder (empty list) |
| `outputs/item2cues.json` | placeholder (empty dict) |
| `outputs/coverage_stats.json` | placeholder (empty dict) |
| `outputs/cue_report.md` | placeholder (template) |
