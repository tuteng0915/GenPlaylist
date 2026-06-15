# WP-A — Reference Input Construction

**Owner:** Student 1
**Goal:** Convert raw user input into a clean reference context `ContextPrefix(item_ids=[m1,...,mK])` that expresses the user's musical preference for personalized generation.

**Read first:** `00_data_schema/schema.py` → `ContextPrefix`, `CatalogItem`
**Interface stub:** `normalizer.py` (unimplemented paths raise `NotImplementedError`)
**Write results to:** `outputs/`

---

## Tasks

- [ ] **Song-only input** — accept a list of item IDs; deduplicate; if too many, select the K most diverse by embedding; if too few, pad by retrieving nearest neighbors from the catalog
- [ ] **Text-only input** — embed a natural-language query; retrieve K nearest catalog items by cosine similarity (use sentence-transformers or CLAP as encoder)
- [ ] **Hybrid input** — combine song IDs + text query; merge candidates; select K
- [ ] **Validation** — every output must pass `ContextPrefix.validate()`
- [ ] **Export** — write 20+ example outputs covering all input types to `outputs/context_prefix_examples.json`
- [ ] **Report** — fill in `outputs/report.md`

---

## Metrics

| Metric | Target |
|--------|--------|
| `validate()` pass rate | 100% |
| Zero duplicate item_ids in any output | 100% |
| All output IDs exist in `clhe_token.json` | 100% |
| text_only: cosine sim of query emb to top-1 result | > 0.5 |
| padded: cosine sim of padded items to input centroid | > 0.4 |

---

## Result files

| File | Status |
|------|--------|
| `outputs/context_prefix_examples.json` | placeholder (empty list) |
| `outputs/retrieval_stats.json` | placeholder (empty dict) |
| `outputs/report.md` | placeholder (template) |
