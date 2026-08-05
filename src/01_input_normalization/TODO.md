# WP-A — status and remaining TODO

## Implemented

- [x] Song-only filtering, deduplication, diversity selection, and centroid padding.
- [x] Text-only cosine retrieval through an injected encoder.
- [x] Hybrid retrieval, including seed-only hybrid padding.
- [x] Validate dimensions, duplicate IDs, blank/zero queries, K, and exact output length.
- [x] Return a validated `ContextPrefix` with metadata in matching order.
- [x] Freeze retrieval evaluation to first 20 songs: 15 references and 5 targets.

## Remaining experiments

- [ ] Build the final retrieval matrix aligned to `item_id_to_row.json`.
- [ ] Choose and version the production query encoder.
- [ ] Rebuild the 468 eligible frozen test examples; remove placeholder metrics.
- [ ] Report Recall@K/cosine statistics on a versioned evaluation set.
