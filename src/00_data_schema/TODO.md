# 00_data_schema — status and remaining TODO

## Implemented

- [x] Freeze `genplaylist-v1`: opaque item IDs, 64-D CLHE, 3×256 RVQ, 74 conflict values, 8×2048 cues.
- [x] Freeze the 13-token item stride and all token offsets in `TokenLayout`.
- [x] Support dict-form and list-form catalog metadata.
- [x] Require `item_id_to_row.json`; never infer a row with `int(item_id)`.
- [x] Validate catalog/embedding/mapping alignment and reject codebooks used as item embeddings.
- [x] Add `build_catalog_artifacts.py` and contract tests.

## Remaining / blocked by artifacts

- [ ] Produce `catalog_item_embeddings.npy` with shape `(5119, 64)` in catalog row order.
- [ ] Rerun the builder with `--embeddings` so its hash enters `artifact_manifest.json`.
- [ ] Produce/version `rvq_codebook_weights.npy` `(768, 64)` and semantic-token mapping.
- [ ] Record CLHE checkpoint, preprocessing commit, and build date before release.

Server procedure: `docs/SERVER_MIGRATION.md` and
`scripts/prepare_server_artifacts.py`.
