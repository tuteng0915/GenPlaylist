# Server artifact migration and training handoff

This document is the server-side portion of the GenPlaylist-v1 migration. The
repository code does not require model files to exist during local development,
but training and live generation must pass the checks below.

## What `catalog_item_embeddings.npy` means

It is the per-song CLHE table for the current frozen catalog:

- shape: `(5119, 64)`
- row `item_id_to_row[item_id]`: the CLHE vector for that song
- dtype: `float32`

There are 5,119 catalog entries but only 5,077 unique items in the playlist
splits. Keeping all 5,119 rows is intentional: WP-A retrieval and the demo use
the full catalog, while WP-D training uses the split subset.

Do not use `clhe_weight.npy` as this file. The expected legacy meanings are:

| Legacy file | Meaning | Typical shape |
|---|---|---:|
| `clhe.pt` | per-item CLHE vectors | `(full_catalog_size, 64)` |
| `clhe_weight.npy` | three merged RVQ codebooks | `(768, 64)` |
| `clhe_token.json` | item → 3 RVQ tokens + conflict token | JSON mapping |

The old full embedding table may use `int(item_id)` as its row. That convention
is accepted only during the one-time migration; runtime code always uses an
explicit mapping.

## 1. Locate and inspect the old server files

The old code referenced `/home/sjj/wenhao/DISCO/datasets/spotify`, but verify
the actual mount instead of assuming it:

```bash
find /home /workspace /data -type f \
  \( -name 'clhe.pt' -o -name 'clhe_weight.npy' -o -name 'clhe_token.json' \) \
  2>/dev/null
```

Inspect shapes without changing anything:

```bash
python - <<'PY'
import json, numpy as np, torch
from pathlib import Path

d = Path('/replace/with/legacy/spotify')
x = torch.load(d / 'clhe.pt', map_location='cpu')
if hasattr(x, 'shape'):
    print('clhe.pt:', tuple(x.shape), x.dtype)
print('clhe_weight.npy:', np.load(d / 'clhe_weight.npy', mmap_mode='r').shape)
t = json.loads((d / 'clhe_token.json').read_text())
print('clhe_token.json:', len(t), 'items')
print('first:', next(iter(t.items())))
PY
```

## 2. Convert to canonical artifacts

Best case: the server already has a source item-to-row mapping:

```bash
python scripts/prepare_server_artifacts.py \
  --legacy-dir /replace/with/legacy/spotify \
  --source-item-id-to-row /replace/with/legacy/item_id_to_row.json
```

If inspection confirms that the old full `clhe.pt` really uses numeric item ID
as its row, make that otherwise-dangerous assumption explicit:

```bash
python scripts/prepare_server_artifacts.py \
  --legacy-dir /replace/with/legacy/spotify \
  --legacy-dense-numeric-ids
```

This produces, atomically, under `data/dataset/`:

- `catalog_item_embeddings.npy` — `(5119, 64)`
- `rvq_codebook_weights.npy` — `(768, 64)`
- `semantic_tokens.json` — exactly four semantic tokens per catalog item
- `item_id_to_row.json` — contiguous rows `0..5118`
- `wpd_artifact_manifest.json` — source information and SHA-256 hashes

The script fails instead of truncating conflict codes. If it reports a conflict
token above 842, inspect the source distribution and update the shared schema;
do not silently clamp it.

## 3. Produce eight-cue WP-B artifacts

Run the production preset, not the 18-cue research preset:

```bash
python src/02_creative_cues/run_production.py --config default
```

The required files are written to
`src/02_creative_cues/outputs/production/latest/`. Its `cue_manifest.json` must
say `wp_d_compatible: true`, `cues_per_item: 8`, and `cue_vocab_size: 2048`.

## 4. Run the fail-fast preflight

```bash
python scripts/validate_server_artifacts.py
```

This validates every catalog ID, matrix shape, semantic token, cue ID, split,
token offset, and one encoded training sequence. Do not start training until it
prints:

```text
[validate] all GenPlaylist-v1 artifacts are mutually compatible
```

## 5. Train a new checkpoint

The old checkpoint is incompatible because vocabulary size, item stride,
conditioning layers, and loss masking changed.

```bash
bash src/03_backbone_recommender/scripts/train_spotify.sh
```

The default Spotify configuration now selects `GenPlaylistTokenizer`, reads the
canonical data directory, uses a 13-token stride, and conditions DiT on
`mu_c`/`sigma_c2`. For each playlist, all songs except the last are ordered
references and the last song is the only next-item target; records with fewer
than two references plus one target are filtered out.

Before a long run, use Hydra overrides for a one-batch smoke test and confirm:

- batch sequence shape is `B × (2 + 13n)`;
- exactly twelve target payload positions are masked for one next item;
- loss is finite;
- generated clean tokens obey their position-specific ranges.

## 6. Live pipeline adapter

The bundled coordinator runtime needs only the new checkpoint path:

```bash
export GENPLAYLIST_BACKBONE_CKPT=/replace/with/genplaylist-v1.ckpt
export GENPLAYLIST_DEVICE=cuda:0
```

`backbone_runtime.py` loads the tokenizer and model lazily, computes playlist
structure, appends one explicit `[BOI, MASK x 12, EOS]` slot, reverse-denoises
only those twelve payload positions, and returns validated `GeneratedItem`
objects. An advanced deployment can override it with:

```bash
export GENPLAYLIST_BACKBONE_RUNNER='your_runtime_module:run_backbone'
```

The override function contract is:

```python
def run_backbone(
    context_prefix,
    n_samples,
    catalog_embs,
    catalog_metadata,
    item_id_to_row,
) -> list[GeneratedItem]:
    ...
```

It must load the new checkpoint, encode the context with
`GenPlaylistTokenizer`, call structure-conditioned full-mask completion, decode
the single generated 13-token item, and return exactly `n_samples` validated
`GeneratedItem` instances. Do not use the legacy multi-stride semi-AR wrapper
for production next-song inference. The central pipeline performs a second
validation before verbalization and synthesis.

## 7. ACE-Step runtime

Set the source path/device only on the synthesis server:

```bash
export ACE_STEP_PATH=/replace/with/ACE-Step
export ACE_STEP_DEVICE=0
export ACE_STEP_DTYPE=bfloat16
```

ACE-Step is loaded lazily on the first request. The demo no longer emits random
embeddings or placeholder candidates when a runtime component is missing.

## 8. End-to-end pipeline smoke test

Validate lightweight startup artifacts first; this does not load DDBC or
ACE-Step:

```bash
python src/pipeline/run.py --preflight-only
```

Then run one complete request with two or more real catalog IDs:

```bash
python src/pipeline/run.py \
  --references REFERENCE_ID_1 REFERENCE_ID_2 \
  --instruction 'one next song with a smooth nocturnal transition'
```

The command runs WP-A reference normalization, DDBC full-mask next-item
completion, WP-C verbalization, and ACE-Step synthesis, then prints the output
audio path and generation metadata as JSON.
