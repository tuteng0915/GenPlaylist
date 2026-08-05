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
the full catalog, while WP-C training uses the split subset.

Do not use `clhe_weight.npy` as this file. The expected legacy meanings are:

| Legacy file | Meaning | Typical shape |
|---|---|---:|
| `clhe.pt` | per-item CLHE vectors | `(full_catalog_size, 64)` |
| `clhe_weight.npy` | three merged RVQ codebooks | `(768, 64)` |
| `clhe_token.json` | item → 3 RVQ tokens + conflict token | JSON mapping |

The old full embedding table may use `int(item_id)` as its row. That convention
is accepted only during the one-time migration; runtime code always uses an
explicit mapping.

## 1. Download and inspect the official DDBC checkpoint

The official `spotify30.ckpt` contains more than DiT weights: its serialized
tokenizer also carries the full `(254155, 64)` CLHE item table, the `(768, 64)`
three-codebook RVQ weights, and semantic tokens for all 254,155 source items.
Download it into the canonical repository:

```bash
conda run -n music python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='liaialley/DDBC', filename='spotify30.ckpt', local_dir='checkpoints/pretrained/ddbc')"
```

The currently verified checkpoint has SHA-256
`3a44f44dcdd850dbe8c441389c754fae9ae5e3673ab8fe9ce1621e7175ac229f`.
Treat checkpoints as trusted pickle inputs; do not run the extraction command
on an untrusted file.

## 2. Extract canonical CLHE/RVQ artifacts

On the NCL server the source catalog is outside the repository. Extract the
5,119 catalog rows directly from the checkpoint:

```bash
conda run -n music python scripts/extract_ddbc_checkpoint_artifacts.py \
  --checkpoint checkpoints/pretrained/ddbc/spotify30.ckpt \
  --catalog /home/wjzhang/tt_workspace/data/data/dataset/catalog_metadata.json \
  --output-dir data/dataset \
  --confirm-dense-item-ids
```

The explicit confirmation is required even though the official tokenizer was
verified to contain exhaustive dense IDs `"0".."254154"`. For the current
catalog all 5,119 IDs are covered, selected CLHE rows are finite `(5119, 64)`,
the RVQ weights are finite `(768, 64)`, and observed conflict tokens are only
769–776.

Copy the authoritative catalog alongside those generated artifacts; keep audio
and lyrics in their external data directory:

```bash
cp /home/wjzhang/tt_workspace/data/data/dataset/catalog_metadata.json data/dataset/
```

The CLHE source is the implementation and dataset contract published by
`Xiaohao-Liu/CLHE`; no separate CLHE retraining is necessary for this subset
because the official DDBC checkpoint already embeds its trained CLHE table.

## Alternative: convert standalone legacy files

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

If standalone legacy files are found, convert them as follows.

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
retain the legacy field `wp_d_compatible: true`, declare
`stored_cues_per_item: 16`, `default_active_cues: 8`, and
`cue_vocab_size: 2048`.

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

## 5. Warm-start and fine-tune a new checkpoint

The old checkpoint cannot be resumed directly because vocabulary size, item
stride, conditioning layers, and loss masking changed. It can, however, be
used as a semantic warm start: the six DiT blocks and other shape-compatible
weights are copied; the original 1,028-token embedding/output rows are remapped
to shared token meanings; the 2,048 cue rows and the new playlist-structure
conditioning layers retain fresh initialization. Optimizer, EMA, epoch, and
step state start from zero.

```bash
bash src/03_backbone_recommender/scripts/train_spotify.sh
```

The script defaults to `GENPLAYLIST_TRAIN_MODE=warmstart` and the downloaded
`checkpoints/pretrained/ddbc/spotify30.ckpt`. Subsequent continuation is
explicit:

```bash
GENPLAYLIST_TRAIN_MODE=resume \
  bash src/03_backbone_recommender/scripts/train_spotify.sh
```

Use `GENPLAYLIST_TRAIN_MODE=scratch` only for the from-scratch ablation. The
warm start enables time conditioning to match the pretrained DDBC checkpoint.

Before launching the trainer, instantiate the current model and verify the real
row remapping on CPU:

```bash
conda run -n music python scripts/check_ddbc_warmstart.py --backward-smoke
```

The default Spotify configuration now selects `GenPlaylistTokenizer`, reads the
canonical data directory, uses a 13-token stride, and conditions DiT on
`mu_c`/`sigma_c2`. Training expands every chronological prefix into at most 15
recent references plus one next-item target (16 songs total). Evaluation keeps
only test playlists with at least 20 songs, takes the first 20, uses songs 1–15
as references and songs 16–20 as targets, and draws the same next-one slot five
times for 5x5 matching. `src/shared/protocol.py` rejects configuration drift.

Before a long run, use Hydra overrides for a one-batch smoke test and confirm:

- batch sequence shape is `B × (2 + 13n)`;
- exactly twelve target payload positions are masked for one next item;
- loss is finite;
- generated clean tokens obey their position-specific ranges.

Prepare the frozen Arrow datasets and checkpoint-independent evaluation vectors
once before launching training:

```bash
conda run -n music python scripts/prepare_wp_c_data.py \
  --data-dir /home/wjzhang/tt_workspace/data/data/dataset \
  --artifact-dir data/dataset \
  --output-dir /home/wjzhang/tt_workspace/data/data/processed/genplaylist-v1-16item-15to5
```

`train_spotify.sh` resolves that directory from `GENPLAYLIST_DATA_ROOT` by
default. Override it with `GENPLAYLIST_PREPARED_DATA_ROOT` only when using a
separately versioned cache.

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

The command runs WP-A reference normalization, WP-C DDBC full-mask next-item
completion, WP-D verbalization, and ACE-Step synthesis, then prints the output
audio path and generation metadata as JSON.
