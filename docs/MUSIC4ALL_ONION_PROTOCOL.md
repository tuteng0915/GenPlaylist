# Music4All-Onion Sequential Protocol

## Role in the paper

Music4All-Onion is the primary sequential-listening dataset for the revised
paper. Spotify MPD remains a curated-playlist proxy, a source of DDBC
initialization, and an external comparison; it is no longer the only evidence
for history-conditioned preference modeling.

The sequential experiment uses only Music4All tracks that can be mapped
conservatively to the frozen 5,119-item GenPlaylist catalog. This preserves the
existing 64-dimensional CLHE table, three-level RVQ tokenizer, 2,048-entry cue
vocabulary, eight-cue item layout, catalog audio, and WP-D adapter. The DDBC
backbone must nevertheless be fine-tuned again on the new chronological
windows.

## Frozen source

- Dataset: Music4All-Onion v2, Zenodo record
  [15394646](https://zenodo.org/records/15394646)
- Timestamp table: `userid_trackid_timestamp.tsv.bz2`
- Events: 252,984,396
- Users: 119,140
- Official timestamp-table MD5: `dfe82201036765f7463e6f3ce3d0f991`
- Timestamp-table SHA-256:
  `05f22c3c316cd62ee6f41ec82cab2c32486d9b575577518a6473c60e7ddfb1c1`
- Identity-table SHA-256 (`id_information.csv`):
  `11b7638e54c6f1bbb69746a087dcc62622cc9f04a7d3de13e98ffd073685be59`
- Spotify-metadata SHA-256 (`id_metadata.csv`):
  `fc32e2ce1b6af0f1dd7b68ea36ed0b2949191fd39a375edf416940de8a63e970`

The raw timestamp file is stored privately at
`data/raw/music4all-onion/`. Do not commit the logs, metadata tables, audio, or
lyrics to Git.

## Catalog alignment

Matching is deterministic and deliberately conservative:

1. Unicode-normalize and ASCII-fold artist and title, case-fold, normalize
   ampersands, remove punctuation, and collapse whitespace.
2. Accept a strict key only when it is one-to-one in both catalogs.
3. Separately identify one-to-one version matches after removing common
   remaster, live, edit, mix, version, and featured-artist suffixes.
4. Never guess among duplicate keys. The relaxed matches must be manually
   audited before they can enter a primary run.

The completed feasibility audit finds:

| Quantity | Value |
|---|---:|
| GenPlaylist catalog | 5,119 tracks |
| Music4All catalog | 109,269 tracks |
| Strict one-to-one overlap | 2,754 tracks |
| Additional version-normalized overlap | 155 tracks |
| Conservative accepted overlap | 2,909 tracks (56.83%) |
| Mapped listening events | 29,645,294 (11.72%) |
| Users with at least 20 mapped events | 76,340 |
| Raw-order contiguous length-20 windows | 527,409 |
| Users with a raw-order contiguous window | 4,462 |

The mapping and audit hashes are:

- `item_mapping.csv`:
  `12de6214f6a0450a7b8a24faef8c46e2bdab68eeb6720f867e420d318d0cdd9c`
- `overlap_audit.json`:
  `5d645a091e8d62b6c4c87a5a58df1425ec8c5795fe6c273cba7641149df5a51b`

The source rows are mostly reverse-chronological but are not globally grouped
by user and contain a small number of within-user order violations. The raw
contiguous-window count is therefore a feasibility estimate, not the final
training count. Dataset preparation must group by user and stable-sort by
timestamp before constructing any window.

## Primary sequence definition

The primary setting uses strict one-to-one matches only. After sorting every
user chronologically, an unsupported Music4All event breaks the run. Rolling
windows are created only inside supported runs of at least 20 adjacent events:
15 visible reference listens followed by five future listens. This retains the
meaning of an immediate sequential continuation and does not silently join
events across missing catalog items.

A secondary scale analysis may remove unsupported events and evaluate the next
supported listen. It must be labeled as a filtered-catalog subsequence and
must not be mixed into the primary result.

Repeated listens are real preference signals and are retained. Consequently,
the sequential evaluator must not exclude tracks merely because they appeared
among the 15 references, and exact-overlap metrics must preserve multiplicity.
The generated waveform remains a new composition; retaining repeats in proxy
supervision does not authorize WP-D to copy a reference recording.

## Split and weighting to freeze after sorted-window audit

Use a seeded user-level split so that evaluation listeners are absent from
training. The model receives no user ID; it must infer preference solely from
the 15 visible events. Within validation and test users, select the latest
eligible window as the primary context so highly active listeners do not
dominate evaluation. Training may use rolling windows, with a fixed per-user
cap chosen before inspecting model results.

The exact train/validation/test fractions, per-user training cap, and resulting
counts will be frozen after the correctly sorted strict-match dataset has been
built. No checkpoint training should begin from the raw-order feasibility
counts above.

## Reproducible commands

Download and verify the official timestamp table:

```bash
conda run -n music python scripts/download_music4all_onion.py \
  --output /home/wjzhang/tt_workspace/data/data/raw/music4all-onion/userid_trackid_timestamp.tsv.bz2 \
  --workers 8
```

Reproduce the overlap audit:

```bash
conda run -n music python scripts/audit_music4all_overlap.py \
  --catalog-metadata /home/wjzhang/tt_workspace/data/data/dataset/catalog_metadata.json \
  --music4all-information /home/wjzhang/tt_workspace/data/data/raw/music4all-onion/identity/id_information.csv \
  --music4all-metadata /home/wjzhang/tt_workspace/data/data/raw/music4all-onion/identity/id_metadata.csv \
  --interactions /home/wjzhang/tt_workspace/data/data/raw/music4all-onion/userid_trackid_timestamp.tsv.bz2 \
  --output /home/wjzhang/tt_workspace/data/data/processed/music4all-onion-overlap-v1/overlap_audit.json \
  --mapping-output /home/wjzhang/tt_workspace/data/data/processed/music4all-onion-overlap-v1/item_mapping.csv
```
