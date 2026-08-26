# Music4All-Onion Sequential Protocol

## Role in the paper

Music4All-Onion is the primary sequential-listening dataset for the revised
paper. Spotify MPD remains a curated-playlist proxy, a source of DDBC
initialization, and an external comparison; it is no longer the only evidence
for history-conditioned preference modeling.

The sequential experiment uses only Music4All tracks that can be mapped
conservatively to the frozen 5,119-item GenPlaylist catalog. The primary run
uses the 2,754 strict one-to-one matches; the 155 version-normalized matches
remain excluded pending manual audit. This preserves the
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

Every user is stable-sorted by timestamp, with the original source row breaking
timestamp ties. The primary setting is a bounded catalog projection: two
adjacent mapped listens may have at most five unsupported events between them;
a larger gap breaks the run. This avoids claiming that filtered events are
literally adjacent while retaining substantially more representative histories
than strict zero-gap matching.

The zero-gap audit was rejected as the primary setting because catalog coverage
made it select almost only looped playback: 976,837 of 979,983 candidate
windows contained a repeated song, and only 6.81% had five distinct targets.
At a five-event gap, 41.70% of candidate windows have five distinct targets.
To ensure that the task actually receives multiple reference songs, a window
must contain at least eight distinct tracks among its 15 references and at
least three distinct tracks among its five targets.

Repeated listens that survive this diversity constraint are real preference
signals and are retained. Sequential evaluators may therefore emit tracks that
appeared in the references or earlier predictions, and exact-overlap metrics
use multiset intersection. The generated waveform remains a new composition;
retaining repeats in proxy supervision does not authorize WP-D to copy a
reference recording.

## Frozen split and weighting

- Split unit: user, assigned by seeded SHA-256 with seed 42.
- Source split: 80% train / 20% test; validation is empty.
- Training: seeded window sampling capped at 16 windows per user.
- Testing: the latest eligible window for every held-out eligible user.
- User IDs: replaced in split record IDs by deterministic 16-hex pseudonyms;
  no user ID is supplied to the model.

The frozen output contains 231,422 training windows from 19,552 users and
4,725 test windows from 4,725 disjoint users. Before capping, the qualifying
pool contains 2,157,666 train-user windows and 579,072 test-user windows.
The 20,000-step, global-batch-512 schedule therefore draws 10.24 million
training examples, or about 44.2 passes over the capped training set.

## Frozen artifacts

| Artifact | SHA-256 |
|---|---|
| Sequence manifest | `773025b803aa3ec02813b09802cca5dc1553f3505eb7ee31072ae4a4e98a6927` |
| Train split | `81f5ee449c9a56831a957f432d10d0ba93acc6286bc79fd0b982b04c0e5eae6c` |
| Test split | `cca5ce769ef87624a8dcf025cdc60efd9233b7043822d320ae976c53a1550652` |
| Gap audit | `46bb40b479bb6ec53a381146f401a846df469b3182298d26118db965dfc98d59` |
| Dataset materialization manifest | `c8849643be4388f513b66a4c956227669f7b05a48f022e28d383f665d929b885` |
| WP-C prepared manifest (8 cues, primary) | `cbf94e313ea21b98915ebc455edea2e4ef2e74b65af010cd6426aa044468fa11` |
| WP-C prepared manifest (0 cues, ablation) | `cb5556dcce1bd83555e63edeb651c6fbc1937ddf9f78db64ecf1cb033f412644` |

## Reproducible commands

Download and verify the official timestamp table:

```bash
conda run -n music python scripts/download_music4all_onion.py \
  --output /home/wjzhang/tt_workspace/model/GenPlaylist/data/raw/music4all-onion/userid_trackid_timestamp.tsv.bz2 \
  --workers 8
```

Reproduce the overlap audit:

```bash
conda run -n music python scripts/audit_music4all_overlap.py \
  --catalog-metadata /home/wjzhang/tt_workspace/model/GenPlaylist/data/dataset/catalog_metadata.json \
  --music4all-information /home/wjzhang/tt_workspace/model/GenPlaylist/data/raw/music4all-onion/identity/id_information.csv \
  --music4all-metadata /home/wjzhang/tt_workspace/model/GenPlaylist/data/raw/music4all-onion/identity/id_metadata.csv \
  --interactions /home/wjzhang/tt_workspace/model/GenPlaylist/data/raw/music4all-onion/userid_trackid_timestamp.tsv.bz2 \
  --output /home/wjzhang/tt_workspace/model/GenPlaylist/data/processed/music4all-onion-overlap-v1/overlap_audit.json \
  --mapping-output /home/wjzhang/tt_workspace/model/GenPlaylist/data/processed/music4all-onion-overlap-v1/item_mapping.csv
```

Build, materialize, and validate the frozen sequential dataset on the server:

```bash
conda run -n music python scripts/prepare_music4all_sequences.py \
  --interactions /home/wjzhang/tt_workspace/model/GenPlaylist/data/raw/music4all-onion/userid_trackid_timestamp.tsv.bz2 \
  --mapping /home/wjzhang/tt_workspace/model/GenPlaylist/data/processed/music4all-onion-overlap-v1/item_mapping.csv \
  --output-dir /home/wjzhang/tt_workspace/model/GenPlaylist/data/processed/music4all-onion-sequential-v1-k5-u8-u3-cap16 \
  --work-dir /home/wjzhang/tt_workspace/model/GenPlaylist/data/processed/music4all-onion-sort-work-v1 \
  --seed 42 --test-fraction 0.2 --train-user-cap 16 \
  --max-skipped-events 5 --min-unique-references 8 --min-unique-targets 3

conda run -n music python scripts/materialize_music4all_dataset.py \
  --sequence-dir /home/wjzhang/tt_workspace/model/GenPlaylist/data/processed/music4all-onion-sequential-v1-k5-u8-u3-cap16 \
  --catalog-dir /home/wjzhang/tt_workspace/model/GenPlaylist/data/dataset \
  --output-dir /home/wjzhang/tt_workspace/model/GenPlaylist/data/dataset-music4all-onion-v1

conda run -n music python scripts/prepare_wp_c_data.py \
  --data-dir /home/wjzhang/tt_workspace/model/GenPlaylist/data/dataset-music4all-onion-v1 \
  --artifact-dir /home/wjzhang/tt_workspace/model/GenPlaylist/data/dataset \
  --output-dir /home/wjzhang/tt_workspace/model/GenPlaylist/data/processed/genplaylist-music4all-onion-v1-8cue-k5-u8-u3-cap16 \
  --active-cues 8

conda run -n music python scripts/validate_wp_c_prepared_data.py \
  --data-dir /home/wjzhang/tt_workspace/model/GenPlaylist/data/dataset-music4all-onion-v1 \
  --artifact-dir /home/wjzhang/tt_workspace/model/GenPlaylist/data/dataset \
  --prepared-dir /home/wjzhang/tt_workspace/model/GenPlaylist/data/processed/genplaylist-music4all-onion-v1-8cue-k5-u8-u3-cap16 \
  --active-cues 8
```

The 0-cue component ablation uses the same split, windows, catalog tokens, and
targets. Re-run the two WP-C preparation commands above with `--active-cues 0`
and output/prepared directory
`genplaylist-music4all-onion-v1-0cue-k5-u8-u3-cap16`.

The training/evaluation runners keep their historical filenames but select the
Music4All Hydra data configuration explicitly:

```bash
export GENPLAYLIST_DATA_CONFIG=music4all
export GENPLAYLIST_DATA_ROOT=/home/wjzhang/tt_workspace/model/GenPlaylist/data/dataset-music4all-onion-v1
export GENPLAYLIST_ARTIFACT_ROOT=/home/wjzhang/tt_workspace/model/GenPlaylist/data/dataset
export GENPLAYLIST_PREPARED_DATA_ROOT=/home/wjzhang/tt_workspace/model/GenPlaylist/data/processed/genplaylist-music4all-onion-v1-8cue-k5-u8-u3-cap16
bash src/03_backbone_recommender/scripts/train_spotify.sh
```
