# Cue Vocabulary and Assignment Freeze

## Frozen production contract

- Vocabulary size: 2,048 IDs (`0=<unk>`, `1..2047` real cues)
- Frozen vocabulary SHA256:
  `cd2294920171aec7338f18dcfaecab03495f6a2e4d74b282116863612607f6e9`
- Stored candidates per catalog item: 16
- Default candidates consumed by GenPlaylist-v1: first 8
- Assignment score: cosine similarity in `all-MiniLM-L6-v2` space
- Ordering: relevance descending, then cue ID ascending for exact ties
- Recall: at least top-64; expand to the requested width; trailing `<unk>` only
- Catalog items: 5,119
- Lyrics available during assignment: 5,119 / 5,119

The immutable server run is:

```text
/home/wjzhang/tt_workspace/model/GenPlaylist/
  src/02_creative_cues/outputs/production/20260804_172128/
```

`outputs/production/latest/` mirrors that run.

## Frozen artifact hashes

```text
cue_vocab.json
  cd2294920171aec7338f18dcfaecab03495f6a2e4d74b282116863612607f6e9
item2cues.json
  281469f8f47f2d2fbdceb46930013d23ea102e277b6d56f270984a28b5ac31cc
item2cue_scores.json
  88185069a640634385dba0ec7dc6e63ed1338c521ab7259bd8ff74b382a92b29
```

The versioned directory also contains `item_cues.tsv`, `cue_manifest.json`,
`run_config.json`, `health_report.md`, and `artifact_audit.json`.

## Validation results

- Every one of the 5,119 items stores exactly 16 cue IDs.
- No invalid cue IDs, duplicate non-UNK IDs within a row, or `<unk>` slots.
- No relevance-order violations and no score-shape mismatches.
- 1,683 / 2,047 non-UNK vocabulary entries are used (82.22%).
- WP-C reads only `cue_ids[:8]`, so the token stride remains 13 per item.
- Official `spotify30.ckpt` warm-start and a target-only CPU backward pass both pass.

## Known vocabulary-quality limitations

The vocabulary is frozen for reproducibility, not claimed to be perfectly
curated. Its most frequent assignments include broad genre labels, lyric-like
phrases, and some explicit language. For example, `pop rap` appears in 50.75%
of 16-candidate rows and `never let me go` in 35.28%. These are retained so cue
IDs do not change silently. Any future content-policy or hubness cleanup must
produce a new versioned vocabulary and new hashes rather than editing this v1
vocabulary in place.

## Rejected run

`rejected_20260804_171350_no_lyrics` was generated before the shared server
lyrics path was supplied. It used zero lyrics and must not be consumed. The
production runner now fails immediately when it finds a non-empty catalog but
zero lyrics, and records the resolved catalog and lyrics paths in every run.
