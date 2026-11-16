# Cue-v2 pooled-control Cue F1 (2026-11-16)

Status: complete.  This record supersedes the earlier diagnostic that attached
each returned item's catalog cues to SID-only, Random, and Popularity rows.
It changes only the Cue F1 controls: returned songs, Recall@5, Diversity, and
audio conditions are unchanged.

## Protocol

All results use the fixed 941-context cohort on each dataset and maximum-weight
one-to-one matching of five predicted cue sets to five target cue sets, with
within-item cue-multiset F1 as the assignment weight.

- **Joint DDBC/TIGER + cues:** use the model's five predicted cue sets from the
  full active v2 cue vocabulary.
- **SID-only:** for each context, form the distinct-cue union over its 15
  history items and that backbone's five already returned items.  Uniformly
  draw 40 cues, then partition the draw order into five eight-cue sets.
- **Random:** uniformly draw 40 active v2 vocabulary cues and partition into
  five eight-cue sets.
- **Popularity:** take the 40 active v2 cues with the highest catalog-song
  frequency and partition their rank order into five eight-cue sets.

Random and SID-only report the mean of ten fixed draws; no seed was selected
after observing F1.  Sampling is without replacement when the available pool
contains at least 40 distinct cues.  Otherwise it samples with replacement
*inside that same pool* to preserve the five-by-eight output format: this
occurs for 11 DDBC and 26 TIGER Music4All contexts, and for no MPD context.
Popularity is deterministic.

## Results

All values below are overall Cue F1 multiplied by 100.  The category values and
paired intervals are in `summary.json`.

| Dataset | Random | Popularity | DDBC SID-only | DDBC + cues | TIGER SID-only | TIGER + cues |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MPD | 1.84 | 12.49 | 8.67 | 24.66 | 8.68 | 27.62 |
| Music4All-Onion | 1.77 | 16.47 | 13.20 | 17.32 | 14.31 | 22.26 |

The paired overall Cue-F1 differences between each joint model and its own
SID-only pooled-cue control are positive on both datasets: DDBC is +15.99 pp
(95% CI [15.58, 16.41]) on MPD and +4.12 pp ([3.66, 4.60]) on Music4All;
TIGER is +18.94 pp ([18.33, 19.53]) on MPD and +7.95 pp ([7.27, 8.65]) on
Music4All.  Intervals use 10,000 paired bootstrap replicates with seed 42.

## Provenance

The immutable remote evidence root is
`/storage_fast/ttu/GenPlaylist/outputs/cue-v2-r31e-evaluation-20261115/`.
The raw remote summary SHA-256 is
`f3911ed697bad4e23673a74b25907ba73792993b2e4a07603d26e3e177ddea7d`.
`summary.json` records the source files and their SHA-256 values.
