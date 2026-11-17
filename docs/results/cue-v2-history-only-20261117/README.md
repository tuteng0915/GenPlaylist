# v2 History-only full-catalog report

This is a deterministic recommendation-only baseline on the fixed 941-context
cohort. For each listener, it returns the five most recent distinct songs in
the 15-song history. Training-target popularity would fill any shortage, but
neither fixed cohort has fewer than five distinct historical songs.

It has no generated cue output.  To make its cue-control score comparable with
the SID-only rows, we draw 40 cues from the distinct union of its 15 history
songs and five returned songs, split them into five sets of eight, then use the
same optimal one-to-one matching against the five target cue sets.  The result
is explicitly a **pooled-cue control**, not a cue-generation score.  We average
ten fixed draws; no seed is selected after seeing the result.

| Dataset | Recall@5 | Hit@5 | Diversity | CLHE OAS | Pooled Cue F1 | Sound / emotion / theme F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MPD | 0.00% | 0.00% | 44.62% | 63.83% | 8.95% | 3.55 / 18.47 / 7.36% |
| Music4All-Onion | 14.20% | 42.30% | 30.16% | 48.91% | 15.03% | 20.59 / 7.56 / 9.11% |

Every returned song is drawn from the visible history (history-repeat rate
100%).  MPD consequently has zero future-target overlap, while Music4All has
some overlap because its held-out future listening events can repeat a recent
history item.  The row is a retrieval/cue-control baseline only: it has no
predicted cues and no newly rendered v2 audio, so generated-audio and generated
text metrics remain unavailable.

The immutable remote reports are under
`outputs/cue-v2-r31e-evaluation-20261115/history-only/`.
The newly computed full-table control reports are
`outputs/cue-v2-r31e-evaluation-20261115/{mpd,m4a}/history-only-pool-one-to-one.json`;
the local copies and their hashes are in `raw/`.
