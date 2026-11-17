# v2 decoder and TIGER audit — 2026-11-17

This handoff records verified results only.  It does not change the manuscript
or authorize replacing any table row before the indicated follow-up is done.

## DDBC v2: inference protocol correction

The original r3.1e DDBC evaluations used one joint full-mask completion.  They
omitted the selected v1 inference policy: legal Beam4 SID decoding, five-song
deduplication, probability interpolation, and cue completion after fixing the
selected SIDs.  Re-decoding uses the same checkpoint, fixed 941-context cohort,
full catalog, 256 diffusion steps, and no retraining.

| Dataset | decoder | Recall@5 | generated-cue F1 | output duplicates |
| --- | --- | ---: | ---: | ---: |
| MPD | original v2 | 1.70% | 30.58% | 63 / 941 |
| MPD | Beam4+dedup, alpha=1 | 2.21% | 31.24% | 0 / 941 |
| MPD | Beam4+dedup, alpha=0.2 | **3.66%** | **31.28%** | 0 / 941 |
| Music4All | original v2 | 0.81% | 21.86% | 38 / 941 |
| Music4All | Beam4+dedup, alpha=1 | 1.79% | 22.47% | 0 / 941 |
| Music4All | Beam4+dedup, alpha=0.2 | **2.95%** | **23.94%** | 0 / 941 |

The DDBC v2 Recall result in a future table must use the alpha=0.2 decoder,
not the original v2 completion.  The cue F1 in this table is the decoder's
many-to-many generated-vs-target cue metric; it is not the manuscript's
pool-control Cue F1.

## Music4All DDBC training checks

The all-steps cue-weight=0.1 replacement was trained for 5k+1k steps in a
separate namespace.  Its alpha=0.2 re-decode reached 2.89% Recall@5 and 22.87%
generated-cue F1, below the earlier stage-1=1.0/stage-2=0.1 checkpoint (2.95%,
23.94%).  A 5k-to-6k continuation also showed late loss rebound.  Neither is a
candidate replacement.

## Music4All TIGER: corrected SID-only control

The older TIGER `sid` implementation encoded catalog cues in its history, so it
was not a SID-only baseline.  A corrected control was trained from scratch for
20k steps with SID4-only history and SID4-only output.  Its inference code was
then fixed before its first prediction batch to convert the SID catalog table to
a Torch tensor; the completed model/optimizer state is unchanged and the new
checkpoint records its parent SHA-256.

Fixed 941-context full-catalog Top-5 comparison:

| Model | Recall@5 | output-list diversity | unique songs / 4,705 | Cue F1 |
| --- | ---: | ---: | ---: | ---: |
| old cue-input `sid` control (invalid main baseline) | 12.56% | 53.41% | 2,513 | 14.31% |
| corrected SID-only | **5.50%** | **28.48%** | **1,340** | 13.46% (pool control) |
| TIGER+cues | 3.61% | 23.12% | 1,088 | **22.26%** (generated) |

Thus the defensible joint-vs-SID-only gap is -1.89 percentage points in Recall,
not -8.95 points.  TIGER+cues improves cue prediction by 8.80 points but loses
both Recall and diversity.  Do **not** retain 12.56% as a standard SID-only
main-table result.

## Active follow-ups (not results)

The next controlled TIGER experiments are: SID/joint score fusion and candidate
reranking selected on a training holdout; SID-checkpoint warm start with cue-loss
ramp; PCGrad only if measured SID/cue gradients conflict; and scheduled sampling
for the five-step autoregressive rollout.  None has a verified result yet.

## Remote evidence

- DDBC re-decodes: `/storage_fast/ttu/GenPlaylist/outputs/cue-v2-r31e-redecode-20261116/`
- all-0.1 DDBC replacement: `/storage_fast/ttu/GenPlaylist/outputs/cue-v2-r31e-redecode-cue01all-20261116/`
- corrected TIGER control: `/storage_fast/ttu/GenPlaylist/outputs/cue-v2-r31e-tiger-sid-inputfix-r2-20261116/`
- joint TIGER: `/storage_fast/ttu/GenPlaylist/outputs/cue-v2-r31e-tiger-20261113/m4a/joint/audio/plan.json`
