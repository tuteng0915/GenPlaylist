# Writing handoff: standard SID-only baselines

Date: 2026-11-10 (SGT). This is an experiment-to-writing handoff, not a manuscript edit. The current paper repository assigns manuscript editing exclusively to Claude Code. No manuscript source, figure, bibliography, or PDF was changed for this handoff.

## Author decision to apply

The main comparison on each backbone is the standard SID-only recommender against GenPlaylist on that backbone:

| Main-table method | Recommender input | Recommender output |
|---|---|---|
| DDBC | History SIDs | Song SIDs |
| DDBC + cues (GenPlaylist) | History SIDs and catalog cues | Song SIDs and predicted cues |
| TIGER | History SIDs | Song SIDs |
| TIGER + cues (GenPlaylist) | History SIDs and catalog cues | Song SIDs and predicted cues |

Catalog cues retrieved for rendering enter the downstream renderer, not the SID-only recommender. The primary comparison changes both the cue information in the input and the output supervision. It is not an output-loss-only ablation.

Remove Released DDBC from the main table, retaining its experimental records. Use the existing standard TIGER row as the primary TIGER baseline, named TIGER rather than Native TIGER. Remove the extra, unevaluated auxiliary TIGER SID-only row; do not retain both a primary standard row and a duplicate auxiliary row. Preserve the auxiliary SID+cue-input/SID-output checkpoints and records outside the main comparison.

DDBC SID-only already uses SID-only input and does not need retraining. The standard TIGER recommendation results and 941-context audio results already exist for each dataset. Both TIGER variants use 20k updates and beam 20, but their initialization and training implementations differ. Standard TIGER uses Hugging Face generation with EOS; the joint implementation scores four SID tokens before emitting cues. Do not describe the comparison as identical initialization, identical decoding implementation, or a loss-only ablation. No completed joint TIGER result was supplied in this packet.

## Main-table cue cells

Retrieved cues can be scored with the same target Cue F1 and cue–song grounding F1 as generated cues. The old baseline N/A values came from omitted scorer inputs. Use these verified values for the fixed 941 contexts of each dataset. Numbers below follow the current table convention: F1 values multiplied by 100 and rounded to two decimals.

| Dataset | Method | Cue F1 | Cue–song grounding F1 |
|---|---|---:|---:|
| MPD | DDBC | 30.88 | 100.00† |
| MPD | DDBC + cues (GenPlaylist) | 27.08 | 20.37 |
| MPD | TIGER | 30.79 | 100.00† |
| MPD | TIGER + cues (GenPlaylist) | — | — |
| Music4All-Onion | DDBC | 47.01 | 100.00† |
| Music4All-Onion | DDBC + cues (GenPlaylist) | 43.10 | 37.91 |
| Music4All-Onion | TIGER | 49.49 | 100.00† |
| Music4All-Onion | TIGER + cues (GenPlaylist) | — | — |

† Retrieved-cue grounding is 1 by construction (100 under this display scale), not evidence of learned grounding superiority. Use “Cue evaluation” for the column group because it includes retrieved cues. History-only has no explicit cue output and remains N/A for these cells. Joint TIGER cells remain not evaluated, not N/A.

The legacy scorer key `sid_matched` identifies the existing standard SID-only DDBC results; it does not mean cue-conditioned input. The key `tiger` identifies standard SID-only TIGER. `conditional_mask` identifies the reported joint DDBC configuration. Keep these legacy file identifiers in evidence but use current method names in the paper.

Joint DDBC has lower target Cue F1 than the retrieved baselines on both datasets. With the same joint DDBC song predictions, retrieved cues obtain 30.07 versus 27.08 on MPD and 46.30 versus 43.10 on Music4All-Onion. Preserve this observed counterevidence; cue agreement does not itself measure audio preference or creativity. The supporting bootstrap is exploratory and checkpoint-conditional, without training-seed uncertainty or multiplicity adjustment.

## Current source locations requiring writer integration

The paper source inspected for this handoff still contains Released DDBC and Native TIGER rows, an empty auxiliary TIGER SID-only row, and N/A cue cells for retrieved baselines in `Figures/main_results_v2.tex`. In `Sections/5_experiments_v2.tex`, Training and comparisons still calls standard TIGER an external reference; Recommendation and cue prediction and Generated music still call it native TIGER. These are writer-side changes, not changes made by this handoff. Preserve the measured recommendation/audio numbers when moving the existing TIGER row.

The code-side five-item rendering request also predates this baseline decision: its separate Native TIGER and future matched-TIGER SID-only entries must not become duplicate primary-baseline jobs. This handoff does not revise that request or start or change any experiment queue.

Keep the main comparison on the fixed 941-context cohorts per dataset. The separate Music4All-Onion full test has 19,771 contexts; its recommendation metrics must not replace the 941-context main-table values.

## Evidence and execution status

The authoritative correction is paper-repository commit `12a29e017e39bf58ff2d45f67d22076262c33e46` on `origin/codex/two-backbone-handoff-20261110`. Its result folder has been copied unchanged to `docs/results/standard-sid-correction-20261110/` in this code repository. The remote-tracking branch was fetched; the paper branch and its dirty worktree were not merged or edited.

The packet records the experiment-owner queue correction at 2026-11-10 16:52 SGT: eight unstarted auxiliary tasks removed, 238 existing TIGER audio tasks changed to joint-only, completed auxiliary trainings preserved, and live joint/non-TIGER work unchanged. Its recorded queue has 1,607 tasks and SHA-256 `37f6c1869bda5bbf35a2705376f1d2a6e96673b6a02376ea20369431b52ba791`. These are the packet's recorded execution facts, not a fresh server inspection by this writing handoff.

Cue results come from paper-repository commit `ef204f4`, `docs/results/retrieved-cue-evaluation-20261110/report.json` and `table-rows.tsv`. The independent scorer checks and first-item rendering-cue checks are recorded there. Preserve all old experiment files and negative/null outcomes. No training, inference, rendering, or queue operation was run by this handoff.
