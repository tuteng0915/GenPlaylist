# Reproducibility code map (v2 experiments)

This repository contains the source snapshots needed to reproduce the v2
experiments.  Source code is versioned here; catalogs, lyrics, model weights,
checkpoints, generated audio, feature caches, and metric outputs are deliberately
external because they are large or license-controlled.

## Cue construction

- `src/02_creative_cues/` contains the reusable facet extraction, normalization,
  assignment, export, and evaluation components.
- `experiments/cue-v2-stageA-20261112/` through
  `experiments/cue-v2-r31e-20261113/` preserve the Stage-A diagnostics, repair,
  candidate-evidence, mapping, vocabulary, and r3.1e selection programs used
  for the final v2 cue release.
- `experiments/cue-v2-mapping-audit-20261113/` and
  `experiments/cue-v2-repair-verification-20261113/` are the standalone audit
  and verification entry points.

## SID / RVQ and recommendation backbones

- `src/03_backbone_recommender/` implements the item tokenizer, RVQ semantic
  tokens, dataset preparation, diffusion backbone, autoregressive backbone,
  and training runtime.
- `scripts/build_music4all_ddbc_catalog.py`,
  `scripts/extract_ddbc_checkpoint_artifacts.py`, and
  `scripts/train_eval_tiger.py` are the dataset and backbone entry points.
- `experiments/cue-v2-r31e-ddbc-*/` contains the v2 DDBC training objectives,
  staged cue weighting, schedules, and fixed-941 evaluators.
- `experiments/cue-v2-r31e-tiger-20261113/`,
  `experiments/cue-v2-r31e-tiger-sid-inputfix-r2-20261116/`, and
  `experiments/cue-v2-r31e-tiger-tune-20261117/` preserve the joint TIGER,
  corrected SID-only, and warm-start/ramp experiments.

## DDBC inference policy

The legal v2 decoding path is split intentionally:

1. `experiments/cue-first-20261110/ddbc_order.py` selects SIDs with Beam-4 and
   list de-duplication.
2. `experiments/ddbc-preserve-20261111/decoder.py` supplies cue-masked versus
   conditioned probability interpolation (`alpha`) during that SID search.
3. `experiments/fixed-sid-cues-20261110/decoder_v2/finalize_cues.py` fills cues
   only after the five SIDs are fixed.

This is the Beam4+dedup, alpha-interpolation, then cue-completion protocol;
it is not a single joint full-mask sample.

## Music generation and metrics

- `src/04_synthesis/` and `scripts/run_end_to_end_verbalization.py` provide
  Qwen verbalization and ACE-Step conditioning.
- `experiments/cue-v2-r31e-list5-audio-20261117/` is the 100-list × five-item
  v2 audio production and MERT/FAD workflow.
- `experiments/cue-v2-r31e-retrieval-reference-audio-20261117/` and
  `experiments/cue-v2-r31e-simple-reference-audio-20261117/` are the no-cue
  retrieval-reference controls.
- `experiments/cue-v2-r31e-content-20261113/` and
  `experiments/cue-v2-followup-20261113/` implement lyric/metadata content-fit
  diagnostics and paired comparisons.

## Reproduction requirements

The programs use absolute experiment roots on the NExT shared storage and read
the frozen catalog, candidate cohorts, and versioned cue files from those
roots.  Each experiment directory records its fixed seeds and input paths in
its protocol or source.  Reproducing a run requires an authorized copy of the
underlying datasets and the cited Qwen, ACE-Step, MERT, and FAD dependencies;
none are bundled in this repository.
