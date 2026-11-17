# Cue v2 — Stage A only

Author clarification, 2026-11-12: Stage A contains the pilot and generation gate.
Full-catalog v2 personalization diagnostics are deferred to Stage B. Stage B
requires another explicit author decision; this pipeline cannot start training.

The immutable cohort has 100 pilot songs and 200 generation contexts per dataset.
A larger stratified pilot vocabulary pool is allowed by the request. Extraction
covers that pool and the gate target songs needed for content evaluation only.
No v1 file is overwritten. The pilot vocabulary may contain fewer than 2,047
supported concepts: unused token IDs remain reserved and are never assigned.
This is not the formal Stage B vocabulary.

Generation uses the first target SID, the same history, original Qwen/ACE
renderer and `42000 + original_context_index` seed in all three arms. There are
200 × 3 × 2 = 1,200 newly generated clips. Target access makes this a controlled
cue rendering diagnostic, not an end-to-end recommendation result.

- MERT-OAS uses the DDBC rectangular assignment definition: one generated clip
  matched against five targets. With one prediction, this equals maximum target
  cosine. It is **not** the earlier five-by-five playlist score. Mean-five and
  first-target similarities are secondary diagnostics, reported alongside it.
- Content fit uses an independent MPNet text encoder, all text chunks, an equal
  mixture of lyric and attribute embeddings, and mean similarity against five
  target cue descriptions. Both v1 and v2 references are reported. Missing
  supported cue descriptions contribute zero, not exclusion from the cohort.
- MERT history fit averages similarity to the 15 history songs.
- FAD uses VGGish and 200 first-target occurrences, following the first-item
  reference convention. It is not computed against an available-audio subset.
- All scalar CIs resample paired contexts, 10,000 draws. FAD CIs resample the
  same contexts with all within-clip frames, 1,000 draws. No CLAP is run.

The arm-blind Qwen judge is from the same family as the extractor; its support
rate is an automatic diagnostic rather than independent human ground truth.
The 40-song comparison is exported for the author's review. No participants
are contacted and no human ratings are invented.

The queue is separate from all earlier experiment queues, has explicit
dependencies, per-task locks, output hashes, failure records and a booking
deadline. A failed task stops this run; it does not trigger silent resampling.
Completed tasks can be resumed without overwriting their outputs after inspection.

Only experimental code/results are maintained here. No paper files, pushes,
session messages, Stage B runs or recurring automations are authorized by this run.
