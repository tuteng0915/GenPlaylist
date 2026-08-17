# Frozen Listener-Study Protocol

## Scope

The paper's listener study is a blinded paired comparison between the primary
GenPlaylist waveform and the immediate real next track. It follows the rating
questions in the existing WP-D interface but does not use WP-D's legacy
curated cases: those cases contain five references and none matches the frozen
941-context, 15-reference evaluation suite. The WP-D demo remains unchanged.

The study evaluates the generated waveform, not catalog recommendation. Each
participant sees the title and artist of the 15 ordered references, hears two
30-second candidates in randomized A/B order, rates each candidate for history
fit, audio quality, and novelty on five-point scales, and chooses Song A, Song
B, or no preference. The candidates are:

- the primary GenPlaylist waveform for that context; and
- a centered 30-second clip of the compatible held-out item at position 16.

The held-out item is a calibration comparator, not the unique correct
composition. DDBC-SFT and ACE-Step-Direct are not part of this human study.

## Frozen cases and assignment

Run the case builder on the server:

```bash
conda run -n music python scripts/prepare_listener_study.py \
  --prepared-dir /home/wjzhang/tt_workspace/data/data/processed/genplaylist-v4-8cue-20item-joint-15to5 \
  --generated-audio-dir /home/wjzhang/tt_workspace/data/data/processed/genplaylist-end-to-end/ace-step-v1 \
  --catalog-audio-dir /home/wjzhang/tt_workspace/data/data/audio/spotify \
  --catalog-metadata data/dataset/catalog_metadata.json \
  --output-dir /home/wjzhang/tt_workspace/data/data/processed/genplaylist-listener-study-v1 \
  --generated-system GenPlaylist \
  --cases 25 \
  --seed 42 \
  --clip-seconds 30 \
  --sample-rate 44100
```

The builder uniformly samples 25 contexts without replacement from all 941
frozen contexts. It refuses to overwrite a nonempty output directory, verifies
the generated-audio record hashes, center-clips both candidates, converts both
through the same 44.1-kHz stereo PCM16 WAV path, balances the generated
candidate between A and B to within one case, and records every input and
output hash. This identical conversion prevents the codec or file format from
revealing which candidate is generated. `public_manifest.json` contains only reference labels
and blinded asset paths. `private_manifest.json` contains the unblinding key
and must not be exposed to participants.

"Public" describes the manifest content, not its filesystem permissions. The
builder creates the package with owner-only directories and files. The study
host should serve only the selected reference labels and two candidate assets;
it must never expose directory listings or the private manifest.

The collection host should assign one case uniformly to each participant and
randomize A/B independently at presentation time, recording
`song_a_is_generated`. Each participant should submit one complete response so
the response is the participant-level sampling unit. If repeated responses are
allowed, a stable pseudonymous participant ID is required; the analysis script
then averages within participant before bootstrapping.

## Collection fields

Required fields are compatible with the current WP-D CSV:

- pseudonymous `session_id` (or a separately configured participant column);
- case identifier and A/B assignment;
- `fit_a`, `fit_b`, `quality_a`, `quality_b`, `novelty_a`, and `novelty_b`;
- `preference` in `{Song A, Song B, No preference}`;
- listening frequency and formal musical training;
- optional free-text comments.

The deployment should additionally record successful playback of both
candidates. Target participant count, recruitment source, compensation,
duplicate-response policy, playback-based exclusion, and any attention check
must be fixed before inspecting responses. Exclusions must never depend on
which system a participant preferred.

## Consent and data handling

Do not begin collection before the institution's required ethics review or
exemption and informed-consent process are complete. The consent page should
state the study purpose, approximate duration, voluntary nature, withdrawal
procedure, compensation, data retention period, researcher contact, and that
one candidate is machine-generated. Avoid collecting names, email addresses,
raw listening histories, or other unnecessary identifiers. Treat optional
comments as potentially identifying free text.

The package contains copyrighted real-audio excerpts and is for controlled
evaluation only. Do not publish `participant_assets`, the private manifest, or
raw response logs. Store the unblinding key separately from the participant UI
and restrict access to the minimum research team.

## Analysis

After applying only the predeclared exclusions, run:

```bash
python scripts/analyze_listener_study.py \
  --responses /path/to/user_study.csv \
  --output /path/to/listener-study-analysis.json \
  --participant-column session_id \
  --bootstrap-samples 10000 \
  --bootstrap-seed 42
```

The analysis unblinds A/B ratings, reports system means and paired
GenPlaylist-minus-real differences for all three scales, and bootstraps by
participant. Preference is the generated choice share with no-preference ties
contributing 0.5 to each candidate; raw generated/real/tie counts are retained.
Results are also stratified by formal musical training. The response CSV hash,
exclusions, bootstrap settings, and complete results are written to the output
JSON.
