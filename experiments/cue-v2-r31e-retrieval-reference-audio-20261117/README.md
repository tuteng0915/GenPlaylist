# Retrieval-reference, no-cue audio baselines

For each fixed 100-playlist cohort, the retriever returns five songs.  The
generator receives only those five songs as its reference set: it derives the
reference embedding and style summary directly from their catalog entries.
It never reads catalog cues, predicted cues, a cue vocabulary, or a cue-derived
prompt.  Five seeded samples are rendered per retrieved list, yielding 500
clips per retriever/dataset arm.

`ddbc_reference` uses the frozen standard DDBC SID-only retrieval plan.
`tiger_reference` uses only the corrected Music4All standard TIGER SID-only
plan.  A corresponding MPD TIGER arm must wait for a corrected MPD SID-only
checkpoint; the invalid historical MPD TIGER output is intentionally excluded.
