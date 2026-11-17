# Random/Popularity retrieval-reference audio controls

Each arm uses only its frozen full-catalog Top-5 retrieval output as a five-song
reference for Qwen and ACE-Step. It supplies no catalog or predicted cue. The
fixed diagnostic cohort has 20 lists per dataset and five generated clips per
list, i.e. 100 clips per arm. It is intentionally a small lower-bound audio
control and does not replace the 100-list model comparisons.
