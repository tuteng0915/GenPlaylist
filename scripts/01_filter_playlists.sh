#!/usr/bin/env bash
# 01_filter_playlists.sh — Iterative playlist filtering (R1 length → freq convergence → audio filter)
# Produces: data/playlists/mpd_subset/
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

BACKBONE_DIR="/Users/tut/migration_buffer/model/datasets/spotify"
ALL_PLAYLISTS="/tmp/all_playlists.txt"

# Merge backbone splits
cat "$BACKBONE_DIR/train.txt" "$BACKBONE_DIR/valid.txt" "$BACKBONE_DIR/test.txt" \
    > "$ALL_PLAYLISTS"

# R1: length 30-90, freq>=10, audio filter, post-filter length 10-60
# Iterate until convergence (run manually or extend this script with a loop)
python "$ROOT/src/00_data_schema/sample_playlists.py" \
    --input   "$ALL_PLAYLISTS" \
    --output  "$ROOT/data/playlists/mpd_subset/" \
    --min-len 30 --max-len 90 \
    --min-freq 10 \
    --min-len-after-filter 10 \
    --max-len-after-filter 60 \
    --audio-dir "$ROOT/data/audio/spotify/" \
    --freq-input "$ALL_PLAYLISTS"
