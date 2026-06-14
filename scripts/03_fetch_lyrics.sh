#!/usr/bin/env bash
# 03_fetch_lyrics.sh — Fetch lyrics via LRCLIB (primary) + Genius (fallback)
# Requires: GENIUS_ACCESS_TOKEN env var
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

BACKBONE_DIR="/Users/tut/migration_buffer/model/datasets/spotify"

if [ -z "$GENIUS_ACCESS_TOKEN" ]; then
    echo "Warning: GENIUS_ACCESS_TOKEN not set — Genius fallback disabled"
fi

python "$ROOT/src/00_data_schema/fetch_lyrics.py" \
    --metadata "$BACKBONE_DIR/metadata.json" \
    --ids-file "$ROOT/data/playlists/mpd_subset/item_ids.txt" \
    --output   "$ROOT/data/lyrics/spotify/" \
    --resume
