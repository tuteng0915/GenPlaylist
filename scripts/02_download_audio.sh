#!/usr/bin/env bash
# 02_download_audio.sh — Download MP3s for all items in the final subset
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

BACKBONE_DIR="/Users/tut/migration_buffer/model/datasets/spotify"

python "$ROOT/src/00_data_schema/download_audio.py" \
    --metadata "$BACKBONE_DIR/metadata.json" \
    --ids-file "$ROOT/data/playlists/mpd_subset/item_ids.txt" \
    --output   "$ROOT/data/audio/spotify/" \
    --format   mp3
