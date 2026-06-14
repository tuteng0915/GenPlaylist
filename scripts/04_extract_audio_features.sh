#!/usr/bin/env bash
# 04_extract_audio_features.sh — Extract librosa features (tempo, key, energy, mood) from MP3s
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python "$ROOT/src/00_data_schema/extract_audio_features.py" \
    --audio-dir "$ROOT/data/audio/spotify/" \
    --ids-file  "$ROOT/data/playlists/mpd_subset/item_ids.txt" \
    --output    "$ROOT/data/tags/audio_features.json" \
    --workers   4
