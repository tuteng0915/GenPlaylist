#!/usr/bin/env bash
# 05_fetch_musicbrainz.sh — Fetch genre/style tags from MusicBrainz (no API key, 1 req/s)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

BACKBONE_DIR="/Users/tut/migration_buffer/model/datasets/spotify"

python "$ROOT/src/00_data_schema/fetch_musicbrainz_tags.py" \
    --metadata "$BACKBONE_DIR/metadata.json" \
    --ids-file "$ROOT/data/playlists/mpd_subset/item_ids.txt" \
    --output   "$ROOT/data/tags/musicbrainz_tags.json"
