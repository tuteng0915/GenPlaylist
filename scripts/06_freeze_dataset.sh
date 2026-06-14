#!/usr/bin/env bash
# 06_freeze_dataset.sh — Build final catalog, freeze splits, produce catalog_metadata.json
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

BACKBONE_DIR="/Users/tut/migration_buffer/model/datasets/spotify"

# Step 1: cross-validate assets → catalog.json
python "$ROOT/src/00_data_schema/build_final_catalog.py" \
    --ids-file   "$ROOT/data/playlists/mpd_subset/item_ids.txt" \
    --metadata   "$BACKBONE_DIR/metadata.json" \
    --audio-dir  "$ROOT/data/audio/spotify/" \
    --lyrics-dir "$ROOT/data/lyrics/spotify/" \
    --output     "$ROOT/data/dataset/"

# Step 2: filter playlists to complete items, 80/10/10 split
python "$ROOT/src/00_data_schema/freeze_dataset.py" \
    --catalog   "$ROOT/data/dataset/catalog.json" \
    --playlists "$ROOT/data/playlists/mpd_subset/playlists.txt" \
    --output    "$ROOT/data/dataset/" \
    --min-complete 5

# Step 3: pipeline-ready catalog_metadata.json
python "$ROOT/src/00_data_schema/build_catalog_metadata.py" \
    --catalog    "$ROOT/data/dataset/catalog.json" \
    --output     "$ROOT/data/dataset/catalog_metadata.json" \
    --lyrics-dir "$ROOT/data/lyrics/spotify/" \
    --tags       "$ROOT/data/tags/audio_features.json"
