#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible entry point. The old Yelp/DISCO experiment paths and
# multi-item settings were removed; all GenPlaylist evaluation now goes through
# the frozen Spotify 15->5 runner.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/eval_spotify.sh" "$@"
