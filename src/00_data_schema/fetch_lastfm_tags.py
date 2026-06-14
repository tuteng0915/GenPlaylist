#!/usr/bin/env python3
"""scripts/fetch_lastfm_tags.py — Fetch top tags per track from Last.fm API.

For each item, calls track.getTopTags and stores the top-N tags
(by weight) as a list.  Falls back to artist.getTopTags if the
track lookup returns no tags.

Writes:
  data/tags/lastfm_tags.json    — {"item_id": {"tags": [...], "source": "track"|"artist"}}
  data/tags/lastfm_coverage.json

Usage
-----
export LASTFM_API_KEY=your_key_here

python scripts/fetch_lastfm_tags.py \\
    --metadata  /path/to/metadata.json \\
    --ids-file  data/playlists/r3_10_30_freq3/item_ids.txt \\
    --output    data/tags/lastfm_tags.json \\
    --top-n 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

_BASE = "https://ws.audioscrobbler.com/2.0/"
_META_RE = re.compile(r"^'(.+)'\s+by\s+(.+?)\s+in\s+album'(.+)'$")


def _parse_meta(raw: str) -> tuple[str, str]:
    m = _META_RE.match(raw.strip())
    if m:
        return m.group(1), m.group(2)
    return raw.strip(), ""


def _get_track_tags(title: str, artist: str, api_key: str, top_n: int) -> tuple[list[str], str]:
    """Returns (tags, source) where source is 'track', 'artist', or 'miss'."""
    def _call(method: str, **params) -> dict:
        r = requests.get(_BASE, params={
            "method": method, "api_key": api_key, "format": "json",
            "autocorrect": 1, **params,
        }, timeout=10)
        r.raise_for_status()
        return r.json()

    # Try track tags first
    try:
        data = _call("track.getTopTags", track=title, artist=artist)
        tags_raw = data.get("toptags", {}).get("tag", [])
        if isinstance(tags_raw, dict):
            tags_raw = [tags_raw]
        tags = [t["name"].lower() for t in tags_raw
                if int(t.get("count", 0)) > 0][:top_n]
        if tags:
            return tags, "track"
    except Exception:
        pass

    # Fallback: artist tags
    if artist:
        try:
            data = _call("artist.getTopTags", artist=artist)
            tags_raw = data.get("toptags", {}).get("tag", [])
            if isinstance(tags_raw, dict):
                tags_raw = [tags_raw]
            tags = [t["name"].lower() for t in tags_raw][:top_n]
            if tags:
                return tags, "artist"
        except Exception:
            pass

    return [], "miss"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata",  required=True)
    parser.add_argument("--ids-file",  required=True,
                        help="One item_id per line")
    parser.add_argument("--output",    default="data/tags/lastfm_tags.json")
    parser.add_argument("--top-n",     type=int, default=5,
                        help="Max tags to keep per item (default: 5)")
    parser.add_argument("--sleep",     type=float, default=0.25,
                        help="Seconds between API calls (default: 0.25; Last.fm allows ~5 rps)")
    parser.add_argument("--api-key",   default=None,
                        help="Last.fm API key (or set LASTFM_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("LASTFM_API_KEY")
    if not api_key:
        print("[lastfm] ERROR: set LASTFM_API_KEY or pass --api-key")
        raise SystemExit(1)

    # Load metadata
    raw_meta: dict = json.load(open(args.metadata))
    ids = [l.strip() for l in open(args.ids_file) if l.strip()]
    print(f"[lastfm] Items to fetch : {len(ids):,}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results for resume
    existing: dict = {}
    if out_path.is_file():
        existing = json.load(open(out_path))
        print(f"[lastfm] Resuming — {len(existing):,} already fetched")

    results = dict(existing)
    n_track = n_artist = n_miss = 0

    todo = [iid for iid in ids if iid not in results]
    print(f"[lastfm] Remaining      : {len(todo):,}")

    for i, item_id in enumerate(todo, 1):
        raw = raw_meta.get(item_id, "")
        title, artist = _parse_meta(raw)

        tags, source = _get_track_tags(title, artist, api_key, args.top_n)
        results[item_id] = {"tags": tags, "source": source}

        if source == "track":   n_track += 1
        elif source == "artist": n_artist += 1
        else:                    n_miss += 1

        if i % 200 == 0 or i == len(todo):
            done = len(existing) + i
            total = len(ids)
            pct = done / total * 100
            print(f"  [{done:>6}/{total}] {pct:4.1f}%  track={n_track} artist={n_artist} miss={n_miss}")
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

        time.sleep(args.sleep)

    # Final save
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    total = len(ids)
    fetched = sum(1 for v in results.values() if v["source"] != "miss")
    cov_path = out_path.with_name("lastfm_coverage.json")
    cov_path.write_text(json.dumps({
        "total": total, "track_hit": n_track, "artist_fallback": n_artist,
        "miss": n_miss, "coverage_rate": round(fetched / total, 4),
    }, indent=2))

    print(f"\n[lastfm] Done. Coverage: {fetched}/{total} ({fetched/total*100:.1f}%)")
    print(f"[lastfm] Written to {out_path}")


if __name__ == "__main__":
    main()
