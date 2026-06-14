#!/usr/bin/env python3
"""scripts/fetch_musicbrainz_tags.py — Fetch genre/style tags from MusicBrainz.

No API key required. Uses the MusicBrainz public JSON API.
Rate limit: 1 request/second (enforced by --sleep, default 1.1s).

For each item, searches recordings by title + artist, then reads
the genre/tag folksonomy from the top match.

Writes:
  data/tags/musicbrainz_tags.json     — {"item_id": {"genres": [...], "mb_id": "..."}}
  data/tags/musicbrainz_coverage.json

Usage
-----
python scripts/fetch_musicbrainz_tags.py \\
    --metadata /path/to/metadata.json \\
    --ids-file data/playlists/r3_10_30_freq3/item_ids.txt \\
    --output   data/tags/musicbrainz_tags.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests

_BASE    = "https://musicbrainz.org/ws/2"
_HEADERS = {"User-Agent": "GenPlaylist/0.1 (research; tuteng0915@gmail.com)"}
_META_RE = re.compile(r"^'(.+)'\s+by\s+(.+?)\s+in\s+album'(.+)'$")


def _parse_meta(raw: str) -> tuple[str, str]:
    m = _META_RE.match(raw.strip())
    if m:
        return m.group(1), m.group(2)
    return raw.strip(), ""


def _search_recording(title: str, artist: str, sleep: float) -> dict | None:
    q = f'recording:"{title}"'
    if artist:
        q += f' AND artist:"{artist}"'
    try:
        r = requests.get(f"{_BASE}/recording",
                         headers=_HEADERS,
                         params={"query": q, "limit": 1, "fmt": "json"},
                         timeout=10)
        time.sleep(sleep)
        r.raise_for_status()
        recordings = r.json().get("recordings", [])
        return recordings[0] if recordings else None
    except Exception:
        return None


def _get_tags(mb_id: str, sleep: float) -> list[str]:
    """Fetch genre/tag folksonomy for a recording MB ID."""
    try:
        r = requests.get(f"{_BASE}/recording/{mb_id}",
                         headers=_HEADERS,
                         params={"inc": "genres+tags", "fmt": "json"},
                         timeout=10)
        time.sleep(sleep)
        r.raise_for_status()
        data = r.json()
        genres = [g["name"] for g in data.get("genres", [])]
        tags   = [t["name"] for t in data.get("tags", []) if t.get("count", 0) > 0]
        # Prefer genres; fall back to tags if empty
        return genres[:5] if genres else tags[:5]
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--ids-file", required=True,
                        help="One item_id per line")
    parser.add_argument("--output",   default="data/tags/musicbrainz_tags.json")
    parser.add_argument("--sleep",    type=float, default=1.1,
                        help="Seconds between requests (default: 1.1 — MB allows 1 rps)")
    args = parser.parse_args()

    raw_meta: dict = json.load(open(args.metadata))
    ids = [l.strip() for l in open(args.ids_file) if l.strip()]
    print(f"[mb] Items to fetch : {len(ids):,}")
    print(f"[mb] Est. time      : ~{len(ids) * args.sleep * 2 / 3600:.1f} hours")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if out_path.is_file():
        existing = json.load(open(out_path))
        print(f"[mb] Resuming — {len(existing):,} already fetched")

    results = dict(existing)
    n_hit = n_tagged = n_miss = 0

    todo = [iid for iid in ids if iid not in results]
    print(f"[mb] Remaining      : {len(todo):,}")

    for i, item_id in enumerate(todo, 1):
        raw = raw_meta.get(item_id, "")
        title, artist = _parse_meta(raw)

        rec = _search_recording(title, artist, args.sleep)
        if rec:
            n_hit += 1
            mb_id = rec.get("id", "")
            tags  = _get_tags(mb_id, args.sleep) if mb_id else []
            if tags:
                n_tagged += 1
            results[item_id] = {"genres": tags, "mb_id": mb_id}
        else:
            n_miss += 1
            results[item_id] = {"genres": [], "mb_id": ""}

        if i % 100 == 0 or i == len(todo):
            done = len(existing) + i
            pct  = done / len(ids) * 100
            print(f"  [{done:>6}/{len(ids)}] {pct:4.1f}%  hit={n_hit} tagged={n_tagged} miss={n_miss}")
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    fetched = sum(1 for v in results.values() if v.get("genres"))
    cov = fetched / len(ids) if ids else 0
    cov_path = out_path.with_name("musicbrainz_coverage.json")
    cov_path.write_text(json.dumps({
        "total": len(ids), "search_hit": n_hit,
        "with_genres": fetched, "miss": n_miss,
        "coverage_rate": round(cov, 4),
    }, indent=2))

    print(f"\n[mb] Done. Genre coverage: {fetched}/{len(ids)} ({cov*100:.1f}%)")
    print(f"[mb] Written to {out_path}")


if __name__ == "__main__":
    main()
