#!/usr/bin/env python3
"""scripts/fetch_lyrics.py — Download lyrics for catalog items.

Primary source : LRCLIB (https://lrclib.net) — free, no API key, ~6M tracks.
Fallback source: Genius API — needs GENIUS_ACCESS_TOKEN env var.

Output layout
-------------
    {output_dir}/
        {item_id}.txt          # plain-text lyrics (LRC timestamps stripped)
        coverage.json          # {"total": N, "found": M, "coverage_rate": ...,
                               #  "source_breakdown": {"lrclib": x, "genius": y, "miss": z}}
        fetch_log.jsonl        # one JSON line per attempt

Usage examples
--------------
# Dry run — print queries without fetching
python scripts/fetch_lyrics.py \\
    --metadata src/03_backbone_recommender/datasets/spotify/metadata.json \\
    --output   data/lyrics/spotify/ \\
    --limit    10 --dry-run

# Fetch lyrics for first 500 items
python scripts/fetch_lyrics.py \\
    --metadata src/03_backbone_recommender/datasets/spotify/metadata.json \\
    --output   data/lyrics/spotify/ \\
    --limit    500

# Fetch only items in test split
python scripts/fetch_lyrics.py \\
    --metadata src/03_backbone_recommender/datasets/spotify/metadata.json \\
    --ids-file src/03_backbone_recommender/datasets/spotify/test.txt \\
    --output   data/lyrics/spotify/

# Resume (skips items that already have a .txt file)
python scripts/fetch_lyrics.py ... --resume
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

# ── optional Genius import ────────────────────────────────────────────────────
try:
    import lyricsgenius
    _GENIUS_AVAILABLE = True
except ImportError:
    _GENIUS_AVAILABLE = False

# ── reuse metadata parser from download_audio.py ─────────────────────────────
_SCRIPTS = os.path.dirname(__file__)
sys.path.insert(0, _SCRIPTS)
from download_audio import parse_metadata_json, load_ids_from_playlist_file  # noqa: E402


# ---------------------------------------------------------------------------
# LRC timestamp stripping
# ---------------------------------------------------------------------------

_LRC_RE = re.compile(r"\[\d+:\d+\.\d+\]")

def strip_lrc_timestamps(text: str) -> str:
    """Remove [mm:ss.xx] timestamps from synced LRC lyrics."""
    lines = [_LRC_RE.sub("", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------------------
# LRCLIB (primary)
# ---------------------------------------------------------------------------

def _lrclib(title: str, artist: str) -> str | None:
    """Fetch plain lyrics from LRCLIB.  Returns None on miss."""
    try:
        r = requests.get(
            "https://lrclib.net/api/search",
            params={"track_name": title, "artist_name": artist},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        results = r.json()
        if not results:
            return None
        hit = results[0]
        plain = hit.get("plainLyrics") or ""
        synced = hit.get("syncedLyrics") or ""
        if plain.strip():
            return plain.strip()
        if synced.strip():
            return strip_lrc_timestamps(synced)
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Genius (fallback)
# ---------------------------------------------------------------------------

_genius_client = None

def _init_genius() -> bool:
    global _genius_client
    if not _GENIUS_AVAILABLE:
        return False
    token = os.getenv("GENIUS_ACCESS_TOKEN")
    if not token:
        return False
    if _genius_client is None:
        _genius_client = lyricsgenius.Genius(token, retries=2, remove_section_headers=True)
    return True


def _genius(title: str, artist: str) -> str | None:
    """Fetch lyrics from Genius.  Returns None on miss or if client unavailable."""
    if not _init_genius():
        return None
    try:
        song = _genius_client.search_song(title, artist)
        if song and song.lyrics:
            return song.lyrics.strip()
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-item fetch
# ---------------------------------------------------------------------------

def fetch_one(
    item_id: str,
    item_info: dict,
    output_dir: Path,
    dry_run: bool = False,
) -> dict:
    """Fetch lyrics for one item. Returns a log-entry dict."""
    out_path = output_dir / f"{item_id}.txt"
    if out_path.exists():
        return {"item_id": item_id, "status": "skip", "source": "", "path": str(out_path)}

    title  = item_info.get("title", "").strip()
    artist = item_info.get("artist", "").strip()

    if dry_run:
        return {"item_id": item_id, "status": "dry", "source": "",
                "query": f"{title} – {artist}", "path": str(out_path)}

    # 1. Try LRCLIB
    lyrics = _lrclib(title, artist)
    source = "lrclib" if lyrics else ""

    # 2. Fallback to Genius
    if not lyrics:
        lyrics = _genius(title, artist)
        source = "genius" if lyrics else ""

    if lyrics:
        out_path.write_text(lyrics, encoding="utf-8")
        return {"item_id": item_id, "status": "ok", "source": source, "path": str(out_path)}

    return {"item_id": item_id, "status": "miss", "source": "", "path": ""}


# ---------------------------------------------------------------------------
# Coverage stats
# ---------------------------------------------------------------------------

def _write_coverage(output_dir: Path, log_path: Path) -> None:
    entries = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    total  = len([e for e in entries if e["status"] != "skip"])
    found  = len([e for e in entries if e["status"] == "ok"])
    skipped = len([e for e in entries if e["status"] == "skip"])
    breakdown = {"lrclib": 0, "genius": 0, "miss": 0}
    for e in entries:
        if e["status"] == "ok":
            breakdown[e.get("source", "lrclib")] += 1
        elif e["status"] == "miss":
            breakdown["miss"] += 1

    cov = {
        "total_attempted": total,
        "found": found,
        "skipped_existing": skipped,
        "coverage_rate": round(found / total, 4) if total else 0.0,
        "source_breakdown": breakdown,
    }
    (output_dir / "coverage.json").write_text(json.dumps(cov, indent=2))
    print(f"\n[lyrics] Coverage: {found}/{total} ({cov['coverage_rate']:.1%})")
    print(f"         Sources : lrclib={breakdown['lrclib']}  genius={breakdown['genius']}  miss={breakdown['miss']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch lyrics via LRCLIB + Genius.")
    parser.add_argument("--metadata", required=True, help="Path to backbone metadata.json")
    parser.add_argument("--output",   required=True, help="Output directory for .txt files")
    parser.add_argument("--ids-file", help="Playlist txt file; only fetch items in this file")
    parser.add_argument("--limit",    type=int, default=None)
    parser.add_argument("--resume",   action="store_true",
                        help="Skip items that already have a .txt file")
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "fetch_log.jsonl"

    print(f"[lyrics] Loading metadata ...")
    catalog = parse_metadata_json(args.metadata)

    if args.ids_file:
        target_ids = load_ids_from_playlist_file(args.ids_file)
        target_ids = [i for i in target_ids if i in catalog]
    else:
        target_ids = sorted(catalog.keys(), key=lambda x: int(x) if x.isdigit() else x)

    if args.limit:
        target_ids = target_ids[: args.limit]

    genius_ready = _init_genius()
    print(f"[lyrics] Items     : {len(target_ids)}")
    print(f"[lyrics] LRCLIB    : enabled (no key needed)")
    print(f"[lyrics] Genius    : {'enabled' if genius_ready else 'disabled (set GENIUS_ACCESS_TOKEN to enable)'}")
    print(f"[lyrics] Dry run   : {args.dry_run}\n")

    n_ok = n_skip = n_miss = 0
    with open(log_path, "a", encoding="utf-8") as log_f:
        for i, item_id in enumerate(target_ids, 1):
            entry = fetch_one(item_id, catalog[item_id], output_dir, dry_run=args.dry_run)
            log_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log_f.flush()

            status = entry["status"]
            tag = {"ok": "OK  ", "skip": "SKIP", "miss": "MISS", "dry": "DRY "}.get(status, status)
            src = f"[{entry.get('source','?')}]" if status == "ok" else ""
            title = catalog[item_id].get("title", "")[:40]
            print(f"  [{i:>6}/{len(target_ids)}] {tag} {src:8} {item_id:>8}  {title}")

            if status == "ok":    n_ok += 1
            elif status == "skip": n_skip += 1
            elif status == "miss": n_miss += 1

            # Small polite delay for LRCLIB (no hard rate limit, but be considerate)
            if status not in ("skip", "dry"):
                time.sleep(0.2)

    if not args.dry_run:
        _write_coverage(output_dir, log_path)

    print(f"\n[lyrics] Done.  OK={n_ok}  Skip={n_skip}  Miss={n_miss}")
    print(f"[lyrics] Log  : {log_path}")


if __name__ == "__main__":
    main()
