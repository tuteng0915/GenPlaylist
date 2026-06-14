#!/usr/bin/env python3
"""scripts/download_audio.py — Download audio for catalog items via yt-dlp.

For each catalog song (title + artist from metadata.json), search YouTube
and download the best-quality audio track.  Uses yt-dlp's built-in ytsearch
so no API key is needed.

Dependencies
------------
    pip install yt-dlp
    # ffmpeg required for wav/mp3 conversion:
    brew install ffmpeg      # macOS
    apt install ffmpeg       # Linux

Usage examples
--------------
# Download first 200 items from Spotify catalog
python scripts/download_audio.py \\
    --metadata src/03_backbone_recommender/datasets/spotify/metadata.json \\
    --output   data/audio/spotify/ \\
    --limit    200

# Download only the songs in the test split
python scripts/download_audio.py \\
    --metadata src/03_backbone_recommender/datasets/spotify/metadata.json \\
    --ids-file src/03_backbone_recommender/datasets/spotify/test.txt \\
    --output   data/audio/spotify/

# Resume after interruption (already-downloaded item_ids are skipped)
python scripts/download_audio.py \\
    --metadata src/03_backbone_recommender/datasets/spotify/metadata.json \\
    --output   data/audio/spotify/ \\
    --resume

# Dry run — print what would be downloaded without actually downloading
python scripts/download_audio.py \\
    --metadata src/03_backbone_recommender/datasets/spotify/metadata.json \\
    --limit    10 --dry-run

Output layout
-------------
    {output_dir}/
        {item_id}.wav          # downloaded audio, named by item_id
        download_log.jsonl     # one JSON line per attempt (success + failure)
        failed.txt             # item_ids that failed; feed back via --ids-file for retry

Notes
-----
- Search query: "{title} {artist}" — yt-dlp picks the top YouTube result.
  The first result is usually correct for well-known tracks; obscure tracks
  may match poorly.  Consider verifying a sample manually.
- Rate limiting: a random sleep of 1–3 s between requests avoids 429 errors.
  Use --sleep to increase this for larger runs.
- yt-dlp is the actively maintained fork of youtube-dl.  Install the latest
  version to avoid YouTube bot-detection issues:
      pip install -U yt-dlp
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Metadata parsing  (mirrors CatalogItem.from_metadata_string)
# ---------------------------------------------------------------------------

def parse_metadata_json(metadata_path: str) -> dict[str, dict]:
    """Load backbone metadata.json → {item_id: {"title": ..., "artist": ...}}.

    Spotify format: "'Title' by Artist in album'Album'"
    """
    with open(metadata_path, encoding="utf-8") as f:
        raw = json.load(f)

    pattern = re.compile(r"'(.+?)'\s+by\s+(.+?)\s+in\s+album'(.+?)'$")
    items = {}
    for item_id, meta_str in raw.items():
        m = pattern.match(meta_str)
        if m:
            items[item_id] = {
                "title":  m.group(1),
                "artist": m.group(2),
                "album":  m.group(3),
            }
        else:
            # Fallback: use full string as title, empty artist
            items[item_id] = {"title": meta_str.strip("'"), "artist": "", "album": ""}
    return items


def load_ids_from_playlist_file(ids_file: str) -> list[str]:
    """Extract unique item IDs from either:
      - backbone playlist format: "playlist_id, item1, item2, ..." (skips first token)
      - simple one-per-line format: one item_id per line (e.g. item_ids.txt from sample_playlists.py)

    Auto-detected: if the first non-empty line contains a comma, treat as playlist format.
    """
    with open(ids_file, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]

    playlist_fmt = "," in lines[0] if lines else False

    seen: set = set()
    ids: list = []
    for line in lines:
        if playlist_fmt:
            parts = line.strip().split(",")
            tokens = parts[1:]   # skip playlist_id
        else:
            tokens = [line.strip()]
        for p in tokens:
            iid = p.strip()
            if iid and iid not in seen:
                ids.append(iid)
                seen.add(iid)
    return ids


# ---------------------------------------------------------------------------
# yt-dlp wrapper
# ---------------------------------------------------------------------------

def build_search_query(item_info: dict) -> str:
    """Build a YouTube search string from title + artist."""
    title  = item_info.get("title", "").strip()
    artist = item_info.get("artist", "").strip()
    if title and artist:
        return f"{title} {artist}"
    return title or artist or "unknown"


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _find_existing(output_dir: Path, item_id: str) -> Path | None:
    """Return any existing audio file for item_id, regardless of extension."""
    audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".webm", ".opus", ".ogg"}
    for p in output_dir.glob(f"{item_id}.*"):
        if p.suffix.lower() in audio_exts:
            return p
    return None


def download_one(
    item_id: str,
    item_info: dict,
    output_dir: Path,
    audio_format: str = "wav",
    sleep_range: tuple[float, float] = (1.0, 3.0),
    dry_run: bool = False,
    cookies_from_browser: str | None = None,
) -> dict:
    """Download audio for one item via yt-dlp ytsearch.

    Returns a log entry dict with keys:
        item_id, query, status ('ok'|'fail'|'skip'|'dry'), path, error

    If ffmpeg is not installed, yt-dlp will download the best native audio
    stream (usually .webm/opus) without converting to the requested format.
    This is still usable for evaluation and synthesis.
    """
    # Skip if any audio file for this item_id already exists
    existing = _find_existing(output_dir, item_id)
    if existing:
        return {"item_id": item_id, "status": "skip", "path": str(existing), "error": ""}

    query = build_search_query(item_info)

    if dry_run:
        return {"item_id": item_id, "query": query, "status": "dry",
                "path": str(output_dir / f"{item_id}.{audio_format}"), "error": ""}

    # Build yt-dlp command.
    # --audio-format conversion requires ffmpeg; if absent, yt-dlp downloads
    # the best native stream. We still pass --audio-format so the conversion
    # happens automatically once ffmpeg is installed.
    cmd = [
        "yt-dlp",
        f"ytsearch1:{query}",
        "--no-playlist",
        "-x",
        "--audio-format", audio_format,
        "--audio-quality", "0",
        "-o", str(output_dir / f"{item_id}.%(ext)s"),
        "--quiet",
        "--no-warnings",
        "--socket-timeout", "30",
    ]

    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]

    if not _ffmpeg_available():
        # Without ffmpeg, remove --audio-format to avoid a confusing error;
        # yt-dlp will pick the best native audio container.
        cmd = [c for c in cmd if c not in ("--audio-format", audio_format)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        downloaded = _find_existing(output_dir, item_id)
        if downloaded:
            time.sleep(random.uniform(*sleep_range))
            return {"item_id": item_id, "query": query, "status": "ok",
                    "path": str(downloaded), "error": ""}
        err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no output file"
        return {"item_id": item_id, "query": query, "status": "fail",
                "path": "", "error": err}
    except subprocess.TimeoutExpired:
        return {"item_id": item_id, "query": query, "status": "fail",
                "path": "", "error": "timeout after 120s"}
    except FileNotFoundError:
        return {"item_id": item_id, "query": query, "status": "fail",
                "path": "", "error": "yt-dlp not found — install with: pip install yt-dlp"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download audio for catalog items via yt-dlp YouTube search."
    )
    parser.add_argument(
        "--metadata", required=True,
        help="Path to backbone metadata.json (Spotify format).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for audio files.",
    )
    parser.add_argument(
        "--ids-file",
        help="Path to a backbone playlist txt file; only download items in this file. "
             "If omitted, download all items in metadata.json.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of items to download (useful for testing).",
    )
    parser.add_argument(
        "--format", default="wav", choices=["wav", "mp3", "flac", "m4a"],
        help="Audio format (default: wav). Requires ffmpeg for wav/mp3/flac.",
    )
    parser.add_argument(
        "--sleep", type=float, default=1.5,
        help="Mean sleep in seconds between downloads (default: 1.5). "
             "Actual sleep = uniform(sleep * 0.5, sleep * 1.5).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip items whose output file already exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be downloaded without actually downloading.",
    )
    parser.add_argument(
        "--cookies-from-browser", default=None, metavar="BROWSER",
        help="Pass cookies from a browser to yt-dlp (e.g. 'chrome'). "
             "Useful for age-restricted videos. Requires the browser to be closed or use a profile.",
    )
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path    = output_dir / "download_log.jsonl"
    failed_path = output_dir / "failed.txt"

    # Load metadata
    print(f"[download] Loading metadata from {args.metadata} ...")
    catalog = parse_metadata_json(args.metadata)
    print(f"[download] {len(catalog)} items in catalog.")

    # Determine which item IDs to process
    if args.ids_file:
        target_ids = load_ids_from_playlist_file(args.ids_file)
        target_ids = [i for i in target_ids if i in catalog]
        print(f"[download] {len(target_ids)} items from --ids-file (after filtering to catalog).")
    else:
        target_ids = sorted(catalog.keys(), key=lambda x: int(x) if x.isdigit() else x)

    if args.limit:
        target_ids = target_ids[: args.limit]
        print(f"[download] Limited to {len(target_ids)} items via --limit.")

    sleep_range = (args.sleep * 0.5, args.sleep * 1.5)

    # Run
    n_ok = n_skip = n_fail = 0
    failed_ids: list[str] = []

    print(f"[download] Output dir : {output_dir}")
    print(f"[download] Format     : {args.format}")
    print(f"[download] Dry run    : {args.dry_run}")
    print(f"[download] Starting download of {len(target_ids)} items ...\n")

    with open(log_path, "a", encoding="utf-8") as log_f:
        for i, item_id in enumerate(target_ids, 1):
            item_info = catalog[item_id]
            entry = download_one(
                item_id=item_id,
                item_info=item_info,
                output_dir=output_dir,
                audio_format=args.format,
                sleep_range=sleep_range,
                dry_run=args.dry_run,
                cookies_from_browser=args.cookies_from_browser,
            )
            log_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log_f.flush()

            status = entry["status"]
            if status == "ok":
                n_ok += 1
                print(f"  [{i:>6}/{len(target_ids)}] OK     {item_id:>8}  {entry['query'][:60]}")
            elif status == "skip":
                n_skip += 1
                if i % 500 == 0:   # only print skips periodically
                    print(f"  [{i:>6}/{len(target_ids)}] SKIP   {item_id:>8}")
            elif status == "dry":
                n_ok += 1
                print(f"  [{i:>6}/{len(target_ids)}] DRY    {item_id:>8}  {entry['query'][:60]}")
            else:
                n_fail += 1
                failed_ids.append(item_id)
                print(f"  [{i:>6}/{len(target_ids)}] FAIL   {item_id:>8}  {entry['error'][:80]}")

    # Write failed.txt for easy retry
    if failed_ids:
        with open(failed_path, "w", encoding="utf-8") as f:
            f.write("\n".join(failed_ids) + "\n")
        print(f"\n[download] {len(failed_ids)} failures written to {failed_path}")

    print(f"\n[download] Done.")
    print(f"           OK:   {n_ok}")
    print(f"           Skip: {n_skip}")
    print(f"           Fail: {n_fail}")
    print(f"           Log:  {log_path}")


if __name__ == "__main__":
    main()
