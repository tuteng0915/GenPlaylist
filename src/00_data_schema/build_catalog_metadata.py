#!/usr/bin/env python3
"""src/00_data_schema/build_catalog_metadata.py — Build catalog_metadata.json for pipeline use.

Converts the frozen catalog.json (from scripts/build_final_catalog.py) into
a flat dict keyed by item_id, with fields matching CatalogItem:
  title, artist, album, audio_path, lyrics_path, lyric_excerpt

Optionally merges in Spotify audio features from data/tags/spotify_tags.json
(genre, mood, tempo, key) when available.

Writes:
  data/dataset/catalog_metadata.json   — {item_id: {...}} used by normalizer, verbalization

Usage
-----
python src/00_data_schema/build_catalog_metadata.py \\
    --catalog    data/dataset/catalog.json \\
    --output     data/dataset/catalog_metadata.json \\
    --tags       data/tags/spotify_tags.json        # optional
    --lyrics-dir data/lyrics/spotify/               # optional, for excerpt
    --excerpt-len 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow importing schema from this directory
sys.path.insert(0, os.path.dirname(__file__))
from schema import CatalogItem  # noqa: F401 — imported for field reference


def _load_lyric_excerpt(lyrics_path: str, max_chars: int) -> str:
    if not lyrics_path or not Path(lyrics_path).is_file():
        return ""
    try:
        text = Path(lyrics_path).read_text(encoding="utf-8").strip()
        # Take first non-empty lines up to max_chars
        lines = [l for l in text.splitlines() if l.strip()]
        excerpt = ""
        for line in lines:
            if len(excerpt) + len(line) + 1 > max_chars:
                break
            excerpt = (excerpt + "\n" + line).strip()
        return excerpt
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog",     required=True,
                        help="data/dataset/catalog.json from build_final_catalog.py")
    parser.add_argument("--output",      default="data/dataset/catalog_metadata.json")
    parser.add_argument("--tags",        default=None,
                        help="data/tags/audio_features.json (optional)")
    parser.add_argument("--mb-tags",     default=None,
                        help="data/tags/musicbrainz_tags.json (optional)")
    parser.add_argument("--lyrics-dir",  default=None,
                        help="Directory with {item_id}.txt lyrics files, for excerpt extraction")
    parser.add_argument("--excerpt-len", type=int, default=200,
                        help="Max chars for lyric_excerpt (default: 200)")
    parser.add_argument("--complete-only", action="store_true",
                        help="Only include items where complete=True (default: include all)")
    args = parser.parse_args()

    catalog: list[dict] = json.load(open(args.catalog))
    print(f"[meta] Loaded {len(catalog):,} catalog entries")

    tags: dict = {}
    if args.tags and Path(args.tags).is_file():
        tags = json.load(open(args.tags))
        print(f"[meta] Loaded audio tags for {len(tags):,} items")
    else:
        print("[meta] No audio tags file — tempo/key/mood will be empty strings")

    mb_tags: dict = {}
    if args.mb_tags and Path(args.mb_tags).is_file():
        mb_tags = json.load(open(args.mb_tags))
        print(f"[meta] Loaded MusicBrainz tags for {len(mb_tags):,} items")

    lyrics_dir = Path(args.lyrics_dir) if args.lyrics_dir else None

    out: dict = {}
    n_complete = n_tags = n_excerpt = 0

    for entry in catalog:
        if args.complete_only and not entry.get("complete"):
            continue

        item_id = entry["item_id"]
        tag = tags.get(item_id, {})
        mb = mb_tags.get(item_id, {})

        # Resolve lyrics path: prefer catalog's recorded path, fall back to lyrics_dir
        lp = entry.get("lyrics_path", "")
        if not lp and lyrics_dir:
            candidate = lyrics_dir / f"{item_id}.txt"
            if candidate.is_file():
                lp = str(candidate)

        excerpt = ""
        if lp:
            excerpt = _load_lyric_excerpt(lp, args.excerpt_len)
            if excerpt:
                n_excerpt += 1

        if tag:
            n_tags += 1

        out[item_id] = {
            "item_id":      item_id,
            "title":        entry.get("title", ""),
            "artist":       entry.get("artist", ""),
            "album":        entry.get("album", ""),
            "audio_path":   entry.get("audio_path", ""),
            "lyrics_path":  lp,
            "lyric_excerpt": excerpt,
            # Tags — populated if spotify_tags.json is available
            "genre":        mb.get("genres", [tag.get("genre", "")])[0] if mb.get("genres") else tag.get("genre", ""),
            "tags":         mb.get("genres", []),
            "mood":         tag.get("mood", ""),
            "tempo":        tag.get("tempo", ""),
            "key":          tag.get("key", ""),
            "language":     tag.get("language", ""),
        }
        n_complete += 1

    print(f"[meta] Items written      : {n_complete:,}")
    print(f"[meta] With tags          : {n_tags:,}")
    print(f"[meta] With lyric excerpt : {n_excerpt:,}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[meta] Written to {out_path}")
    print(f"\n[meta] Load in pipeline:")
    print(f"  import json")
    print(f"  catalog_metadata = json.load(open('{out_path}'))")


if __name__ == "__main__":
    main()
