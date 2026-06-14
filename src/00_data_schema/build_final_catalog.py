#!/usr/bin/env python3
"""scripts/build_final_catalog.py — Cross-validate all data assets and produce catalog.json.

For each item in the target subset, checks:
  - metadata exists in metadata.json
  - audio file exists in --audio-dir
  - lyrics file exists in --lyrics-dir

Writes:
  data/dataset/catalog.json        — full per-item record (all items, with flags)
  data/dataset/complete_ids.txt    — item IDs with ALL three assets
  data/dataset/stats.json          — coverage summary

Usage
-----
python scripts/build_final_catalog.py \\
    --ids-file  data/playlists/r3_10_30_freq3/item_ids.txt \\
    --metadata  /path/to/metadata.json \\
    --audio-dir data/audio/spotify/ \\
    --lyrics-dir data/lyrics/spotify/ \\
    --output    data/dataset/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".webm", ".opus", ".ogg"}
_META_RE = re.compile(r"^'(.+)'\s+by\s+(.+?)\s+in\s+album'(.+)'$")


def _find_audio(audio_dir: Path, item_id: str) -> str | None:
    for p in audio_dir.glob(f"{item_id}.*"):
        if p.suffix.lower() in _AUDIO_EXTS:
            return str(p)
    return None


def _parse_meta(raw: str) -> dict:
    m = _META_RE.match(raw.strip())
    if m:
        return {"title": m.group(1), "artist": m.group(2), "album": m.group(3)}
    return {"title": raw.strip(), "artist": "", "album": ""}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-file",   required=True,
                        help="One item_id per line (e.g. data/playlists/r3_10_30_freq3/item_ids.txt)")
    parser.add_argument("--metadata",   required=True,
                        help="Path to metadata.json from backbone")
    parser.add_argument("--audio-dir",  required=True,
                        help="Directory containing downloaded audio files")
    parser.add_argument("--lyrics-dir", required=True,
                        help="Directory containing fetched lyrics .txt files")
    parser.add_argument("--output",     default="data/dataset/",
                        help="Output directory (default: data/dataset/)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_dir  = Path(args.audio_dir)
    lyrics_dir = Path(args.lyrics_dir)

    # Load target IDs
    ids = [l.strip() for l in open(args.ids_file) if l.strip()]
    print(f"[catalog] Target items : {len(ids):,}")

    # Load metadata
    print(f"[catalog] Loading metadata ...")
    raw_meta: dict = json.load(open(args.metadata))
    print(f"[catalog] Metadata entries: {len(raw_meta):,}")

    # Build catalog
    catalog: list[dict] = []
    n_audio = n_lyrics = n_meta = n_complete = 0

    for item_id in ids:
        has_meta   = item_id in raw_meta
        audio_path = _find_audio(audio_dir, item_id) if has_meta else None
        lyrics_path = str(lyrics_dir / f"{item_id}.txt")
        has_lyrics  = Path(lyrics_path).is_file()
        has_audio   = audio_path is not None

        if has_meta:
            n_meta += 1
            parsed = _parse_meta(raw_meta[item_id])
        else:
            parsed = {"title": "", "artist": "", "album": ""}

        if has_audio:
            n_audio += 1
        if has_lyrics:
            n_lyrics += 1
        if has_meta and has_audio and has_lyrics:
            n_complete += 1

        catalog.append({
            "item_id":     item_id,
            "title":       parsed["title"],
            "artist":      parsed["artist"],
            "album":       parsed["album"],
            "has_meta":    has_meta,
            "has_audio":   has_audio,
            "has_lyrics":  has_lyrics,
            "complete":    has_meta and has_audio and has_lyrics,
            "audio_path":  audio_path or "",
            "lyrics_path": lyrics_path if has_lyrics else "",
        })

    total = len(ids)
    print(f"\n[catalog] Asset coverage:")
    print(f"  metadata : {n_meta:>6,} / {total:,}  ({n_meta/total*100:.1f}%)")
    print(f"  audio    : {n_audio:>6,} / {total:,}  ({n_audio/total*100:.1f}%)")
    print(f"  lyrics   : {n_lyrics:>6,} / {total:,}  ({n_lyrics/total*100:.1f}%)")
    print(f"  complete : {n_complete:>6,} / {total:,}  ({n_complete/total*100:.1f}%)")

    # Write catalog.json
    catalog_path = output_dir / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))

    # Write complete_ids.txt
    complete_ids = [e["item_id"] for e in catalog if e["complete"]]
    (output_dir / "complete_ids.txt").write_text("\n".join(complete_ids) + "\n")

    stats = {
        "total_items":    total,
        "has_meta":       n_meta,
        "has_audio":      n_audio,
        "has_lyrics":     n_lyrics,
        "complete":       n_complete,
        "coverage_rate":  round(n_complete / total, 4),
        "sources": {
            "ids_file":    args.ids_file,
            "metadata":    args.metadata,
            "audio_dir":   args.audio_dir,
            "lyrics_dir":  args.lyrics_dir,
        },
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\n[catalog] Written to {output_dir}/")
    print(f"  catalog.json      — {total:,} items")
    print(f"  complete_ids.txt  — {n_complete:,} items with all 3 assets")
    print(f"  stats.json        — coverage summary")
    print(f"\n[catalog] Next: run freeze_dataset.py to build train/val/test splits")
    print(f"  python scripts/freeze_dataset.py \\")
    print(f"      --catalog    {catalog_path} \\")
    print(f"      --playlists  data/playlists/r3_10_30_freq3/playlists.txt \\")
    print(f"      --output     {output_dir}")


if __name__ == "__main__":
    main()
