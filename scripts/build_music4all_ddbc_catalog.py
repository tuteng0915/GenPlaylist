#!/usr/bin/env python3
"""Build an event-supported Music4All catalog in the original DDBC ID space.

The overlap audit maps Music4All item IDs to the dense numeric item IDs used by
the official DDBC Spotify checkpoint.  This script keeps only mapped tracks
that actually occur in the timestamp table, preserves the DDBC IDs required by
the embedded CLHE/RVQ tokenizer, and enriches their catalog rows with
Music4All/Spotify metadata.

Strict one-to-one matches are the default.  Relaxed version matches require an
explicit flag and are intended for sensitivity analysis rather than the main
dataset.
"""

from __future__ import annotations

import argparse
import ast
import bz2
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


SCHEMA = "genplaylist-music4all-ddbc-catalog-v1"
PITCH_CLASSES = (
    "C", "C-sharp/D-flat", "D", "D-sharp/E-flat", "E", "F",
    "F-sharp/G-flat", "G", "G-sharp/A-flat", "A", "A-sharp/B-flat", "B",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_tsv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    output = {str(row["id"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"Duplicate Music4All IDs in {path}")
    return output


def _read_weighted_tags(
    path: Path, selected_ids: set[str], max_tags: int,
) -> dict[str, list[str]]:
    if max_tags < 1:
        raise ValueError("max-tags must be positive")
    output: dict[str, list[str]] = {}
    handle = (
        bz2.open(path, mode="rt", encoding="utf-8", newline="")
        if path.suffix == ".bz2"
        else path.open(mode="rt", encoding="utf-8", newline="")
    )
    with handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"id", "(tag, weight)"}.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"Unexpected weighted-tag columns in {path}")
        for row in reader:
            item_id = row["id"]
            if item_id not in selected_ids:
                continue
            try:
                raw = ast.literal_eval(row["(tag, weight)"])
            except (SyntaxError, ValueError) as error:
                raise ValueError(f"Malformed tag dictionary for {item_id}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"Tag payload for {item_id} is not a dictionary")
            ranked = sorted(
                ((str(tag).strip(), float(weight)) for tag, weight in raw.items()),
                key=lambda value: (-value[1], value[0].casefold()),
            )
            output[item_id] = [tag for tag, _ in ranked[:max_tags] if tag]
    return output


def _load_mapping(path: Path, include_relaxed: bool) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "music4all_id", "genplaylist_item_id", "match_type", "event_count",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Mapping is missing columns {sorted(required)}")
        rows = [dict(row) for row in reader]
    selected = []
    source_ids: set[str] = set()
    target_ids: set[str] = set()
    for row in rows:
        if row["match_type"] != "strict" and not include_relaxed:
            continue
        try:
            event_count = int(row["event_count"])
        except ValueError as error:
            raise ValueError(
                f"Invalid event_count for {row['music4all_id']}: "
                f"{row['event_count']!r}"
            ) from error
        if event_count <= 0:
            continue
        source = row["music4all_id"]
        target = row["genplaylist_item_id"]
        if source in source_ids or target in target_ids:
            raise ValueError(
                f"Observed mapping is not one-to-one: source={source}, target={target}"
            )
        if not target.isdigit():
            raise ValueError(f"DDBC item ID must be numeric, got {target!r}")
        row["event_count"] = str(event_count)
        source_ids.add(source)
        target_ids.add(target)
        selected.append(row)
    if not selected:
        raise ValueError("No event-supported mapping rows")
    return sorted(selected, key=lambda row: int(row["genplaylist_item_id"]))


def _key_label(metadata: dict[str, str]) -> str:
    try:
        key = int(float(metadata["key"]))
        mode = int(float(metadata["mode"]))
    except (KeyError, TypeError, ValueError):
        return ""
    if not 0 <= key < len(PITCH_CLASSES) or mode not in {0, 1}:
        return ""
    return f"{PITCH_CLASSES[key]} {'major' if mode else 'minor'}"


def _optional_float(value: str | None) -> float | None:
    try:
        parsed = float(value) if value not in {None, ""} else None
    except ValueError:
        return None
    return parsed


def build_catalog(
    mapping_rows: list[dict[str, str]],
    ddbc_metadata: dict[str, str],
    ddbc_tokens: dict[str, list[int]],
    information: dict[str, dict[str, str]],
    metadata: dict[str, dict[str, str]],
    tags: dict[str, list[str]] | None = None,
) -> tuple[dict[str, dict], list[dict], list[dict[str, str]]]:
    tags = tags or {}
    catalog_metadata: dict[str, dict] = {}
    catalog: list[dict] = []
    for row in mapping_rows:
        source = row["music4all_id"]
        target = row["genplaylist_item_id"]
        if target not in ddbc_metadata:
            raise ValueError(f"DDBC metadata is missing mapped item {target}")
        if target not in ddbc_tokens:
            raise ValueError(f"DDBC semantic tokens are missing mapped item {target}")
        if source not in information or source not in metadata:
            raise ValueError(f"Music4All metadata is missing mapped item {source}")
        info = information[source]
        meta = metadata[source]
        tempo = _optional_float(meta.get("tempo"))
        common = {
            "item_id": target,
            "title": info.get("song", ""),
            "artist": info.get("artist", ""),
            "album": info.get("album_name", ""),
            "genre": "",
            "tags": tags.get(source, []),
            "mood": "",
            "tempo": tempo,
            "key": _key_label(meta),
            "language": "",
            "lyric_excerpt": "",
            "audio_path": "",
            "source_music4all_id": source,
            "spotify_id": meta.get("spotify_id", ""),
            "event_count": int(row["event_count"]),
            "match_type": row["match_type"],
            "release": meta.get("release", ""),
            "duration_ms": _optional_float(meta.get("duration_ms")),
            "danceability": _optional_float(meta.get("danceability")),
            "energy": _optional_float(meta.get("energy")),
            "valence": _optional_float(meta.get("valence")),
        }
        catalog_metadata[target] = common
        catalog.append({**common, "complete": True})
    return catalog_metadata, catalog, mapping_rows


def _atomic_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--ddbc-metadata", type=Path, required=True)
    parser.add_argument("--ddbc-semantic-tokens", type=Path, required=True)
    parser.add_argument("--music4all-information", type=Path, required=True)
    parser.add_argument("--music4all-metadata", type=Path, required=True)
    parser.add_argument("--music4all-tags", type=Path)
    parser.add_argument("--music4all-lyrics-dir", type=Path)
    parser.add_argument("--max-tags", type=int, default=16)
    parser.add_argument("--lyric-excerpt-chars", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-relaxed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = {
        "mapping": args.mapping.expanduser().resolve(),
        "ddbc_metadata": args.ddbc_metadata.expanduser().resolve(),
        "ddbc_semantic_tokens": args.ddbc_semantic_tokens.expanduser().resolve(),
        "music4all_information": args.music4all_information.expanduser().resolve(),
        "music4all_metadata": args.music4all_metadata.expanduser().resolve(),
    }
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    tags_path = (
        args.music4all_tags.expanduser().resolve()
        if args.music4all_tags is not None else None
    )
    if tags_path is not None and not tags_path.is_file():
        raise FileNotFoundError(tags_path)
    lyrics_dir = (
        args.music4all_lyrics_dir.expanduser().resolve()
        if args.music4all_lyrics_dir is not None else None
    )
    if lyrics_dir is not None and not lyrics_dir.is_dir():
        raise FileNotFoundError(lyrics_dir)
    if args.lyric_excerpt_chars < 0:
        raise ValueError("lyric-excerpt-chars must be nonnegative")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; use --overwrite"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent,
    ))
    try:
        mapping_rows = _load_mapping(sources["mapping"], args.include_relaxed)
        ddbc_metadata = {
            str(key): str(value) for key, value in json.loads(
                sources["ddbc_metadata"].read_text(encoding="utf-8")
            ).items()
        }
        ddbc_tokens = {
            str(key): value for key, value in json.loads(
                sources["ddbc_semantic_tokens"].read_text(encoding="utf-8")
            ).items()
        }
        information = _read_tsv(sources["music4all_information"])
        metadata = _read_tsv(sources["music4all_metadata"])
        selected_source_ids = {row["music4all_id"] for row in mapping_rows}
        tags = (
            _read_weighted_tags(tags_path, selected_source_ids, args.max_tags)
            if tags_path is not None else {}
        )
        catalog_metadata, catalog, observed_mapping = build_catalog(
            mapping_rows, ddbc_metadata, ddbc_tokens, information, metadata, tags,
        )
        lyric_count = 0
        if lyrics_dir is not None:
            destination_dir = temporary / "lyrics"
            destination_dir.mkdir()
            for item in catalog:
                source = lyrics_dir / f"{item['source_music4all_id']}.txt"
                if not source.is_file():
                    continue
                destination = destination_dir / f"{item['item_id']}.txt"
                os.link(source, destination)
                final_path = output_dir / "lyrics" / destination.name
                item["lyrics_path"] = str(final_path)
                item["lyric_excerpt"] = source.read_text(
                    encoding="utf-8", errors="ignore"
                ).strip()[:args.lyric_excerpt_chars]
                catalog_metadata[item["item_id"]]["lyrics_path"] = str(final_path)
                catalog_metadata[item["item_id"]]["lyric_excerpt"] = item[
                    "lyric_excerpt"
                ]
                lyric_count += 1
        _atomic_json(temporary / "catalog_metadata.json", catalog_metadata)
        _atomic_json(temporary / "catalog.json", catalog)
        (temporary / "complete_ids.txt").write_text(
            "\n".join(catalog_metadata) + "\n", encoding="utf-8"
        )
        mapping_path = temporary / "item_mapping.csv"
        with mapping_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(observed_mapping[0]))
            writer.writeheader()
            writer.writerows(observed_mapping)
        total_events = sum(item["event_count"] for item in catalog)
        stats = {
            "result_schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "match_policy": (
                "strict-plus-relaxed-version" if args.include_relaxed else "strict-only"
            ),
            "catalog_items": len(catalog),
            "mapped_events": total_events,
            "items_with_weighted_tags": sum(bool(item["tags"]) for item in catalog),
            "items_with_processed_lyrics": lyric_count,
            "source_music4all_items": len({
                item["source_music4all_id"] for item in catalog
            }),
            "ddbc_item_id_range": [
                min(int(item_id) for item_id in catalog_metadata),
                max(int(item_id) for item_id in catalog_metadata),
            ],
            "sources": {
                name: {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for name, path in sources.items()
            },
        }
        for name, path in {
            "music4all_tags": tags_path,
        }.items():
            if path is not None:
                stats["sources"][name] = {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
        _atomic_json(temporary / "stats.json", stats)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
