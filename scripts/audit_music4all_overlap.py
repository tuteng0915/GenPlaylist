#!/usr/bin/env python3
"""Audit Music4All-Onion overlap with the frozen GenPlaylist MPD catalog."""

from __future__ import annotations

import argparse
import bz2
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import unicodedata


WINDOW_ITEMS = 20
VERSION_WORDS = "remaster|live|version|edit|mix|mono|stereo|feat|from"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").casefold()
    text = text.replace("&", " and ")
    text = re.sub(
        r"\b(?:[a-z]\.){2,}[a-z]?\.?",
        lambda match: match.group(0).replace(".", ""),
        text,
    )
    text = re.sub(r"(?<=\w)[.'](?=\w)", "", text)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _relaxed_title(value: object) -> str:
    text = str(value or "")
    text = re.sub(
        rf"[\[(](?:[^\])]*(?:{VERSION_WORDS})[^\])]*)[\])]",
        " ", text, flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"\s+(?:-|–|—)\s+.*(?:{VERSION_WORDS}).*$",
        " ", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(?:feat|ft)\.?\s+.*$", " ", text, flags=re.IGNORECASE)
    return _normalize(text)


def _load_catalog(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        output = {}
        metadata_pattern = re.compile(
            r"'(.+?)'\s+by\s+(.+?)\s+in\s+album'(.+?)'$"
        )
        for item_id, value in raw.items():
            if isinstance(value, dict):
                output[str(item_id)] = dict(value)
                continue
            if isinstance(value, str):
                match = metadata_pattern.match(value)
                if match is None:
                    raise ValueError(
                        f"Cannot parse legacy DDBC metadata for item {item_id}: "
                        f"{value!r}"
                    )
                output[str(item_id)] = {
                    "title": match.group(1),
                    "artist": match.group(2),
                    "album": match.group(3),
                }
                continue
            raise ValueError(
                f"Catalog item {item_id} must be an object or DDBC metadata string"
            )
        return output
    if isinstance(raw, list):
        return {str(value["item_id"]): dict(value) for value in raw}
    raise ValueError("GenPlaylist catalog must be a JSON object or list")


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _unique_key_matches(
    current: dict[str, dict], music4all: list[dict[str, str]], relaxed: bool,
) -> dict[str, str]:
    current_by_key: dict[tuple[str, str], list[str]] = {}
    music4all_by_key: dict[tuple[str, str], list[str]] = {}
    title = _relaxed_title if relaxed else _normalize
    for item_id, item in current.items():
        key = (_normalize(item.get("artist")), title(item.get("title")))
        current_by_key.setdefault(key, []).append(item_id)
    for item in music4all:
        key = (_normalize(item.get("artist")), title(item.get("song")))
        music4all_by_key.setdefault(key, []).append(str(item["id"]))
    return {
        music4all_ids[0]: current_ids[0]
        for key, current_ids in current_by_key.items()
        if len(current_ids) == 1
        and len(music4all_ids := music4all_by_key.get(key, [])) == 1
    }


def _unique_album_matches(
    current: dict[str, dict], music4all: list[dict[str, str]],
) -> dict[str, str]:
    """Resolve remaining artist/title duplicates with a non-empty album key."""
    current_by_key: dict[tuple[str, str, str], list[str]] = {}
    music4all_by_key: dict[tuple[str, str, str], list[str]] = {}
    for item_id, item in current.items():
        album = _normalize(item.get("album"))
        if not album:
            continue
        key = (
            _normalize(item.get("artist")),
            _normalize(item.get("title")),
            album,
        )
        current_by_key.setdefault(key, []).append(item_id)
    for item in music4all:
        album = _normalize(item.get("album_name"))
        if not album:
            continue
        key = (_normalize(item.get("artist")), _normalize(item.get("song")), album)
        music4all_by_key.setdefault(key, []).append(str(item["id"]))
    return {
        music4all_ids[0]: current_ids[0]
        for key, current_ids in current_by_key.items()
        if len(current_ids) == 1
        and len(music4all_ids := music4all_by_key.get(key, [])) == 1
    }


def _build_mapping(
    current: dict[str, dict],
    music4all: list[dict[str, str]],
    metadata: list[dict[str, str]],
) -> tuple[dict[str, str], list[dict], dict]:
    strict_direct = _unique_key_matches(current, music4all, relaxed=False)
    remaining_current = {
        item_id: item for item_id, item in current.items()
        if item_id not in set(strict_direct.values())
    }
    remaining_music4all = [
        item for item in music4all if item["id"] not in strict_direct
    ]
    strict_album = _unique_album_matches(remaining_current, remaining_music4all)
    strict = {**strict_direct, **strict_album}
    remaining_current = {
        item_id: item for item_id, item in remaining_current.items()
        if item_id not in set(strict_album.values())
    }
    remaining_music4all = [
        item for item in remaining_music4all if item["id"] not in strict_album
    ]
    relaxed = _unique_key_matches(
        remaining_current, remaining_music4all, relaxed=True)
    mapping = {**strict, **relaxed}
    metadata_by_id = {str(item["id"]): item for item in metadata}
    information_by_id = {str(item["id"]): item for item in music4all}
    records = []
    for music4all_id, current_id in sorted(mapping.items()):
        info = information_by_id[music4all_id]
        records.append({
            "music4all_id": music4all_id,
            "genplaylist_item_id": current_id,
            "match_type": "strict" if music4all_id in strict else "relaxed-version",
            "match_detail": (
                "artist-title-one-to-one"
                if music4all_id in strict_direct
                else "artist-title-album-one-to-one"
                if music4all_id in strict_album
                else "version-normalized-one-to-one"
            ),
            "artist": info["artist"],
            "title": info["song"],
            "spotify_id": metadata_by_id.get(music4all_id, {}).get("spotify_id", ""),
        })
    stats = {
        "genplaylist_catalog_items": len(current),
        "music4all_catalog_items": len(music4all),
        "strict_one_to_one_matches": len(strict),
        "strict_artist_title_matches": len(strict_direct),
        "strict_album_resolved_matches": len(strict_album),
        "additional_relaxed_version_matches": len(relaxed),
        "accepted_one_to_one_matches": len(mapping),
        "accepted_genplaylist_catalog_fraction": len(mapping) / len(current),
        "music4all_rows_with_spotify_id": sum(
            bool(item.get("spotify_id")) for item in metadata),
    }
    return mapping, records, stats


def _interaction_stats(
    path: Path,
    mapped_ids: set[str],
    event_counts: dict[str, int] | None = None,
) -> dict:
    total_events = 0
    mapped_events = 0
    users: set[str] = set()
    mapped_counts: dict[str, int] = {}
    contiguous_windows = 0
    users_with_contiguous_window: set[str] = set()
    current_user = None
    current_streak = 0
    departed_users: set[str] = set()
    grouped_by_user = True
    timestamp_increases = 0
    timestamp_decreases = 0
    previous_timestamp = None

    with bz2.open(path, mode="rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        required = {"user_id", "track_id", "timestamp"}
        if not required.issubset(header):
            raise ValueError(
                f"Music4All interactions need columns {sorted(required)}, "
                f"got {header}"
            )
        user_column = header.index("user_id")
        track_column = header.index("track_id")
        timestamp_column = header.index("timestamp")
        for line in handle:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != len(header):
                raise ValueError(f"Malformed Music4All interaction row {total_events + 2}")
            total_events += 1
            user_id = fields[user_column]
            track_id = fields[track_column]
            timestamp = fields[timestamp_column]
            users.add(user_id)
            if user_id != current_user:
                if current_user is not None:
                    departed_users.add(current_user)
                if user_id in departed_users:
                    grouped_by_user = False
                current_user = user_id
                current_streak = 0
                previous_timestamp = None
            if previous_timestamp is not None:
                timestamp_increases += timestamp > previous_timestamp
                timestamp_decreases += timestamp < previous_timestamp
            previous_timestamp = timestamp
            if track_id in mapped_ids:
                mapped_events += 1
                if event_counts is not None:
                    event_counts[track_id] = event_counts.get(track_id, 0) + 1
                mapped_counts[user_id] = mapped_counts.get(user_id, 0) + 1
                current_streak += 1
                if current_streak >= WINDOW_ITEMS:
                    contiguous_windows += 1
                    users_with_contiguous_window.add(user_id)
            else:
                current_streak = 0

    thresholds = (20, 40, 60, 100, 200)
    filtered_windows = sum(
        max(0, count - WINDOW_ITEMS + 1) for count in mapped_counts.values())
    return {
        "total_events": total_events,
        "mapped_events": mapped_events,
        "mapped_event_fraction": mapped_events / total_events if total_events else 0.0,
        "total_users": len(users),
        "users_with_mapped_event": len(mapped_counts),
        "users_by_minimum_mapped_events": {
            str(threshold): sum(count >= threshold for count in mapped_counts.values())
            for threshold in thresholds
        },
        "filtered_subsequence_length20_windows": filtered_windows,
        "contiguous_supported_length20_windows": contiguous_windows,
        "users_with_contiguous_supported_window": len(users_with_contiguous_window),
        "rows_grouped_by_user": grouped_by_user,
        "within_user_timestamp_increases": timestamp_increases,
        "within_user_timestamp_decreases": timestamp_decreases,
        "timestamp_order": (
            "descending" if timestamp_decreases and not timestamp_increases
            else "ascending" if timestamp_increases and not timestamp_decreases
            else "constant" if not timestamp_increases and not timestamp_decreases
            else "mixed"
        ),
        "interpretation": {
            "filtered_subsequence": (
                "unmatched Music4All items are omitted before rolling length-20 windows"
            ),
            "contiguous_supported": (
                "an unmatched item breaks the run; only adjacent supported events count"
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-metadata", type=Path, required=True)
    parser.add_argument("--music4all-information", type=Path, required=True)
    parser.add_argument("--music4all-metadata", type=Path, required=True)
    parser.add_argument("--interactions", type=Path)
    parser.add_argument(
        "--catalog-semantic-tokens", type=Path,
        help=(
            "Optional DDBC item-to-token JSON. When supplied, every mapped "
            "GenPlaylist item must have a semantic-token row."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog_path = args.catalog_metadata.expanduser().resolve()
    information_path = args.music4all_information.expanduser().resolve()
    metadata_path = args.music4all_metadata.expanduser().resolve()
    current = _load_catalog(catalog_path)
    music4all = _read_tsv(information_path)
    metadata = _read_tsv(metadata_path)
    mapping, records, overlap = _build_mapping(current, music4all, metadata)
    if args.catalog_semantic_tokens is not None:
        semantic_path = args.catalog_semantic_tokens.expanduser().resolve()
        semantic_tokens = json.loads(semantic_path.read_text(encoding="utf-8"))
        missing = sorted(set(mapping.values()) - {str(key) for key in semantic_tokens})
        if missing:
            raise ValueError(
                f"Catalog semantic tokens are missing {len(missing)} mapped items: "
                f"{missing[:10]}"
            )
        overlap["mapped_items_with_semantic_tokens"] = len(mapping)
        overlap["catalog_semantic_token_items"] = len(semantic_tokens)
    payload = {
        "result_schema": "genplaylist-music4all-overlap-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "matching": {
            "strict": "ASCII-folded case-insensitive artist and title; punctuation removed",
            "relaxed": "strict matching after removing common version/remaster/feature suffixes",
            "ambiguity_rule": "accept only one-to-one keys; never guess among duplicates",
        },
        "overlap": overlap,
    }
    if args.interactions is not None:
        event_counts: dict[str, int] = {}
        payload["interactions"] = _interaction_stats(
            args.interactions.expanduser().resolve(), set(mapping), event_counts)
        match_type_by_id = {
            record["music4all_id"]: record["match_type"] for record in records
        }
        event_summary = {
            match_type: {
                "items_with_events": sum(
                    event_counts.get(item_id, 0) > 0
                    for item_id, current_type in match_type_by_id.items()
                    if current_type == match_type
                ),
                "events": sum(
                    event_counts.get(item_id, 0)
                    for item_id, current_type in match_type_by_id.items()
                    if current_type == match_type
                ),
            }
            for match_type in ("strict", "relaxed-version")
        }
        payload["interactions"]["mapped_by_match_type"] = event_summary
        for record in records:
            record["event_count"] = event_counts.get(record["music4all_id"], 0)
    output_path = args.output.expanduser().resolve()
    _atomic_json(output_path, payload)
    mapping_path = args.mapping_output.expanduser().resolve()
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = mapping_path.with_suffix(mapping_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, mapping_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
