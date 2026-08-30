#!/usr/bin/env python3
"""Dependency-free tests for Music4All-to-DDBC catalog materialization."""

from __future__ import annotations

import bz2
import importlib.util
from pathlib import Path
import tempfile


SCRIPT = Path(__file__).with_name("build_music4all_ddbc_catalog.py")
SPEC = importlib.util.spec_from_file_location("build_music4all_ddbc_catalog", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_catalog_keeps_only_observed_strict_rows() -> None:
    rows = [
        {
            "music4all_id": "m1", "genplaylist_item_id": "7",
            "match_type": "strict", "event_count": "3",
        },
    ]
    metadata, catalog, observed = MODULE.build_catalog(
        rows,
        {"7": "'Song' by Artist in album'Album'"},
        {"7": [1, 2, 3, 4]},
        {"m1": {"id": "m1", "artist": "Artist", "song": "Song",
                "album_name": "Album"}},
        {"m1": {"id": "m1", "spotify_id": "spotify", "tempo": "120",
                "key": "0", "mode": "1", "duration_ms": "1000",
                "danceability": "0.5", "energy": "0.6", "valence": "0.7"}},
        {"m1": ["rock", "alternative"]},
    )
    assert list(metadata) == ["7"]
    assert catalog[0]["source_music4all_id"] == "m1"
    assert catalog[0]["event_count"] == 3
    assert catalog[0]["key"] == "C major"
    assert catalog[0]["tags"] == ["rock", "alternative"]
    assert observed == rows


def test_mapping_filter_rejects_unobserved_and_relaxed_rows() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "mapping.csv"
        path.write_text(
            "music4all_id,genplaylist_item_id,match_type,event_count\n"
            "m1,1,strict,4\n"
            "m2,2,strict,0\n"
            "m3,3,relaxed-version,5\n",
            encoding="utf-8",
        )
        assert [
            row["genplaylist_item_id"]
            for row in MODULE._load_mapping(path, include_relaxed=False)
        ] == ["1"]
        assert [
            row["genplaylist_item_id"]
            for row in MODULE._load_mapping(path, include_relaxed=True)
        ] == ["1", "3"]


def test_weighted_tags_are_ranked_deterministically() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tags.tsv.bz2"
        with bz2.open(path, "wt", encoding="utf-8") as handle:
            handle.write("id\t(tag, weight)\n")
            handle.write("m1\t{'zeta': 50, 'alpha': 50, 'rock': 100}\n")
            handle.write("m2\t{'ignored': 100}\n")
        assert MODULE._read_weighted_tags(path, {"m1"}, max_tags=2) == {
            "m1": ["rock", "alpha"],
        }


if __name__ == "__main__":
    test_catalog_keeps_only_observed_strict_rows()
    test_mapping_filter_rejects_unobserved_and_relaxed_rows()
    test_weighted_tags_are_ranked_deterministically()
    print("Music4All DDBC catalog tests passed")
