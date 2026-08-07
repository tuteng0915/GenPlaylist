"""Tests that WP-A retrieval evaluation uses the frozen 15->5 split."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "build_recall_eval", HERE / "build_recall_eval.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_split_playlist_is_fixed_first_twenty():
    references, targets = module.split_playlist([str(index) for index in range(25)])
    assert references == [str(index) for index in range(15)]
    assert targets == [str(index) for index in range(15, 20)]


def test_loader_excludes_short_rows_and_strips_playlist_id():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "test.txt"
        long_items = [str(index) for index in range(25)]
        short_items = [str(index) for index in range(19)]
        path.write_text(
            "long," + ",".join(long_items) + "\n" +
            "short," + ",".join(short_items) + "\n",
            encoding="utf-8",
        )
        playlists = module.load_test_playlists(path, set(long_items))
        assert playlists == [long_items[:20]]


def test_loader_merges_sources_in_order():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        items_a = [str(index) for index in range(20)]
        items_b = [str(index) for index in range(20, 40)]
        val_path = root / "val.txt"
        test_path = root / "test.txt"
        val_path.write_text("v," + ",".join(items_a) + "\n", encoding="utf-8")
        test_path.write_text("t," + ",".join(items_b) + "\n", encoding="utf-8")

        playlists = module.load_test_playlists(
            [str(val_path), str(test_path)], set(items_a + items_b))
        assert playlists == [items_a, items_b]


def test_item_id_to_row_mapping_is_loaded_in_row_order():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "item_id_to_row.json"
        path.write_text('{"song-b": 1, "song-a": 0}', encoding="utf-8")
        assert module.load_row_ids(path) == ["song-a", "song-b"]


def test_encoder_alignment_rejects_partial_catalog():
    import numpy as np

    try:
        module.validate_encoder_alignment(
            "fixture", np.ones((2, 4), dtype=np.float32),
            ["a", "b"], {"a", "b", "c"})
    except ValueError as exc:
        assert "complete frozen catalog" in str(exc)
    else:
        raise AssertionError("Expected partial-catalog embeddings to be rejected")


def test_mrr_uses_rank_beyond_top_fifty():
    ranked = [f"wrong-{index}" for index in range(75)] + ["target"]
    assert module.reciprocal_rank(ranked, {"target"}) == 1 / 76


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
