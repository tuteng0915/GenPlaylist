"""Tests for sparse-ID split parsing without requiring Hugging Face datasets."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("backbone_dataset", HERE / "dataset.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_split_parser_preserves_sparse_ids():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "train.txt"
        path.write_text("p1, 18996, 48262\n", encoding="utf-8")
        assert module.read_split_file(path) == [("p1", ["18996", "48262"])]


def test_short_record_rejected():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "bad.txt"
        path.write_text("p1, 18996\n", encoding="utf-8")
        try:
            module.read_split_file(path)
        except ValueError as exc:
            assert "at least two" in str(exc)
        else:
            raise AssertionError("Expected short split record to fail")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
