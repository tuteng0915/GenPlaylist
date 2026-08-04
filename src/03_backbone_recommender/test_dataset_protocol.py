"""Dependency-light tests for the frozen 16-item/15->5 dataset protocol."""

from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("wp_c_dataset", HERE / "dataset.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
AbstractDataset = module.AbstractDataset


def make_dataset(records):
    dataset = object.__new__(AbstractDataset)
    dataset.config = {
        "protocol": {
            "min_reference_items": 2,
            "eval_reference_items": 15,
            "eval_target_items": 5,
        }
    }
    dataset._records_for_split = lambda split: records[split]
    return dataset


def test_training_expands_every_chronological_next_item():
    items = [str(index) for index in range(20)]
    dataset = make_dataset({"train": [("p", items)]})
    result = dataset.convert_txt_to_dataset("train", 0.0, 16, if_train=True)

    assert len(result["item_seq"]) == 18
    assert result["item_seq"][0] == ["0", "1", "2"]
    assert result["item_seq"][13] == items[:16]
    assert result["item_seq"][-1] == items[4:20]
    assert all(3 <= len(row) <= 16 for row in result["item_seq"])


def test_training_rejects_order_destroying_swap_augmentation():
    dataset = make_dataset({"train": [("p", ["0", "1", "2"])]})
    try:
        dataset.convert_txt_to_dataset("train", 0.4, 16, if_train=True)
    except ValueError as exc:
        assert "swap_ratio must be 0" in str(exc)
    else:
        raise AssertionError("Expected non-zero swap augmentation to be rejected")


def test_test_split_keeps_only_first_twenty_items():
    records = {
        "test": [
            ("short", [str(index) for index in range(19)]),
            ("long", [str(index) for index in range(25)]),
        ]
    }
    dataset = make_dataset(records)
    result = dataset.convert_txt_to_dataset("test", 0.0, 16)

    assert result["bundle"] == ["long"]
    assert result["item_seq"] == [[str(index) for index in range(20)]]


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
