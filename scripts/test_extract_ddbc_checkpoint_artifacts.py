"""Dependency-light checks for DDBC embedded-artifact extraction."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "extract_ddbc_checkpoint_artifacts", HERE / "extract_ddbc_checkpoint_artifacts.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from shared.schema import CatalogItem  # noqa: E402


def make_embedded():
    features = np.arange(4 * 64, dtype=np.float32).reshape(4, 64)
    codebook = np.arange(768 * 64, dtype=np.float32).reshape(768, 64)
    tokens = {
        str(index): [1 + index, 257 + index, 513 + index, 769 + index]
        for index in range(4)
    }
    return SimpleNamespace(feature=features, weight=codebook, token=tokens)


def test_selects_catalog_order_and_writes_manifest():
    embedded = make_embedded()
    items = [CatalogItem("2"), CatalogItem("0")]
    selected, codebook, semantic, mapping = module.build_artifact_arrays(
        embedded, items, confirm_dense_item_ids=True)
    assert np.array_equal(selected[0], embedded.feature[2])
    assert np.array_equal(selected[1], embedded.feature[0])
    assert codebook.shape == (768, 64)
    assert list(semantic) == ["2", "0"]
    assert mapping == {"2": 0, "0": 1}

    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir)
        manifest_path = module.write_artifacts(
            output, selected, codebook, semantic, mapping,
            source_manifest={"checkpoint": "fixture.ckpt"})
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["catalog_items"] == 2
        assert manifest["selection_mode"] == "checkpoint-validated-dense-item-ids"
        assert np.load(output / "catalog_item_embeddings.npy").shape == (2, 64)


def test_requires_explicit_dense_id_confirmation():
    try:
        module.build_artifact_arrays(
            make_embedded(), [CatalogItem("0")], confirm_dense_item_ids=False)
    except ValueError as exc:
        assert "confirm-dense-item-ids" in str(exc)
    else:
        raise AssertionError("Expected dense item-ID selection to require confirmation")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
