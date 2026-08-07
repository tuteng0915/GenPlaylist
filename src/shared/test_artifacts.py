"""CPU-only contract tests for shared catalog artifacts."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.artifacts import build_item_id_to_row, load_catalog_artifacts, validate_catalog_alignment
from shared.schema import CLHE_EMB_DIM, RQ_CODEBOOK_SIZE, RQ_N_CODEBOOKS, CatalogItem


def test_sparse_ids_are_mapped_not_cast() -> None:
    items = [CatalogItem(item_id="18996"), CatalogItem(item_id="48262")]
    mapping = build_item_id_to_row(items)
    assert mapping == {"18996": 0, "48262": 1}
    assert items[0].feature_index == -1


def test_alignment_assigns_feature_rows() -> None:
    items = [CatalogItem(item_id="18996"), CatalogItem(item_id="48262")]
    mapping = build_item_id_to_row(items)
    embeddings = np.zeros((2, CLHE_EMB_DIM), dtype=np.float32)
    validate_catalog_alignment(items, embeddings, mapping)
    assert [item.feature_index for item in items] == [0, 1]


def test_codebook_is_rejected_as_catalog_embeddings() -> None:
    items = [CatalogItem(item_id=str(i)) for i in range(10)]
    mapping = build_item_id_to_row(items)
    codebook = np.zeros((RQ_N_CODEBOOKS * RQ_CODEBOOK_SIZE, CLHE_EMB_DIM), dtype=np.float32)
    try:
        validate_catalog_alignment(items, codebook, mapping)
        raise AssertionError("RVQ codebook should not validate as catalog embeddings")
    except ValueError as exc:
        assert "codebook" in str(exc).lower()


def test_dict_catalog_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        catalog_path = root / "catalog.json"
        mapping_path = root / "item_id_to_row.json"
        embeddings_path = root / "embeddings.npy"
        catalog_path.write_text(json.dumps({
            "18996": {"item_id": "wrong-value-is-overridden", "title": "Badfish"},
            "48262": {"title": "Santeria"},
        }))
        mapping_path.write_text(json.dumps({"18996": 0, "48262": 1}))
        np.save(embeddings_path, np.zeros((2, CLHE_EMB_DIM), dtype=np.float32))

        artifacts = load_catalog_artifacts(catalog_path, embeddings_path, mapping_path)
        assert [item.item_id for item in artifacts.items] == ["18996", "48262"]
        assert [item.feature_index for item in artifacts.items] == [0, 1]


if __name__ == "__main__":
    tests = [
        test_sparse_ids_are_mapped_not_cast,
        test_alignment_assigns_feature_rows,
        test_codebook_is_rejected_as_catalog_embeddings,
        test_dict_catalog_round_trip,
    ]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
