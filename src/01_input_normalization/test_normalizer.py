"""Small deterministic tests for WP-A normalization."""

from __future__ import annotations

import numpy as np

from normalizer import normalize, retrieve_by_embedding
from schema import CatalogItem


class EchoEncoder:
    def encode(self, text, normalize_embeddings=True):
        return np.asarray([1.0, 0.0], dtype=np.float32)


ITEMS = [CatalogItem(str(i), title=f"song {i}") for i in (10, 30, 90, 120)]
EMBS = np.asarray([[1, 0], [0.9, 0.1], [0, 1], [-1, 0]], dtype=np.float32)


def test_song_dedup_and_padding():
    result = normalize(["10", "10", "bad"], ITEMS, EMBS, K=3)
    assert result.item_ids == ["10", "30", "90"]
    assert result.source == "padded"


def test_text_query():
    result = normalize("bright guitar", ITEMS, EMBS, K=2, text_encoder=EchoEncoder())
    assert result.item_ids == ["10", "30"]
    assert result.source == "text_only"


def test_hybrid_without_text_still_fills():
    result = normalize({"item_ids": ["90"]}, ITEMS, EMBS, K=3)
    assert result.item_ids[0] == "90"
    assert len(result.item_ids) == 3


def test_zero_query_rejected():
    try:
        retrieve_by_embedding(np.zeros(2), EMBS, [item.item_id for item in ITEMS], 2)
    except ValueError as exc:
        assert "non-zero" in str(exc)
    else:
        raise AssertionError("Expected zero query to fail")


def test_reference_set_requires_at_least_two_items():
    try:
        normalize(["10"], ITEMS, EMBS, K=1)
    except ValueError as exc:
        assert "K >= 2" in str(exc)
    else:
        raise AssertionError("Expected K=1 to be rejected")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
