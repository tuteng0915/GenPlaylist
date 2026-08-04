"""Dependency-light checks for deterministic production cue assignment."""

from __future__ import annotations

import numpy as np

import cue_assign
from schema import CatalogItem


def _fixture():
    catalog = [CatalogItem(item_id="song-1", title="Fixture")]
    vocab = ["<unk>", "one", "two", "three", "four"]
    cue_embeddings = np.asarray([
        [1.0, 0.0],
        [0.8, 0.6],
        [0.8, -0.6],
        [0.0, 1.0],
    ], dtype=np.float32)

    def embed_fn(texts):
        return np.repeat(
            np.asarray([[1.0, 0.0]], dtype=np.float32), len(texts), axis=0)

    return catalog, vocab, cue_embeddings, embed_fn


def test_relevance_order_and_tie_break_are_deterministic():
    catalog, vocab, cue_embeddings, embed_fn = _fixture()
    mapping, scores = cue_assign.assign_all(
        catalog, {}, vocab, cue_embeddings, n_cues=3, candidate_k=2,
        strategy="relevance", embed_fn=embed_fn, return_scores=True,
        verbose=False)
    assert mapping["song-1"].cue_ids == [1, 2, 3]
    assert np.allclose(scores["song-1"], [1.0, 0.8, 0.8])


def test_short_vocab_is_padded_with_trailing_unk():
    catalog, vocab, cue_embeddings, embed_fn = _fixture()
    mapping, scores = cue_assign.assign_all(
        catalog, {}, vocab, cue_embeddings, n_cues=6, candidate_k=2,
        strategy="relevance", embed_fn=embed_fn, return_scores=True,
        verbose=False)
    assert mapping["song-1"].cue_ids == [1, 2, 3, 4, 0, 0]
    assert scores["song-1"][-2:] == [None, None]


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
