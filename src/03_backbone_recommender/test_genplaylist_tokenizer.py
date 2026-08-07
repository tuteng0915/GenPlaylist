"""Dependency-light contract tests for the GenPlaylist-v1 tokenizer."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "genplaylist_tokenizer", HERE / "genplaylist_tokenizer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
GenPlaylistTokenizer = module.GenPlaylistTokenizer

from shared.schema import CatalogItem, TOKEN_LAYOUT  # noqa: E402


def make_tokenizer():
    items = [CatalogItem("18996"), CatalogItem("48262"), CatalogItem("73001")]
    semantic = {
        "18996": [1, 257, 513, 769],
        "48262": [256, 512, 768, 842],
        "73001": [2, 258, 514, 770],
    }
    cues = {
        "18996": list(range(16)),
        "48262": list(range(16, 32)),
        "73001": list(range(32, 48)),
    }
    embeddings = np.stack([
        np.zeros(64, dtype=np.float32),
        np.ones(64, dtype=np.float32),
        np.full(64, 2.0, dtype=np.float32),
    ])
    weights = np.arange(768 * 64, dtype=np.float32).reshape(768, 64)
    return GenPlaylistTokenizer(
        semantic, cues, items, embeddings,
        {"18996": 0, "48262": 1, "73001": 2}, weights)


def make_twenty_item_tokenizer():
    item_ids = [str(10000 + index) for index in range(20)]
    items = [CatalogItem(item_id) for item_id in item_ids]
    semantic = {
        item_id: [1 + index, 257 + index, 513 + index, 769 + index]
        for index, item_id in enumerate(item_ids)
    }
    cues = {item_id: list(range(16)) for item_id in item_ids}
    embeddings = np.stack([
        np.full(64, float(index), dtype=np.float32)
        for index in range(20)
    ])
    weights = np.arange(768 * 64, dtype=np.float32).reshape(768, 64)
    tokenizer = GenPlaylistTokenizer(
        semantic, cues, items, embeddings,
        {item_id: index for index, item_id in enumerate(item_ids)}, weights)
    tokenizer.config = {
        "rq_codebook_size": 256,
        "protocol": {
            "eval_reference_items": 15,
            "eval_target_items": 5,
        },
    }
    return tokenizer, item_ids


class FakeRows:
    """Small subset of the HF Dataset interface used by tokenizer tests."""

    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0]) if rows else []

    def filter(self, function):
        return FakeRows([row for row in self.rows if function(row)])

    def map(self, function, **_kwargs):
        return FakeRows([function(row) for row in self.rows])

    def set_format(self, **_kwargs):
        return None

    def __getitem__(self, index):
        return self.rows[index]


def test_encode_layout_and_target_mask():
    tokenizer = make_tokenizer()
    encoded = tokenizer.encode_playlist(
        ["18996", "48262", "73001"], context_items=2)
    assert len(encoded.input_ids) == 1 + 3 * TOKEN_LAYOUT.tokens_per_item + 1
    assert encoded.input_ids[0] == 0
    assert encoded.input_ids[1] == TOKEN_LAYOUT.boi_token
    assert encoded.input_ids[14] == TOKEN_LAYOUT.boi_token
    assert encoded.input_ids[-1] == TOKEN_LAYOUT.eos_token
    assert encoded.target_mask[:27].sum() == 0
    assert encoded.target_mask.sum() == 12
    assert np.allclose(encoded.mu_c, 0.5)
    assert encoded.sigma_c2 > 0.0


def test_decode_item():
    tokenizer = make_tokenizer()
    payload = tokenizer.encode_item("48262")
    generated = tokenizer.decode_item(
        payload, mu_c=np.zeros(64, dtype=np.float32), sigma_c2=1.5)
    assert generated.rvq_codes == (255, 255, 255)
    assert generated.conflict_code == 73
    assert generated.cue_ids == list(range(16, 24))
    assert tokenizer.stored_item2cues["48262"] == list(range(16, 32))
    assert tokenizer.item2cues["48262"] == list(range(16, 24))
    expected = tokenizer.codebook_weights[[255, 511, 767]].sum(axis=0)
    assert np.array_equal(generated.z_hat_emb, expected)
    assert np.array_equal(
        tokenizer._token_to_feature([256, 512, 768, 842]), expected)


def test_sparse_ids_are_never_array_indices():
    tokenizer = make_tokenizer()
    encoded = tokenizer.encode_playlist(
        ["48262", "73001", "18996"], context_items=2)
    assert np.allclose(encoded.mu_c, 1.5)


def test_type_mask_matches_stride():
    tokenizer = make_tokenizer()
    legal = tokenizer.make_type_mask(15)
    assert legal.shape == (15, TOKEN_LAYOUT.runtime_vocab_size)
    assert np.flatnonzero(legal[0]).tolist() == [0]
    assert np.flatnonzero(legal[1]).tolist() == [TOKEN_LAYOUT.boi_token]
    assert legal[2, 1:257].all() and legal[2].sum() == 256
    assert legal[5, 769:843].all() and legal[5].sum() == 74
    assert legal[6, 845:2893].all() and legal[6].sum() == 2048
    assert np.flatnonzero(legal[-1]).tolist() == [TOKEN_LAYOUT.eos_token]


def test_file_contract_loads_sixteen_and_activates_first_eight():
    source = make_tokenizer()
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / "semantic.json").write_text(
            json.dumps(source.semantic_tokens), encoding="utf-8")
        (root / "item2cues.json").write_text(
            json.dumps(source.stored_item2cues), encoding="utf-8")
        (root / "manifest.json").write_text(json.dumps({
            "schema_version": TOKEN_LAYOUT.schema_version,
            "wp_d_compatible": True,
            "stored_cues_per_item": 16,
            "default_active_cues": 8,
        }), encoding="utf-8")
        np.save(root / "weights.npy", source.codebook_weights)
        loaded = GenPlaylistTokenizer.from_files(
            semantic_tokens_path=root / "semantic.json",
            item2cues_path=root / "item2cues.json",
            cue_manifest_path=root / "manifest.json",
            catalog_items=source.catalog_items,
            catalog_embeddings=source.catalog_embeddings,
            item_id_to_row=source.item_id_to_row,
            codebook_weights_path=root / "weights.npy",
        )
        assert len(loaded.stored_item2cues["18996"]) == 16
        assert loaded.item2cues["18996"] == list(range(8))


def test_builds_explicit_full_mask_next_item_slot():
    tokenizer = make_tokenizer()
    context = [tokenizer.bos_token]
    context.extend(tokenizer.encode_item("18996"))
    context.extend(tokenizer.encode_item("48262"))
    context.append(tokenizer.eos_token)
    completed, completion_mask = tokenizer.build_next_item_completion(context)
    assert len(completed) == 2 + 3 * tokenizer.tokens_per_item
    assert completed[:len(context) - 1].tolist() == context[:-1]
    assert completed[len(context) - 1] == tokenizer.boi_token
    assert completion_mask.sum() == tokenizer.tokens_per_item - 1
    assert np.all(completed[completion_mask] == tokenizer.mask_token_id)
    assert completed[-1] == tokenizer.eos_token


def test_full_mask_completion_handles_reference_lengths_and_rejects_one_reference():
    tokenizer = make_tokenizer()

    def context_for(item_ids):
        tokens = [tokenizer.bos_token]
        for item_id in item_ids:
            tokens.extend(tokenizer.encode_item(item_id))
        tokens.append(tokenizer.eos_token)
        return tokens

    two_refs, two_mask = tokenizer.build_next_item_completion(
        context_for(["18996", "48262"]))
    three_refs, three_mask = tokenizer.build_next_item_completion(
        context_for(["18996", "48262", "73001"]))
    assert len(three_refs) - len(two_refs) == tokenizer.tokens_per_item
    assert two_mask.sum() == three_mask.sum() == tokenizer.tokens_per_item - 1
    try:
        tokenizer.build_next_item_completion(context_for(["18996"]))
    except ValueError as exc:
        assert "at least two" in str(exc)
    else:
        raise AssertionError("Expected a one-reference completion to be rejected")


def test_test_tokenization_exposes_fifteen_references_and_five_labels():
    tokenizer, item_ids = make_twenty_item_tokenizer()
    source = FakeRows([{"bundle": "p", "item_seq": item_ids}])
    row = tokenizer.tokenize({"test": source})["test"][0]

    assert len(row["input_ids"]) == 2 + 15 * tokenizer.tokens_per_item
    assert row["input_ids"][0] == tokenizer.bos_token
    assert row["input_ids"][-1] == tokenizer.eos_token
    assert len(row["labels"]) == 5
    assert row["labels"][0] == tokenizer.semantic_tokens[item_ids[15]]
    assert row["labels"][-1] == tokenizer.semantic_tokens[item_ids[19]]
    assert np.allclose(row["mu_c"], 7.0)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
