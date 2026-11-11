"""Dependency-light tests for end-to-end plan conversion."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("run_end_to_end_verbalization.py")
SPEC = importlib.util.spec_from_file_location("run_end_to_end_verbalization", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_semantic_plan_reconstructs_three_codebooks():
    tokens = [
        MODULE.TOKEN_LAYOUT.rvq_token(0, 2),
        MODULE.TOKEN_LAYOUT.rvq_token(1, 3),
        MODULE.TOKEN_LAYOUT.rvq_token(2, 4),
        MODULE.TOKEN_LAYOUT.conflict_token(1),
    ]
    weights = np.arange(3 * 256 * 2, dtype=np.float32).reshape(3 * 256, 2)
    codes, conflict, reconstructed = MODULE._decode_semantic_plan(tokens, weights)
    assert codes == (2, 3, 4)
    assert conflict == 1
    expected = weights[[2, 256 + 3, 512 + 4]].sum(axis=0)
    assert np.array_equal(reconstructed, expected)


def test_reference_structure_is_mean_and_mean_squared_distance():
    embeddings = np.asarray([
        [1.0, 0.0],
        [3.0, 0.0],
        [0.0, 8.0],
    ], dtype=np.float32)
    mu, sigma = MODULE._reference_structure(
        ["a", "b"], {"a": 0, "b": 1, "c": 2}, embeddings)
    assert np.array_equal(mu, np.asarray([2.0, 0.0], dtype=np.float32))
    assert np.isclose(sigma, 1.0)


def test_resume_identity_ignores_only_time_and_repository_head():
    original = {"git_commit": "old", "created_utc": "then", "seed": 42}
    resumed = {"git_commit": "new", "created_utc": "now", "seed": 42}
    assert MODULE._resume_identity(original) == MODULE._resume_identity(resumed)
    resumed["seed"] = 7
    assert MODULE._resume_identity(original) != MODULE._resume_identity(resumed)


def test_cue_drop_preserves_plan_and_all_noncue_prompt_inputs():
    from copy import deepcopy

    tokens = [MODULE.TOKEN_LAYOUT.rvq_token(i, i) for i in range(3)]
    tokens.append(MODULE.TOKEN_LAYOUT.conflict_token(0))
    plan = {"predictions": {"semantic_token_ids": [[tokens]],
                            "item_ids": [["candidate"]],
                            "cue_ids": [[[1, 2, 3, 4, 5, 6, 7, 8]]]}}
    untouched = deepcopy(plan)
    rng = np.random.default_rng(42)
    kwargs = dict(system="GenPlaylist", example_index=0,
                  reference_ids=["reference"], mu_c=np.ones(64, dtype=np.float32),
                  sigma_c2=0.5, plan=plan,
                  codebook_weights=rng.normal(size=(768, 64)).astype(np.float32),
                  item2cues={})
    full = MODULE._generated_item(**kwargs)
    dropped = MODULE._generated_item(**kwargs, cue_intervention="drop")
    assert plan == untouched
    assert full.cue_ids == list(range(1, 9)) and dropped.cue_ids == []
    assert full.rvq_codes == dropped.rvq_codes
    assert full.conflict_code == dropped.conflict_code
    np.testing.assert_array_equal(full.z_hat_emb, dropped.z_hat_emb)
    np.testing.assert_array_equal(full.mu_c_emb, dropped.mu_c_emb)
    assert full.context_prefix == dropped.context_prefix
    assert full.sigma_c2 == dropped.sigma_c2
    assert full._end_to_end_metadata["retrieved_item_id"] == dropped._end_to_end_metadata["retrieved_item_id"]
    verbalizer = MODULE._load_verbalization_module()
    metadata = [MODULE.CatalogItem(item_id="reference", title="Reference", artist="Artist")]
    vocab = [f"term_{i}" for i in range(2048)]
    prompts = []
    def capture(prompt, system=""):
        prompts.append((prompt, system))
        return "fixed response"
    for item in (full, dropped):
        verbalizer.verbalize(item, np.ones((1, 64), dtype=np.float32), metadata,
                            k=1, cue_vocab=vocab, llm_call=capture)
    assert len(prompts) == 4
    cue_text = ", ".join(vocab[i] for i in range(1, 9))
    for full_prompt, dropped_prompt in zip(prompts[:2], prompts[2:]):
        assert cue_text in full_prompt[0]
        assert full_prompt[0].replace(cue_text, "<unk>") == dropped_prompt[0]
        assert full_prompt[1] == dropped_prompt[1]


if __name__ == "__main__":
    test_semantic_plan_reconstructs_three_codebooks()
    test_reference_structure_is_mean_and_mean_squared_distance()
    test_resume_identity_ignores_only_time_and_repository_head()
    test_cue_drop_preserves_plan_and_all_noncue_prompt_inputs()
    print("  PASS  end-to-end verbalization plan tests")
