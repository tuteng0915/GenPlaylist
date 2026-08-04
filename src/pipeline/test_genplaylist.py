"""CPU-only end-to-end coordinator test with explicit dependency injection."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.genplaylist import GenPlaylistPipeline, generate_next_song
from shared.artifacts import CatalogArtifacts
from shared.schema import CatalogItem, ContextPrefix, GeneratedItem


def test_pipeline_uses_context_and_returns_valid_result():
    items = [CatalogItem("18996"), CatalogItem("48262")]
    embeddings = np.stack([
        np.ones(64, dtype=np.float32),
        np.full(64, 2.0, dtype=np.float32),
    ])
    mapping = {"18996": 0, "48262": 1}
    context = ContextPrefix(["48262", "18996"], source="song_only")

    def runner(ctx, n_samples, catalog_embs, catalog_items, item_id_to_row):
        assert ctx.item_ids == ["48262", "18996"]
        mu = catalog_embs.mean(axis=0)
        return [GeneratedItem(
            rvq_codes=(1, 2, 3), conflict_code=0,
            z_hat_emb=catalog_embs[0].copy(), mu_c_emb=mu,
            sigma_c2=0.0, cue_ids=list(range(8)),
            context_prefix=ctx,
        ) for _ in range(n_samples)]

    prompts = []
    def llm_call(prompt, system=""):
        prompts.append(prompt)
        return "indie pop, calm, 100 BPM" if "style analyst" in system else "[verse]\nhello"

    with tempfile.TemporaryDirectory() as temp_dir:
        def synthesizer(**kwargs):
            path = Path(temp_dir) / "generated.wav"
            path.write_bytes(b"RIFF-test")
            return str(path)

        result = generate_next_song(
            context, k_neighbors=1,
            catalog_embs=embeddings, catalog_metadata=items,
            item_id_to_row=mapping, backbone_runner=runner,
            llm_call=llm_call, synthesizer=synthesizer)
        result.validate()
        assert np.allclose(result.generated_item.mu_c_emb, 1.5)
        assert len(prompts) == 2
        assert "cue_5" in prompts[0]
        assert "Ordered reference music" in prompts[0]
        assert prompts[0].index("item_48262") < prompts[0].index("item_18996")


def test_next_song_rejects_single_reference():
    items = [CatalogItem("18996"), CatalogItem("48262")]
    embeddings = np.stack([
        np.ones(64, dtype=np.float32),
        np.full(64, 2.0, dtype=np.float32),
    ])
    try:
        generate_next_song(
            ContextPrefix(["18996"]),
            catalog_embs=embeddings,
            catalog_metadata=items,
            item_id_to_row={"18996": 0, "48262": 1},
        )
    except ValueError as exc:
        assert "at least two reference" in str(exc)
    else:
        raise AssertionError("Expected a single reference to be rejected")


def test_reusable_pipeline_connects_wp_a_d_and_c():
    items = [
        CatalogItem("18996", title="Reference A"),
        CatalogItem("48262", title="Reference B"),
        CatalogItem("73001", title="Catalog Neighbor"),
    ]
    embeddings = np.stack([
        np.ones(64, dtype=np.float32),
        np.full(64, 2.0, dtype=np.float32),
        np.full(64, 3.0, dtype=np.float32),
    ])
    artifacts = CatalogArtifacts(
        items, embeddings, {item.item_id: row for row, item in enumerate(items)})
    observed = {}

    def runner(ctx, n_samples, catalog_embs, catalog_items, item_id_to_row):
        observed["context"] = ctx
        assert n_samples == 1
        reference_rows = [item_id_to_row[item_id] for item_id in ctx.item_ids]
        mu = catalog_embs[reference_rows].mean(axis=0)
        return [GeneratedItem(
            rvq_codes=(1, 2, 3), conflict_code=0,
            z_hat_emb=catalog_embs[2].copy(), mu_c_emb=mu,
            sigma_c2=0.25, cue_ids=list(range(8)),
            context_prefix=ctx,
        )]

    prompts = []
    def llm_call(prompt, system=""):
        prompts.append(prompt)
        return "dream pop, calm, 96 BPM" if "style analyst" in system else "[verse]\nhello"

    with tempfile.TemporaryDirectory() as temp_dir:
        def synthesizer(**kwargs):
            path = Path(temp_dir) / "next.wav"
            path.write_bytes(b"RIFF-next")
            return str(path)

        pipeline = GenPlaylistPipeline(
            artifacts,
            backbone_runner=runner,
            llm_call=llm_call,
            synthesizer=synthesizer,
        )
        summary = pipeline.preflight()
        assert summary["task"] == "multiple_references_to_one_next_song"
        assert summary["sampler"] == "ddbc_full_mask_completion"
        result = pipeline.generate(
            ["48262", "18996"],
            user_instruction="make the transition feel nocturnal",
            k_neighbors=1,
        )
        result.validate()
        context = observed["context"]
        assert context.item_ids == ["48262", "18996"]
        assert [item.item_id for item in context.items] == context.item_ids
        assert context.raw_input == "make the transition feel nocturnal"
        assert "make the transition feel nocturnal" in prompts[0]
        assert "['48262', '18996']" not in prompts[0]


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
