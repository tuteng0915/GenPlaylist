#!/usr/bin/env python3
"""Create reproducible Qwen3 verbalizations for the end-to-end audio study.

The script consumes schema-v4 plans from DDBC-SFT and GenPlaylist, plus the
same frozen reference histories, and writes one resumable JSON record per
history and system.  It does not modify or invoke the WP-D demo.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
WP_D_ROOT = SRC_ROOT / "04_synthesis"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(WP_D_ROOT))

from shared.artifacts import sha256_file  # noqa: E402
from shared.schema import (  # noqa: E402
    RQ_N_CODEBOOKS,
    TOKEN_LAYOUT,
    CatalogItem,
    ContextPrefix,
    GeneratedItem,
)


REFERENCE_ITEMS = 15
TARGET_ITEMS = 5
EXPECTED_EXAMPLES = 941
SYSTEMS = ("ACE-Step-Direct", "DDBC-SFT", "GenPlaylist")


def _load_verbalization_module():
    path = WP_D_ROOT / "verbalization.py"
    specification = importlib.util.spec_from_file_location(
        "genplaylist_wp_d_verbalization", path)
    module = importlib.util.module_from_spec(specification)
    if specification.loader is None:
        raise RuntimeError(f"Cannot load verbalization module from {path}")
    specification.loader.exec_module(module)
    return module


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resume_identity(value: dict) -> dict:
    """Return fields that must match when resuming a multi-hour run."""
    comparable = dict(value)
    comparable.pop("created_utc", None)
    # Git provenance is retained in the manifest, but unrelated evaluation-only
    # commits may land while generation shards are running. Frozen input hashes
    # and decoding settings, rather than the moving repository HEAD, determine
    # whether existing records are compatible.
    comparable.pop("git_commit", None)
    return comparable


def _load_plan(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("result_schema") != "genplaylist-wp-c-joint-15to5-eval-v4":
        raise ValueError(f"End-to-end evaluation requires a schema-v4 plan: {path}")
    predictions = payload.get("predictions", {})
    expected = (EXPECTED_EXAMPLES, TARGET_ITEMS)
    for key in ("item_ids", "target_item_ids", "semantic_token_ids", "cue_ids"):
        values = predictions.get(key)
        if not isinstance(values, list) or len(values) != expected[0]:
            raise ValueError(f"{path.name}: invalid {key} example count")
        if any(not isinstance(row, list) or len(row) != expected[1] for row in values):
            raise ValueError(f"{path.name}: invalid {key} target count")
    return payload


def _load_test_item_ids(prepared_dir: Path) -> list[list[str]]:
    try:
        from datasets import load_from_disk
    except ImportError as error:
        raise RuntimeError("The datasets package is required") from error
    dataset = load_from_disk(str(prepared_dir / "raw_dataset"))["test"]
    sequences = [[str(item) for item in row] for row in dataset["item_seq"]]
    if len(sequences) != EXPECTED_EXAMPLES or any(len(row) != 20 for row in sequences):
        raise ValueError("Frozen end-to-end histories must have shape [941, 20]")
    return sequences


def _decode_semantic_plan(
    semantic_token_ids: list[int],
    codebook_weights: np.ndarray,
) -> tuple[tuple[int, ...], int, np.ndarray]:
    if len(semantic_token_ids) != RQ_N_CODEBOOKS + 1:
        raise ValueError(f"Expected four semantic tokens, got {semantic_token_ids}")
    rvq_codes = tuple(
        int(semantic_token_ids[level]) - TOKEN_LAYOUT.rvq_token(level, 0)
        for level in range(RQ_N_CODEBOOKS)
    )
    conflict_code = (
        int(semantic_token_ids[-1]) - TOKEN_LAYOUT.conflict_token(0))
    for level, code in enumerate(rvq_codes):
        TOKEN_LAYOUT.rvq_token(level, code)
    TOKEN_LAYOUT.conflict_token(conflict_code)
    rows = [level * TOKEN_LAYOUT.rq_codebook_size + code
            for level, code in enumerate(rvq_codes)]
    z_hat = np.asarray(codebook_weights[rows].sum(axis=0), dtype=np.float32)
    return rvq_codes, conflict_code, z_hat


def _reference_structure(
    reference_ids: list[str],
    item_to_row: dict[str, int],
    catalog_embeddings: np.ndarray,
) -> tuple[np.ndarray, float]:
    rows = np.asarray([item_to_row[item] for item in reference_ids], dtype=np.int64)
    values = np.asarray(catalog_embeddings[rows], dtype=np.float32)
    mu = values.mean(axis=0).astype(np.float32)
    sigma_c2 = float(np.mean(np.sum((values - mu) ** 2, axis=1)))
    return mu, sigma_c2


def _generated_item(
    *,
    system: str,
    example_index: int,
    reference_ids: list[str],
    mu_c: np.ndarray,
    sigma_c2: float,
    plan: dict | None,
    codebook_weights: np.ndarray,
    item2cues: dict[str, list[int]],
) -> GeneratedItem:
    context = ContextPrefix(
        item_ids=reference_ids,
        source="frozen-15-song-history",
        raw_input="",
    )
    if system == "ACE-Step-Direct":
        return GeneratedItem(
            rvq_codes=(0, 0, 0),
            conflict_code=0,
            z_hat_emb=np.asarray(mu_c, dtype=np.float32),
            mu_c_emb=np.asarray(mu_c, dtype=np.float32),
            sigma_c2=sigma_c2,
            cue_ids=[],
            sample_idx=0,
            context_prefix=context,
        ).validate(allow_missing_cues=True)
    if plan is None:
        raise ValueError(f"{system} requires a schema-v4 plan")
    prediction = plan["predictions"]
    semantic = [int(value) for value in prediction["semantic_token_ids"][example_index][0]]
    rvq_codes, conflict_code, z_hat = _decode_semantic_plan(
        semantic, codebook_weights)
    predicted_item = str(prediction["item_ids"][example_index][0])
    if system == "GenPlaylist":
        cue_ids = [int(value) for value in prediction["cue_ids"][example_index][0]]
        if len(cue_ids) != 8:
            raise ValueError("GenPlaylist end-to-end plan must contain eight predicted cues")
        cue_source = "model-predicted"
    elif system == "DDBC-SFT":
        cue_ids = [int(value) for value in item2cues[predicted_item][:8]]
        cue_source = "retrieved-catalog-item"
    else:
        raise ValueError(f"Unknown system {system}")
    generated = GeneratedItem(
        rvq_codes=rvq_codes,
        conflict_code=conflict_code,
        z_hat_emb=z_hat,
        mu_c_emb=np.asarray(mu_c, dtype=np.float32),
        sigma_c2=sigma_c2,
        cue_ids=cue_ids,
        sample_idx=0,
        context_prefix=context,
    ).validate()
    generated._end_to_end_metadata = {  # type: ignore[attr-defined]
        "retrieved_item_id": predicted_item,
        "cue_source": cue_source,
    }
    return generated


class LocalQwen:
    def __init__(
        self,
        *,
        model_name: str,
        revision: str | None,
        device: str,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {"revision": revision} if revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            **kwargs,
        ).eval().to(device)
        self.device = device
        self.model_name = model_name
        self.model_revision = (
            getattr(self.model.config, "_commit_hash", None) or revision)

    def __call__(self, prompt: str, system: str = "") -> str:
        import torch

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.device)
        max_new_tokens = 128 if "style analyst" in system else 384
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not text:
            raise ValueError("Qwen3 returned an empty verbalization")
        return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--ddbc-sft-result", type=Path, required=True)
    parser.add_argument("--genplaylist-result", type=Path, required=True)
    parser.add_argument("--catalog-metadata", type=Path, required=True)
    parser.add_argument("--catalog-embeddings", type=Path, required=True)
    parser.add_argument("--item-id-to-row", type=Path, required=True)
    parser.add_argument("--codebook-weights", type=Path, required=True)
    parser.add_argument("--item2cues", type=Path, required=True)
    parser.add_argument("--cue-vocab", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepared_dir = args.prepared_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    plans = {
        "DDBC-SFT": _load_plan(args.ddbc_sft_result.expanduser().resolve()),
        "GenPlaylist": _load_plan(args.genplaylist_result.expanduser().resolve()),
    }
    test_sequences = _load_test_item_ids(prepared_dir)
    item_to_row = {
        str(item): int(row) for item, row in json.loads(
            args.item_id_to_row.read_text(encoding="utf-8")).items()
    }
    catalog_embeddings = np.load(
        args.catalog_embeddings, allow_pickle=False).astype(np.float32)
    codebook_weights = np.load(
        args.codebook_weights, allow_pickle=False).astype(np.float32)
    metadata_by_id = CatalogItem.load_catalog(str(args.catalog_metadata))
    if isinstance(metadata_by_id, list):
        metadata_by_id = {item.item_id: item for item in metadata_by_id}
    catalog_metadata = [None] * len(item_to_row)
    for item_id, row in item_to_row.items():
        catalog_metadata[row] = metadata_by_id[item_id]
    if any(item is None for item in catalog_metadata):
        raise ValueError("Catalog metadata does not align with item rows")
    item2cues = {
        str(item): [int(value) for value in cues]
        for item, cues in json.loads(args.item2cues.read_text(encoding="utf-8")).items()
    }
    cue_vocab = [
        str(value) for value in json.loads(args.cue_vocab.read_text(encoding="utf-8"))]

    for name, plan in plans.items():
        targets = plan["predictions"]["target_item_ids"]
        expected = [row[REFERENCE_ITEMS:] for row in test_sequences]
        if targets != expected:
            raise ValueError(f"{name} target IDs differ from frozen test histories")

    random.seed(args.seed)
    np.random.seed(args.seed)
    caller = LocalQwen(
        model_name=args.model_name,
        revision=args.revision,
        device=args.device,
    )
    verbalization = _load_verbalization_module()
    start = args.start_index
    stop = EXPECTED_EXAMPLES if args.max_examples is None else min(
        EXPECTED_EXAMPLES, start + args.max_examples)
    if not 0 <= start < stop <= EXPECTED_EXAMPLES:
        raise ValueError(f"Invalid example interval {start}:{stop}")

    manifest_path = output_dir / "verbalization_manifest.json"
    identity = {
        "result_schema": "genplaylist-end-to-end-verbalization-v1",
        "git_commit": _git_commit(),
        "model_name": caller.model_name,
        "model_revision": caller.model_revision,
        "decoding": "greedy",
        "seed": args.seed,
        "systems": list(SYSTEMS),
        "examples": EXPECTED_EXAMPLES,
        "reference_items": REFERENCE_ITEMS,
        "selected_prediction": 1,
        "inputs": {
            "prepared_manifest_sha256": sha256_file(
                prepared_dir / "prepared_manifest.json"),
            "ddbc_sft_result_sha256": sha256_file(args.ddbc_sft_result),
            "genplaylist_result_sha256": sha256_file(args.genplaylist_result),
            "catalog_metadata_sha256": sha256_file(args.catalog_metadata),
            "catalog_embeddings_sha256": sha256_file(args.catalog_embeddings),
            "item2cues_sha256": sha256_file(args.item2cues),
            "cue_vocab_sha256": sha256_file(args.cue_vocab),
        },
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _resume_identity(existing) != _resume_identity(identity):
            raise ValueError("Existing verbalization directory has a different identity")
    else:
        _atomic_json(manifest_path, {
            **identity,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })

    for example_index in range(start, stop):
        reference_ids = test_sequences[example_index][:REFERENCE_ITEMS]
        target_id = test_sequences[example_index][REFERENCE_ITEMS]
        mu_c, sigma_c2 = _reference_structure(
            reference_ids, item_to_row, catalog_embeddings)
        for system in SYSTEMS:
            destination = output_dir / system / f"{example_index:04d}.json"
            if destination.is_file():
                continue
            generated = _generated_item(
                system=system,
                example_index=example_index,
                reference_ids=reference_ids,
                mu_c=mu_c,
                sigma_c2=sigma_c2,
                plan=plans.get(system),
                codebook_weights=codebook_weights,
                item2cues=item2cues,
            )
            result = verbalization.verbalize(
                generated=generated,
                catalog_embs=catalog_embeddings,
                catalog_metadata=catalog_metadata,
                k=5,
                cue_vocab=cue_vocab,
                llm_call=caller,
            )
            metadata = getattr(generated, "_end_to_end_metadata", {})
            record = {
                "result_schema": "genplaylist-end-to-end-verbalization-record-v1",
                "system": system,
                "example_index": example_index,
                "reference_item_ids": reference_ids,
                "target_item_id": target_id,
                "retrieved_item_id": metadata.get("retrieved_item_id"),
                "cue_source": metadata.get("cue_source", "reference-only"),
                "cue_ids": generated.cue_ids,
                "cue_terms": result["cue_terms"],
                "neighbor_item_ids": [item.item_id for item in result["neighbors"]],
                "style_item_ids": [item.item_id for item in result["style_summary"]],
                "music_attributes": result["music_attributes"],
                "lyric_draft": result["lyric_draft"],
            }
            _atomic_json(destination, record)
            print(f"[Qwen3] {system} {example_index + 1}/{stop}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
