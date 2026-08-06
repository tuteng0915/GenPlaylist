#!/usr/bin/env python3
"""Evaluate the untouched Spotify30 DDBC checkpoint without creative cues.

This is a zero-shot baseline for the frozen GenPlaylist test set.  It restores
the checkpoint's native 1,028-token vocabulary and five-token item layout, then
jointly inpaints either one or five items after the same 15 references.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
WP_C_ROOT = REPO_ROOT / "src" / "03_backbone_recommender"
sys.path.insert(0, str(WP_C_ROOT))

from diffusion import Diffusion  # noqa: E402
from many_to_many_metrics import calculate_many_to_many_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=REPO_ROOT / "data" / "dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def legacy_type_mask(length: int, vocab_size: int, tokenizer) -> np.ndarray:
    """Return the native DDBC positional vocabulary for a legacy sequence."""
    codebook_size = int(tokenizer.config["rq_codebook_size"])
    item_width = int(tokenizer.n_digit) + 2
    legal = np.zeros((length, vocab_size), dtype=bool)
    legal[0, int(tokenizer.bos_token)] = True
    legal[-1, int(tokenizer.eos_token)] = True
    for position in range(1, length - 1):
        offset = position % item_width
        if offset == 1:
            legal[position, int(tokenizer.boi_token)] = True
        elif 2 <= offset <= int(tokenizer.n_digit) + 1:
            digit = offset - 2
            start = digit * codebook_size + 1
            legal[position, start:start + codebook_size] = True
        else:
            start = int(tokenizer.n_digit) * codebook_size + 1
            legal[position, start:start + codebook_size] = True
    if not bool(legal.any(axis=1).all()):
        raise ValueError("Legacy type mask left an empty position")
    return legal


def load_native_model(checkpoint_path: Path, data_dir: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    tokenizer = hparams.get("tokenizer")
    config_dict = hparams.get("config")
    if tokenizer is None or config_dict is None:
        raise ValueError("Expected official DDBC Lightning checkpoint metadata")
    if len(tokenizer.token) != 254155:
        raise ValueError(f"Unexpected source catalog size: {len(tokenizer.token)}")

    resolvers = {
        "cwd": os.getcwd,
        "device_count": torch.cuda.device_count,
        "eval": eval,
        "div_up": lambda x, y: (x + y - 1) // y,
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)
    config = OmegaConf.create(config_dict)
    OmegaConf.set_struct(config, False)
    config.data_root = str(data_dir.resolve())
    config.codebook_weights_path = str(
        (data_dir / "rvq_codebook_weights.npy").resolve())
    config.sampling.cfg_enabled = False
    config.sampling.structure_conditioning = False
    tokenizer.make_type_mask = lambda length: legacy_type_mask(
        length, int(tokenizer.vocab_size) + 1, tokenizer)

    model = Diffusion(config=config, tokenizer=tokenizer)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    expected_missing = {
        "backbone.mu_c_map.weight",
        "backbone.sigma_c2_map.weight",
        "backbone.context_map.weight",
    }
    if set(missing) != expected_missing or unexpected:
        raise ValueError(
            f"Unexpected checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    # The current runtime has three additional, unused conditioning parameters.
    # Apply the legacy EMA shadows by the original shared parameter order rather
    # than passing them through the longer current EMA parameter list.
    named_parameters = dict(model.named_parameters())
    shared_names = [
        name for name in named_parameters if name in checkpoint["state_dict"]]
    shadows = checkpoint["ema"]["shadow_params"]
    if len(shared_names) != len(shadows):
        raise ValueError(
            f"EMA parameter mismatch: {len(shared_names)} names vs {len(shadows)} shadows")
    for name, shadow in zip(shared_names, shadows):
        parameter = named_parameters[name]
        if tuple(parameter.shape) != tuple(shadow.shape):
            raise ValueError(
                f"EMA shape mismatch at {name}: {tuple(parameter.shape)} vs "
                f"{tuple(shadow.shape)}")
        parameter.data.copy_(shadow)

    model.ema = None
    model.eval().to(device)
    return model, tokenizer, config, checkpoint


def validate_catalog_compatibility(
    tokenizer, item_ids: list[str], semantic_tokens: np.ndarray,
    catalog_embeddings: np.ndarray,
) -> None:
    if len(item_ids) != len(semantic_tokens) or len(item_ids) != len(catalog_embeddings):
        raise ValueError("Catalog ID, semantic-token, and embedding counts differ")
    mismatches = [
        item_id for item_id, tokens in zip(item_ids, semantic_tokens)
        if item_id not in tokenizer.token
        or tuple(tokenizer.token[item_id]) != tuple(tokens.tolist())]
    if mismatches:
        raise ValueError(
            f"Current catalog is not an exact semantic subset; examples={mismatches[:5]}")
    source_features = np.stack(
        [np.asarray(tokenizer.feature[int(item_id)]) for item_id in item_ids])
    if not np.array_equal(source_features, catalog_embeddings):
        difference = float(np.max(np.abs(source_features - catalog_embeddings)))
        raise ValueError(f"Current/source CLHE embeddings differ; max_abs={difference}")


def build_masked_batch(
    reference_semantic_tokens: np.ndarray,
    *,
    generated_items: int,
    tokenizer,
    mask_index: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, reference_items, semantic_width = reference_semantic_tokens.shape
    expected_width = int(tokenizer.n_digit) + 1
    if reference_items != 15 or semantic_width != expected_width:
        raise ValueError(
            f"Expected [batch, 15, {expected_width}], got "
            f"{reference_semantic_tokens.shape}")
    rows = []
    completion_masks = []
    for reference_row in reference_semantic_tokens:
        values = [int(tokenizer.bos_token)]
        target_mask = [False]
        for item_tokens in reference_row:
            values.extend([int(tokenizer.boi_token), *map(int, item_tokens)])
            target_mask.extend([False] * (semantic_width + 1))
        for _ in range(generated_items):
            values.extend([int(tokenizer.boi_token)] + [mask_index] * semantic_width)
            target_mask.extend([False] + [True] * semantic_width)
        values.append(int(tokenizer.eos_token))
        target_mask.append(False)
        rows.append(values)
        completion_masks.append(target_mask)
    return (
        torch.as_tensor(rows, dtype=torch.long, device=device),
        torch.as_tensor(completion_masks, dtype=torch.bool, device=device),
    )


def extract_generated_semantics(
    completed: torch.Tensor, *, reference_items: int, generated_items: int,
    semantic_width: int,
) -> np.ndarray:
    item_width = semantic_width + 1
    first_target_boi = 1 + reference_items * item_width
    output = []
    for target_index in range(generated_items):
        start = first_target_boi + target_index * item_width + 1
        output.append(completed[:, start:start + semantic_width])
    return torch.stack(output, dim=1).detach().cpu().numpy()


def retrieve_catalog(
    predicted_tokens: np.ndarray,
    *,
    semantic_tokens: np.ndarray,
    codebook_weights: np.ndarray,
    catalog_embeddings: np.ndarray,
    catalog_embeddings_l2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    token_to_row = {
        tuple(map(int, tokens)): row for row, tokens in enumerate(semantic_tokens)}
    flat_tokens = predicted_tokens.reshape(-1, predicted_tokens.shape[-1])
    rows = np.empty(len(flat_tokens), dtype=np.int64)
    retrieval_cosines = np.empty(len(flat_tokens), dtype=np.float32)
    direct_hits = 0
    for index, tokens in enumerate(flat_tokens):
        token_tuple = tuple(map(int, tokens))
        direct_row = token_to_row.get(token_tuple)
        reconstructed = codebook_weights[tokens[:3] - 1].sum(axis=0)
        norm = max(float(np.linalg.norm(reconstructed)), np.finfo(np.float32).eps)
        if direct_row is None:
            similarities = catalog_embeddings_l2 @ (reconstructed / norm)
            row = int(np.argmax(similarities))
        else:
            row = int(direct_row)
            direct_hits += 1
        rows[index] = row
        retrieval_cosines[index] = float(
            catalog_embeddings_l2[row] @ (reconstructed / norm))
    output_shape = predicted_tokens.shape[:2]
    rows = rows.reshape(output_shape)
    features = catalog_embeddings[rows]
    diagnostics = {
        "rvq_direct_hit_rate": direct_hits / max(len(flat_tokens), 1),
        "retrieval_cosine": float(retrieval_cosines.mean()),
    }
    return rows, features, diagnostics


def evaluate_setting(
    *, model, tokenizer, reference_rows: np.ndarray, target_rows: np.ndarray,
    semantic_tokens: np.ndarray, codebook_weights: np.ndarray,
    catalog_embeddings: np.ndarray, catalog_embeddings_l2: np.ndarray,
    item_ids: list[str], generated_items: int, batch_size: int, steps: int,
    seed: int, device: str,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    all_metrics: dict[str, list[np.ndarray]] = {}
    direct_hit_weighted = 0.0
    retrieval_cosine_weighted = 0.0
    prediction_count = 0
    for start in range(0, len(reference_rows), batch_size):
        stop = min(start + batch_size, len(reference_rows))
        references = semantic_tokens[reference_rows[start:stop]]
        masked_ids, completion_mask = build_masked_batch(
            references, generated_items=generated_items, tokenizer=tokenizer,
            mask_index=model.mask_index, device=device)
        completed = model.sample_masked_completion(
            masked_ids, completion_mask, num_steps=steps,
            sequence_mask=torch.ones_like(masked_ids, dtype=torch.bool))
        predicted_tokens = extract_generated_semantics(
            completed, reference_items=15, generated_items=generated_items,
            semantic_width=semantic_tokens.shape[1])
        predicted_rows, prediction_features, diagnostics = retrieve_catalog(
            predicted_tokens, semantic_tokens=semantic_tokens,
            codebook_weights=codebook_weights,
            catalog_embeddings=catalog_embeddings,
            catalog_embeddings_l2=catalog_embeddings_l2)
        selected_targets = target_rows[start:stop, :generated_items]
        target_features = catalog_embeddings[selected_targets]
        prediction_ids = np.asarray(item_ids, dtype=object)[predicted_rows]
        target_ids = np.asarray(item_ids, dtype=object)[selected_targets]
        metrics = calculate_many_to_many_metrics(
            prediction_features, target_features, prediction_ids, target_ids)
        for name, values in metrics.items():
            all_metrics.setdefault(name, []).append(values)

        count = predicted_rows.size
        direct_hit_weighted += diagnostics["rvq_direct_hit_rate"] * count
        retrieval_cosine_weighted += diagnostics["retrieval_cosine"] * count
        prediction_count += count
        print(
            f"15->{generated_items}: {stop}/{len(reference_rows)} examples",
            flush=True)

    output = {
        name: float(np.concatenate(chunks).mean())
        for name, chunks in all_metrics.items()
    }
    output.update({
        "rvq_direct_hit_rate": direct_hit_weighted / prediction_count,
        "retrieval_cosine": retrieval_cosine_weighted / prediction_count,
    })
    return output


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    prepared_dir = args.prepared_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model, tokenizer, source_config, checkpoint = load_native_model(
        checkpoint_path, data_dir, args.device)
    source_steps = int(source_config.sampling.steps)
    steps = source_steps if args.steps is None else args.steps
    if steps <= 0:
        raise ValueError("--steps must be positive")

    vectors = prepared_dir / "vectors"
    reference_rows = np.load(vectors / "eval_reference_rows.npy", allow_pickle=False)
    target_rows = np.load(vectors / "eval_target_rows.npy", allow_pickle=False)
    semantic_tokens = np.load(
        vectors / "catalog_semantic_tokens.npy", allow_pickle=False).astype(np.int64)
    catalog_embeddings = np.load(
        data_dir / "catalog_item_embeddings.npy", allow_pickle=False).astype(np.float32)
    catalog_embeddings_l2 = np.load(
        vectors / "catalog_embeddings_l2.npy", allow_pickle=False).astype(np.float32)
    codebook_weights = np.load(
        data_dir / "rvq_codebook_weights.npy", allow_pickle=False).astype(np.float32)
    item_ids = json.loads(
        (vectors / "catalog_item_ids.json").read_text(encoding="utf-8"))
    item_ids = [str(item_id) for item_id in item_ids]
    validate_catalog_compatibility(
        tokenizer, item_ids, semantic_tokens, catalog_embeddings)

    if args.max_examples is not None:
        reference_rows = reference_rows[:args.max_examples]
        target_rows = target_rows[:args.max_examples]

    settings = {}
    for generated_items in (1, 5):
        settings[f"15_to_{generated_items}_joint_full_mask"] = evaluate_setting(
            model=model, tokenizer=tokenizer,
            reference_rows=reference_rows, target_rows=target_rows,
            semantic_tokens=semantic_tokens, codebook_weights=codebook_weights,
            catalog_embeddings=catalog_embeddings,
            catalog_embeddings_l2=catalog_embeddings_l2,
            item_ids=item_ids, generated_items=generated_items,
            batch_size=args.batch_size, steps=steps, seed=args.seed,
            device=args.device)

    manifest_path = prepared_dir / "prepared_manifest.json"
    payload = {
        "result_schema": "genplaylist-ddbc-base-zero-shot-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "runtime_vocab_size": int(model.vocab_size),
            "source_catalog_items": len(tokenizer.token),
        },
        "prepared_data": {
            "path": str(prepared_dir),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "evaluation": {
            "test_examples": len(reference_rows),
            "reference_items": 15,
            "catalog_items": len(item_ids),
            "creative_cues": False,
            "structure_conditioning": False,
            "sampling_steps": steps,
            "checkpoint_native_sampling_steps": source_steps,
            "seed": args.seed,
            "ema_enabled": True,
            "sampler": str(source_config.sampling.predictor),
            "full_catalog_retrieval": True,
            "generation": "joint_full_mask",
            "targets": {
                "15_to_1": "song_16",
                "15_to_5": "songs_16_through_20",
            },
        },
        "settings": settings,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
