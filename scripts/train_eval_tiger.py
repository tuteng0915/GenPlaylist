#!/usr/bin/env python3
"""Train and evaluate a TIGER adaptation on the frozen 15-to-5 protocol.

This implementation follows TIGER's T5 encoder-decoder formulation and consumes
the already-frozen DDBC/RQ semantic IDs (three residual codes plus one collision
code).  Each update samples next-item transitions from positions 16--20.  At
test time, constrained beam search produces five valid, unseen catalog items
autoregressively and saves their IDs for the common MERT evaluator.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.nn.utils import clip_grad_norm_
from transformers import Adafactor, T5Config, T5ForConditionalGeneration


REPO_ROOT = Path(__file__).resolve().parents[1]
WP_ROOT = REPO_ROOT / "src" / "03_backbone_recommender"
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(WP_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from many_to_many_metrics import calculate_many_to_many_metrics  # noqa: E402
from shared.artifacts import sha256_file  # noqa: E402


PAD_TOKEN = 0
EOS_TOKEN = 1
REFERENCE_ITEMS = 15
TARGET_ITEMS = 5
SEMANTIC_WIDTH = 4
SEQUENCE_ITEMS = REFERENCE_ITEMS + TARGET_ITEMS
MAX_HISTORY_TOKENS = (SEQUENCE_ITEMS - 1) * SEMANTIC_WIDTH


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_row_sequences(
    prepared_dir: Path,
    item_to_row: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from datasets import load_from_disk
    except ImportError as error:
        raise RuntimeError("The datasets package is required for TIGER training") from error
    dataset = load_from_disk(str(prepared_dir / "raw_dataset"))
    output = []
    for split in ("train", "test"):
        item_sequences = dataset[split]["item_seq"]
        if any(len(sequence) != SEQUENCE_ITEMS for sequence in item_sequences):
            raise ValueError(f"{split} contains a sequence whose length is not 20")
        unknown = sorted(
            {str(item) for sequence in item_sequences for item in sequence} - set(item_to_row))
        if unknown:
            raise ValueError(f"{split} contains unknown catalog IDs: {unknown[:10]}")
        output.append(np.asarray([
            [item_to_row[str(item)] for item in sequence]
            for sequence in item_sequences
        ], dtype=np.int64))
    train, test = output
    if test.shape != (941, SEQUENCE_ITEMS):
        raise ValueError(f"Frozen test shape drifted: {test.shape}")
    return train, test


def _validate_semantic_tokens(tokens: np.ndarray) -> None:
    if tokens.ndim != 2 or tokens.shape[1] != SEMANTIC_WIDTH:
        raise ValueError(f"Expected catalog semantic IDs [items, 4], got {tokens.shape}")
    if tokens.min() <= EOS_TOKEN:
        raise ValueError("Catalog semantic tokens overlap PAD/EOS")
    if len({tuple(map(int, row)) for row in tokens}) != len(tokens):
        raise ValueError("Collision-resolved catalog semantic IDs are not unique")
    for offset in range(SEMANTIC_WIDTH - 1):
        if set(tokens[:, offset]).intersection(tokens[:, offset + 1]):
            raise ValueError("Adjacent semantic-ID codebooks are not token-disjoint")


def _make_training_batch(
    row_sequences: torch.Tensor,
    semantic_tokens: torch.Tensor,
    *,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample next-item cases from the five suffix transitions."""
    indices = torch.randint(
        len(row_sequences), (batch_size,), generator=generator)
    offsets = torch.randint(
        TARGET_ITEMS, (batch_size,), generator=generator)
    selected = row_sequences[indices]
    inputs = torch.zeros(
        (batch_size, MAX_HISTORY_TOKENS), dtype=torch.long)
    attention = torch.zeros_like(inputs)
    labels = torch.empty((batch_size, SEMANTIC_WIDTH + 1), dtype=torch.long)
    for offset in range(TARGET_ITEMS):
        mask = offsets.eq(offset)
        if not bool(mask.any()):
            continue
        history_items = REFERENCE_ITEMS + offset
        history = semantic_tokens[selected[mask, :history_items]].reshape(
            int(mask.sum()), history_items * SEMANTIC_WIDTH)
        inputs[mask, :history.shape[1]] = history
        attention[mask, :history.shape[1]] = 1
        labels[mask, :SEMANTIC_WIDTH] = semantic_tokens[
            selected[mask, history_items]]
    labels[:, -1] = EOS_TOKEN
    return (
        inputs.to(device, non_blocking=True),
        attention.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
    )


class SemanticTrie:
    """Prefix index used to constrain every decoded ID to an unseen item."""

    def __init__(self, semantic_tokens: np.ndarray) -> None:
        semantic_tokens = np.asarray(semantic_tokens, dtype=np.int64)
        _validate_semantic_tokens(semantic_tokens)
        self.tokens = semantic_tokens
        self.tuple_to_row = {
            tuple(map(int, values)): row
            for row, values in enumerate(semantic_tokens)
        }
        self.children: dict[tuple[int, ...], dict[int, tuple[int, ...]]] = {}
        for depth in range(SEMANTIC_WIDTH):
            temporary: dict[tuple[int, ...], dict[int, list[int]]] = {}
            for row, values in enumerate(semantic_tokens):
                prefix = tuple(map(int, values[:depth]))
                token = int(values[depth])
                temporary.setdefault(prefix, {}).setdefault(token, []).append(row)
            for prefix, choices in temporary.items():
                self.children[prefix] = {
                    token: tuple(rows) for token, rows in choices.items()
                }

    def allowed(
        self,
        prefix: tuple[int, ...],
        excluded_rows: set[int],
    ) -> list[int]:
        if len(prefix) == SEMANTIC_WIDTH:
            return [EOS_TOKEN]
        choices = self.children.get(prefix, {})
        allowed = [
            token for token, rows in choices.items()
            if any(row not in excluded_rows for row in rows)
        ]
        if not allowed:
            raise ValueError(
                f"No valid unseen semantic continuation for prefix {prefix}")
        return sorted(allowed)


def _build_model(args: argparse.Namespace, vocab_size: int) -> T5ForConditionalGeneration:
    config = T5Config(
        vocab_size=vocab_size,
        d_model=args.d_model,
        d_kv=args.d_kv,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout_rate=args.dropout,
        feed_forward_proj="relu",
        pad_token_id=PAD_TOKEN,
        eos_token_id=EOS_TOKEN,
        decoder_start_token_id=PAD_TOKEN,
        tie_word_embeddings=True,
    )
    return T5ForConditionalGeneration(config)


@torch.no_grad()
def _generate_next_rows(
    model: T5ForConditionalGeneration,
    history_rows: np.ndarray,
    semantic_tokens: np.ndarray,
    trie: SemanticTrie,
    *,
    beam_size: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    history_rows = np.asarray(history_rows, dtype=np.int64)
    output_rows = []
    model.eval()
    for start in range(0, len(history_rows), batch_size):
        rows = history_rows[start:start + batch_size]
        flat = semantic_tokens[rows].reshape(len(rows), -1)
        inputs = torch.as_tensor(flat, dtype=torch.long, device=device)
        attention = torch.ones_like(inputs)
        excluded = [set(map(int, values)) for values in rows]

        def allowed_tokens(batch_id: int, decoder_ids: torch.Tensor) -> list[int]:
            values = decoder_ids.tolist()
            if values and values[0] == PAD_TOKEN:
                values = values[1:]
            prefix = tuple(map(int, values[:SEMANTIC_WIDTH]))
            return trie.allowed(prefix, excluded[batch_id])

        generated = model.generate(
            input_ids=inputs,
            attention_mask=attention,
            max_new_tokens=SEMANTIC_WIDTH + 1,
            num_beams=beam_size,
            num_return_sequences=1,
            do_sample=False,
            early_stopping=True,
            prefix_allowed_tokens_fn=allowed_tokens,
        )
        semantic_ids = generated[:, 1:1 + SEMANTIC_WIDTH].cpu().numpy()
        try:
            decoded = np.asarray([
                trie.tuple_to_row[tuple(map(int, values))]
                for values in semantic_ids
            ], dtype=np.int64)
        except KeyError as error:
            raise ValueError(f"Constrained TIGER emitted invalid ID {error.args[0]}") from error
        if any(row in blocked for row, blocked in zip(decoded, excluded)):
            raise ValueError("Constrained TIGER emitted a visible item")
        output_rows.append(decoded)
    return np.concatenate(output_rows)


@torch.no_grad()
def _autoregressive_five(
    model: T5ForConditionalGeneration,
    reference_rows: np.ndarray,
    semantic_tokens: np.ndarray,
    trie: SemanticTrie,
    *,
    beam_size: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    histories = np.asarray(reference_rows, dtype=np.int64)
    predictions = []
    for _ in range(TARGET_ITEMS):
        selected = _generate_next_rows(
            model,
            histories,
            semantic_tokens,
            trie,
            beam_size=beam_size,
            batch_size=batch_size,
            device=device,
        )
        predictions.append(selected)
        histories = np.concatenate([histories, selected[:, None]], axis=1)
    return np.stack(predictions, axis=1)


def _learning_rate(step: int, *, peak: float, constant_steps: int) -> float:
    if step <= constant_steps:
        return peak
    return peak * math.sqrt(constant_steps / step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--beam-size", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-kv", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--constant-lr-steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(
        args.steps, args.batch_size, args.eval_batch_size, args.beam_size,
        args.constant_lr_steps,
    ) <= 0:
        raise ValueError("Steps, batches, beams, and schedule lengths must be positive")
    prepared_dir = args.prepared_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    vectors = prepared_dir / "vectors"
    item_ids = [str(item) for item in json.loads(
        (vectors / "catalog_item_ids.json").read_text(encoding="utf-8"))]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Catalog item IDs are not unique")
    item_to_row = {item: row for row, item in enumerate(item_ids)}
    semantic_tokens = np.load(
        vectors / "catalog_semantic_tokens.npy", allow_pickle=False).astype(np.int64)
    if len(semantic_tokens) != len(item_ids):
        raise ValueError("Catalog IDs and semantic-token rows differ")
    _validate_semantic_tokens(semantic_tokens)
    train_rows, test_rows = _load_row_sequences(prepared_dir, item_to_row)
    if args.max_train_examples is not None:
        train_rows = train_rows[:args.max_train_examples]
    if args.max_eval_examples is not None:
        test_rows = test_rows[:args.max_eval_examples]
    if not len(train_rows) or not len(test_rows):
        raise ValueError("Training and evaluation splits must be non-empty")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    vocab_size = int(semantic_tokens.max()) + 1
    model = _build_model(args, vocab_size).to(device)
    optimizer = Adafactor(
        model.parameters(),
        lr=args.learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )
    sampling_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    start_step = 0
    model_config = {
        "vocab_size": vocab_size,
        "d_model": args.d_model,
        "d_kv": args.d_kv,
        "d_ff": args.d_ff,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
    }
    if checkpoint_path.exists() and not args.no_resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_config") != model_config:
            raise ValueError("Existing TIGER checkpoint configuration differs")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        sampling_generator.set_state(checkpoint["sampling_generator_state"])
        start_step = int(checkpoint["step"])
        print(f"Resuming TIGER from step {start_step}", flush=True)

    train_tensor = torch.from_numpy(train_rows)
    semantic_tensor = torch.from_numpy(semantic_tokens)
    model.train()
    interval_loss = 0.0
    interval_start = time.monotonic()
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for step in range(start_step + 1, args.steps + 1):
        learning_rate = _learning_rate(
            step, peak=args.learning_rate, constant_steps=args.constant_lr_steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        inputs, attention, labels = _make_training_batch(
            train_tensor,
            semantic_tensor,
            batch_size=args.batch_size,
            generator=sampling_generator,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            loss = model(
                input_ids=inputs,
                attention_mask=attention,
                labels=labels,
            ).loss
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        interval_loss += float(loss.detach())
        if step % args.log_every == 0 or step == args.steps:
            elapsed = time.monotonic() - interval_start
            count = args.log_every if step % args.log_every == 0 else step % args.log_every
            print(
                f"step={step}/{args.steps} loss={interval_loss / count:.6f} "
                f"lr={learning_rate:.8f} steps_per_second={count / max(elapsed, 1e-9):.2f}",
                flush=True,
            )
            interval_loss = 0.0
            interval_start = time.monotonic()
        if step % args.save_every == 0 or step == args.steps:
            _atomic_torch_save({
                "result_schema": "genplaylist-tiger-checkpoint-v1",
                "step": step,
                "model_config": model_config,
                "training_config": vars(args),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "sampling_generator_state": sampling_generator.get_state(),
                "prepared_manifest_sha256": sha256_file(
                    prepared_dir / "prepared_manifest.json"),
                "git_commit": _git_commit(),
            }, checkpoint_path)

    trie = SemanticTrie(semantic_tokens)
    reference_rows = test_rows[:, :REFERENCE_ITEMS]
    target_rows = test_rows[:, REFERENCE_ITEMS:]
    prediction_rows = _autoregressive_five(
        model,
        reference_rows,
        semantic_tokens,
        trie,
        beam_size=args.beam_size,
        batch_size=args.eval_batch_size,
        device=device,
    )
    catalog_embeddings_l2 = np.load(
        vectors / "catalog_embeddings_l2.npy", allow_pickle=False).astype(np.float32)
    id_array = np.asarray(item_ids, dtype=object)
    prediction_ids = id_array[prediction_rows]
    target_ids = id_array[target_rows]
    block = calculate_many_to_many_metrics(
        catalog_embeddings_l2[prediction_rows],
        catalog_embeddings_l2[target_rows],
        prediction_ids,
        target_ids,
    )
    metrics = {name: float(values.mean()) for name, values in block.items()}
    unique_predictions = len(set(prediction_ids.reshape(-1).tolist()))
    metrics.update({
        "coverage_at_5": unique_predictions / len(item_ids),
        "unique_predicted_items": unique_predictions,
    })
    payload = {
        "result_schema": "genplaylist-baseline-predictions-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "method": "TIGER adaptation",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "step": args.steps,
        },
        "prepared_data": {
            "path": str(prepared_dir),
            "manifest_sha256": sha256_file(prepared_dir / "prepared_manifest.json"),
        },
        "training": {
            "train_examples": len(train_rows),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "sampled_target_positions": [16, 17, 18, 19, 20],
            "objective": "teacher-forced next semantic-ID cross entropy",
            "semantic_id": "three frozen DDBC RVQ codes plus collision code",
            "optimizer": "Adafactor",
            "learning_rate": {
                "peak": args.learning_rate,
                "constant_through_step": args.constant_lr_steps,
                "after_constant": "inverse-square-root decay",
            },
            "seed": args.seed,
            "model_config": model_config,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "user_id_token": False,
        },
        "evaluation": {
            "test_examples": len(test_rows),
            "reference_items": REFERENCE_ITEMS,
            "generated_items": TARGET_ITEMS,
            "catalog_items": len(item_ids),
            "generation": "autoregressive constrained beam search",
            "beam_size": args.beam_size,
            "visible_and_generated_items_excluded": True,
        },
        "metrics_clhe_diagnostic": metrics,
        "predictions": {
            "item_ids": prediction_ids.tolist(),
            "target_item_ids": target_ids.tolist(),
            "shape": [len(test_rows), TARGET_ITEMS],
        },
    }
    _atomic_json_dump(payload, output_path)
    print(json.dumps(payload["metrics_clhe_diagnostic"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
