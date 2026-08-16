#!/usr/bin/env python3
"""Train and evaluate a SASRec baseline on the frozen 15-to-5 protocol.

The model receives the first fifteen catalog items and predicts items 16--20
autoregressively.  During training, the five target transitions are optimized
with teacher forcing and full-catalog cross entropy.  Evaluation greedily
selects five unseen catalog items and persists their IDs for the common MERT
postprocessor.
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
import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
WP_ROOT = REPO_ROOT / "src" / "03_backbone_recommender"
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(WP_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from many_to_many_metrics import calculate_many_to_many_metrics  # noqa: E402
from shared.artifacts import sha256_file  # noqa: E402


REFERENCE_ITEMS = 15
TARGET_ITEMS = 5
SEQUENCE_ITEMS = REFERENCE_ITEMS + TARGET_ITEMS


class SASRecBlock(nn.Module):
    """Pre-norm causal self-attention block used by SASRec."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        values: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.attention_norm(values)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        values = values + self.attention_dropout(attended)
        values = values + self.ffn(self.ffn_norm(values))
        return values.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class SASRec(nn.Module):
    """Compact SASRec encoder with tied full-catalog output embeddings."""

    def __init__(
        self,
        *,
        catalog_size: int,
        max_length: int,
        hidden_size: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.catalog_size = catalog_size
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.item_embedding = nn.Embedding(
            catalog_size + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_size, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.apply(self._initialize_module)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        # Match the small Gaussian initialization used by the reference SASRec
        # implementation.  PyTorch's default Embedding N(0, 1) makes tied
        # full-catalog logits unusably large at initialization.
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, item_indices: torch.Tensor) -> torch.Tensor:
        if item_indices.ndim != 2 or item_indices.shape[1] != self.max_length:
            raise ValueError(
                f"Expected [batch, {self.max_length}], got {tuple(item_indices.shape)}")
        padding_mask = item_indices.eq(0)
        positions = torch.arange(
            self.max_length, device=item_indices.device).unsqueeze(0)
        values = self.item_embedding(item_indices) * math.sqrt(self.hidden_size)
        values = self.embedding_dropout(values + self.position_embedding(positions))
        values = values.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        causal_mask = torch.triu(
            torch.ones(
                self.max_length,
                self.max_length,
                dtype=torch.bool,
                device=item_indices.device,
            ),
            diagonal=1,
        )
        for block in self.blocks:
            values = block(
                values, causal_mask=causal_mask, padding_mask=padding_mask)
        return self.final_norm(values).masked_fill(
            padding_mask.unsqueeze(-1), 0.0)

    def catalog_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.item_embedding.weight[1:].T


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


def _load_sequences(
    prepared_dir: Path,
    item_to_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from datasets import load_from_disk
    except ImportError as error:
        raise RuntimeError("The datasets package is required for SASRec training") from error
    dataset = load_from_disk(str(prepared_dir / "raw_dataset"))
    output = []
    for split in ("train", "test"):
        rows = dataset[split]["item_seq"]
        if any(len(row) != SEQUENCE_ITEMS for row in rows):
            raise ValueError(f"{split} contains a sequence whose length is not 20")
        unknown = sorted({str(item) for row in rows for item in row} - set(item_to_index))
        if unknown:
            raise ValueError(f"{split} contains unknown catalog IDs: {unknown[:10]}")
        output.append(np.asarray([
            [item_to_index[str(item)] + 1 for item in row] for row in rows
        ], dtype=np.int64))
    train, test = output
    if test.shape != (941, SEQUENCE_ITEMS):
        raise ValueError(f"Frozen test shape drifted: {test.shape}")
    return train, test


def _training_loss(model: SASRec, sequences: torch.Tensor) -> torch.Tensor:
    """Return full-catalog CE over the five protocol target transitions."""
    if sequences.ndim != 2 or sequences.shape[1] != SEQUENCE_ITEMS:
        raise ValueError(f"Expected [batch, 20], got {tuple(sequences.shape)}")
    inputs = sequences[:, :-1]
    encoded = model.encode(inputs)
    target_hidden = encoded[:, REFERENCE_ITEMS - 1:]
    targets = sequences[:, REFERENCE_ITEMS:] - 1
    logits = model.catalog_logits(target_hidden)
    return F.cross_entropy(
        logits.reshape(-1, model.catalog_size), targets.reshape(-1))


@torch.no_grad()
def _autoregressive_topk(
    model: SASRec,
    reference_indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Greedily generate five items, excluding visible and prior predictions."""
    reference_indices = np.asarray(reference_indices, dtype=np.int64)
    if reference_indices.ndim != 2 or reference_indices.shape[1] != REFERENCE_ITEMS:
        raise ValueError(f"Expected [examples, 15], got {reference_indices.shape}")
    predictions = []
    model.eval()
    for start in range(0, len(reference_indices), batch_size):
        prefix = torch.as_tensor(
            reference_indices[start:start + batch_size], dtype=torch.long, device=device)
        generated = []
        for _ in range(TARGET_ITEMS):
            inputs = torch.zeros(
                (len(prefix), model.max_length), dtype=torch.long, device=device)
            inputs[:, :prefix.shape[1]] = prefix
            encoded = model.encode(inputs)
            hidden = encoded[:, prefix.shape[1] - 1]
            scores = model.catalog_logits(hidden)
            seen = torch.zeros_like(scores, dtype=torch.bool)
            seen.scatter_(1, prefix - 1, True)
            scores.masked_fill_(seen, -torch.inf)
            selected = scores.argmax(dim=1) + 1
            generated.append(selected)
            prefix = torch.cat([prefix, selected.unsqueeze(1)], dim=1)
        predictions.append(torch.stack(generated, dim=1).cpu().numpy() - 1)
    return np.concatenate(predictions, axis=0)


def _build_model(args: argparse.Namespace, catalog_size: int) -> SASRec:
    return SASRec(
        catalog_size=catalog_size,
        max_length=SEQUENCE_ITEMS - 1,
        hidden_size=args.hidden_size,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
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
    if args.steps <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Step and batch counts must be positive")
    if args.hidden_size % args.num_heads:
        raise ValueError("hidden-size must be divisible by num-heads")
    prepared_dir = args.prepared_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    vectors = prepared_dir / "vectors"
    item_ids = [str(item) for item in json.loads(
        (vectors / "catalog_item_ids.json").read_text(encoding="utf-8"))]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Catalog item IDs are not unique")
    item_to_index = {item: index for index, item in enumerate(item_ids)}
    train_sequences, test_sequences = _load_sequences(prepared_dir, item_to_index)
    if args.max_train_examples is not None:
        train_sequences = train_sequences[:args.max_train_examples]
    if args.max_eval_examples is not None:
        test_sequences = test_sequences[:args.max_eval_examples]
    if not len(train_sequences) or not len(test_sequences):
        raise ValueError("Training and evaluation splits must be non-empty")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model = _build_model(args, len(item_ids)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )
    sampling_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    start_step = 0
    if checkpoint_path.exists() and not args.no_resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected = {
            "catalog_size": len(item_ids),
            "hidden_size": args.hidden_size,
            "num_blocks": args.num_blocks,
            "num_heads": args.num_heads,
            "max_length": SEQUENCE_ITEMS - 1,
        }
        if checkpoint.get("model_config") != expected:
            raise ValueError("Existing SASRec checkpoint configuration differs")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        sampling_generator.set_state(checkpoint["sampling_generator_state"])
        start_step = int(checkpoint["step"])
        print(f"Resuming SASRec from step {start_step}", flush=True)
    train_tensor = torch.from_numpy(train_sequences)
    model.train()
    interval_loss = 0.0
    interval_start = time.monotonic()
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for step in range(start_step + 1, args.steps + 1):
        indices = torch.randint(
            len(train_tensor),
            (args.batch_size,),
            generator=sampling_generator,
        )
        batch = train_tensor[indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            loss = _training_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        interval_loss += float(loss.detach())
        if step % args.log_every == 0 or step == args.steps:
            elapsed = time.monotonic() - interval_start
            count = args.log_every if step % args.log_every == 0 else step % args.log_every
            print(
                f"step={step}/{args.steps} loss={interval_loss / count:.6f} "
                f"steps_per_second={count / max(elapsed, 1e-9):.2f}",
                flush=True,
            )
            interval_loss = 0.0
            interval_start = time.monotonic()
        if step % args.save_every == 0 or step == args.steps:
            _atomic_torch_save({
                "result_schema": "genplaylist-sasrec-checkpoint-v1",
                "step": step,
                "model_config": {
                    "catalog_size": len(item_ids),
                    "hidden_size": args.hidden_size,
                    "num_blocks": args.num_blocks,
                    "num_heads": args.num_heads,
                    "max_length": SEQUENCE_ITEMS - 1,
                },
                "training_config": vars(args),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "sampling_generator_state": sampling_generator.get_state(),
                "prepared_manifest_sha256": sha256_file(
                    prepared_dir / "prepared_manifest.json"),
                "git_commit": _git_commit(),
            }, checkpoint_path)

    reference_indices = test_sequences[:, :REFERENCE_ITEMS] - 1
    target_indices = test_sequences[:, REFERENCE_ITEMS:] - 1
    prediction_indices = _autoregressive_topk(
        model,
        reference_indices + 1,
        batch_size=args.eval_batch_size,
        device=device,
    )
    catalog_embeddings_l2 = np.load(
        vectors / "catalog_embeddings_l2.npy", allow_pickle=False).astype(np.float32)
    id_array = np.asarray(item_ids, dtype=object)
    prediction_ids = id_array[prediction_indices]
    target_ids = id_array[target_indices]
    block = calculate_many_to_many_metrics(
        catalog_embeddings_l2[prediction_indices],
        catalog_embeddings_l2[target_indices],
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
        "method": "SASRec",
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
            "train_examples": len(train_sequences),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "target_transitions_per_example": TARGET_ITEMS,
            "objective": "teacher-forced full-catalog next-item cross entropy",
            "seed": args.seed,
            "hidden_size": args.hidden_size,
            "num_blocks": args.num_blocks,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
        },
        "evaluation": {
            "test_examples": len(test_sequences),
            "reference_items": REFERENCE_ITEMS,
            "generated_items": TARGET_ITEMS,
            "catalog_items": len(item_ids),
            "generation": "greedy autoregressive",
            "visible_and_generated_items_excluded": True,
        },
        "metrics_clhe_diagnostic": metrics,
        "predictions": {
            "item_ids": prediction_ids.tolist(),
            "target_item_ids": target_ids.tolist(),
            "shape": [len(test_sequences), TARGET_ITEMS],
        },
    }
    _atomic_json_dump(payload, output_path)
    print(json.dumps(payload["metrics_clhe_diagnostic"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
