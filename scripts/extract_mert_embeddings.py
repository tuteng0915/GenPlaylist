#!/usr/bin/env python3
"""Extract reproducible catalog audio embeddings with frozen MERT-v1-95M."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from shared.artifacts import sha256_file  # noqa: E402


DEFAULT_MODEL = "m-a-p/MERT-v1-95M"
DEFAULT_SAMPLE_RATE = 24_000
DEFAULT_CLIP_SECONDS = 30.0


def _center_crop_bounds(length: int, target: int) -> tuple[int, int]:
    """Return deterministic centered bounds, leaving short clips unpadded."""
    if length < 0 or target <= 0:
        raise ValueError(f"Invalid crop sizes: length={length}, target={target}")
    if length <= target:
        return 0, length
    start = (length - target) // 2
    return start, start + target


def _masked_mean_numpy(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reference implementation used by dependency-light regression tests."""
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.bool_)
    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ValueError(f"Incompatible values/mask shapes: {values.shape}, {mask.shape}")
    weights = mask[..., None].astype(np.float32)
    return (values * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1.0)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _atomic_json(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_item_ids(mapping_path: Path) -> list[str]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    item_ids: list[str | None] = [None] * len(mapping)
    for item_id, row in mapping.items():
        row = int(row)
        if not 0 <= row < len(item_ids) or item_ids[row] is not None:
            raise ValueError("item_id_to_row.json is not a contiguous one-to-one mapping")
        item_ids[row] = str(item_id)
    if any(item_id is None for item_id in item_ids):
        raise ValueError("item_id_to_row.json contains missing rows")
    return [str(item_id) for item_id in item_ids]


def _audio_path(audio_dir: Path, item_id: str) -> Path:
    candidates = [audio_dir / f"{item_id}{suffix}" for suffix in (".mp3", ".wav", ".flac")]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one audio file for item {item_id}, found {matches}")
    return matches[0]


def _load_center_audio(path: Path, sample_rate: int, clip_samples: int) -> np.ndarray:
    import torch
    import torchaudio

    waveform, source_rate = torchaudio.load(str(path))
    if waveform.numel() == 0:
        raise ValueError(f"Empty audio file: {path}")
    waveform = waveform.to(torch.float32).mean(dim=0)
    if int(source_rate) != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, int(source_rate), sample_rate)
    start, stop = _center_crop_bounds(int(waveform.numel()), clip_samples)
    waveform = waveform[start:stop]
    if not torch.isfinite(waveform).all():
        raise ValueError(f"Non-finite samples in audio file: {path}")
    return waveform.cpu().numpy()


def _pool_valid_frames(model, hidden, sample_attention_mask):
    import torch

    if sample_attention_mask is None:
        return hidden.mean(dim=1)
    if not hasattr(model, "_get_feature_vector_attention_mask"):
        raise RuntimeError(
            "MERT model does not expose feature-vector attention-mask conversion")
    frame_mask = model._get_feature_vector_attention_mask(
        hidden.shape[1], sample_attention_mask).to(hidden.device)
    weights = frame_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item-id-to-row", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--clip-seconds", type=float, default=DEFAULT_CLIP_SECONDS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-items", type=int, default=None,
        help="Smoke-test prefix only; outputs are marked incomplete and cannot be resumed as full.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_rate <= 0 or args.clip_seconds <= 0 or args.batch_size <= 0:
        raise ValueError("sample-rate, clip-seconds, and batch-size must be positive")

    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    mapping_path = args.item_id_to_row.expanduser().resolve()
    audio_dir = args.audio_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "catalog_mert_embeddings_l2.npy"
    manifest_path = output_dir / "mert_manifest.json"
    partial_path = output_dir / "catalog_mert_embeddings_l2.partial.npy"
    progress_path = output_dir / "mert_progress.json"
    ids_path = output_dir / "catalog_item_ids.json"
    if final_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite completed MERT artifacts in {output_dir}")

    item_ids = _load_item_ids(mapping_path)
    full_count = len(item_ids)
    if args.max_items is not None:
        if not 0 < args.max_items <= full_count:
            raise ValueError(f"Invalid --max-items={args.max_items} for {full_count} items")
        item_ids = item_ids[:args.max_items]
    for item_id in item_ids:
        _audio_path(audio_dir, item_id)

    model_kwargs = {"trust_remote_code": True}
    if args.revision is not None:
        model_kwargs["revision"] = args.revision
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name, **model_kwargs)
    model = AutoModel.from_pretrained(args.model_name, **model_kwargs)
    model.eval().to(args.device)
    hidden_size = int(model.config.hidden_size)
    resolved_revision = getattr(model.config, "_commit_hash", None) or args.revision

    identity = {
        "model_name": args.model_name,
        "model_revision": resolved_revision,
        "sample_rate": args.sample_rate,
        "clip_seconds": args.clip_seconds,
        "catalog_count": len(item_ids),
        "catalog_mapping_sha256": sha256_file(mapping_path),
        "hidden_size": hidden_size,
        "complete_catalog": len(item_ids) == full_count,
    }
    completed = 0
    if progress_path.exists() or partial_path.exists():
        if not progress_path.exists() or not partial_path.exists():
            raise RuntimeError("Incomplete MERT resume state: progress/partial pair differs")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError("Existing MERT resume state belongs to a different extraction")
        completed = int(progress["completed"])
        embeddings = np.lib.format.open_memmap(partial_path, mode="r+")
        if embeddings.shape != (len(item_ids), hidden_size):
            raise ValueError(f"Partial MERT shape drifted: {embeddings.shape}")
    else:
        embeddings = np.lib.format.open_memmap(
            partial_path, mode="w+", dtype=np.float32,
            shape=(len(item_ids), hidden_size))
        _atomic_json(ids_path, item_ids)
        _atomic_json(progress_path, {"identity": identity, "completed": 0})

    clip_samples = int(round(args.sample_rate * args.clip_seconds))
    with torch.inference_mode():
        for start in range(completed, len(item_ids), args.batch_size):
            stop = min(start + args.batch_size, len(item_ids))
            audio = [
                _load_center_audio(_audio_path(audio_dir, item_id), args.sample_rate, clip_samples)
                for item_id in item_ids[start:stop]
            ]
            inputs = processor(
                audio, sampling_rate=args.sample_rate, padding=True,
                return_attention_mask=True, return_tensors="pt")
            inputs = {name: value.to(args.device) for name, value in inputs.items()}
            outputs = model(**inputs)
            pooled = _pool_valid_frames(
                model, outputs.last_hidden_state, inputs.get("attention_mask"))
            pooled = torch.nn.functional.normalize(pooled.to(torch.float32), dim=-1)
            values = pooled.cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite MERT embedding in rows {start}:{stop}")
            embeddings[start:stop] = values
            embeddings.flush()
            _atomic_json(progress_path, {"identity": identity, "completed": stop})
            print(f"[MERT] {stop}/{len(item_ids)}", flush=True)

    del embeddings
    os.replace(partial_path, final_path)
    progress_path.unlink()
    manifest = {
        **identity,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "audio_root": str(audio_dir),
        "center_crop": True,
        "short_clip_padding": "processor zero padding excluded by attention mask",
        "pooling": "attention-masked mean of final hidden layer",
        "normalization": "L2",
        "dtype": "float32",
        "shape": [len(item_ids), hidden_size],
        "outputs": {
            final_path.name: sha256_file(final_path),
            ids_path.name: sha256_file(ids_path),
        },
    }
    _atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
