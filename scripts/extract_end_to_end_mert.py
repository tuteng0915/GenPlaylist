#!/usr/bin/env python3
"""Extract frozen MERT embeddings for one generated-audio system."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from shared.artifacts import sha256_file  # noqa: E402
from extract_mert_embeddings import (  # noqa: E402
    DEFAULT_CLIP_SECONDS,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    _git_commit,
    _load_center_audio,
    _pool_valid_frames,
)


SYSTEMS = ("ACE-Step-Direct", "DDBC-SFT", "GenPlaylist")
EXPECTED_EXAMPLES = 941


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _record_fingerprint(audio_dir: Path, system: str) -> str:
    digest = hashlib.sha256()
    for index in range(EXPECTED_EXAMPLES):
        record_path = audio_dir / system / f"{index:04d}.json"
        audio_path = audio_dir / system / f"{index:04d}.mp3"
        if not record_path.is_file() or not audio_path.is_file():
            raise FileNotFoundError(
                f"Incomplete {system} audio at history {index}: "
                f"{record_path}, {audio_path}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("system") != system or record.get("example_index") != index:
            raise ValueError(f"Audio record identity mismatch: {record_path}")
        digest.update(sha256_file(record_path).encode("ascii"))
    return digest.hexdigest()


def _slug(system: str) -> str:
    return system.lower().replace("-", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system", required=True, choices=SYSTEMS)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--clip-seconds", type=float, default=DEFAULT_CLIP_SECONDS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_rate <= 0 or args.clip_seconds <= 0 or args.batch_size <= 0:
        raise ValueError("Sample rate, clip length, and batch size must be positive")
    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    audio_dir = args.audio_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_manifest = audio_dir / "audio_manifest.json"
    if not audio_manifest.is_file():
        raise FileNotFoundError(audio_manifest)
    records_sha256 = _record_fingerprint(audio_dir, args.system)

    model_kwargs = {"trust_remote_code": True}
    if args.revision:
        model_kwargs["revision"] = args.revision
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.model_name, **model_kwargs)
    model = AutoModel.from_pretrained(args.model_name, **model_kwargs).eval().to(args.device)
    hidden_size = int(model.config.hidden_size)
    resolved_revision = getattr(model.config, "_commit_hash", None) or args.revision
    slug = _slug(args.system)
    final_path = output_dir / f"{slug}_mert_embeddings_l2.npy"
    partial_path = output_dir / f"{slug}_mert_embeddings_l2.partial.npy"
    progress_path = output_dir / f"{slug}_mert_progress.json"
    manifest_path = output_dir / f"{slug}_mert_manifest.json"
    if final_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Completed MERT artifact already exists for {args.system}")

    identity = {
        "system": args.system,
        "examples": EXPECTED_EXAMPLES,
        "audio_manifest_sha256": sha256_file(audio_manifest),
        "audio_records_fingerprint": records_sha256,
        "model_name": args.model_name,
        "model_revision": resolved_revision,
        "sample_rate": args.sample_rate,
        "clip_seconds": args.clip_seconds,
        "hidden_size": hidden_size,
    }
    if progress_path.exists() or partial_path.exists():
        if not progress_path.exists() or not partial_path.exists():
            raise RuntimeError("Incomplete MERT resume state")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError("Existing MERT resume state has a different identity")
        completed = int(progress["completed"])
        embeddings = np.lib.format.open_memmap(partial_path, mode="r+")
        if embeddings.shape != (EXPECTED_EXAMPLES, hidden_size):
            raise ValueError(f"Partial embedding shape drifted: {embeddings.shape}")
    else:
        completed = 0
        embeddings = np.lib.format.open_memmap(
            partial_path, mode="w+", dtype=np.float32,
            shape=(EXPECTED_EXAMPLES, hidden_size))
        _atomic_json(progress_path, {"identity": identity, "completed": completed})

    clip_samples = int(round(args.sample_rate * args.clip_seconds))
    with torch.inference_mode():
        for start in range(completed, EXPECTED_EXAMPLES, args.batch_size):
            stop = min(start + args.batch_size, EXPECTED_EXAMPLES)
            paths = [audio_dir / args.system / f"{index:04d}.mp3"
                     for index in range(start, stop)]
            for index, path in zip(range(start, stop), paths):
                record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                if record["audio_sha256"] != sha256_file(path):
                    raise ValueError(f"Generated audio hash mismatch: {path}")
            audio = [_load_center_audio(path, args.sample_rate, clip_samples)
                     for path in paths]
            inputs = processor(
                audio, sampling_rate=args.sample_rate, padding=True,
                return_attention_mask=True, return_tensors="pt")
            inputs = {name: value.to(args.device) for name, value in inputs.items()}
            hidden = model(**inputs).last_hidden_state
            pooled = _pool_valid_frames(model, hidden, inputs.get("attention_mask"))
            values = torch.nn.functional.normalize(
                pooled.to(torch.float32), dim=-1).cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite MERT embedding at rows {start}:{stop}")
            embeddings[start:stop] = values
            embeddings.flush()
            _atomic_json(progress_path, {"identity": identity, "completed": stop})
            print(f"[MERT] {args.system} {stop}/{EXPECTED_EXAMPLES}", flush=True)

    del embeddings
    os.replace(partial_path, final_path)
    progress_path.unlink()
    _atomic_json(manifest_path, {
        "result_schema": "genplaylist-end-to-end-mert-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        **identity,
        "pooling": "attention-masked mean of final hidden layer",
        "normalization": "L2",
        "shape": [EXPECTED_EXAMPLES, hidden_size],
        "output_sha256": sha256_file(final_path),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
