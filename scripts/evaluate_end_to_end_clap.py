#!/usr/bin/env python3
"""Compute reproducible CLAP audio--attribute alignment for one system."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from shared.artifacts import sha256_file  # noqa: E402
from evaluate_mert_proxy import _bootstrap_mean_interval  # noqa: E402
from extract_end_to_end_mert import EXPECTED_EXAMPLES, SYSTEMS, _slug  # noqa: E402


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 1e-12):
        raise ValueError("CLAP returned a non-finite or zero embedding")
    return values / norms


def _load_condition(
    audio_dir: Path, verbalization_dir: Path, system: str, index: int,
) -> tuple[Path, str]:
    audio = audio_dir / system / f"{index:04d}.mp3"
    audio_record_path = audio.with_suffix(".json")
    verbalization_path = verbalization_dir / system / f"{index:04d}.json"
    if not audio.is_file() or not audio_record_path.is_file() or not verbalization_path.is_file():
        raise FileNotFoundError(f"Incomplete CLAP inputs for {system} history {index}")
    audio_record = json.loads(audio_record_path.read_text(encoding="utf-8"))
    if audio_record.get("audio_sha256") != sha256_file(audio):
        raise ValueError(f"Generated audio hash mismatch: {audio}")
    verbalization = json.loads(verbalization_path.read_text(encoding="utf-8"))
    attributes = str(verbalization.get("music_attributes", "")).strip()
    if not attributes:
        raise ValueError(f"Empty music-attribute condition: {verbalization_path}")
    return audio, attributes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--verbalization-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system", required=True, choices=SYSTEMS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-base", type=int, default=42000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seed_base < 0 or args.bootstrap_samples <= 0:
        raise ValueError("Seeds must be nonnegative and bootstrap samples positive")
    import torch
    import laion_clap

    audio_dir = args.audio_dir.expanduser().resolve()
    verbalization_dir = args.verbalization_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    audio_manifest = audio_dir / "audio_manifest.json"
    verbalization_manifest = verbalization_dir / "verbalization_manifest.json"
    if not audio_manifest.is_file() or not verbalization_manifest.is_file():
        raise FileNotFoundError("Frozen audio/verbalization manifest is missing")

    slug = _slug(args.system)
    final_path = output_dir / f"{slug}_clap.json"
    audio_embedding_path = output_dir / f"{slug}_clap_audio_l2.npy"
    text_embedding_path = output_dir / f"{slug}_clap_attribute_l2.npy"
    partial_audio_path = output_dir / f"{slug}_clap_audio_l2.partial.npy"
    partial_text_path = output_dir / f"{slug}_clap_attribute_l2.partial.npy"
    progress_path = output_dir / f"{slug}_clap_progress.json"
    if final_path.exists() or audio_embedding_path.exists() or text_embedding_path.exists():
        raise FileExistsError(f"Completed CLAP artifact already exists for {args.system}")

    model = laion_clap.CLAP_Module(
        enable_fusion=False,
        device=args.device,
        amodel="HTSAT-tiny",
        tmodel="roberta",
    )
    model.load_ckpt(str(checkpoint), verbose=False)
    embedding_dim = int(model.model.text_projection.shape[-1])
    tokenizer_revision = model.tokenize.init_kwargs.get("_commit_hash")
    identity = {
        "system": args.system,
        "examples": EXPECTED_EXAMPLES,
        "audio_manifest_sha256": sha256_file(audio_manifest),
        "verbalization_manifest_sha256": sha256_file(verbalization_manifest),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_name": checkpoint.name,
        "audio_model": "HTSAT-tiny",
        "text_model": "roberta-base",
        "text_model_revision": tokenizer_revision,
        "enable_fusion": False,
        "audio_crop": "one fixed-seed 10-second random crop from each 30-second waveform",
        "seed_rule": "seed_base + history_index; shared across systems",
        "seed_base": args.seed_base,
        "embedding_dim": embedding_dim,
    }
    if progress_path.exists() or partial_audio_path.exists() or partial_text_path.exists():
        if not all(path.exists() for path in
                   (progress_path, partial_audio_path, partial_text_path)):
            raise RuntimeError("Incomplete CLAP resume state")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError("Existing CLAP resume state has a different identity")
        completed = int(progress["completed"])
        audio_embeddings = np.lib.format.open_memmap(partial_audio_path, mode="r+")
        text_embeddings = np.lib.format.open_memmap(partial_text_path, mode="r+")
    else:
        completed = 0
        audio_embeddings = np.lib.format.open_memmap(
            partial_audio_path, mode="w+", dtype=np.float32,
            shape=(EXPECTED_EXAMPLES, embedding_dim))
        text_embeddings = np.lib.format.open_memmap(
            partial_text_path, mode="w+", dtype=np.float32,
            shape=(EXPECTED_EXAMPLES, embedding_dim))
        _atomic_json(progress_path, {"identity": identity, "completed": 0})
    expected_shape = (EXPECTED_EXAMPLES, embedding_dim)
    if audio_embeddings.shape != expected_shape or text_embeddings.shape != expected_shape:
        raise ValueError("Partial CLAP embedding shape drifted")

    for index in range(completed, EXPECTED_EXAMPLES):
        audio, attributes = _load_condition(
            audio_dir, verbalization_dir, args.system, index)
        seed = args.seed_base + index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            audio_value = model.get_audio_embedding_from_filelist([str(audio)])
            text_value = model.get_text_embedding([attributes])
        audio_embeddings[index] = _normalize(audio_value)[0]
        text_embeddings[index] = _normalize(text_value)[0]
        audio_embeddings.flush()
        text_embeddings.flush()
        _atomic_json(progress_path, {"identity": identity, "completed": index + 1})
        print(f"[CLAP-A] {args.system} {index + 1}/{EXPECTED_EXAMPLES}", flush=True)

    del audio_embeddings, text_embeddings
    os.replace(partial_audio_path, audio_embedding_path)
    os.replace(partial_text_path, text_embedding_path)
    progress_path.unlink()
    audio_embeddings = np.load(audio_embedding_path, allow_pickle=False)
    text_embeddings = np.load(text_embedding_path, allow_pickle=False)
    per_history = np.einsum("bd,bd->b", audio_embeddings, text_embeddings)
    payload = {
        "result_schema": "genplaylist-end-to-end-clap-a-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **identity,
        "package": {
            "name": "laion-clap",
            "version": importlib.metadata.version("laion-clap"),
        },
        "metric": "cosine similarity between generated audio and generated music attributes",
        "clap_a": float(per_history.mean()),
        "confidence_interval_95": _bootstrap_mean_interval(
            per_history, samples=args.bootstrap_samples, seed=args.bootstrap_seed),
        "outputs": {
            audio_embedding_path.name: sha256_file(audio_embedding_path),
            text_embedding_path.name: sha256_file(text_embedding_path),
        },
    }
    _atomic_json(final_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
