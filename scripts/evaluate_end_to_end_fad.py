#!/usr/bin/env python3
"""Compute VGGish Frechet Audio Distance for the three audio systems."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from shared.artifacts import sha256_file  # noqa: E402
from extract_end_to_end_mert import EXPECTED_EXAMPLES, SYSTEMS, _slug  # noqa: E402


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _test_targets(prepared_dir: Path) -> list[str]:
    from datasets import load_from_disk

    sequences = load_from_disk(str(prepared_dir / "raw_dataset"))["test"]["item_seq"]
    if len(sequences) != EXPECTED_EXAMPLES or any(len(row) != 20 for row in sequences):
        raise ValueError("Frozen end-to-end histories must have shape [941, 20]")
    return [str(row[15]) for row in sequences]


def _ensure_link(directory: Path, name: str, source: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / name
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError(f"Staging link points to a different file: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"Refusing to replace staging entry: {destination}")
    destination.symlink_to(source.resolve())


def _validate_staging(directory: Path) -> None:
    entries = sorted(directory.iterdir())
    if len(entries) != EXPECTED_EXAMPLES:
        raise ValueError(f"FAD staging directory must have 941 files: {directory}")
    if any(path.suffix.lower() != ".mp3" or not path.is_file() for path in entries):
        raise ValueError(f"FAD staging directory contains a non-MP3 entry: {directory}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--catalog-audio-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-load-workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.audio_load_workers <= 0:
        raise ValueError("--audio-load-workers must be positive")
    from frechet_audio_distance import FrechetAudioDistance

    audio_dir = args.audio_dir.expanduser().resolve()
    prepared_dir = args.prepared_dir.expanduser().resolve()
    catalog_audio_dir = args.catalog_audio_dir.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    audio_manifest = audio_dir / "audio_manifest.json"
    if not audio_manifest.is_file():
        raise FileNotFoundError(audio_manifest)

    target_dir = work_dir / "staging" / "real-next"
    for index, item_id in enumerate(_test_targets(prepared_dir)):
        _ensure_link(target_dir, f"{index:04d}.mp3", catalog_audio_dir / f"{item_id}.mp3")
    _validate_staging(target_dir)
    for system in SYSTEMS:
        system_dir = work_dir / "staging" / _slug(system)
        for index in range(EXPECTED_EXAMPLES):
            _ensure_link(
                system_dir, f"{index:04d}.mp3",
                audio_dir / system / f"{index:04d}.mp3")
        _validate_staging(system_dir)

    evaluator = FrechetAudioDistance(
        model_name="vggish",
        sample_rate=16_000,
        channels=1,
        use_pca=False,
        use_activation=False,
        verbose=True,
        audio_load_worker=args.audio_load_workers,
    )
    embeddings_dir = work_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    background_embeddings = embeddings_dir / "real_next_vggish.npy"
    scores = {}
    artifact_hashes = {}
    for system in SYSTEMS:
        generated_embeddings = embeddings_dir / f"{_slug(system)}_vggish.npy"
        score = evaluator.score(
            str(target_dir),
            str(work_dir / "staging" / _slug(system)),
            background_embds_path=str(background_embeddings),
            eval_embds_path=str(generated_embeddings),
        )
        scores[system] = float(score)
        artifact_hashes[system] = sha256_file(generated_embeddings)

    checkpoint_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    checkpoint_hashes = {
        path.name: sha256_file(path)
        for path in sorted(checkpoint_dir.glob("vggish*.pth"))
    }
    payload = {
        "result_schema": "genplaylist-end-to-end-fad-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "examples_per_corpus": EXPECTED_EXAMPLES,
        "audio_manifest_sha256": sha256_file(audio_manifest),
        "implementation": {
            "package": "frechet-audio-distance",
            "version": importlib.metadata.version("frechet-audio-distance"),
            "model": "VGGish",
            "sample_rate": 16_000,
            "channels": 1,
            "use_pca": False,
            "use_activation": False,
            "checkpoint_sha256": checkpoint_hashes,
        },
        "embedding_sha256": {
            "real_next": sha256_file(background_embeddings),
            **artifact_hashes,
        },
        "fad": scores,
    }
    _atomic_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
