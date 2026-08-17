#!/usr/bin/env python3
"""Prepare a frozen, blinded GenPlaylist-versus-real listener-study package."""

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


EXPECTED_CONTEXTS = 941
REFERENCE_ITEMS = 15
TARGET_OFFSET = 15


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _select_indices(total: int, count: int, seed: int) -> list[int]:
    if count <= 0 or count > total:
        raise ValueError(f"Study count must be in [1, {total}], got {count}")
    rng = np.random.default_rng(seed)
    return sorted(int(value) for value in rng.choice(total, size=count, replace=False))


def _generated_on_side_a(count: int, seed: int) -> list[bool]:
    """Return a deterministic side assignment balanced to at most one case."""
    values = np.asarray(
        [True] * ((count + 1) // 2) + [False] * (count // 2), dtype=np.bool_)
    np.random.default_rng(seed + 1).shuffle(values)
    return [bool(value) for value in values]


def _opaque_case_id(index: int, seed: int) -> str:
    digest = hashlib.sha256(f"genplaylist-study-v1:{seed}:{index}".encode()).hexdigest()
    return f"case-{digest[:12]}"


def _load_sequences(prepared_dir: Path) -> list[list[str]]:
    from datasets import load_from_disk

    values = [
        [str(item) for item in row]
        for row in load_from_disk(str(prepared_dir / "raw_dataset"))["test"]["item_seq"]
    ]
    if len(values) != EXPECTED_CONTEXTS or any(len(row) != 20 for row in values):
        raise ValueError("Frozen listener-study source must have shape [941, 20]")
    return values


def _load_metadata(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        values = {str(item["item_id"]): item for item in raw}
    elif isinstance(raw, dict):
        values = {str(key): value for key, value in raw.items()}
    else:
        raise ValueError("Catalog metadata must be a list or object")
    return values


def _display_item(item_id: str, metadata: dict[str, dict]) -> dict[str, str]:
    if item_id not in metadata:
        raise ValueError(f"Catalog metadata is missing item {item_id}")
    item = metadata[item_id]
    return {
        "title": str(item.get("title") or "Unknown title"),
        "artist": str(item.get("artist") or "Unknown artist"),
    }


def _prepare_clip(
    source: Path, destination: Path, seconds: float, sample_rate: int,
) -> float:
    """Decode both systems identically and save a fixed stereo PCM16 WAV."""
    import torch
    import torchaudio

    waveform, source_rate = torchaudio.load(str(source))
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError(f"Invalid audio samples: {source}")
    waveform = waveform.to(torch.float32)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    if int(source_rate) != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, int(source_rate), sample_rate)
    target_samples = int(round(seconds * sample_rate))
    missing = target_samples - int(waveform.shape[1])
    if missing > int(round(0.15 * sample_rate)):
        raise ValueError(
            f"Study source is too short: {source} has "
            f"{waveform.shape[1] / sample_rate:.3f}s")
    if missing > 0:
        waveform = torch.nn.functional.pad(waveform, (0, missing))
    elif missing < 0:
        start = (int(waveform.shape[1]) - target_samples) // 2
        waveform = waveform[:, start:start + target_samples]
    torchaudio.save(
        str(destination), waveform, sample_rate,
        encoding="PCM_S", bits_per_sample=16)
    return waveform.shape[1] / sample_rate


def _validated_generated_audio(audio_dir: Path, system: str, index: int) -> Path:
    audio = audio_dir / system / f"{index:04d}.mp3"
    record_path = audio.with_suffix(".json")
    if not audio.is_file() or not record_path.is_file():
        raise FileNotFoundError(f"Missing generated study candidate at example {index}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("system") != system or record.get("example_index") != index:
        raise ValueError(f"Generated-audio identity mismatch: {record_path}")
    if record.get("audio_sha256") != sha256_file(audio):
        raise ValueError(f"Generated-audio hash mismatch: {audio}")
    return audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--generated-audio-dir", type=Path, required=True)
    parser.add_argument("--catalog-audio-dir", type=Path, required=True)
    parser.add_argument("--catalog-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generated-system", default="GenPlaylist")
    parser.add_argument("--cases", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-seconds", type=float, default=30.0)
    parser.add_argument("--sample-rate", type=int, default=44_100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seed < 0 or args.clip_seconds <= 0 or args.sample_rate <= 0:
        raise ValueError("Seed must be nonnegative and audio settings positive")
    prepared_dir = args.prepared_dir.expanduser().resolve()
    generated_dir = args.generated_audio_dir.expanduser().resolve()
    catalog_audio_dir = args.catalog_audio_dir.expanduser().resolve()
    metadata_path = args.catalog_metadata.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to replace nonempty study package: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "participant_assets"
    assets_dir.mkdir()

    sequences = _load_sequences(prepared_dir)
    metadata = _load_metadata(metadata_path)
    indices = _select_indices(len(sequences), args.cases, args.seed)
    side_a = _generated_on_side_a(args.cases, args.seed)
    public_cases = []
    private_cases = []
    for ordinal, (index, generated_is_a) in enumerate(zip(indices, side_a)):
        sequence = sequences[index]
        case_id = _opaque_case_id(index, args.seed)
        case_dir = assets_dir / case_id
        case_dir.mkdir()
        generated_source = _validated_generated_audio(
            generated_dir, args.generated_system, index)
        target_item = sequence[TARGET_OFFSET]
        real_source = catalog_audio_dir / f"{target_item}.mp3"
        if not real_source.is_file():
            raise FileNotFoundError(real_source)

        path_a = case_dir / "song_a.wav"
        path_b = case_dir / "song_b.wav"
        generated_destination = path_a if generated_is_a else path_b
        real_destination = path_b if generated_is_a else path_a
        generated_duration = _prepare_clip(
            generated_source, generated_destination,
            args.clip_seconds, args.sample_rate)
        real_duration = _prepare_clip(
            real_source, real_destination, args.clip_seconds, args.sample_rate)

        public_cases.append({
            "case_id": case_id,
            "reference_music": [
                _display_item(item, metadata) for item in sequence[:REFERENCE_ITEMS]
            ],
            "song_a": str(path_a.relative_to(output_dir)),
            "song_b": str(path_b.relative_to(output_dir)),
        })
        private_cases.append({
            "ordinal": ordinal,
            "case_id": case_id,
            "example_index": index,
            "reference_item_ids": sequence[:REFERENCE_ITEMS],
            "real_next_item_id": target_item,
            "generated_system": args.generated_system,
            "generated_is_song_a": generated_is_a,
            "generated_source_sha256": sha256_file(generated_source),
            "real_source_sha256": sha256_file(real_source),
            "song_a_sha256": sha256_file(path_a),
            "song_b_sha256": sha256_file(path_b),
            "song_a_seconds": args.clip_seconds,
            "song_b_seconds": args.clip_seconds,
            "generated_clip_seconds": generated_duration,
            "real_center_clip_seconds": real_duration,
        })
        print(f"[study] {ordinal + 1}/{args.cases} {case_id}", flush=True)

    public_manifest = {
        "result_schema": "genplaylist-listener-study-public-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design": "blinded paired A/B: GenPlaylist versus immediate real next track",
        "cases": public_cases,
    }
    public_path = output_dir / "public_manifest.json"
    _atomic_json(public_path, public_manifest)
    generated_manifest = generated_dir / "audio_manifest.json"
    private_manifest = {
        "result_schema": "genplaylist-listener-study-private-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "population": EXPECTED_CONTEXTS,
            "method": "uniform without replacement",
            "cases": args.cases,
            "seed": args.seed,
        },
        "protocol": {
            "reference_items": REFERENCE_ITEMS,
            "target_offset_one_based": TARGET_OFFSET + 1,
            "clip_seconds": args.clip_seconds,
            "sample_rate": args.sample_rate,
            "channels": 2,
            "encoding": "PCM16 WAV; identical conversion for both candidates",
            "generated_side_balance": {
                "song_a": sum(side_a),
                "song_b": len(side_a) - sum(side_a),
            },
        },
        "inputs": {
            "prepared_manifest_sha256": sha256_file(
                prepared_dir / "prepared_manifest.json"),
            "generated_audio_manifest_sha256": sha256_file(generated_manifest),
            "catalog_metadata_sha256": sha256_file(metadata_path),
            "public_manifest_sha256": sha256_file(public_path),
        },
        "cases": private_cases,
    }
    _atomic_json(output_dir / "private_manifest.json", private_manifest)
    print(json.dumps(private_manifest["selection"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
