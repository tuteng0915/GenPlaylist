#!/usr/bin/env python3
"""Render frozen Qwen3 verbalizations with ACE-Step for the audio study.

The runner is resumable and shardable.  For a given history index, every
system receives the same explicit ACE-Step diffusion seed, so system
comparisons do not confound planner conditioning with different initial noise.
It does not import or modify the WP-D demo.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("ACE-Step-Direct", "DDBC-SFT", "GenPlaylist")
EXPECTED_EXAMPLES = 941


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _model_revision(checkpoint_dir: Path) -> str:
    if checkpoint_dir.parent.name != "snapshots":
        raise ValueError(
            "--checkpoint-dir must be an immutable Hugging Face snapshot path")
    required = ("ace_step_transformer", "music_dcae_f8c8", "music_vocoder", "umt5-base")
    missing = [name for name in required if not (checkpoint_dir / name).is_dir()]
    if missing:
        raise ValueError(f"Incomplete ACE-Step checkpoint: {missing}")
    return checkpoint_dir.name


def _load_verbalization(path: Path, system: str, index: int) -> dict:
    source = path / system / f"{index:04d}.json"
    if not source.is_file():
        raise FileNotFoundError(f"Missing frozen verbalization: {source}")
    record = json.loads(source.read_text(encoding="utf-8"))
    if record.get("system") != system or record.get("example_index") != index:
        raise ValueError(f"Verbalization identity mismatch: {source}")
    if not str(record.get("music_attributes", "")).strip():
        raise ValueError(f"Empty music attributes: {source}")
    if not str(record.get("lyric_draft", "")).strip():
        raise ValueError(f"Empty lyric draft: {source}")
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbalization-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ace-step-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--inference-steps", type=int, default=60)
    parser.add_argument("--seed-base", type=int, default=42000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--system", action="append", choices=SYSTEMS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verbalization_dir = args.verbalization_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    ace_step_path = args.ace_step_path.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    systems = tuple(args.system or SYSTEMS)
    if not 1.0 <= args.duration <= 600.0:
        raise ValueError("--duration must be in [1, 600]")
    if args.inference_steps <= 0 or args.seed_base < 0:
        raise ValueError("Inference steps must be positive and seed base nonnegative")
    stop = EXPECTED_EXAMPLES if args.max_examples is None else min(
        EXPECTED_EXAMPLES, args.start_index + args.max_examples)
    if not 0 <= args.start_index < stop <= EXPECTED_EXAMPLES:
        raise ValueError(f"Invalid example interval {args.start_index}:{stop}")

    verbalization_manifest = verbalization_dir / "verbalization_manifest.json"
    if not verbalization_manifest.is_file():
        raise FileNotFoundError(verbalization_manifest)
    revision = _model_revision(checkpoint_dir)
    identity = {
        "result_schema": "genplaylist-end-to-end-audio-v1",
        "git_commit": _git_commit(REPO_ROOT),
        "verbalization_manifest_sha256": _sha256(verbalization_manifest),
        "ace_step": {
            "source_git_commit": _git_commit(ace_step_path),
            "model": "ACE-Step/ACE-Step-v1-3.5B",
            "model_revision": revision,
            "duration_seconds": args.duration,
            "inference_steps": args.inference_steps,
            "format": "mp3",
            "dtype": "bfloat16",
            "guidance_scale": 15.0,
            "scheduler_type": "euler",
            "cfg_type": "apg",
            "omega_scale": 10.0,
        },
        "seed": {
            "rule": "seed_base + history_index; shared across systems",
            "seed_base": args.seed_base,
        },
        "systems": list(systems),
        "examples": EXPECTED_EXAMPLES,
    }
    manifest_path = output_dir / "audio_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = dict(existing)
        comparable.pop("created_utc", None)
        if comparable != identity:
            raise ValueError("Existing audio directory has a different identity")
    else:
        _atomic_json(manifest_path, {
            **identity,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })

    sys.path.insert(0, str(ace_step_path))
    from acestep.pipeline_ace_step import ACEStepPipeline

    pipeline = ACEStepPipeline(
        checkpoint_dir=str(checkpoint_dir),
        device_id=args.device_id,
        dtype="bfloat16",
        torch_compile=False,
    )
    for index in range(args.start_index, stop):
        seed = args.seed_base + index
        for system in systems:
            destination = output_dir / system / f"{index:04d}.mp3"
            metadata_path = destination.with_suffix(".json")
            if destination.is_file() and metadata_path.is_file():
                continue
            record = _load_verbalization(verbalization_dir, system, index)
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial_dir = output_dir / ".partial" / system
            partial_dir.mkdir(parents=True, exist_ok=True)
            partial = partial_dir / f"{index:04d}.{os.getpid()}.mp3"
            outputs = pipeline(
                prompt=record["music_attributes"],
                lyrics=record["lyric_draft"],
                audio_duration=args.duration,
                infer_step=args.inference_steps,
                guidance_scale=15.0,
                scheduler_type="euler",
                cfg_type="apg",
                omega_scale=10.0,
                manual_seeds=[seed],
                format="mp3",
                save_path=str(partial),
            )
            actual = Path(outputs[0]).resolve()
            if actual != partial.resolve() or not partial.is_file():
                raise RuntimeError(f"ACE-Step returned an unexpected output: {outputs}")
            os.replace(partial, destination)
            _atomic_json(metadata_path, {
                "result_schema": "genplaylist-end-to-end-audio-record-v1",
                "system": system,
                "example_index": index,
                "seed": seed,
                "audio_sha256": _sha256(destination),
                "verbalization_sha256": _sha256(
                    verbalization_dir / system / f"{index:04d}.json"),
                "audio_path": str(destination),
            })
            print(f"[ACE-Step] {system} {index + 1}/{stop} seed={seed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
