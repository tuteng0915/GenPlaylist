#!/usr/bin/env python3
"""Validate the frozen four-arm audio comparison before evaluation or blinding."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = {"full": "GenPlaylist", "drop": "GenPlaylist", "sft": "DDBC-SFT", "direct": "ACE-Step-Direct"}


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def validate_pair(records, index):
    full = records["full"]
    for arm, system in ARMS.items():
        record = records[arm]
        if record["system"] != system or record["example_index"] != index:
            raise ValueError(f"Wrong arm/history identity: {arm}/{index}")
        for field in ("reference_item_ids", "target_item_id"):
            if record[field] != full[field]:
                raise ValueError(f"Unmatched {field}: {arm}/{index}")
        if len(record["reference_item_ids"]) != 15:
            raise ValueError("Expected 15 reference items")
        if not record["music_attributes"].strip() or not record["lyric_draft"].strip():
            raise ValueError(f"Empty generation conditions: {arm}/{index}")
    for field in ("retrieved_item_id", "neighbor_item_ids", "style_item_ids"):
        if records["drop"][field] != full[field]:
            raise ValueError(f"Cue intervention changed {field}: {index}")
    if len(full["cue_ids"]) != 8 or len(full["cue_terms"]) != 8:
        raise ValueError("Full must retain eight predicted cues")
    if records["drop"]["cue_ids"] or records["drop"]["cue_terms"]:
        raise ValueError("Dropped cues leaked into generation")


def validate(root, indices):
    import soundfile as sf
    manifests = {}
    for arm in ARMS:
        directory = root / arm
        verbal_path = directory / "verbalization/verbalization_manifest.json"
        audio = read(directory / "synthesis/audio_manifest.json")
        if audio["verbalization_manifest_sha256"] != sha(verbal_path):
            raise ValueError(f"Audio manifest references different verbalization: {arm}")
        manifests[arm] = (read(verbal_path), audio)
    for arm, (verbal, audio) in manifests.items():
        for key in ("model_name", "model_revision", "decoding", "seed", "examples", "reference_items", "selected_prediction", "inputs"):
            if verbal[key] != manifests["full"][0][key]:
                raise ValueError(f"Verbalization setting differs: {arm}/{key}")
        for key in ("ace_step", "seed", "examples"):
            if audio[key] != manifests["full"][1][key]:
                raise ValueError(f"Synthesis setting differs: {arm}/{key}")
    result = []
    for index in indices:
        records, checks = {}, {}
        for arm, system in ARMS.items():
            directory = root / arm
            verbal_path = directory / "verbalization" / system / f"{index:04d}.json"
            audio_path = directory / "synthesis" / system / f"{index:04d}.mp3"
            record = read(audio_path.with_suffix(".json"))
            if record["system"] != system or record["example_index"] != index or record["seed"] != 42000 + index:
                raise ValueError(f"Audio identity/seed mismatch: {arm}/{index}")
            if record["audio_sha256"] != sha(audio_path) or record["verbalization_sha256"] != sha(verbal_path):
                raise ValueError(f"Artifact hash mismatch: {arm}/{index}")
            values, rate = sf.read(audio_path, always_2d=True, dtype="float32")
            duration = len(values) / rate
            if not 29.9 <= duration <= 30.1 or not np.isfinite(values).all() or not np.any(values):
                raise ValueError(f"Invalid audio samples/duration: {arm}/{index}")
            records[arm] = read(verbal_path)
            checks[arm] = {"duration": duration, "sample_rate": rate, "channels": values.shape[1],
                           "rms": float(np.sqrt(np.mean(values.astype(np.float64) ** 2))),
                           "audio_sha256": record["audio_sha256"]}
        validate_pair(records, index)
        result.append({"index": index, "arms": checks})
    return {"schema": "genplaylist-matched-audio-validation-v1", "status": "passed",
            "complete_941": indices == list(range(941)), "validated_histories": len(indices),
            "records": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", help="Omit for all 941 histories")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    indices = sorted(set(args.indices)) if args.indices is not None else list(range(941))
    if not indices or any(i < 0 or i >= 941 for i in indices):
        raise ValueError("History index outside 0..940")
    if args.output.exists():
        raise FileExistsError(args.output)
    result = validate(args.root, indices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"Validated {len(indices)} histories across all four arms")


if __name__ == "__main__":
    main()
