#!/usr/bin/env python3
"""Prepare offline blinded Full-versus-generated-control assets, without collection.

This separate v2 format must not be fed to the legacy Full-versus-real service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_listener_study import (
    _atomic_json, _display_item, _generated_on_side_a, _load_metadata,
    _load_sequences, _prepare_clip, _select_indices, _validated_generated_audio,
)
from validate_matched_audio import ARMS, sha, validate


def assignments(indices, seed, control):
    side_a = _generated_on_side_a(len(indices), seed)
    return [{"index": index, "full_is_a": is_a,
             "case_id": "case-" + hashlib.sha256(
                 f"matched-listener-v2:{seed}:{control}:{index}".encode()).hexdigest()[:16]}
            for index, is_a in zip(indices, side_a)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Four-arm generation root")
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--catalog-metadata", type=Path, required=True)
    parser.add_argument("--control", choices=("drop", "sft", "direct"), required=True)
    parser.add_argument("--design", type=Path, required=True, help="Frozen listener-study-design.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh directory; existing study assets are immutable")
    design = json.loads(args.design.read_text())
    indices = design["context_indices"]
    expected = _select_indices(941, 25, 42)
    if indices != expected or design["context_seed"] != 42:
        raise ValueError("Study contexts differ from the frozen 25-case seed42 selection")
    validation = validate(args.root, indices)
    sequences = _load_sequences(args.prepared_dir)
    metadata = _load_metadata(args.catalog_metadata)
    # Fail before making a package if any visible metadata is unavailable.
    for index in indices:
        for item in sequences[index][:15]:
            _display_item(item, metadata)
    args.output_dir.mkdir(parents=True, mode=0o700)
    public_cases, private_cases = [], []
    for row in assignments(indices, 42, args.control):
        index, case_id = row["index"], row["case_id"]
        directory = args.output_dir / "participant_assets" / case_id
        directory.mkdir(parents=True, mode=0o700)
        directory.parent.chmod(0o700)
        full_source = _validated_generated_audio(args.root / "full/synthesis", ARMS["full"], index)
        control_source = _validated_generated_audio(args.root / args.control / "synthesis", ARMS[args.control], index)
        sources = [full_source, control_source] if row["full_is_a"] else [control_source, full_source]
        paths = [directory / "song_a.wav", directory / "song_b.wav"]
        for source, destination in zip(sources, paths):
            _prepare_clip(source, destination, 30.0, 44100)
            destination.chmod(0o600)
        public_cases.append({"case_id": case_id,
            "reference_music": [_display_item(item, metadata) for item in sequences[index][:15]],
            "song_a": str(paths[0].relative_to(args.output_dir)),
            "song_b": str(paths[1].relative_to(args.output_dir))})
        private_cases.append({**row, "control": args.control,
            "full_source_sha256": sha(full_source), "control_source_sha256": sha(control_source),
            "song_a_sha256": sha(paths[0]), "song_b_sha256": sha(paths[1])})
    public = {"result_schema": "genplaylist-matched-listener-public-v2",
              "design": "Blinded comparison of two generated candidates", "cases": public_cases}
    _atomic_json(args.output_dir / "public_manifest.json", public)
    private = {"result_schema": "genplaylist-matched-listener-private-v2",
               "control": args.control, "selection_seed": 42, "cases": private_cases,
               "full_side_a_count": sum(x["full_is_a"] for x in private_cases),
               "clip_seconds": 30, "sample_rate": 44100, "channels": 2, "format": "PCM16 WAV",
               "collection_started": False, "design_sha256": sha(args.design),
               "prepared_manifest_sha256": sha(args.prepared_dir / "prepared_manifest.json"),
               "public_manifest_sha256": sha(args.output_dir / "public_manifest.json")}
    _atomic_json(args.output_dir / "private_manifest.json", private)
    _atomic_json(args.output_dir / "input_validation.json", validation)
    for path in args.output_dir.glob("*.json"):
        path.chmod(0o600)
    print(f"Prepared {len(indices)} blinded offline cases; no responses collected")


if __name__ == "__main__":
    main()
