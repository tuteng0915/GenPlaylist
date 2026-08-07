#!/usr/bin/env python3
"""Fail-fast validation for all artifacts required before WP-C training."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from shared.artifacts import load_catalog_artifacts  # noqa: E402
from shared.protocol import FROZEN_NEXT_SONG_PROTOCOL  # noqa: E402

TOKENIZER_PATH = SRC_ROOT / "03_backbone_recommender" / "genplaylist_tokenizer.py"
spec = importlib.util.spec_from_file_location("genplaylist_tokenizer", TOKENIZER_PATH)
tokenizer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tokenizer_module)
GenPlaylistTokenizer = tokenizer_module.GenPlaylistTokenizer


def read_split(path: Path) -> list[list[str]]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = [value.strip() for value in raw.split(",")]
        if len(fields) < 3:
            raise ValueError(f"{path}:{line_no}: malformed playlist")
        rows.append(fields[1:])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=REPO_ROOT / "data" / "dataset")
    parser.add_argument(
        "--cue-dir", type=Path,
        default=SRC_ROOT / "02_creative_cues" / "outputs" / "production" / "latest")
    args = parser.parse_args()

    data = args.data_dir
    artifacts = load_catalog_artifacts(
        data / "catalog_metadata.json",
        data / "catalog_item_embeddings.npy",
        data / "item_id_to_row.json")
    tokenizer = GenPlaylistTokenizer.from_files(
        semantic_tokens_path=data / "semantic_tokens.json",
        item2cues_path=args.cue_dir / "item2cues.json",
        cue_manifest_path=args.cue_dir / "cue_manifest.json",
        catalog_items=artifacts.items,
        catalog_embeddings=artifacts.item_embeddings,
        item_id_to_row=artifacts.item_id_to_row,
        codebook_weights_path=data / "rvq_codebook_weights.npy")

    known = set(artifacts.item_id_to_row)
    report = {"catalog_items": len(known), "splits": {}}
    for split, filename in (("train", "train.txt"), ("val", "val.txt"), ("test", "test.txt")):
        playlists = read_split(data / "splits" / filename)
        missing = sorted({item for playlist in playlists for item in playlist} - known)
        if missing:
            raise ValueError(f"{split} references catalog-missing IDs: {missing[:10]}")
        lengths = np.asarray([len(playlist) for playlist in playlists])
        report["splits"][split] = {
            "playlists": len(playlists),
            "min_length": int(lengths.min()),
            "max_length": int(lengths.max()),
            "mean_length": float(lengths.mean()),
        }

    first = read_split(data / "splits" / "train.txt")[0]
    train_window = first[:FROZEN_NEXT_SONG_PROTOCOL.train_total_items]
    smoke = tokenizer.encode_playlist(
        train_window, context_items=len(train_window) - 1)
    report["smoke_sequence_length"] = int(len(smoke.input_ids))
    report["smoke_target_tokens"] = int(smoke.target_mask.sum())
    print(json.dumps(report, indent=2))
    print("[validate] all GenPlaylist-v1 artifacts are mutually compatible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
