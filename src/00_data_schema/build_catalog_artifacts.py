#!/usr/bin/env python3
"""Build the catalog row mapping and a versioned artifact manifest.

This does not create CLHE embeddings. It establishes the authoritative row
order that any per-item embedding builder must follow.

Run from the repository root:

    python src/00_data_schema/build_catalog_artifacts.py

Optionally validate an existing per-item embedding matrix:

    python src/00_data_schema/build_catalog_artifacts.py \
      --embeddings data/dataset/catalog_item_embeddings.npy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from shared.artifacts import (  # noqa: E402
    build_item_id_to_row,
    sha256_file,
    validate_catalog_alignment,
)
from shared.schema import CLHE_EMB_DIM, SCHEMA_VERSION, TOKEN_LAYOUT, CatalogItem  # noqa: E402


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPO_ROOT / "data" / "dataset" / "catalog_metadata.json",
    )
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=REPO_ROOT / "data" / "dataset" / "item_id_to_row.json",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=REPO_ROOT / "data" / "dataset" / "artifact_manifest.json",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Optional N x 64 per-item CLHE matrix to validate and record",
    )
    args = parser.parse_args()

    items = CatalogItem.load_catalog(str(args.catalog))
    mapping = build_item_id_to_row(items)
    _atomic_write_json(args.mapping_output, mapping)

    embedding_record = None
    if args.embeddings is not None:
        embeddings = np.load(args.embeddings, allow_pickle=False).astype(np.float32)
        validate_catalog_alignment(items, embeddings, mapping)
        embedding_record = {
            "path": str(args.embeddings),
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "sha256": sha256_file(args.embeddings),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "catalog": {
            "path": str(args.catalog),
            "n_items": len(items),
            "sha256": sha256_file(args.catalog),
        },
        "item_id_to_row": {
            "path": str(args.mapping_output),
            "n_items": len(mapping),
            "sha256": sha256_file(args.mapping_output),
        },
        "catalog_item_embeddings": embedding_record,
        "embedding_dim": CLHE_EMB_DIM,
        "token_layout": {
            "rq_n_codebooks": TOKEN_LAYOUT.rq_n_codebooks,
            "rq_codebook_size": TOKEN_LAYOUT.rq_codebook_size,
            "conflict_vocab_size": TOKEN_LAYOUT.conflict_vocab_size,
            "cue_tokens": TOKEN_LAYOUT.cue_tokens,
            "cue_vocab_size": TOKEN_LAYOUT.cue_vocab_size,
            "tokens_per_item": TOKEN_LAYOUT.tokens_per_item,
            "boi_token": TOKEN_LAYOUT.boi_token,
            "eos_token": TOKEN_LAYOUT.eos_token,
            "cue_token_start": TOKEN_LAYOUT.cue_token_start,
            "mask_token": TOKEN_LAYOUT.mask_token,
            "vocab_size_excluding_mask": TOKEN_LAYOUT.vocab_size,
            "runtime_vocab_size": TOKEN_LAYOUT.runtime_vocab_size,
        },
    }
    _atomic_write_json(args.manifest_output, manifest)

    print(f"[artifacts] catalog items : {len(items)}")
    print(f"[artifacts] row mapping   : {args.mapping_output}")
    print(f"[artifacts] manifest      : {args.manifest_output}")
    if embedding_record is None:
        print("[artifacts] embeddings    : not provided; manifest records null")
    else:
        print(f"[artifacts] embeddings    : validated {embedding_record['shape']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
