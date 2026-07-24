"""Decode creative cue ids into cue text for each song.

Example:
    python scripts/reconstruct_cue_mappings.py --method tfidf
    python scripts/reconstruct_cue_mappings.py --method llm --format md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / "src" / "02_creative_cues" / "outputs"
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "dataset" / "catalog_metadata.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    """Return item_id -> metadata, accepting either dict or list JSON shapes."""
    if not path.exists():
        return {}

    raw = load_json(path)
    if isinstance(raw, dict):
        return {str(item_id): entry for item_id, entry in raw.items() if isinstance(entry, dict)}
    if isinstance(raw, list):
        catalog = {}
        for entry in raw:
            if isinstance(entry, dict) and "item_id" in entry:
                catalog[str(entry["item_id"])] = entry
        return catalog
    raise ValueError(f"Unsupported catalog format in {path}")


def cue_text_for_id(cue_id: Any, vocab: list[str]) -> str:
    try:
        index = int(cue_id)
    except (TypeError, ValueError):
        return f"<invalid:{cue_id}>"

    if 0 <= index < len(vocab):
        return vocab[index]
    return f"<missing:{index}>"


def build_rows(
    item2cues: dict[str, list[int]],
    vocab: list[str],
    catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for item_id, cue_ids in item2cues.items():
        meta = catalog.get(str(item_id), {})
        cues = [cue_text_for_id(cue_id, vocab) for cue_id in cue_ids]
        rows.append(
            {
                "item_id": str(item_id),
                "title": meta.get("title", ""),
                "artist": meta.get("artist", ""),
                "cue_ids": cue_ids,
                "cues": cues,
            }
        )
    return rows


def write_json(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "title", "artist", "cues", "cue_ids"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "item_id": row["item_id"],
                    "title": row["title"],
                    "artist": row["artist"],
                    "cues": ", ".join(row["cues"]),
                    "cue_ids": " ".join(str(cue_id) for cue_id in row["cue_ids"]),
                }
            )


def write_md(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Song Cue Mappings\n\n")
        for row in rows:
            label = row["item_id"]
            if row["title"] and row["artist"]:
                label = f'{row["title"]} - {row["artist"]} ({row["item_id"]})'
            elif row["title"]:
                label = f'{row["title"]} ({row["item_id"]})'
            f.write(f"- **{label}**: {', '.join(row['cues'])}\n")


WRITERS = {
    "json": write_json,
    "jsonl": write_jsonl,
    "csv": write_csv,
    "md": write_md,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct song-to-cue mappings as cue text instead of cue ids."
    )
    parser.add_argument(
        "--method",
        default="tfidf",
        help="Creative cue output subdirectory to read, e.g. tfidf, yake, keybert, llm.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing item2cues.json and cue_vocab.json. Overrides --method.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help="Path to catalog_metadata.json for song titles and artists.",
    )
    parser.add_argument(
        "--format",
        choices=sorted(WRITERS),
        default="json",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path. Defaults to item2cues_text.<format> next to the inputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir or (DEFAULT_OUTPUTS_ROOT / args.method)
    output_path = args.output or (input_dir / f"item2cues_text.{args.format}")

    item2cues_path = input_dir / "item2cues.json"
    vocab_path = input_dir / "cue_vocab.json"

    if not item2cues_path.exists():
        raise FileNotFoundError(f"Missing item2cues file: {item2cues_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing cue vocab file: {vocab_path}")

    item2cues = load_json(item2cues_path)
    vocab = load_json(vocab_path)
    if not isinstance(item2cues, dict):
        raise ValueError(f"Expected dict in {item2cues_path}")
    if not isinstance(vocab, list):
        raise ValueError(f"Expected list in {vocab_path}")

    rows = build_rows(item2cues, vocab, load_catalog(args.catalog))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    WRITERS[args.format](rows, output_path)

    print(f"Wrote {len(rows)} decoded cue mappings to {output_path}")


if __name__ == "__main__":
    main()
