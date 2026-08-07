"""Command-line entry point for the complete GenPlaylist next-song pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from pipeline import GenPlaylistPipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one next song from multiple ordered catalog references. "
            "Artifact, checkpoint, and synthesis paths use GENPLAYLIST_* / "
            "ACE_STEP_* environment variables."))
    parser.add_argument(
        "--references", nargs="+", metavar="ITEM_ID",
        help="Ordered catalog item IDs; at least two are required")
    parser.add_argument(
        "--instruction", default="",
        help="Optional creative instruction forwarded to WP-D")
    parser.add_argument("--audio-duration", type=int, default=30)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Validate lightweight artifacts without loading DDBC or ACE-Step")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    pipeline = GenPlaylistPipeline.from_environment()
    print(json.dumps(pipeline.preflight(), indent=2, ensure_ascii=False))
    if args.preflight_only:
        return 0
    if not args.references or len(args.references) < 2:
        raise SystemExit("--references requires at least two ordered item IDs")

    result = pipeline.generate(
        args.references,
        reference_count=len(args.references),
        user_instruction=args.instruction,
        audio_duration=args.audio_duration,
        k_neighbors=args.k_neighbors,
    )
    print(json.dumps({
        "audio_path": result.audio_path,
        "music_attributes": result.music_attributes,
        "lyric_draft": result.lyric_draft,
        "neighbor_item_ids": [item.item_id for item in result.neighbors],
        "reference_item_ids": result.generated_item.context_prefix.item_ids,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
