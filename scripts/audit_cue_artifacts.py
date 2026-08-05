#!/usr/bin/env python3
"""Audit a frozen cue vocabulary and ranked per-song cue table."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shared.protocol import FROZEN_NEXT_SONG_PROTOCOL


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_test_windows(mapping: dict, test_split: Path) -> dict:
    eligible = 0
    reference_items = 0
    target_items = 0
    missing: set[str] = set()
    with test_split.open("r", encoding="utf-8") as handle:
        for raw in handle:
            fields = [value.strip() for value in raw.strip().split(",")]
            if len(fields) <= FROZEN_NEXT_SONG_PROTOCOL.eval_total_items:
                continue
            songs = fields[1:]
            if len(songs) < FROZEN_NEXT_SONG_PROTOCOL.eval_total_items:
                continue
            references, targets = FROZEN_NEXT_SONG_PROTOCOL.split_evaluation_items(songs)
            eligible += 1
            reference_items += len(references)
            target_items += len(targets)
            missing.update(item_id for item_id in [*references, *targets]
                           if item_id not in mapping)
    return {
        "test_split": str(test_split.resolve()),
        "eligible_playlists": eligible,
        "reference_slots": reference_items,
        "target_slots": target_items,
        "missing_item_count": len(missing),
        "missing_items": sorted(missing)[:25],
    }


def audit(cue_dir: Path, test_split: Path | None = None) -> dict:
    vocab_path = cue_dir / "cue_vocab.json"
    mapping_path = cue_dir / "item2cues.json"
    scores_path = cue_dir / "item2cue_scores.json"
    vocab = _load(vocab_path)
    mapping = _load(mapping_path)
    scores = _load(scores_path) if scores_path.is_file() else None

    if not isinstance(vocab, list) or not vocab or vocab[0] != "<unk>":
        raise ValueError("cue_vocab.json must be a non-empty list with <unk> at ID 0")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("item2cues.json must be a non-empty object")

    lengths = collections.Counter(len(row) for row in mapping.values())
    invalid_ids = []
    duplicate_rows = 0
    unk_slots = 0
    counts: collections.Counter[int] = collections.Counter()
    for item_id, row in mapping.items():
        if len(row) != len(set(cue_id for cue_id in row if cue_id != 0)) + row.count(0):
            duplicate_rows += 1
        for cue_id in row:
            if not isinstance(cue_id, int) or not 0 <= cue_id < len(vocab):
                invalid_ids.append([item_id, cue_id])
                continue
            counts[cue_id] += 1
            unk_slots += cue_id == 0

    score_order_violations = None
    score_shape_mismatches = None
    if scores is not None:
        if set(scores) != set(mapping):
            raise ValueError("item2cue_scores IDs do not match item2cues IDs")
        score_order_violations = 0
        score_shape_mismatches = 0
        for item_id, row in mapping.items():
            score_row = scores[item_id]
            if len(score_row) != len(row):
                score_shape_mismatches += 1
                continue
            real_scores = [score for cue_id, score in zip(row, score_row) if cue_id != 0]
            if any(score is None for score in real_scores) or any(
                left < right for left, right in zip(real_scores, real_scores[1:])
            ):
                score_order_violations += 1

    used_non_unk = {cue_id for cue_id in counts if cue_id != 0}
    top = [
        {"cue_id": cue_id, "cue": vocab[cue_id], "count": count,
         "item_fraction": round(count / len(mapping), 6)}
        for cue_id, count in counts.most_common(50) if cue_id != 0
    ][:25]
    report = {
        "cue_dir": str(cue_dir.resolve()),
        "cue_vocab_sha256": _sha256(vocab_path),
        "item2cues_sha256": _sha256(mapping_path),
        "item2cue_scores_sha256": _sha256(scores_path) if scores_path.is_file() else None,
        "vocab_size": len(vocab),
        "pad_slots": sum(cue.startswith("<pad_") for cue in vocab),
        "n_items": len(mapping),
        "stored_length_counts": dict(sorted(lengths.items())),
        "total_slots": sum(counts.values()),
        "unk_slots": unk_slots,
        "duplicate_rows": duplicate_rows,
        "invalid_id_count": len(invalid_ids),
        "distinct_non_unk_assigned": len(used_non_unk),
        "vocab_utilization": round(len(used_non_unk) / max(len(vocab) - 1, 1), 6),
        "score_order_violations": score_order_violations,
        "score_shape_mismatches": score_shape_mismatches,
        "top_assigned_cues": top,
    }
    if test_split is not None:
        report["frozen_15_to_5_coverage"] = _audit_test_windows(
            mapping, test_split)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cue_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--test-split", type=Path,
        help="Also verify cue coverage for every first-20 15->5 evaluation window")
    args = parser.parse_args()
    report = audit(args.cue_dir, test_split=args.test_split)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
