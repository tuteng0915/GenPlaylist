#!/usr/bin/env python3
"""Audit catalog-projected Music4All windows under skipped-event limits.

Input is the stable user/timestamp/source-row sorted table produced by
``prepare_music4all_sequences.py``.  A threshold of K permits at most K
unsupported listening events between adjacent mapped events; ``all`` is the
fully catalog-projected subsequence.  Repeated mapped listens are retained.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


WINDOW_ITEMS = 20
REFERENCE_ITEMS = 15
TARGET_ITEMS = 5
CAP_CANDIDATES = (16, 32, 64, 100, 200)
SCHEMA = "genplaylist-music4all-gap-audit-v1"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _parse_thresholds(value: str) -> tuple[int | None, ...]:
    output: list[int | None] = []
    for field in value.split(","):
        field = field.strip().casefold()
        threshold = None if field in {"all", "none", "unbounded"} else int(field)
        if threshold is not None and threshold < 0:
            raise ValueError("Gap thresholds must be nonnegative or 'all'")
        if threshold not in output:
            output.append(threshold)
    if not output:
        raise ValueError("At least one threshold is required")
    return tuple(output)


def _split(user_id: str, seed: int, test_fraction: float) -> str:
    digest = hashlib.sha256(
        f"split\0{seed}\0{user_id}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "test" if value < test_fraction else "train"


def _new_stats() -> dict:
    return {
        "windows": 0,
        "windows_with_any_repeat": 0,
        "windows_with_target_reference_overlap": 0,
        "windows_with_duplicate_targets": 0,
        "unique_items_histogram": Counter(),
        "unique_targets_histogram": Counter(),
        "eligible_users_by_split": Counter(),
        "eligible_windows_by_split": Counter(),
        "train_window_counts": [],
        "qualifying_windows": 0,
        "qualifying_windows_with_any_repeat": 0,
        "qualifying_windows_with_target_reference_overlap": 0,
        "qualifying_windows_with_duplicate_targets": 0,
        "qualifying_users_by_split": Counter(),
        "qualifying_windows_by_split": Counter(),
        "qualifying_train_window_counts": [],
    }


def _audit_sorted(
    path: Path,
    thresholds: tuple[int | None, ...],
    seed: int,
    test_fraction: float,
    progress_every: int,
    min_unique_references: int = 1,
    min_unique_targets: int = 1,
) -> dict:
    states = {
        threshold: deque(maxlen=WINDOW_ITEMS) for threshold in thresholds
    }
    stats = {threshold: _new_stats() for threshold in thresholds}
    user_windows = {threshold: 0 for threshold in thresholds}
    user_qualifying_windows = {threshold: 0 for threshold in thresholds}
    current_user: str | None = None
    gap_since_mapped = 0
    seen_mapped = False
    total_events = 0
    mapped_events = 0

    def finalize_user() -> None:
        if current_user is None:
            return
        user_split = _split(current_user, seed, test_fraction)
        for threshold in thresholds:
            count = user_windows[threshold]
            if not count:
                continue
            stats[threshold]["eligible_users_by_split"][user_split] += 1
            stats[threshold]["eligible_windows_by_split"][user_split] += count
            if user_split == "train":
                stats[threshold]["train_window_counts"].append(count)
            qualifying_count = user_qualifying_windows[threshold]
            if not qualifying_count:
                continue
            stats[threshold]["qualifying_users_by_split"][user_split] += 1
            stats[threshold]["qualifying_windows_by_split"][user_split] += (
                qualifying_count
            )
            if user_split == "train":
                stats[threshold]["qualifying_train_window_counts"].append(
                    qualifying_count
                )

    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"Malformed sorted row {line_number}")
            user_id, _, _, item_id = fields
            total_events += 1
            if user_id != current_user:
                finalize_user()
                current_user = user_id
                gap_since_mapped = 0
                seen_mapped = False
                states = {
                    threshold: deque(maxlen=WINDOW_ITEMS)
                    for threshold in thresholds
                }
                user_windows = {threshold: 0 for threshold in thresholds}
                user_qualifying_windows = {
                    threshold: 0 for threshold in thresholds
                }
            if not item_id:
                if seen_mapped:
                    gap_since_mapped += 1
                continue

            mapped_events += 1
            for threshold in thresholds:
                sequence = states[threshold]
                if (
                    seen_mapped
                    and threshold is not None
                    and gap_since_mapped > threshold
                ):
                    sequence.clear()
                sequence.append(item_id)
                if len(sequence) != WINDOW_ITEMS:
                    continue
                items = tuple(sequence)
                targets = items[REFERENCE_ITEMS:]
                unique_items = len(set(items))
                unique_targets = len(set(targets))
                reference_set = set(items[:REFERENCE_ITEMS])
                current = stats[threshold]
                current["windows"] += 1
                user_windows[threshold] += 1
                current["unique_items_histogram"][unique_items] += 1
                current["unique_targets_histogram"][unique_targets] += 1
                current["windows_with_any_repeat"] += unique_items != WINDOW_ITEMS
                current["windows_with_duplicate_targets"] += (
                    unique_targets != TARGET_ITEMS
                )
                current["windows_with_target_reference_overlap"] += any(
                    item in reference_set for item in targets
                )
                unique_references = len(reference_set)
                if (
                    unique_references >= min_unique_references
                    and unique_targets >= min_unique_targets
                ):
                    current["qualifying_windows"] += 1
                    user_qualifying_windows[threshold] += 1
                    current["qualifying_windows_with_any_repeat"] += (
                        unique_items != WINDOW_ITEMS
                    )
                    current["qualifying_windows_with_duplicate_targets"] += (
                        unique_targets != TARGET_ITEMS
                    )
                    current[
                        "qualifying_windows_with_target_reference_overlap"
                    ] += any(item in reference_set for item in targets)
            seen_mapped = True
            gap_since_mapped = 0
            if progress_every and total_events % progress_every == 0:
                print(
                    f"[gap-audit] events={total_events:,} mapped={mapped_events:,}",
                    flush=True,
                )
        finalize_user()

    output = {}
    for threshold in thresholds:
        current = stats[threshold]
        windows = current["windows"]
        train_counts = current.pop("train_window_counts")
        qualifying_windows = current["qualifying_windows"]
        qualifying_train_counts = current.pop("qualifying_train_window_counts")
        for name in (
            "unique_items_histogram", "unique_targets_histogram",
            "eligible_users_by_split", "eligible_windows_by_split",
            "qualifying_users_by_split", "qualifying_windows_by_split",
        ):
            current[name] = {
                str(key): value for key, value in sorted(current[name].items())
            }
        current["candidate_train_rows_by_cap"] = {
            str(cap): sum(min(value, cap) for value in train_counts)
            for cap in CAP_CANDIDATES
        }
        current["qualifying_train_rows_by_cap"] = {
            str(cap): sum(min(value, cap) for value in qualifying_train_counts)
            for cap in CAP_CANDIDATES
        }
        current["fractions"] = {
            "all_20_items_unique": (
                current["unique_items_histogram"].get("20", 0) / windows
                if windows else 0.0
            ),
            "all_5_targets_unique": (
                current["unique_targets_histogram"].get("5", 0) / windows
                if windows else 0.0
            ),
            "target_reference_disjoint": (
                1 - current["windows_with_target_reference_overlap"] / windows
                if windows else 0.0
            ),
        }
        current["qualifying_fractions"] = {
            "of_candidate_windows": (
                qualifying_windows / windows if windows else 0.0
            ),
            "all_20_items_unique": (
                1 - current["qualifying_windows_with_any_repeat"]
                / qualifying_windows
                if qualifying_windows else 0.0
            ),
            "all_5_targets_unique": (
                1 - current["qualifying_windows_with_duplicate_targets"]
                / qualifying_windows
                if qualifying_windows else 0.0
            ),
            "target_reference_disjoint": (
                1 - current[
                    "qualifying_windows_with_target_reference_overlap"
                ] / qualifying_windows
                if qualifying_windows else 0.0
            ),
        }
        output["all" if threshold is None else str(threshold)] = current
    return {
        "total_events": total_events,
        "mapped_events": mapped_events,
        "thresholds": output,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sorted-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", default="0,1,5,20,all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--min-unique-references", type=int, default=1)
    parser.add_argument("--min-unique-targets", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.test_fraction < 1:
        raise ValueError("test-fraction must be in [0, 1)")
    if not 1 <= args.min_unique_references <= REFERENCE_ITEMS:
        raise ValueError(
            f"min-unique-references must be in [1, {REFERENCE_ITEMS}]"
        )
    if not 1 <= args.min_unique_targets <= TARGET_ITEMS:
        raise ValueError(
            f"min-unique-targets must be in [1, {TARGET_ITEMS}]"
        )
    sorted_path = args.sorted_events.expanduser().resolve()
    if not sorted_path.is_file():
        raise FileNotFoundError(sorted_path)
    thresholds = _parse_thresholds(args.thresholds)
    result = _audit_sorted(
        sorted_path, thresholds, args.seed, args.test_fraction,
        args.progress_every, args.min_unique_references, args.min_unique_targets,
    )
    payload = {
        "result_schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(sorted_path),
        "definition": (
            "K is the maximum unsupported-event count permitted between "
            "adjacent mapped events; all is the catalog-projected subsequence"
        ),
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "diversity_filter": {
            "min_unique_references": args.min_unique_references,
            "min_unique_targets": args.min_unique_targets,
        },
        **result,
    }
    _atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
