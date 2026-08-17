#!/usr/bin/env python3
"""Build user-disjoint Music4All-Onion 15-to-5 sequential splits.

The primary protocol keeps only strict catalog matches.  Events are first
stable-sorted by user, timestamp, and original row number.  An unsupported
event breaks the current run, repeated listens are retained, and windows are
created only inside runs containing at least 20 adjacent supported events.
"""

from __future__ import annotations

import argparse
import bz2
import csv
from datetime import datetime, timezone
import hashlib
import heapq
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import TextIO


WINDOW_ITEMS = 20
REFERENCE_ITEMS = 15
TARGET_ITEMS = 5
SCHEMA = "genplaylist-music4all-sequences-v1"
CAP_CANDIDATES = (16, 32, 64, 100, 200)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
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


def _load_mapping(path: Path, include_relaxed: bool) -> dict[str, str]:
    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "music4all_id", "genplaylist_item_id", "match_type",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Mapping is missing columns {sorted(required)}")
        for row in reader:
            if row["match_type"] != "strict" and not include_relaxed:
                continue
            source = row["music4all_id"]
            target = row["genplaylist_item_id"]
            if source in mapping:
                raise ValueError(f"Duplicate Music4All mapping: {source}")
            if target in reverse:
                raise ValueError(
                    f"Non one-to-one mapping: {target} maps from "
                    f"{reverse[target]} and {source}"
                )
            mapping[source] = target
            reverse[target] = source
    if not mapping:
        raise ValueError("No accepted mapping rows")
    return mapping


def _spool_events(
    interactions: Path,
    mapping: dict[str, str],
    spool: Path,
    progress_every: int,
) -> dict:
    """Write the minimal sortable record for every event.

    Unsupported events must remain in the spool because their chronological
    position defines a run boundary after sorting.
    """
    spool.parent.mkdir(parents=True, exist_ok=True)
    temporary = spool.with_suffix(spool.suffix + ".tmp")
    total = 0
    mapped = 0
    users: set[str] = set()
    with bz2.open(interactions, "rt", encoding="utf-8", newline="") as source, \
            temporary.open("w", encoding="utf-8", newline="") as target:
        header = source.readline().rstrip("\r\n").split("\t")
        required = {"user_id", "track_id", "timestamp"}
        if not required.issubset(header):
            raise ValueError(
                f"Interaction table needs {sorted(required)}, got {header}"
            )
        user_column = header.index("user_id")
        track_column = header.index("track_id")
        timestamp_column = header.index("timestamp")
        for source_row, line in enumerate(source, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != len(header):
                raise ValueError(f"Malformed interaction row {source_row + 1}")
            user_id = fields[user_column]
            timestamp = fields[timestamp_column]
            if "\t" in user_id or "\t" in timestamp:
                raise ValueError(f"Tab in sortable field at row {source_row + 1}")
            item_id = mapping.get(fields[track_column], "")
            total += 1
            mapped += bool(item_id)
            users.add(user_id)
            target.write(f"{user_id}\t{timestamp}\t{source_row}\t{item_id}\n")
            if progress_every and total % progress_every == 0:
                print(
                    f"[spool] events={total:,} mapped={mapped:,} "
                    f"users={len(users):,}",
                    flush=True,
                )
    os.replace(temporary, spool)
    return {
        "total_events": total,
        "mapped_events": mapped,
        "users": len(users),
    }


def _external_sort(
    spool: Path,
    sorted_path: Path,
    temporary_dir: Path,
    executable: str,
    buffer_size: str,
    parallel: int,
) -> None:
    temporary_dir.mkdir(parents=True, exist_ok=True)
    sorted_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "--stable",
        "--field-separator=\t",
        "--key=1,1",
        "--key=2,2",
        "--key=3,3n",
        f"--buffer-size={buffer_size}",
        f"--parallel={parallel}",
        f"--temporary-directory={temporary_dir}",
        f"--output={sorted_path}",
        str(spool),
    ]
    environment = dict(os.environ)
    environment["LC_ALL"] = "C"
    print("[sort] " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def _unit_interval(seed: int, namespace: str, value: str) -> float:
    digest = hashlib.sha256(
        f"{namespace}\0{seed}\0{value}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _user_split(user_id: str, seed: int, test_fraction: float) -> str:
    return (
        "test"
        if _unit_interval(seed, "split", user_id) < test_fraction
        else "train"
    )


def _pseudonym(user_id: str, seed: int) -> str:
    digest = hashlib.sha256(
        f"pseudonym\0{seed}\0{user_id}".encode("utf-8")
    ).hexdigest()
    return digest[:16]


def _percentiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {name: 0 for name in ("min", "p25", "p50", "p75", "p90", "p95", "p99", "max")}
    ordered = sorted(values)

    def nearest(fraction: float) -> int:
        return ordered[round(fraction * (len(ordered) - 1))]

    return {
        "min": ordered[0],
        "p25": nearest(0.25),
        "p50": nearest(0.50),
        "p75": nearest(0.75),
        "p90": nearest(0.90),
        "p95": nearest(0.95),
        "p99": nearest(0.99),
        "max": ordered[-1],
    }


def _write_record(
    handle: TextIO,
    split: str,
    user_id: str,
    seed: int,
    run_index: int,
    start_index: int,
    items: tuple[str, ...],
) -> None:
    if len(items) != WINDOW_ITEMS:
        raise AssertionError(f"Expected {WINDOW_ITEMS} items, got {len(items)}")
    record_id = (
        f"m4a-{split}-{_pseudonym(user_id, seed)}-"
        f"r{run_index:04d}-s{start_index:08d}"
    )
    handle.write(",".join((record_id, *items)) + "\n")


def _scan_sorted_events(
    sorted_path: Path,
    train_output: Path,
    test_output: Path,
    seed: int,
    test_fraction: float,
    train_user_cap: int,
    max_skipped_events: int,
    min_unique_references: int,
    min_unique_targets: int,
    progress_every: int,
) -> dict:
    """Consume a sorted event table and write capped train/latest-test windows."""
    train_output.parent.mkdir(parents=True, exist_ok=True)
    train_temp = train_output.with_suffix(train_output.suffix + ".tmp")
    test_temp = test_output.with_suffix(test_output.suffix + ".tmp")

    total_events = 0
    mapped_events = 0
    total_users = 0
    eligible_users = {"train": 0, "test": 0}
    all_users = {"train": 0, "test": 0}
    eligible_windows = {"train": 0, "test": 0}
    output_windows = {"train": 0, "test": 0}
    windows_per_user = {"train": [], "test": []}
    repeat_stats = {
        "windows_with_any_repeated_item": 0,
        "windows_with_target_reference_overlap": 0,
        "windows_with_duplicate_targets": 0,
        "target_occurrences_also_in_reference": 0,
    }
    diversity_filter = {
        "candidate_windows_before_filter": 0,
        "rejected_unique_references": 0,
        "rejected_unique_targets": 0,
    }

    current_user: str | None = None
    current_split = ""
    run: list[tuple[str, str, int]] = []
    run_index = -1
    run_position = -1
    skipped_events = 0
    user_window_count = 0
    train_heap: list[tuple[int, int, tuple]] = []
    latest_test: tuple | None = None
    candidate_serial = 0

    def finalize_user(train_handle: TextIO, test_handle: TextIO) -> None:
        nonlocal latest_test
        if current_user is None:
            return
        all_users[current_split] += 1
        windows_per_user[current_split].append(user_window_count)
        if not user_window_count:
            return
        eligible_users[current_split] += 1
        eligible_windows[current_split] += user_window_count
        if current_split == "train":
            selected = [entry[2] for entry in train_heap]
            selected.sort(key=lambda value: (value[0], value[1], value[2]))
            for candidate in selected:
                candidate_run, candidate_start, _, items = candidate
                _write_record(
                    train_handle, "train", current_user, seed,
                    candidate_run, candidate_start, items,
                )
                output_windows["train"] += 1
        else:
            if latest_test is None:
                raise AssertionError("Eligible test user has no latest window")
            candidate_run, candidate_start, _, items = latest_test
            _write_record(
                test_handle, "test", current_user, seed,
                candidate_run, candidate_start, items,
            )
            output_windows["test"] += 1

    with sorted_path.open("r", encoding="utf-8", newline="") as source, \
            train_temp.open("w", encoding="utf-8", newline="") as train_handle, \
            test_temp.open("w", encoding="utf-8", newline="") as test_handle:
        for line_number, line in enumerate(source, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"Malformed sorted event row {line_number}")
            user_id, timestamp, source_row_text, item_id = fields
            try:
                source_row = int(source_row_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid source row at sorted row {line_number}"
                ) from error
            total_events += 1
            mapped_events += bool(item_id)
            if user_id != current_user:
                finalize_user(train_handle, test_handle)
                total_users += 1
                current_user = user_id
                current_split = _user_split(user_id, seed, test_fraction)
                run = []
                run_index = -1
                run_position = -1
                skipped_events = 0
                user_window_count = 0
                train_heap = []
                latest_test = None
            if not item_id:
                if run:
                    skipped_events += 1
                continue
            if run and skipped_events > max_skipped_events:
                run = []
                run_position = -1
            skipped_events = 0
            if not run:
                run_index += 1
            run_position += 1
            run.append((item_id, timestamp, source_row))
            if len(run) > WINDOW_ITEMS:
                del run[0]
            if len(run) < WINDOW_ITEMS:
                continue

            items = tuple(value[0] for value in run)
            start_index = run_position - WINDOW_ITEMS + 1
            references = items[:REFERENCE_ITEMS]
            targets = items[REFERENCE_ITEMS:]
            diversity_filter["candidate_windows_before_filter"] += 1
            if len(set(references)) < min_unique_references:
                diversity_filter["rejected_unique_references"] += 1
                continue
            if len(set(targets)) < min_unique_targets:
                diversity_filter["rejected_unique_targets"] += 1
                continue
            candidate = (run_index, start_index, timestamp, items)
            user_window_count += 1
            candidate_serial += 1
            reference_set = set(references)
            repeat_stats["windows_with_any_repeated_item"] += (
                len(set(items)) != WINDOW_ITEMS
            )
            overlap_count = sum(item in reference_set for item in targets)
            repeat_stats["target_occurrences_also_in_reference"] += overlap_count
            repeat_stats["windows_with_target_reference_overlap"] += overlap_count > 0
            repeat_stats["windows_with_duplicate_targets"] += (
                len(set(targets)) != TARGET_ITEMS
            )

            if current_split == "test":
                latest_test = candidate
            else:
                priority = int.from_bytes(hashlib.sha256(
                    f"window\0{seed}\0{user_id}\0{run_index}\0{start_index}".encode(
                        "utf-8"
                    )
                ).digest()[:8], "big")
                heap_entry = (-priority, -candidate_serial, candidate)
                if train_user_cap <= 0 or len(train_heap) < train_user_cap:
                    heapq.heappush(train_heap, heap_entry)
                elif priority < -train_heap[0][0]:
                    heapq.heapreplace(train_heap, heap_entry)

            if progress_every and total_events % progress_every == 0:
                print(
                    f"[scan] events={total_events:,} users={total_users:,} "
                    f"eligible_windows={sum(eligible_windows.values()) + user_window_count:,}",
                    flush=True,
                )
        finalize_user(train_handle, test_handle)

    os.replace(train_temp, train_output)
    os.replace(test_temp, test_output)
    train_counts = windows_per_user["train"]
    positive_train_counts = [value for value in train_counts if value]
    return {
        "sorted_events": total_events,
        "mapped_events": mapped_events,
        "total_users": total_users,
        "all_users_by_split": all_users,
        "eligible_users_by_split": eligible_users,
        "eligible_windows_by_split": eligible_windows,
        "output_windows_by_split": output_windows,
        "eligible_window_count_percentiles": {
            split: _percentiles([value for value in values if value])
            for split, values in windows_per_user.items()
        },
        "candidate_train_rows_by_cap": {
            str(cap): sum(min(value, cap) for value in positive_train_counts)
            for cap in CAP_CANDIDATES
        },
        "diversity_filter": diversity_filter,
        "repeat_statistics_over_all_eligible_windows": repeat_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--train-user-cap", type=int, default=100)
    parser.add_argument(
        "--max-skipped-events", type=int, default=0,
        help="Maximum unsupported events between adjacent mapped events.")
    parser.add_argument("--min-unique-references", type=int, default=1)
    parser.add_argument("--min-unique-targets", type=int, default=1)
    parser.add_argument("--include-relaxed", action="store_true")
    parser.add_argument("--reuse-spool", action="store_true")
    parser.add_argument("--reuse-sorted", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sort-executable", default="sort")
    parser.add_argument("--sort-buffer-size", default="128G")
    parser.add_argument("--sort-parallel", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=5_000_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.test_fraction < 1:
        raise ValueError("test-fraction must be in [0, 1)")
    if args.train_user_cap < 0:
        raise ValueError("train-user-cap must be nonnegative (0 means uncapped)")
    if args.max_skipped_events < 0:
        raise ValueError("max-skipped-events must be nonnegative")
    if not 1 <= args.min_unique_references <= REFERENCE_ITEMS:
        raise ValueError(
            f"min-unique-references must be in [1, {REFERENCE_ITEMS}]")
    if not 1 <= args.min_unique_targets <= TARGET_ITEMS:
        raise ValueError(f"min-unique-targets must be in [1, {TARGET_ITEMS}]")
    interactions = args.interactions.expanduser().resolve()
    mapping_path = args.mapping.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    if not interactions.is_file() or not mapping_path.is_file():
        raise FileNotFoundError("Interactions and mapping files must exist")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    spool = work_dir / "events.unsorted.tsv"
    sorted_path = work_dir / "events.sorted.tsv"
    mapping = _load_mapping(mapping_path, args.include_relaxed)
    print(f"[mapping] accepted={len(mapping):,}", flush=True)

    spool_stats = None
    if args.reuse_sorted:
        if not sorted_path.is_file():
            raise FileNotFoundError(sorted_path)
    else:
        if args.reuse_spool:
            if not spool.is_file():
                raise FileNotFoundError(spool)
        else:
            spool_stats = _spool_events(
                interactions, mapping, spool, args.progress_every,
            )
        _external_sort(
            spool,
            sorted_path,
            work_dir / "sort-tmp",
            args.sort_executable,
            args.sort_buffer_size,
            args.sort_parallel,
        )

    splits_dir = output_dir / "splits"
    scan = _scan_sorted_events(
        sorted_path,
        splits_dir / "train.txt",
        splits_dir / "test.txt",
        args.seed,
        args.test_fraction,
        args.train_user_cap,
        args.max_skipped_events,
        args.min_unique_references,
        args.min_unique_targets,
        args.progress_every,
    )
    (splits_dir / "val.txt").write_text("", encoding="utf-8")
    manifest = {
        "result_schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "sequence": "chronological ascending by timestamp; source row breaks ties",
            "window": "15 reference events followed by 5 target events",
            "unsupported_event": (
                f"up to {args.max_skipped_events} skipped events are allowed "
                "between mapped events; a larger gap breaks the run"
            ),
            "repeated_listens": "retained",
            "diversity_filter": (
                f"at least {args.min_unique_references}/15 unique references and "
                f"{args.min_unique_targets}/5 unique targets"
            ),
            "mapping": "strict one-to-one" if not args.include_relaxed else "strict plus relaxed-version",
            "split": "seeded SHA-256 user-disjoint 80/20 train/test" if args.test_fraction == 0.2 else "seeded SHA-256 user-disjoint train/test",
            "test_context": "latest eligible window per test user",
            "train_selection": "lowest seeded SHA-256 window priorities per user",
        },
        "configuration": {
            "seed": args.seed,
            "test_fraction": args.test_fraction,
            "train_user_cap": args.train_user_cap,
            "max_skipped_events": args.max_skipped_events,
            "min_unique_references": args.min_unique_references,
            "min_unique_targets": args.min_unique_targets,
            "accepted_mapping_items": len(mapping),
        },
        "sources": {
            "interactions": {
                "path": str(interactions),
                "size_bytes": interactions.stat().st_size,
                "sha256": _sha256(interactions),
            },
            "mapping": {
                "path": str(mapping_path),
                "size_bytes": mapping_path.stat().st_size,
                "sha256": _sha256(mapping_path),
            },
        },
        "spool": spool_stats,
        "scan": scan,
        "outputs": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in {
                "train": splits_dir / "train.txt",
                "val": splits_dir / "val.txt",
                "test": splits_dir / "test.txt",
            }.items()
        },
    }
    _atomic_json(output_dir / "sequence_manifest.json", manifest)
    print(json.dumps(scan, indent=2, sort_keys=True), flush=True)

    if not args.keep_intermediate:
        spool.unlink(missing_ok=True)
        sorted_path.unlink(missing_ok=True)
        shutil.rmtree(work_dir / "sort-tmp", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
