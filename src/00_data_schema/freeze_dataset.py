#!/usr/bin/env python3
"""scripts/freeze_dataset.py — Filter playlists to complete items and produce train/val/test splits.

Reads catalog.json (from build_final_catalog.py) and the subset playlists.txt,
keeps only items that are "complete" (audio + lyrics + metadata), drops playlists
that fall below a minimum complete-item threshold, then splits into train/val/test.

Writes to --output/:
  playlists_clean.txt    — playlists with only complete items (backbone format)
  splits/
    train.txt            — training playlists
    val.txt              — validation playlists
    test.txt             — test playlists
  dataset_card.json      — final frozen dataset statistics

Usage
-----
python scripts/freeze_dataset.py \\
    --catalog    data/dataset/catalog.json \\
    --playlists  data/playlists/r3_10_30_freq3/playlists.txt \\
    --output     data/dataset/ \\
    --min-complete 5 \\
    --val-frac 0.1 --test-frac 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog",       required=True,
                        help="catalog.json from build_final_catalog.py")
    parser.add_argument("--playlists",     required=True,
                        help="playlists.txt in backbone comma-separated format")
    parser.add_argument("--output",        default="data/dataset/",
                        help="Output directory (default: data/dataset/)")
    parser.add_argument("--min-complete",  type=int, default=5,
                        help="Min complete items per playlist to keep it (default: 5)")
    parser.add_argument("--val-frac",      type=float, default=0.1,
                        help="Fraction for validation split (default: 0.1)")
    parser.add_argument("--test-frac",     type=float, default=0.1,
                        help="Fraction for test split (default: 0.1)")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Load complete item set
    catalog = json.load(open(args.catalog))
    complete_ids: set = {e["item_id"] for e in catalog if e["complete"]}
    print(f"[freeze] Complete items   : {len(complete_ids):,}")

    # Load and filter playlists
    raw_playlists: list[list[str]] = []
    with open(args.playlists, encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",") if p.strip()]
            raw_playlists.append(parts)

    print(f"[freeze] Input playlists  : {len(raw_playlists):,}")

    # Keep only complete items within each playlist, then filter by min-complete
    clean_playlists: list[list[str]] = []
    for parts in raw_playlists:
        pid = parts[0]
        kept = [iid for iid in parts[1:] if iid in complete_ids]
        if len(kept) >= args.min_complete:
            clean_playlists.append([pid] + kept)

    print(f"[freeze] After min-complete={args.min_complete} filter: {len(clean_playlists):,} playlists")

    # Write playlists_clean.txt
    clean_path = output_dir / "playlists_clean.txt"
    with open(clean_path, "w", encoding="utf-8") as f:
        for pl in clean_playlists:
            f.write(", ".join(pl) + "\n")

    # Train / val / test split (stratify by playlist length bucket to keep distribution)
    random.seed(args.seed)
    shuffled = clean_playlists[:]
    random.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, math.floor(n * args.test_frac))
    n_val  = max(1, math.floor(n * args.val_frac))
    n_train = n - n_val - n_test

    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train + n_val]
    test  = shuffled[n_train + n_val:]

    def write_split(name: str, playlists: list[list[str]]) -> None:
        path = splits_dir / f"{name}.txt"
        with open(path, "w", encoding="utf-8") as f:
            for pl in playlists:
                f.write(", ".join(pl) + "\n")
        items = {iid for pl in playlists for iid in pl[1:]}
        print(f"  {name:5s}: {len(playlists):>5,} playlists  {len(items):>6,} unique items")

    print(f"\n[freeze] Splits (seed={args.seed}):")
    write_split("train", train)
    write_split("val",   val)
    write_split("test",  test)

    # Unique items per split for the dataset card
    def split_items(pls): return sorted({iid for pl in pls for iid in pl[1:]})
    train_items = split_items(train)
    val_items   = split_items(val)
    test_items  = split_items(test)

    # Playlist length stats
    lengths = [len(pl) - 1 for pl in clean_playlists]
    avg_len = sum(lengths) / len(lengths) if lengths else 0

    # Item frequency across clean playlists
    freq: Counter = Counter()
    for pl in clean_playlists:
        for iid in pl[1:]:
            freq[iid] += 1

    dataset_card = {
        "frozen": True,
        "seed": args.seed,
        "n_playlists": {
            "total": len(clean_playlists),
            "train": len(train),
            "val":   len(val),
            "test":  len(test),
        },
        "n_unique_items": {
            "total": len({iid for pl in clean_playlists for iid in pl[1:]}),
            "train": len(train_items),
            "val":   len(val_items),
            "test":  len(test_items),
        },
        "playlist_length": {
            "min": min(lengths),
            "max": max(lengths),
            "avg": round(avg_len, 1),
        },
        "item_freq": {
            "min":    min(freq.values()) if freq else 0,
            "max":    max(freq.values()) if freq else 0,
            "median": sorted(freq.values())[len(freq)//2] if freq else 0,
        },
        "min_complete_per_playlist": args.min_complete,
        "split_fractions": {
            "train": round(1 - args.val_frac - args.test_frac, 2),
            "val":   args.val_frac,
            "test":  args.test_frac,
        },
        "sources": {
            "catalog":    args.catalog,
            "playlists":  args.playlists,
        },
    }
    (output_dir / "dataset_card.json").write_text(json.dumps(dataset_card, indent=2))

    print(f"\n[freeze] Written to {output_dir}/")
    print(f"  playlists_clean.txt — {len(clean_playlists):,} playlists")
    print(f"  splits/train.txt    — {len(train):,} playlists")
    print(f"  splits/val.txt      — {len(val):,} playlists")
    print(f"  splits/test.txt     — {len(test):,} playlists")
    print(f"  dataset_card.json   — frozen dataset statistics")
    print(f"\n[freeze] Dataset is frozen. Load with:")
    print(f"  catalog  = json.load(open('{output_dir}/catalog.json'))")
    print(f"  splits   = {{s: open('{output_dir}/splits/{{s}}.txt').readlines() for s in ['train','val','test']}}")


if __name__ == "__main__":
    main()
