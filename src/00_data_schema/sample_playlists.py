#!/usr/bin/env python3
"""scripts/sample_playlists.py — Filter and sample playlists by length and item popularity.

Reads a backbone playlist file (bi_full.txt / train.txt / test.txt),
applies up to two rounds of filtering, and writes:
  - {output_dir}/playlists.txt      — filtered playlists (backbone comma-separated format)
  - {output_dir}/item_ids.txt       — one item_id per line (deduplicated, popularity-filtered)
  - {output_dir}/item_freq.json     — {"item_id": freq, ...} for all items in output
  - {output_dir}/stats.json         — summary stats

Filtering pipeline
------------------
Round 1 — length filter:
    Keep playlists with min-len <= len <= max-len.

Round 2 (optional) — popularity filter (requires --min-freq):
    Count how many playlists each item appears in across the full catalog
    (--freq-input, defaults to --input).  Drop items below the threshold.
    Then re-apply the length check on each playlist's *remaining* items:
    playlists that fall below --min-len-after-filter are discarded.

This can be run iteratively: run once to inspect item_freq.json, choose a
threshold, then re-run with --min-freq to produce the clean subset.

Usage
-----
# Step 1 — explore (no frequency filter yet)
python scripts/sample_playlists.py \\
    --input  /path/to/bi_full.txt \\
    --output data/playlists/subset_10_30/

# Step 2 — apply popularity filter, re-check length
python scripts/sample_playlists.py \\
    --input      /path/to/bi_full.txt \\
    --freq-input /path/to/bi_full.txt \\
    --output     data/playlists/subset_10_30_freq3/ \\
    --min-len 10 --max-len 30 --min-freq 3

# Random sample of 500 after all filters
python scripts/sample_playlists.py \\
    --input  /path/to/bi_full.txt \\
    --output data/playlists/subset_10_30_freq3_500/ \\
    --min-len 10 --max-len 30 --min-freq 3 --sample 500 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def _count_item_freq(path: str) -> Counter:
    """Count how many playlists each item appears in."""
    freq: Counter = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",") if p.strip()]
            for iid in parts[1:]:
                freq[iid] += 1
    return freq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True,
                        help="Backbone playlist txt file (bi_full.txt etc.)")
    parser.add_argument("--freq-input",  default=None,
                        help="File to count item frequencies from (default: same as --input).")
    parser.add_argument("--output",      required=True, help="Output directory")
    parser.add_argument("--min-len",     type=int, default=10,
                        help="Min playlist length for initial filter (default: 10)")
    parser.add_argument("--max-len",     type=int, default=30,
                        help="Max playlist length for initial filter (default: 30)")
    parser.add_argument("--min-freq",    type=int, default=None,
                        help="Min number of playlists an item must appear in to be kept. "
                             "If set, triggers Round 2 filtering.")
    parser.add_argument("--min-len-after-filter", type=int, default=None,
                        help="After removing low-freq items, drop playlists shorter than this. "
                             "Defaults to --min-len if --min-freq is set.")
    parser.add_argument("--max-len-after-filter", type=int, default=None,
                        help="After removing low-freq items, drop playlists longer than this.")
    parser.add_argument("--audio-dir",   default=None,
                        help="If set, only count items that have an audio file in this directory "
                             "when checking playlist length constraints. Ensures every kept playlist "
                             "has >= min-len items with actual audio.")
    parser.add_argument("--sample",      type=int, default=None,
                        help="Randomly sample N playlists after all filters. Default: use all.")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Round 1: length filter ---
    all_lines = 0
    filtered = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            all_lines += 1
            parts = [p.strip() for p in line.strip().split(",") if p.strip()]
            if args.min_len <= len(parts) - 1 <= args.max_len:
                filtered.append(parts)

    print(f"[sample] Total playlists   : {all_lines:,}")
    print(f"[sample] After len filter  : {len(filtered):,}  "
          f"({args.min_len}–{args.max_len} songs)")

    # --- Build audio-available set (optional) ---
    audio_ids: set | None = None
    if args.audio_dir:
        _audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".webm", ".opus", ".ogg"}
        audio_ids = {p.stem for p in Path(args.audio_dir).iterdir()
                     if p.suffix.lower() in _audio_exts}
        print(f"[sample] Audio files found : {len(audio_ids):,}  (in {args.audio_dir})")

    # --- Round 2: popularity + audio filter ---
    item_freq: Counter = Counter()
    if args.min_freq is not None or audio_ids is not None:
        min_after = args.min_len_after_filter if args.min_len_after_filter is not None \
                    else args.min_len

        if args.min_freq is not None:
            freq_src = args.freq_input or args.input
            print(f"[sample] Counting item freq from {freq_src} ...")
            item_freq = _count_item_freq(freq_src)

        before = len(filtered)
        filtered_pop = []
        for parts in filtered:
            pid = parts[0]
            kept = parts[1:]
            if args.min_freq is not None:
                kept = [iid for iid in kept if item_freq[iid] >= args.min_freq]
            if audio_ids is not None:
                kept = [iid for iid in kept if iid in audio_ids]
            max_after = args.max_len_after_filter
            if len(kept) >= min_after and (max_after is None or len(kept) <= max_after):
                filtered_pop.append([pid] + kept)
        filtered = filtered_pop

        desc = []
        if args.min_freq is not None:
            desc.append(f"min_freq={args.min_freq}")
        if audio_ids is not None:
            desc.append("audio-only")
        print(f"[sample] After filter      : {len(filtered):,}  "
              f"({', '.join(desc)}, "
              f"removed {before - len(filtered):,} playlists below {min_after} items)")

    # --- Optional random sample ---
    if args.sample and args.sample < len(filtered):
        random.seed(args.seed)
        filtered = random.sample(filtered, args.sample)
        print(f"[sample] After sampling    : {len(filtered):,}  (seed={args.seed})")

    # --- Collect unique item IDs (preserve first-seen order) ---
    seen: set = set()
    item_ids: list = []
    for playlist in filtered:
        for iid in playlist[1:]:
            if iid not in seen:
                item_ids.append(iid)
                seen.add(iid)

    print(f"[sample] Unique items      : {len(item_ids):,}")
    print(f"[sample] Est. audio        : ~{len(item_ids) * 7.3 / 1024:.0f} GB (MP3 ~7.3 MB/song)")

    # --- Write outputs ---
    playlists_path = output_dir / "playlists.txt"
    with open(playlists_path, "w", encoding="utf-8") as f:
        for playlist in filtered:
            f.write(", ".join(playlist) + "\n")

    items_path = output_dir / "item_ids.txt"
    with open(items_path, "w", encoding="utf-8") as f:
        f.write("\n".join(item_ids) + "\n")

    # Write per-item frequencies for inspection / further filtering rounds
    freq_out = {iid: int(item_freq[iid]) for iid in item_ids} if item_freq else {}
    if not freq_out and args.min_freq is None:
        # Compute freq anyway so the output is useful for choosing --min-freq next time
        freq_src = args.freq_input or args.input
        item_freq = _count_item_freq(freq_src)
        freq_out = {iid: int(item_freq[iid]) for iid in item_ids}
    (output_dir / "item_freq.json").write_text(json.dumps(freq_out, indent=2))

    lengths = [len(p) - 1 for p in filtered]
    stats = {
        "n_playlists":    len(filtered),
        "n_unique_items": len(item_ids),
        "min_len":        min(lengths),
        "max_len":        max(lengths),
        "avg_len":        round(sum(lengths) / len(lengths), 1),
        "filter": {
            "min_len":   args.min_len,
            "max_len":   args.max_len,
            "min_freq":  args.min_freq,
        },
        "sample":             args.sample,
        "seed":               args.seed,
        "est_audio_gb_mp3":   round(len(item_ids) * 7.3 / 1024, 1),
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\n[sample] Written to {output_dir}/")
    print(f"         playlists.txt  — {len(filtered):,} playlists")
    print(f"         item_ids.txt   — {len(item_ids):,} item IDs")
    print(f"         item_freq.json — per-item playlist-frequency (use to pick --min-freq)")
    print(f"         stats.json     — summary")
    print(f"\n[sample] Next steps:")
    if args.min_freq is None:
        print(f"  # Inspect item_freq.json, then re-run with --min-freq <N> to drop long-tail items")
        print(f"  python scripts/sample_playlists.py \\")
        print(f"      --input {args.input} --output {args.output}_freq3 \\")
        print(f"      --min-len {args.min_len} --max-len {args.max_len} --min-freq 3")
    print(f"\n  # Download audio for the clean subset")
    print(f"  python scripts/download_audio.py \\")
    print(f"      --metadata <metadata.json> \\")
    print(f"      --ids-file {items_path} \\")
    print(f"      --output   data/audio/spotify/ --format mp3")


if __name__ == "__main__":
    main()
