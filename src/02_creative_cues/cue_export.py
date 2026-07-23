"""cue_export.py — shared export/stats utilities for the cue pipeline.

Split out of the old cue_mining.py: this module keeps only the pieces still used
by the active pipeline (pipeline.py / run_compare.py / run_production.py) —
writing cue_vocab.json + item2cues.json, coverage/health stats, and the
markdown health report. The superseded Week-1 vocab-building + PMI-assignment
code (build_vocab / assign_cues / run_full_dataset / make_synthetic_catalog)
now lives in legacy/cue_mining_legacy.py, which only legacy/run_wpb.py still uses.

Interface contract
------------------
  Input  : vocab (list[str]) + {item_id: CueMappingEntry} produced by cue_assign.assign_all
  Output : cue_vocab.json, item2cues.json, cue_report.md

Schema constants (do NOT change without coordinating with 00_data_schema)
-------------------------------------------------------------------------
  CUE_VOCAB_SIZE = 2048    (index 0 = '<unk>')
  CUE_TOKENS     = 6       (c0 … c5, per song; experiments may override via num_cues)
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '00_data_schema'))
from schema import CUE_VOCAB_SIZE  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNK_CUE_ID = 0           # index 0 reserved for '<unk>' (missing coverage fallback)
UNK_CUE_STRING = "<unk>"


def load_vocab(vocab_path: str) -> list[str]:
    """Load cue_vocab.json.  Validates that index 0 == '<unk>'."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    assert isinstance(vocab, list), "cue_vocab.json must be a JSON list."
    assert len(vocab) == CUE_VOCAB_SIZE, \
        f"Expected {CUE_VOCAB_SIZE} entries, got {len(vocab)}."
    assert vocab[0] == UNK_CUE_STRING, \
        f"vocab[0] must be '{UNK_CUE_STRING}', got '{vocab[0]}'."
    return vocab


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_outputs(
    vocab: list[str],
    item2cues: dict,
    output_dir: str,
) -> None:
    """Write cue_vocab.json and item2cues.json to output_dir.

    item2cues.json format: {"item_id": [c0, c1, c2, c3, c4, c5], ...}
    This is the format that CueMappingEntry.load_mapping() reads.

    Parameters
    ----------
    vocab      : list of CUE_VOCAB_SIZE strings (vocab[0] == '<unk>').
    item2cues  : mapping produced by cue_assign.assign_all() (or the legacy
                 run_full_dataset()) — {item_id: CueMappingEntry}.
    output_dir : directory to write files.
    """
    os.makedirs(output_dir, exist_ok=True)

    vocab_path = os.path.join(output_dir, "cue_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"[cue_export] Wrote vocab ({len(vocab)} entries) -> {vocab_path}")

    mapping = {iid: entry.cue_ids for iid, entry in item2cues.items()}
    cues_path = os.path.join(output_dir, "item2cues.json")
    with open(cues_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f)
    print(f"[cue_export] Wrote item2cues ({len(mapping)} items) -> {cues_path}")


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def compute_coverage_stats(
    item2cues: dict,
    vocab: list[str],
) -> dict:
    """Compute coverage and distribution statistics for the cue mapping.

    Metrics
    -------
    - coverage_rate  : fraction of items with zero '<unk>' slots (fully assigned)
    - unk_rate       : mean fraction of cue slots that are '<unk>' (index 0)
    - vocab_coverage : fraction of vocab entries used by at least 1 item
    - top10_cues     : list of (cue_string, count) for the 10 most used cues
    - cue_entropy    : Shannon entropy of cue usage distribution

    Returns
    -------
    dict with the above keys + values.
    """
    import collections

    n_items = len(item2cues)
    if n_items == 0:
        return {}

    cue_counts: dict[int, int] = collections.Counter()
    unk_slots = 0
    items_with_unk = 0

    for entry in item2cues.values():
        has_unk = False
        for cid in entry.cue_ids:
            cue_counts[cid] += 1
            if cid == UNK_CUE_ID:
                unk_slots += 1
                has_unk = True
        if has_unk:
            items_with_unk += 1

    # Sum actual per-entry lengths rather than assuming a fixed count: item2cues may
    # come from a --num-cues override, so slot count isn't always 6/item.
    total_slots = sum(len(entry.cue_ids) for entry in item2cues.values())
    coverage_rate = 1.0 - items_with_unk / n_items
    unk_rate = unk_slots / total_slots

    vocab_coverage = len([c for c in cue_counts if c != UNK_CUE_ID]) / max(len(vocab) - 1, 1)

    top10 = [(vocab[cid], cnt) for cid, cnt in cue_counts.most_common(10) if cid != UNK_CUE_ID]

    total_non_unk = sum(cnt for cid, cnt in cue_counts.items() if cid != UNK_CUE_ID)
    entropy = 0.0
    if total_non_unk > 0:
        for cid, cnt in cue_counts.items():
            if cid == UNK_CUE_ID:
                continue
            p = cnt / total_non_unk
            entropy -= p * math.log2(p + 1e-12)

    return {
        "n_items": n_items,
        "coverage_rate": round(coverage_rate, 4),
        "unk_rate": round(unk_rate, 4),
        "vocab_coverage": round(vocab_coverage, 4),
        "top10_cues": top10,
        "cue_entropy_bits": round(entropy, 4),
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(stats: dict, output_dir: str) -> None:
    """Write cue_report.md from coverage stats."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "cue_report.md")
    lines = [
        "# Cue Mining Report\n",
        f"**Items processed:** {stats.get('n_items', 0)}\n",
        f"**Coverage rate** (≥1 non-unk cue): {stats.get('coverage_rate', 0):.1%}\n",
        f"**UNK slot rate:** {stats.get('unk_rate', 0):.1%}\n",
        f"**Vocab utilization:** {stats.get('vocab_coverage', 0):.1%} of non-unk vocab entries used\n",
        f"**Cue entropy:** {stats.get('cue_entropy_bits', 0):.2f} bits\n",
        "\n## Top-10 most assigned cues\n",
        "| Rank | Cue | Count |\n",
        "|------|-----|-------|\n",
    ]
    for rank, (cue, cnt) in enumerate(stats.get("top10_cues", []), 1):
        lines.append(f"| {rank} | {cue} | {cnt} |\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[cue_export] Wrote report -> {report_path}")
