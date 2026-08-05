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
  CUE_CANDIDATES_PER_ITEM = 16  (ranked master table width)
  CUE_TOKENS              = 8   (default prefix consumed by WP-C)
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '00_data_schema'))
from schema import (  # noqa: E402
    CUE_CANDIDATES_PER_ITEM,
    CUE_TOKENS,
    CUE_VOCAB_SIZE,
    SCHEMA_VERSION,
    TOKEN_LAYOUT,
)

import cue_io  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNK_CUE_ID = 0           # index 0 reserved for '<unk>' (missing coverage fallback)
UNK_CUE_STRING = "<unk>"


def load_vocab(vocab_path: str) -> list[str]:
    """Load cue_vocab.json.  Validates that index 0 == '<unk>'."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    if not isinstance(vocab, list):
        raise ValueError("cue_vocab.json must be a JSON list")
    if len(vocab) != CUE_VOCAB_SIZE:
        raise ValueError(f"Expected {CUE_VOCAB_SIZE} entries, got {len(vocab)}")
    if vocab[0] != UNK_CUE_STRING:
        raise ValueError(
            f"vocab[0] must be '{UNK_CUE_STRING}', got '{vocab[0]}'")
    return vocab


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_outputs(
    vocab: list[str],
    item2cues: dict,
    output_dir: str,
    *,
    score_mapping: dict[str, list[float | None]] | None = None,
    assignment_metadata: dict | None = None,
) -> None:
    """Write cue_vocab.json and item2cues.json to output_dir.

    item2cues.json format: {"item_id": [c0, c1, ..., c15], ...}. The
    GenPlaylist-v1 model consumes the first eight entries.
    This is the format that CueMappingEntry.load_mapping() reads.

    Parameters
    ----------
    vocab      : list of CUE_VOCAB_SIZE strings (vocab[0] == '<unk>').
    item2cues  : mapping produced by cue_assign.assign_all() (or the legacy
                 run_full_dataset()) — {item_id: CueMappingEntry}.
    output_dir : directory to write files.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not vocab or vocab[0] != UNK_CUE_STRING:
        raise ValueError(
            f"Cue vocab must reserve index 0 for {UNK_CUE_STRING!r}")

    cue_counts = {len(entry.cue_ids) for entry in item2cues.values()}
    if len(cue_counts) > 1:
        raise ValueError(f"Inconsistent cue counts in item2cues: {sorted(cue_counts)}")
    num_cues = cue_counts.pop() if cue_counts else CUE_TOKENS
    for item_id, entry in item2cues.items():
        if str(item_id) != str(entry.item_id):
            raise ValueError(
                f"Mapping key {item_id!r} disagrees with entry.item_id {entry.item_id!r}")
        entry.validate(n_cues=num_cues, vocab_size=len(vocab))

    vocab_path = os.path.join(output_dir, "cue_vocab.json")
    cue_io.atomic_write_json(vocab_path, vocab)
    print(f"[cue_export] Wrote vocab ({len(vocab)} entries) -> {vocab_path}")

    mapping = {iid: entry.cue_ids for iid, entry in item2cues.items()}
    cues_path = os.path.join(output_dir, "item2cues.json")
    cue_io.atomic_write_json(cues_path, mapping)
    print(f"[cue_export] Wrote item2cues ({len(mapping)} items) -> {cues_path}")

    score_path = None
    if score_mapping is not None:
        if set(score_mapping) != set(mapping):
            raise ValueError("item2cue score IDs must exactly match item2cues IDs")
        for item_id, scores in score_mapping.items():
            if len(scores) != num_cues:
                raise ValueError(
                    f"Expected {num_cues} cue scores for {item_id}, got {len(scores)}")
        score_path = os.path.join(output_dir, "item2cue_scores.json")
        cue_io.atomic_write_json(score_path, score_mapping)

        table_lines = ["item_id\trank\tcue_id\tcue\trelevance_score\tis_unk\n"]
        for item_id, cue_ids in mapping.items():
            for rank, (cue_id, score) in enumerate(
                zip(cue_ids, score_mapping[item_id]), 1
            ):
                cue_text = vocab[cue_id].replace("\t", " ").replace("\n", " ")
                score_text = "" if score is None else f"{score:.8f}"
                table_lines.append(
                    f"{item_id}\t{rank}\t{cue_id}\t{cue_text}\t{score_text}\t"
                    f"{str(cue_id == UNK_CUE_ID).lower()}\n")
        cue_io.atomic_write_text(
            os.path.join(output_dir, "item_cues.tsv"), "".join(table_lines))

    def sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "wp_d_compatible": (
            num_cues in {CUE_TOKENS, CUE_CANDIDATES_PER_ITEM}
            and len(vocab) == CUE_VOCAB_SIZE),
        "n_items": len(mapping),
        "cue_vocab_size": len(vocab),
        # Keep cues_per_item for older readers while making the new stored vs.
        # active distinction explicit.
        "cues_per_item": num_cues,
        "stored_cues_per_item": num_cues,
        "default_active_cues": CUE_TOKENS,
        "artifact_contract_cues": CUE_CANDIDATES_PER_ITEM,
        "cue_vocab_sha256": sha256(vocab_path),
        "item2cues_sha256": sha256(cues_path),
        "token_layout": {
            "tokens_per_item": TOKEN_LAYOUT.tokens_per_item,
            "bos": 0,
            "boi": TOKEN_LAYOUT.boi_token,
            "eos": TOKEN_LAYOUT.eos_token,
            "cue_start": TOKEN_LAYOUT.cue_token_start,
            "mask": TOKEN_LAYOUT.mask_token,
        },
    }
    if score_path is not None:
        manifest["item2cue_scores_sha256"] = sha256(score_path)
    if assignment_metadata is not None:
        manifest["assignment"] = assignment_metadata
    cue_io.atomic_write_json(os.path.join(output_dir, "cue_manifest.json"), manifest)


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
