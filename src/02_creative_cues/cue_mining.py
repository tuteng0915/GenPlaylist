"""02_creative_cues/cue_mining.py — WP-B: Creative Cue Vocabulary and Assignment.

**Owner:** Student 2 (WP-B)

Goal
----
Build a 2048-entry creative cue vocabulary and assign exactly 6 cue tokens
per catalog song:

    song metadata + lyrics  →  cue_vocab.json + item2cues.json

The output (item2cues.json) extends the backbone tokenizer from 5 tokens per
item (BOI + z0 + z1 + z2 + z_conf) to 12 tokens per item
(BOI + z0 + z1 + z2 + z_conf + c0 + c1 + c2 + c3 + c4 + c5).

Output files (write to this directory's outputs/)
--------------------------------------------------
  outputs/cue_vocab.json   — list of 2048 cue strings; index = cue_id
  outputs/item2cues.json   — {"item_id": [c0, c1, c2, c3, c4, c5], ...}
  outputs/cue_report.md    — coverage / sparsity / distribution stats

Interface contract
------------------
  Input  : CatalogItem list + optional lyrics dict
  Output : item2cues consumed by 00_data_schema.schema.CueMappingEntry
  Invariant: every item_id in the backbone catalog must appear in item2cues
             (unknown = 6× index 0, the '<unk>' cue).

Schema constants (do NOT change without coordinating with 00_data_schema)
-------------------------------------------------------------------------
  CUE_VOCAB_SIZE = 2048    (index 0 = '<unk>')
  CUE_TOKENS     = 6       (c0 … c5, per song)

Implementation roadmap (see TODO.md)
-------------------------------------
  Week 1 : data collection; missing-rate audit; TF-IDF baseline extraction
  Week 2 : KeyBERT / YAKE / LLM-assisted extraction; vocab construction
  Week 3 : assign c0-c5 per song; coverage stats; export files; write report
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '00_data_schema'))
from schema import CatalogItem, CueMappingEntry, CUE_VOCAB_SIZE, CUE_TOKENS  # noqa: E402

from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNK_CUE_ID = 0           # index 0 reserved for '<unk>' (missing coverage fallback)
UNK_CUE_STRING = "<unk>"


# ---------------------------------------------------------------------------
# Step 1: Vocabulary construction
# ---------------------------------------------------------------------------

def build_vocab(
    catalog: list[CatalogItem],
    lyrics_dict: Optional[dict[str, str]] = None,
    vocab_size: int = CUE_VOCAB_SIZE,
    extraction_method: str = "tfidf",
) -> list[str]:
    """Extract and filter a cue vocabulary from catalog metadata + lyrics.

    The vocabulary is a sorted list of `vocab_size` thematic strings.
    Index 0 is always '<unk>'.  Subsequent entries are ranked by coverage
    (how many songs a cue applies to).

    Parameters
    ----------
    catalog:
        List of CatalogItem for all items in the backbone catalog.
    lyrics_dict:
        Optional dict mapping item_id → raw lyrics string.
        If None, extraction falls back to title / artist / genre / tags.
    vocab_size:
        Target vocabulary size (paper uses 2048).
    extraction_method:
        Extraction algorithm:
          'tfidf'   — TF-IDF over concatenated text fields (Week 1 baseline)
          'keybert' — KeyBERT keyphrase extraction (Week 2)
          'yake'    — YAKE keyphrase extraction (Week 2)
          'llm'     — LLM-assisted extraction via Qwen3 (Week 2/3)

    Returns
    -------
    list[str]: vocab_size cue strings.  vocab[0] = '<unk>'.

    TODO (WP-B Week 1): implement 'tfidf' extraction baseline.
    TODO (WP-B Week 2): implement 'keybert', 'yake', 'llm'.
    """
    if extraction_method == "tfidf":
        raise NotImplementedError(
            "TF-IDF vocabulary extraction is a WP-B Week 1 deliverable.\n"
            "Implement: concatenate title+artist+genre+tags+lyrics per item, "
            "run TfidfVectorizer, pick top-(vocab_size-1) terms by df*idf score."
        )
    raise NotImplementedError(
        f"Extraction method '{extraction_method}' is a WP-B Week 2 deliverable."
    )


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
# Step 2: Per-item cue assignment
# ---------------------------------------------------------------------------

def assign_cues(
    item: CatalogItem,
    vocab: list[str],
    lyrics: str = "",
    assignment_method: str = "pmi",
) -> CueMappingEntry:
    """Assign exactly CUE_TOKENS=6 cue indices to a single song.

    Assignment strategy
    -------------------
    'pmi'    (Week 2): select 6 vocab entries with highest PMI score against
             item text fields.  Regularize for diversity (penalize near-duplicates).
    'tfidf'  (Week 1 baseline): top-6 TF-IDF terms from item text.
    'llm'    (Week 2/3): Qwen3 prompt → extract 6 thematic cues → map to vocab.

    Fallback: any item with fewer than 6 matching cues gets remaining slots
    filled with index 0 ('<unk>').

    Parameters
    ----------
    item   : CatalogItem with at minimum item_id set.
    vocab  : vocab list from build_vocab() or load_vocab().
    lyrics : raw lyrics string for this item (may be empty).
    assignment_method: see above.

    Returns
    -------
    CueMappingEntry with exactly 6 cue_ids.
    """
    if assignment_method in ("pmi", "tfidf", "llm"):
        raise NotImplementedError(
            f"Cue assignment (method='{assignment_method}') is a WP-B Week 2 deliverable.\n"
            "Implement: score vocab entries against item text → pick top-6 with diversity.\n"
            "Fallback: pad with UNK_CUE_ID (0) until len == 6."
        )
    raise NotImplementedError(f"Unknown assignment method: {assignment_method}")


def _unk_entry(item_id: str) -> CueMappingEntry:
    """Fallback mapping: all 6 cues are '<unk>' (index 0)."""
    return CueMappingEntry(item_id=item_id, cue_ids=[UNK_CUE_ID] * CUE_TOKENS)


# ---------------------------------------------------------------------------
# Step 3: Full-catalog run
# ---------------------------------------------------------------------------

def run_full_dataset(
    catalog: list[CatalogItem],
    vocab: list[str],
    lyrics_dict: Optional[dict[str, str]] = None,
    assignment_method: str = "pmi",
) -> dict[str, CueMappingEntry]:
    """Assign cues to every item in the catalog.

    Items with no text data get [0,0,0,0,0,0] (full unk fallback).

    Parameters
    ----------
    catalog  : full catalog as list[CatalogItem].
    vocab    : built vocabulary (list of CUE_VOCAB_SIZE strings).
    lyrics_dict: optional item_id → lyrics string.
    assignment_method: passed to assign_cues().

    Returns
    -------
    dict[str, CueMappingEntry]: item_id → CueMappingEntry (validated).
    """
    result: dict[str, CueMappingEntry] = {}
    for item in catalog:
        lyrics = (lyrics_dict or {}).get(item.item_id, "")
        try:
            entry = assign_cues(item, vocab, lyrics, assignment_method)
            entry.validate()
        except NotImplementedError:
            raise
        except Exception:
            entry = _unk_entry(item.item_id)
        result[item.item_id] = entry
    return result


# ---------------------------------------------------------------------------
# Step 4: Export
# ---------------------------------------------------------------------------

def export_outputs(
    vocab: list[str],
    item2cues: dict[str, CueMappingEntry],
    output_dir: str,
) -> None:
    """Write cue_vocab.json and item2cues.json to output_dir.

    item2cues.json format: {"item_id": [c0, c1, c2, c3, c4, c5], ...}
    This is the format that CueMappingEntry.load_mapping() reads.

    Parameters
    ----------
    vocab      : list of CUE_VOCAB_SIZE strings (vocab[0] == '<unk>').
    item2cues  : mapping produced by run_full_dataset().
    output_dir : directory to write files.
    """
    os.makedirs(output_dir, exist_ok=True)

    vocab_path = os.path.join(output_dir, "cue_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"[cue_mining] Wrote vocab ({len(vocab)} entries) → {vocab_path}")

    mapping = {iid: entry.cue_ids for iid, entry in item2cues.items()}
    cues_path = os.path.join(output_dir, "item2cues.json")
    with open(cues_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f)
    print(f"[cue_mining] Wrote item2cues ({len(mapping)} items) → {cues_path}")


# ---------------------------------------------------------------------------
# Step 5: Quality metrics
# ---------------------------------------------------------------------------

def compute_coverage_stats(
    item2cues: dict[str, CueMappingEntry],
    vocab: list[str],
) -> dict:
    """Compute coverage and distribution statistics for the cue mapping.

    Metrics
    -------
    - coverage_rate  : fraction of items with at least 1 non-unk cue
    - unk_rate       : mean fraction of cue slots that are '<unk>' (index 0)
    - vocab_coverage : fraction of vocab entries used by at least 1 item
    - top10_cues     : list of (cue_string, count) for the 10 most used cues
    - cue_entropy    : Shannon entropy of cue usage distribution

    Returns
    -------
    dict with the above keys + values.
    """
    import collections
    import math

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

    total_slots = n_items * CUE_TOKENS
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
# CLI entry point (for WP-B run scripts)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Quick sanity-check: load backbone metadata, run unk-fallback for all items.

    Usage:
        python src/02_creative_cues/cue_mining.py \
            --metadata path/to/datasets/spotify/metadata.json \
            --output   src/02_creative_cues/outputs/
    """
    import argparse
    parser = argparse.ArgumentParser(description="Creative cue mining (WP-B)")
    parser.add_argument("--metadata", required=True, help="Path to backbone metadata.json")
    parser.add_argument("--output",   default="outputs", help="Output directory")
    parser.add_argument("--method",   default="tfidf",
                        choices=["tfidf", "pmi", "keybert", "yake", "llm"])
    args = parser.parse_args()

    catalog = CatalogItem.load_from_backbone_metadata(args.metadata)
    print(f"[cue_mining] Loaded {len(catalog)} catalog items.")

    # TODO: replace with real vocab once WP-B Week 1 is done
    # For now, create a placeholder vocab for structural testing
    dummy_vocab = [UNK_CUE_STRING] + [f"cue_{i}" for i in range(1, CUE_VOCAB_SIZE)]
    print(f"[cue_mining] Using dummy vocab ({len(dummy_vocab)} entries).")

    item2cues = {
        item.item_id: _unk_entry(item.item_id) for item in catalog
    }
    stats = compute_coverage_stats(item2cues, dummy_vocab)
    print(f"[cue_mining] Stats: {stats}")

    export_outputs(dummy_vocab, item2cues, args.output)
    print("[cue_mining] Done. Replace dummy vocab + unk fallback with real assignment.")
