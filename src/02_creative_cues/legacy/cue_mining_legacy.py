"""legacy/cue_mining_legacy.py — superseded Week-1 vocab-building + PMI assignment.

**Superseded.** This is the original TF-IDF/KeyBERT/YAKE/LLM vocab-builder and
PMI-based per-song assignment, kept only because legacy/run_wpb.py still calls
it. The active pipeline is cue_extractors.py (extraction) + cue_normalize.py
(cleaning) + cue_assign.py (MMR assignment) + cue_export.py (export/stats),
orchestrated by pipeline.py and driven by run_production.py / run_compare.py —
that pipeline supports multiple extraction methods with a shared cleaning
pipeline, held-out evaluation, and semantic (not just token-overlap) assignment,
none of which this module does.

Do not add new callers of this module. If you need vocab-building or cue
assignment, use pipeline.py instead.
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '00_data_schema'))
from schema import CatalogItem, CueMappingEntry, CUE_VOCAB_SIZE, CUE_TOKENS  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cue_export import UNK_CUE_ID, UNK_CUE_STRING  # noqa: E402

from typing import Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _item_text(item: CatalogItem, lyrics: str = "", lyrics_cap: int = 0) -> str:
    """Concatenate all text fields of an item into one string.

    lyrics_cap: if > 0, truncate lyrics to this many characters before appending.
    Use a cap during vocab building so lyrics don't drown out metadata signals.
    Use 0 (no cap) during per-item cue assignment.
    """
    # Repeat metadata fields 3x so they're not drowned out by lyrics in TF-IDF
    # Artist is intentionally excluded from cue mining text. Cues should describe
    # song-level content, not performer identity.
    meta = " ".join(p for p in [
        item.title, item.genre, item.mood, item.lyric_excerpt, *item.tags
    ] if p)
    parts = [meta, meta, meta]
    if lyrics:
        lyric_text = lyrics[:lyrics_cap] if lyrics_cap else lyrics
        parts.append(lyric_text)
    return " ".join(parts)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into tokens."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def _check_import(package: str, pip_name: str):
    """Raise an actionable ImportError if `package` is not installed."""
    import importlib.util
    if importlib.util.find_spec(package) is None:
        raise ImportError(
            f"Required package '{package}' is not installed. "
            f"Run: pip install {pip_name}"
        )


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
        If None, extraction falls back to title / genre / mood / tags.
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
    """
    if extraction_method == "tfidf":
        return _build_vocab_tfidf(catalog, lyrics_dict, vocab_size)
    if extraction_method == "keybert":
        return _build_vocab_keybert(catalog, lyrics_dict, vocab_size)
    if extraction_method == "yake":
        return _build_vocab_yake(catalog, lyrics_dict, vocab_size)
    if extraction_method == "llm":
        return _build_vocab_llm(catalog, lyrics_dict, vocab_size)
    raise ValueError(f"Unknown extraction_method: '{extraction_method}'")


def _build_vocab_tfidf(
    catalog: list[CatalogItem],
    lyrics_dict: Optional[dict[str, str]],
    vocab_size: int,
) -> list[str]:
    """TF-IDF baseline: top-(vocab_size-1) terms by df*idf across the corpus."""
    _check_import("sklearn", "scikit-learn")
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    # Cap lyrics at 300 chars so metadata fields (title/genre/tags) dominate the vocab
    corpus = [
        _item_text(item, (lyrics_dict or {}).get(item.item_id, ""), lyrics_cap=300)
        for item in catalog
    ]

    vectorizer = TfidfVectorizer(
        max_features=None,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(corpus)  # (n_docs, n_features)

    feature_names = vectorizer.get_feature_names_out()
    idf = vectorizer.idf_                          # shape (n_features,)
    # df = number of docs containing each term
    df = np.array((X > 0).sum(axis=0)).flatten()  # shape (n_features,)
    scores = df * idf

    n_take = vocab_size - 1  # slot 0 is reserved for <unk>
    if len(feature_names) < n_take:
        # Pad with generic placeholders if corpus is too small
        top_terms = list(feature_names[scores.argsort()[::-1]])
        top_terms += [f"<pad_{i}>" for i in range(n_take - len(top_terms))]
    else:
        top_idx = scores.argsort()[::-1][:n_take]
        top_terms = list(feature_names[top_idx])

    return [UNK_CUE_STRING] + top_terms


def _build_vocab_keybert(
    catalog: list[CatalogItem],
    lyrics_dict: Optional[dict[str, str]],
    vocab_size: int,
) -> list[str]:
    """KeyBERT keyphrase extraction — top-(vocab_size-1) phrases by aggregate score."""
    _check_import("keybert", "keybert")
    from keybert import KeyBERT

    kw_model = KeyBERT()
    phrase_scores: dict[str, float] = {}

    for item in catalog:
        text = _item_text(item, (lyrics_dict or {}).get(item.item_id, ""))
        if not text.strip():
            continue
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=20,
        )
        for phrase, score in keywords:
            phrase_scores[phrase] = phrase_scores.get(phrase, 0.0) + score

    n_take = vocab_size - 1
    top_phrases = sorted(phrase_scores, key=phrase_scores.__getitem__, reverse=True)[:n_take]
    # Pad if needed
    top_phrases += [f"<pad_{i}>" for i in range(n_take - len(top_phrases))]
    return [UNK_CUE_STRING] + top_phrases


def _build_vocab_yake(
    catalog: list[CatalogItem],
    lyrics_dict: Optional[dict[str, str]],
    vocab_size: int,
) -> list[str]:
    """YAKE keyphrase extraction — top-(vocab_size-1) phrases by aggregate score."""
    _check_import("yake", "yake")
    import yake

    extractor = yake.KeywordExtractor(
        lan="en",
        n=2,
        dedupLim=0.9,
        top=20,
    )
    phrase_scores: dict[str, float] = {}

    for item in catalog:
        text = _item_text(item, (lyrics_dict or {}).get(item.item_id, ""))
        if not text.strip():
            continue
        keywords = extractor.extract_keywords(text)
        for phrase, score in keywords:
            # YAKE scores are LOWER = better, so invert for aggregation
            phrase_scores[phrase] = phrase_scores.get(phrase, 0.0) + (1.0 / (score + 1e-9))

    n_take = vocab_size - 1
    top_phrases = sorted(phrase_scores, key=phrase_scores.__getitem__, reverse=True)[:n_take]
    top_phrases += [f"<pad_{i}>" for i in range(n_take - len(top_phrases))]
    return [UNK_CUE_STRING] + top_phrases


# ---------------------------------------------------------------------------
# LLM client (swappable; only invoked when method='llm')
# ---------------------------------------------------------------------------

def _llm_client(prompt: str) -> str:
    """Call the configured LLM.  Swap this function to change backends.

    Raises RuntimeError if no client is configured.
    """
    raise RuntimeError(
        "LLM client is not configured. "
        "Set up a Qwen3 client and replace _llm_client() in cue_mining_legacy.py."
    )


def _build_vocab_llm(
    catalog: list[CatalogItem],
    lyrics_dict: Optional[dict[str, str]],
    vocab_size: int,
) -> list[str]:
    """LLM-assisted vocab extraction — Qwen3 extracts thematic phrases per item."""
    phrase_counts: dict[str, int] = {}
    for item in catalog:
        text = _item_text(item, (lyrics_dict or {}).get(item.item_id, ""))
        prompt = (
            f"Extract 10 thematic music keywords or short phrases from the following "
            f"song metadata. Return only a comma-separated list, no explanations.\n\n"
            f"Metadata: {text}\n\nKeywords:"
        )
        response = _llm_client(prompt)
        for phrase in response.split(","):
            phrase = phrase.strip().lower()
            if phrase:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

    n_take = vocab_size - 1
    top_phrases = sorted(phrase_counts, key=phrase_counts.__getitem__, reverse=True)[:n_take]
    top_phrases += [f"<pad_{i}>" for i in range(n_take - len(top_phrases))]
    return [UNK_CUE_STRING] + top_phrases


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
    'pmi'    (default): select 6 vocab entries with highest PMI score against
             item text fields.  Regularize for diversity (penalize near-duplicates).
    'tfidf'  (baseline): top-6 TF-IDF terms from item text (rank by vocab position).
    'llm'    : Qwen3 prompt → extract 6 thematic cues → map to vocab.

    Fallback: any item with fewer than 6 matching cues gets remaining slots
    filled with index 0 ('<unk>').

    Parameters
    ----------
    item   : CatalogItem with at minimum item_id set.
    vocab  : vocab list from build_vocab() or cue_export.load_vocab().
    lyrics : raw lyrics string for this item (may be empty).
    assignment_method: see above.

    Returns
    -------
    CueMappingEntry with exactly 6 cue_ids.
    """
    if assignment_method == "pmi":
        return _assign_pmi(item, vocab, lyrics)
    if assignment_method == "tfidf":
        return _assign_tfidf(item, vocab, lyrics)
    if assignment_method == "llm":
        return _assign_llm(item, vocab, lyrics)
    raise ValueError(f"Unknown assignment_method: '{assignment_method}'")


def _item_token_set(item: CatalogItem, lyrics: str) -> set[str]:
    """Return the set of tokens present in this item's text."""
    return set(_tokenize(_item_text(item, lyrics)))


def _assign_tfidf(item: CatalogItem, vocab: list[str], lyrics: str) -> CueMappingEntry:
    """Top-6 by vocab rank (position in vocab = df*idf rank from build_vocab)."""
    tokens = _item_token_set(item, lyrics)
    matched: list[int] = []
    for idx, term in enumerate(vocab):
        if idx == UNK_CUE_ID:
            continue
        term_tokens = set(_tokenize(term))
        if term_tokens and term_tokens.issubset(tokens):
            matched.append(idx)
        if len(matched) == CUE_TOKENS:
            break  # vocab is already ordered; first N matches are the best-scoring

    cue_ids = _pad_to_six(matched)
    entry = CueMappingEntry(item_id=item.item_id, cue_ids=cue_ids)
    entry.validate()
    return entry


def _assign_pmi(item: CatalogItem, vocab: list[str], lyrics: str) -> CueMappingEntry:
    """PMI-based: prefer rare (high-index) matched terms; regularize for diversity."""
    tokens = _item_token_set(item, lyrics)
    n_vocab = len(vocab)

    # Score: higher index → rarer term → higher PMI.
    # Only terms whose tokens ALL appear in the item text are candidates.
    candidates: list[tuple[float, int]] = []  # (score, vocab_idx)
    for idx, term in enumerate(vocab):
        if idx == UNK_CUE_ID:
            continue
        term_tokens = set(_tokenize(term))
        if not term_tokens:
            continue
        if term_tokens.issubset(tokens):
            # PMI proxy: idf rank (higher index = rarer)
            score = idx / n_vocab
            candidates.append((score, idx))

    # Greedy MMR-style diversity regularization.
    # λ controls exploration vs. exploitation (0 = pure PMI rank, 1 = pure diversity).
    _lambda = 0.5
    selected_indices: list[int] = []
    selected_term_tokens: list[set[str]] = []

    remaining = list(candidates)
    for _ in range(CUE_TOKENS):
        if not remaining:
            break
        best_score = -1.0
        best_pos = 0
        for pos, (score, idx) in enumerate(remaining):
            term_tok = set(_tokenize(vocab[idx]))
            # Penalize overlap with already-selected terms
            max_sim = max(
                (len(term_tok & sel) / max(len(term_tok | sel), 1)
                 for sel in selected_term_tokens),
                default=0.0,
            )
            mmr = (1 - _lambda) * score - _lambda * max_sim
            if mmr > best_score:
                best_score = mmr
                best_pos = pos
        _, chosen_idx = remaining.pop(best_pos)
        selected_indices.append(chosen_idx)
        selected_term_tokens.append(set(_tokenize(vocab[chosen_idx])))

    cue_ids = _pad_to_six(selected_indices)
    entry = CueMappingEntry(item_id=item.item_id, cue_ids=cue_ids)
    entry.validate()
    return entry


def _assign_llm(item: CatalogItem, vocab: list[str], lyrics: str) -> CueMappingEntry:
    """Qwen3 extracts 6 thematic cue phrases; each is mapped to nearest vocab entry."""
    text = _item_text(item, lyrics)
    prompt = (
        f"You are a music tagging assistant. Given a song's metadata, output exactly "
        f"6 short thematic music keywords or phrases, comma-separated.\n\n"
        f"Song: {text}\n\n6 keywords:"
    )
    response = _llm_client(prompt)
    phrases = [p.strip().lower() for p in response.split(",") if p.strip()][:6]

    # Map each phrase to the nearest vocab entry (longest common token overlap)
    vocab_token_sets = [set(_tokenize(v)) for v in vocab]
    selected: list[int] = []
    for phrase in phrases:
        phrase_tok = set(_tokenize(phrase))
        best_idx = UNK_CUE_ID
        best_overlap = -1
        for idx, vtok in enumerate(vocab_token_sets):
            if idx == UNK_CUE_ID or not vtok:
                continue
            overlap = len(phrase_tok & vtok)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx
        selected.append(best_idx)

    cue_ids = _pad_to_six(selected)
    entry = CueMappingEntry(item_id=item.item_id, cue_ids=cue_ids)
    entry.validate()
    return entry


def _pad_to_six(indices: list[int]) -> list[int]:
    """Pad or trim to exactly CUE_TOKENS=6 entries, filling with UNK_CUE_ID."""
    result = list(indices[:CUE_TOKENS])
    while len(result) < CUE_TOKENS:
        result.append(UNK_CUE_ID)
    return result


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
# Synthetic fixture (30-item catalog for smoke-testing without real data)
# ---------------------------------------------------------------------------

def make_synthetic_catalog() -> tuple[list[CatalogItem], dict[str, str]]:
    """Return a 30-item synthetic catalog + lyrics dict for end-to-end testing."""
    _SONGS = [
        ("Bohemian Rhapsody", "Queen",       "rock",       ["operatic", "epic", "piano"],
         "Is this the real life? Is this just fantasy?"),
        ("Hotel California",  "Eagles",      "rock",       ["guitar", "classic", "dark"],
         "On a dark desert highway, cool wind in my hair"),
        ("Blinding Lights",   "The Weeknd",  "pop",        ["synthwave", "80s", "night"],
         "I've been tryna call, I've been on my own"),
        ("Shape of You",      "Ed Sheeran",  "pop",        ["dance", "love", "rhythm"],
         "The club isn't the best place to find a lover"),
        ("Lose Yourself",     "Eminem",      "hip-hop",    ["rap", "motivation", "battle"],
         "His palms are sweaty, knees weak, arms are heavy"),
        ("HUMBLE.",           "Kendrick Lamar", "hip-hop", ["rap", "consciousness", "power"],
         "Sit down, be humble, sit down be humble"),
        ("God's Plan",        "Drake",       "hip-hop",    ["trap", "chill", "success"],
         "She said do you love me I tell her only partly"),
        ("Shallow",           "Lady Gaga",   "pop",        ["ballad", "emotional", "film"],
         "I'm off the deep end, watch as I dive in"),
        ("Rolling in the Deep", "Adele",     "soul",       ["breakup", "powerful", "piano"],
         "There's a fire starting in my heart"),
        ("Uptown Funk",       "Bruno Mars",  "funk",       ["dance", "groovy", "brass"],
         "Don't believe me just watch"),
        ("Sunflower",         "Post Malone", "pop",        ["chill", "soft", "summer"],
         "Needless to say I keep her in check"),
        ("Old Town Road",     "Lil Nas X",   "country rap",["crossover", "viral", "horses"],
         "I'm gonna take my horse to the old town road"),
        ("Smells Like Teen Spirit", "Nirvana", "grunge",   ["guitar", "angst", "90s"],
         "Load up on guns bring your friends"),
        ("Yesterday",         "The Beatles", "classic rock", ["melodic", "nostalgic", "sad"],
         "Yesterday, all my troubles seemed so far away"),
        ("Watermelon Sugar",  "Harry Styles","pop",        ["summer", "happy", "indie"],
         "Tastes like strawberries on a summer evening"),
        ("Levitating",        "Dua Lipa",    "disco pop",  ["dance", "fun", "space"],
         "If you wanna run away with me I know a galaxy"),
        ("Stay",              "The Kid LAROI","pop",       ["emotional", "love", "youthful"],
         "I do the same thing I told you that I never would"),
        ("Peaches",           "Justin Bieber","pop",       ["chill", "summer", "R&B"],
         "I got my peaches out in Georgia"),
        ("Industry Baby",     "Lil Nas X",   "hip-hop",   ["trap", "confident", "brass"],
         "Baby back, I am not fazed"),
        ("Good 4 U",          "Olivia Rodrigo","pop punk", ["breakup", "angry", "guitar"],
         "Well, good for you, I guess you moved on really easily"),
        ("Drivers License",   "Olivia Rodrigo","pop",     ["sad", "love", "teen"],
         "I got my driver's license last week"),
        ("Mood",              "24kGoldn",    "pop",        ["chill", "sad", "party"],
         "Why you always in a mood"),
        ("Dynamite",          "BTS",         "K-pop",      ["dance", "energetic", "happy"],
         "Cause I'm in the stars tonight"),
        ("Butter",            "BTS",         "K-pop",      ["smooth", "pop", "catchy"],
         "Smooth like butter"),
        ("Save Your Tears",   "The Weeknd",  "synth-pop",  ["80s", "sad", "dance"],
         "I saw you dancing in a crowded room"),
        ("Astronaut in the Ocean","Masked Wolf","hip-hop", ["trap", "dark", "introspective"],
         "What you know about rolling down in the deep"),
        ("Montero",           "Lil Nas X",   "pop rap",    ["bold", "colorful", "experimental"],
         "I caught it bad yesterday"),
        ("Essence",           "Wizkid",      "afrobeats",  ["smooth", "afro", "romantic"],
         "Nobody else can give me what you give me"),
        ("Permission to Dance","BTS",         "K-pop",     ["upbeat", "fun", "freedom"],
         "We don't need to worry"),
        ("Beautiful Mistakes","Maroon 5",    "pop",        ["piano", "romantic", "jazz"],
         "Oh I keep making beautiful mistakes"),
    ]
    catalog: list[CatalogItem] = []
    lyrics_dict: dict[str, str] = {}
    for i, (title, artist, genre, tags, lyric) in enumerate(_SONGS):
        item_id = str(i)
        catalog.append(CatalogItem(
            item_id=item_id,
            title=title,
            artist=artist,
            genre=genre,
            tags=tags,
            lyric_excerpt=lyric,
        ))
        lyrics_dict[item_id] = lyric
    return catalog, lyrics_dict


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Run the legacy Week-1 cue-mining pipeline. Superseded — see module docstring.

    Usage (with real backbone metadata):
        python src/02_creative_cues/legacy/cue_mining_legacy.py \
            --metadata path/to/datasets/spotify/metadata.json \
            --output   src/02_creative_cues/outputs/legacy

    Usage (synthetic fixture, no metadata file needed):
        python src/02_creative_cues/legacy/cue_mining_legacy.py --output src/02_creative_cues/outputs/legacy
    """
    import argparse

    from cue_export import compute_coverage_stats, export_outputs, write_report

    parser = argparse.ArgumentParser(description="Legacy creative cue mining (superseded)")
    parser.add_argument("--metadata", default=None, help="Path to backbone metadata.json (omit to use synthetic fixture)")
    parser.add_argument("--output",   default="outputs", help="Output directory")
    parser.add_argument("--vocab-method",   default="tfidf",
                        choices=["tfidf", "keybert", "yake", "llm"],
                        dest="vocab_method")
    parser.add_argument("--assign-method",  default="pmi",
                        choices=["pmi", "tfidf", "llm"],
                        dest="assign_method")
    args = parser.parse_args()

    # Load or generate catalog
    if args.metadata:
        catalog = CatalogItem.load_from_backbone_metadata(args.metadata)
        lyrics_dict = None
    else:
        print("[cue_mining_legacy] No --metadata supplied; using 30-item synthetic fixture.")
        catalog, lyrics_dict = make_synthetic_catalog()
    print(f"[cue_mining_legacy] Loaded {len(catalog)} catalog items.")

    # Build vocabulary
    print(f"[cue_mining_legacy] Building vocab (method={args.vocab_method}) …")
    vocab = build_vocab(catalog, lyrics_dict, vocab_size=CUE_VOCAB_SIZE,
                        extraction_method=args.vocab_method)
    assert vocab[0] == UNK_CUE_STRING, "vocab[0] must be '<unk>'"
    assert len(vocab) == CUE_VOCAB_SIZE, f"vocab must have {CUE_VOCAB_SIZE} entries"
    print(f"[cue_mining_legacy] Vocab built: {len(vocab)} entries. Sample [1:6]: {vocab[1:6]}")

    # Assign cues to every item
    print(f"[cue_mining_legacy] Assigning cues (method={args.assign_method}) …")
    item2cues = run_full_dataset(catalog, vocab, lyrics_dict, args.assign_method)
    print(f"[cue_mining_legacy] Assigned cues for {len(item2cues)} items.")

    # Coverage stats + report
    stats = compute_coverage_stats(item2cues, vocab)
    print(f"[cue_mining_legacy] Stats: {stats}")

    # Export all three output files
    export_outputs(vocab, item2cues, args.output)
    write_report(stats, args.output)
    print("[cue_mining_legacy] Done.")
