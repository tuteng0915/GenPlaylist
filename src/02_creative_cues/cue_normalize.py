"""cue_normalize.py — WP-B Phase 1: cue normalization pipeline (paper §4.2 step 3).

Extractor-agnostic. Takes RAW candidate cues per song from any front-end
(TF-IDF / KeyBERT / YAKE / LLM) and produces a clean 2048-entry cue vocabulary:

    raw_cues: {item_id -> [phrase, ...]}
        │
        ├─ 0. T5 tokenize           tokenize each cue with the T5 tokenizer; a
        │                           cue's identity from here on is its token-id
        │                           sequence (dedup happens on token equality,
        │                           not raw string equality)
        ├─ 1. df-band filter        5 ≤ df ≤ 0.3·N   (drop hapax + corpus-dominant)
        ├─ 2. POS filter            keep noun / adj+noun phrases (drop verb fragments)
        ├─ 3. embed cues            sentence encoder (MiniLM) — cosine = semantic sim
        ├─ 4. semantic dedup        merge pairs with cosine > 0.92
        └─ 5. rank by IDF, keep top-(vocab_size-1), prepend "<unk>"  → vocab_size

T5 is used only in step 0 to canonicalize/dedup cues by token identity; the
semantic embeddings (steps 3-4 and downstream assignment) come from a sentence
encoder, per the paper's "multilingual sentence encoder" specification.

Graceful degradation (so it runs with zero heavy installs):
  - embedder : sentence-transformers MiniLM if installed, else sklearn char-ngram TF-IDF
  - POS      : spaCy if installed, else NLTK, else a regex/stopword heuristic

The returned (vocab, cue_embeddings) are reused by assignment (PMI + semantic
diversity) and evaluation, so embeddings are computed once.
"""

from __future__ import annotations

import collections
import math
import os
import re
from typing import Callable, Optional

import numpy as np

UNK_CUE_STRING = "<unk>"

# Lead-word blocklist for the heuristic POS fallback: phrases starting with these
# are almost always verb/filler fragments ("need need", "wanna make", "know let").
_VERB_FILLER_LEADS = {
    "need", "want", "wanna", "gonna", "gotta", "know", "let", "make", "get",
    "got", "say", "said", "go", "going", "come", "came", "take", "feel", "tell",
    "give", "put", "keep", "keeps", "kept", "see", "saw", "look", "looking",
    "do", "does", "did", "be", "is", "are", "was", "were", "can", "will",
    "would", "could", "should", "oh", "yeah", "ooh", "la", "na", "hey", "uh",
}
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "of", "to", "in",
    "on", "at", "for", "with", "as", "by", "it", "its", "this", "that", "these",
    "those", "i", "you", "he", "she", "we", "they", "me", "my", "your", "all",
    "just", "like", "up", "down", "out", "no", "not", "now", "what", "when",
}

# Boilerplate / structural tokens that are never valid creative cues. A candidate
# cue is dropped if ANY of its tokens is one of these.
_DOMAIN_HARD_STOP = {
    "artist", "album", "title", "bpm", "key",
    "lyrics", "lyric", "song", "songs", "verse", "chorus", "bridge", "intro",
    "outro", "instrumental", "neutral", "feat", "ft", "remix", "edit", "version",
    "oh", "ooh", "oooh", "yeah", "yea", "hey", "na", "nah", "la", "uh", "uhh",
    "huh", "woah", "whoa", "mmm", "hmm", "ah", "ahh", "gonna", "wanna", "gotta",
}

_EXACT_PHRASE_BLOCK = {
    "love don", "things don", "girl don", "don worry",
    "ain got", "baby girl", "baby want",
    "rock intense", "rock melancholic", "rock pop",
    "major mood", "key major", "bpm key",
}

_FRAGMENT_EDGE_TOKENS = {
    "don", "ain", "got", "gonna", "wanna", "gotta", "yeah", "yea", "oh", "ooh", "oooh",
}

_COLLOQUIAL_TOKEN_NORMALIZATION = {
    "talkin": "talking",
    "mornin": "morning",
    "feelin": "feeling",
    "lookin": "looking",
    "runnin": "running",
    "somethin": "something",
}


_T5_MODEL_NAME = os.environ.get("CUE_T5_MODEL", "t5-small")

_t5_tokenizer_singleton = None


def _get_t5_tokenizer():
    """Lazily load and cache the T5 tokenizer used for step-0 cue canonicalization."""
    global _t5_tokenizer_singleton
    if _t5_tokenizer_singleton is None:
        from transformers import T5TokenizerFast
        _t5_tokenizer_singleton = T5TokenizerFast.from_pretrained(_T5_MODEL_NAME)
    return _t5_tokenizer_singleton


def _normalize_cue_text(cue: str) -> str:
    """Normalize raw cue strings before df counting and exact dedup."""
    cue = cue.lower().replace("\u2019", "'").replace("â€™", "'").replace("`", "'")
    cue = re.sub(r"[^a-z0-9']+", " ", cue)
    cue = re.sub(r"\s+", " ", cue).strip(" '")
    if not cue:
        return ""

    toks = []
    for tok in cue.split():
        tok = tok.rstrip("'")
        toks.append(_COLLOQUIAL_TOKEN_NORMALIZATION.get(tok, tok))
    return " ".join(toks)


def normalize_raw_cues(
    raw_cues: dict[str, list[str]],
    tokenizer=None,
) -> tuple[dict[str, list[str]], dict[str, tuple[int, ...]]]:
    """Normalize cue text, then dedup on T5 token identity (not string equality).

    Each cue goes through the cheap regex/colloquial cleanup first, then is
    tokenized with the T5 tokenizer. From here on a cue's identity is its
    token-id sequence: two spellings that tokenize identically collapse into
    one, even if their surface strings still differ. The decoded, canonicalized
    text is what df-counting / POS filtering / blocklist filtering (steps 1-2)
    operate on; the returned token_ids_of map lets the embedder (step 3) reuse
    these exact tokens instead of re-tokenizing from scratch.
    """
    tokenizer = tokenizer or _get_t5_tokenizer()
    token_ids_of: dict[str, tuple[int, ...]] = {}
    normalized: dict[str, list[str]] = {}
    for item_id, cues in raw_cues.items():
        seen_ids: set[tuple[int, ...]] = set()
        cleaned: list[str] = []
        for cue in cues:
            pre = _normalize_cue_text(str(cue))
            if not pre:
                continue
            ids = tuple(tokenizer(pre, add_special_tokens=False)["input_ids"])
            if not ids or ids in seen_ids:
                continue
            canonical = tokenizer.decode(ids, skip_special_tokens=True)
            canonical = re.sub(r"\s*'\s*", "'", canonical)
            canonical = re.sub(r"\s+", " ", canonical).strip()
            if not canonical:
                continue
            seen_ids.add(ids)
            cleaned.append(canonical)
            token_ids_of[canonical] = ids
        normalized[item_id] = cleaned
    return normalized, token_ids_of


def build_block_tokens(catalog) -> set[str]:
    """Soft-blocked tokens drawn from ARTIST names only.

    Creative cues should be imagery/motifs, not who performed the song. Genre and
    mood are intentionally NOT blocked — they are valid cue categories (the paper
    lists mood/emotion among target cues). A candidate cue is dropped if ALL of its
    tokens are artist tokens (so 'tame impala' / 'marley wailers' go, but 'neon rain'
    and 'reggae' stay).
    """
    blocked: set[str] = set()
    for it in catalog:
        if it.artist:
            blocked.update(re.findall(r"[a-z]+", it.artist.lower()))
    blocked -= _STOPWORDS   # keep generic words available to real cues
    return blocked


def _blocklist_filter(cues: list[str], block_tokens: set[str]) -> list[str]:
    """Drop boilerplate cues (hard stop) and tag/artist-only cues (all tokens blocked)."""
    out = []
    for c in cues:
        toks = c.split()
        if c in _EXACT_PHRASE_BLOCK:
            continue
        if any(t in _DOMAIN_HARD_STOP for t in toks):
            continue
        if len(toks) > 1 and (
            toks[0] in _FRAGMENT_EDGE_TOKENS or toks[-1] in _FRAGMENT_EDGE_TOKENS
        ):
            continue
        if block_tokens and toks and all(t in block_tokens for t in toks):
            continue
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Step 1: document frequency + df-band filter
# ---------------------------------------------------------------------------

def compute_df(raw_cues: dict[str, list[str]]) -> tuple[collections.Counter, int]:
    """Return (df Counter over distinct cues, N = number of songs)."""
    df: collections.Counter = collections.Counter()
    for cues in raw_cues.values():
        for cue in set(cues):           # count each cue once per song
            df[cue] += 1
    return df, len(raw_cues)


def df_band_filter(
    cues: list[str],
    df: collections.Counter,
    n_docs: int,
    min_df: int = 5,
    max_df_frac: float = 0.3,
) -> list[str]:
    """Keep cues with min_df ≤ df ≤ max_df_frac·N (paper: 5 ≤ df ≤ 0.3N)."""
    max_df = max_df_frac * n_docs
    return [c for c in cues if min_df <= df[c] <= max_df]


# ---------------------------------------------------------------------------
# Step 2: POS filter (keep noun-ish phrases, drop verb/filler fragments)
# ---------------------------------------------------------------------------

def _make_pos_filter() -> Callable[[list[str]], list[str]]:
    """Return a POS-filter function, preferring spaCy → NLTK → heuristic."""
    # --- spaCy ---
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

        def keep(cue: str) -> bool:
            doc = nlp(cue)
            if not len(doc):
                return False
            # head token must be a NOUN/PROPN; no leading verb
            if doc[0].pos_ in ("VERB", "AUX", "INTJ"):
                return False
            return any(t.pos_ in ("NOUN", "PROPN") for t in doc)

        return lambda cues: [c for c in cues if keep(c)]
    except Exception:
        pass

    # --- NLTK ---
    try:
        import nltk
        from nltk import pos_tag, word_tokenize
        nltk.data.find("taggers/averaged_perceptron_tagger")

        def keep(cue: str) -> bool:
            toks = word_tokenize(cue)
            if not toks:
                return False
            tags = [t for _, t in pos_tag(toks)]
            if tags[0].startswith(("VB", "UH")):
                return False
            return any(t.startswith("NN") for t in tags)

        return lambda cues: [c for c in cues if keep(c)]
    except Exception:
        pass

    # --- heuristic fallback (no deps) ---
    def keep(cue: str) -> bool:
        toks = cue.split()
        if not toks:
            return False
        if toks[0] in _VERB_FILLER_LEADS:
            return False
        # drop repeated-word bigrams ("need need", "feel feel", "long long")
        if len(toks) == 2 and toks[0] == toks[1]:
            return False
        # require at least one token that isn't a stopword/filler
        return any(t not in _STOPWORDS and t not in _VERB_FILLER_LEADS for t in toks)

    return lambda cues: [c for c in cues if keep(c)]


# ---------------------------------------------------------------------------
# Step 3: embedder (semantic vectors for dedup + downstream reuse)
# ---------------------------------------------------------------------------

def _make_embedder(
    token_ids_of: Optional[dict[str, tuple[int, ...]]] = None,
) -> tuple[Callable[[list[str]], np.ndarray], str]:
    """Return (embed_fn, backend_name). Vectors are L2-normalized.

    Uses a sentence-embedding model (all-MiniLM-L6-v2) — the paper's "multilingual
    sentence encoder" role — so cosine reflects semantic similarity, which is what
    assignment relevance and semantic dedup rely on. Falls back to a sklearn
    char-ngram vectorizer only if sentence-transformers can't load.

    Note: T5 is used ONLY for token-identity canonicalization in normalize_raw_cues
    (step 0); its raw encoder is NOT a semantic-similarity model, so it is not used
    for embeddings here. `token_ids_of` is accepted for signature compatibility but
    unused by the sentence encoder (it re-encodes from strings).
    """
    # --- sentence-transformers MiniLM (preferred; trained for semantic similarity) ---
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        def embed(strs: list[str]) -> np.ndarray:
            if not strs:
                return np.zeros((0, 1), dtype=np.float32)
            v = model.encode(strs, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(v, dtype=np.float32)

        return embed, "sentence-transformers/all-MiniLM-L6-v2"
    except Exception:
        pass

    # --- sklearn char-ngram fallback (lexical, no torch) ---
    from sklearn.feature_extraction.text import TfidfVectorizer

    def embed(strs: list[str]) -> np.ndarray:
        if not strs:
            return np.zeros((0, 1), dtype=np.float32)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        m = vec.fit_transform(strs).astype(np.float32).toarray()
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms

    return embed, "sklearn-char-ngram (fallback)"


# ---------------------------------------------------------------------------
# Step 4: semantic dedup
# ---------------------------------------------------------------------------

def semantic_dedup(
    cues: list[str],
    embeddings: np.ndarray,
    threshold: float = 0.92,
) -> tuple[list[str], np.ndarray]:
    """Greedily merge near-duplicate cues (cosine > threshold).

    Cues are assumed pre-sorted by priority (first = kept representative).
    Returns (kept_cues, kept_embeddings).
    """
    kept_idx: list[int] = []
    kept_vecs: Optional[np.ndarray] = None
    for i in range(len(cues)):
        v = embeddings[i:i + 1]                      # (1, d), L2-normalized
        if kept_vecs is not None and kept_vecs.shape[0]:
            sims = (kept_vecs @ v.T).ravel()         # cosine vs all kept
            if float(sims.max()) > threshold:
                continue                             # duplicate → merge away
        kept_idx.append(i)
        kept_vecs = v if kept_vecs is None else np.vstack([kept_vecs, v])
    return [cues[i] for i in kept_idx], embeddings[kept_idx]


# ---------------------------------------------------------------------------
# Step 5: IDF ranking
# ---------------------------------------------------------------------------

def idf_score(cue: str, df: collections.Counter, n_docs: int) -> float:
    """Standard IDF: higher = rarer = more discriminative."""
    return math.log(n_docs / (1 + df[cue]))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_vocab_normalized(
    raw_cues: dict[str, list[str]],
    vocab_size: int = 2048,
    min_df: int = 5,
    max_df_frac: float = 0.3,
    dedup_threshold: float = 0.92,
    block_tokens: Optional[set[str]] = None,
    verbose: bool = True,
) -> dict:
    """Run the full Phase-1 pipeline.

    Returns a dict with:
      vocab            : list[str] of length vocab_size, vocab[0] == "<unk>"
      embeddings       : np.ndarray (vocab_size-1, d) for cues[1:] (None for <unk>)
      embedder_backend : which embedder was used
      stats            : per-stage survivor counts
    """
    # 0. T5 tokenize (dedup on token identity) — token_ids_of feeds step 3
    raw_cues, token_ids_of = normalize_raw_cues(raw_cues)
    df, n_docs = compute_df(raw_cues)
    candidates = list(df.keys())
    stats = {"raw_candidates": len(candidates), "n_docs": n_docs}

    # 1. df-band filter
    candidates = df_band_filter(candidates, df, n_docs, min_df, max_df_frac)
    stats["after_df_band"] = len(candidates)

    # 2. POS filter
    pos_filter = _make_pos_filter()
    candidates = pos_filter(candidates)
    stats["after_pos"] = len(candidates)

    # 2b. blocklist filter (boilerplate + artist-only cues)
    candidates = _blocklist_filter(candidates, block_tokens or set())
    stats["after_blocklist"] = len(candidates)

    # Sort by IDF descending BEFORE dedup so the most discriminative survives a merge
    candidates.sort(key=lambda c: idf_score(c, df, n_docs), reverse=True)

    # 3. embed (sentence encoder / MiniLM; token_ids_of accepted but unused)
    embed_fn, backend = _make_embedder(token_ids_of)
    emb = embed_fn(candidates) if candidates else np.zeros((0, 1), dtype=np.float32)

    # 4. semantic dedup
    candidates, emb = semantic_dedup(candidates, emb, dedup_threshold)
    stats["after_dedup"] = len(candidates)

    # 5. top-(vocab_size-1) by IDF (already IDF-sorted; dedup preserved order)
    n_take = vocab_size - 1
    top = candidates[:n_take]
    top_emb = emb[:n_take]

    # pad if corpus too small to fill the vocab
    if len(top) < n_take:
        pad = [f"<pad_{i}>" for i in range(n_take - len(top))]
        top = top + pad
        if top_emb.shape[0] < n_take:
            d = top_emb.shape[1] if top_emb.shape[0] else 1
            top_emb = np.vstack([top_emb, np.zeros((n_take - top_emb.shape[0], d), np.float32)])

    vocab = [UNK_CUE_STRING] + top
    stats["final_vocab"] = len(vocab)

    if verbose:
        print(f"[normalize] embedder: {backend}")
        print(f"[normalize] raw candidates : {stats['raw_candidates']}")
        print(f"[normalize] after df-band  : {stats['after_df_band']} "
              f"(kept {min_df} <= df <= {max_df_frac:.0%} of {n_docs})")
        print(f"[normalize] after POS      : {stats['after_pos']}")
        print(f"[normalize] after blocklist: {stats['after_blocklist']} (boilerplate/artist)")
        print(f"[normalize] after dedup    : {stats['after_dedup']} (cos > {dedup_threshold})")
        print(f"[normalize] final vocab    : {stats['final_vocab']} (incl <unk>)")

    return {
        "vocab": vocab,
        "embeddings": top_emb,           # aligns with vocab[1:]
        "embedder_backend": backend,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Demonstrate the normalizer on real data with a quick raw-cue extractor.

    Raw cues here are a placeholder per-song candidate pool (unigrams+bigrams from
    metadata + capped lyrics). Phase 2 (cue_extractors.py) replaces this with the
    real TF-IDF / KeyBERT / YAKE / LLM front-ends.

    Usage:
      .venv/Scripts/python.exe src/02_creative_cues/cue_normalize.py --limit 1000
    """
    import argparse
    import json
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_data_schema"))
    from schema import CatalogItem  # noqa: E402

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--catalog", default="data/dataset/catalog_metadata.json")
    parser.add_argument("--lyrics-dir", default="data/lyrics/spotify")
    args = parser.parse_args()

    with open(args.catalog, encoding="utf-8") as f:
        raw = json.load(f)
    items = [
        CatalogItem(**{k: v for k, v in e.items() if k in CatalogItem.__dataclass_fields__})
        for e in list(raw.values())[:args.limit]
    ]
    print(f"[demo] {len(items)} items")

    # placeholder raw-cue extractor: unigrams + bigrams from metadata + 300-char lyrics
    _tok = lambda t: re.findall(r"[a-z]+", t.lower())
    raw_cues: dict[str, list[str]] = {}
    for it in items:
        meta = " ".join([it.title, it.genre, it.mood, *it.tags])
        lp = os.path.join(args.lyrics_dir, f"{it.item_id}.txt")
        lyr = open(lp, encoding="utf-8", errors="ignore").read()[:300] if os.path.isfile(lp) else ""
        toks = _tok(meta + " " + lyr)
        grams = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
        raw_cues[it.item_id] = grams

    result = build_vocab_normalized(raw_cues, vocab_size=2048)
    print("\n[demo] top 30 cues after normalization:")
    print(result["vocab"][1:31])
