"""cue_extractors.py — WP-B Phase 2: swappable raw-cue extraction front-ends.

Each front-end maps the catalog to RAW candidate cues per song:

    extract_raw_cues(catalog, lyrics_dict, method) -> {item_id: [phrase, ...]}

The four methods (tfidf / yake / keybert / llm) all emit the same shape, then feed
the SAME normalization pipeline (cue_normalize.build_vocab_normalized), so any
difference in the final vocabulary is attributable purely to extraction quality.

Results are cached to outputs/<method>/raw_cues.json so the expensive front-ends
(keybert, llm) only run once. Pass force=True to recompute.

Heavy libraries are imported lazily inside each extractor with a clear pip hint,
so importing this module never requires keybert/torch/openai to be installed.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

UNK = "<unk>"

_OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "outputs")
_TOK = re.compile(r"[a-z][a-z']+")


def _meta_text(item) -> str:
    """Metadata used for cue extraction.

    Artist is intentionally excluded: cue mining should capture song-level
    content, not performer identity or repeated artist-name artifacts.
    """
    return " ".join(p for p in [item.title, item.genre, item.mood,
                                item.lyric_excerpt, *item.tags] if p)


def _full_text(item, lyrics: str, lyrics_cap: int = 0) -> str:
    meta = _meta_text(item)
    if lyrics:
        lyrics = lyrics[:lyrics_cap] if lyrics_cap else lyrics
        return f"{meta} {lyrics}"
    return meta


def _cache_path(method: str, n_items: int, tag: str = "") -> str:
    # include song count (and lyrics-mode tag) so a cache from a different sample
    # size or preprocessing mode is never reused silently
    name = f"raw_cues_{n_items}.json" if not tag else f"raw_cues_{n_items}_{tag}.json"
    return os.path.join(_OUTPUT_ROOT, method, name)


def _llm_prompt(item, lyrics_dict: dict[str, str], top_n: int) -> str:
    # lyrics arrive already preprocessed upstream (cue_lyrics); no extra cap here
    text = _full_text(item, lyrics_dict.get(item.item_id, ""), lyrics_cap=0)
    return (
        f"Song metadata + lyrics:\n{text}\n\n"
        f"List up to {top_n} creative cues found in the song lyrics that describe the song's content as a comma-separated list, "
        f"lowercase, no numbering, no extra text."
    )


def _parse_llm_cues(resp: str) -> list[str]:
    cues = [c.strip().lower() for c in resp.replace("\n", ",").split(",")]
    return [c for c in cues if c and len(c) <= 40]


def _read_cached_raw(method: str, n_items: int, tag: str = "") -> Optional[dict[str, list[str]]]:
    path = _cache_path(method, n_items, tag)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        print(f"[extract:{method}] loaded cached raw cues ({path})")
        return json.load(f)


def _write_raw(method: str, n_items: int, raw: dict[str, list[str]], tag: str = "") -> str:
    path = _cache_path(method, n_items, tag)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False)
    print(f"[extract:{method}] wrote raw cues for {len(raw)} items -> {path}")
    return path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_raw_cues(
    catalog: list,
    lyrics_dict: Optional[dict[str, str]],
    method: str = "tfidf",
    force: bool = False,
    top_n: int = 25,
    cache_tag: str = "",
) -> dict[str, list[str]]:
    """Return {item_id: [raw cue phrases]} for the given method, with disk cache.

    cache_tag distinguishes caches produced under different lyrics-preprocessing
    modes so switching modes never silently reuses stale cues.
    """
    cached = None if force else _read_cached_raw(method, len(catalog), cache_tag)
    if cached is not None:
        return cached

    lyrics_dict = lyrics_dict or {}
    if method == "tfidf":
        raw = _extract_tfidf(catalog, lyrics_dict, top_n)
    elif method == "yake":
        raw = _extract_yake(catalog, lyrics_dict, top_n)
    elif method == "keybert":
        raw = _extract_keybert(catalog, lyrics_dict, top_n)
    elif method == "llm":
        raw = _extract_llm(catalog, lyrics_dict, top_n)
    else:
        raise ValueError(f"Unknown extraction method: '{method}'")

    _write_raw(method, len(catalog), raw, cache_tag)
    return raw


# ---------------------------------------------------------------------------
# tfidf — per-song top terms by TF-IDF weight (corpus-fit, fast)
# ---------------------------------------------------------------------------

def _extract_tfidf(catalog, lyrics_dict, top_n) -> dict[str, list[str]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    ids = [it.item_id for it in catalog]
    corpus = [_full_text(it, lyrics_dict.get(it.item_id, ""), lyrics_cap=0)
              for it in catalog]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                          min_df=2, sublinear_tf=True)
    X = vec.fit_transform(corpus)
    terms = vec.get_feature_names_out()

    raw: dict[str, list[str]] = {}
    for r, iid in enumerate(ids):
        row = X.getrow(r)
        if row.nnz == 0:
            raw[iid] = []
            continue
        idx = row.indices[np.argsort(row.data)[::-1][:top_n]]
        raw[iid] = [terms[j] for j in idx]
    return raw


# ---------------------------------------------------------------------------
# yake — per-song statistical keyphrases (pure python)
# ---------------------------------------------------------------------------

def _extract_yake(catalog, lyrics_dict, top_n) -> dict[str, list[str]]:
    try:
        import yake
    except ImportError:
        raise ImportError("yake not installed. Run: pip install yake")
    extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=top_n)

    raw: dict[str, list[str]] = {}
    for it in catalog:
        text = _full_text(it, lyrics_dict.get(it.item_id, ""), lyrics_cap=0)
        if not text.strip():
            raw[it.item_id] = []
            continue
        raw[it.item_id] = [kw.lower() for kw, _score in extractor.extract_keywords(text)]
    return raw


# ---------------------------------------------------------------------------
# keybert — per-song semantic keyphrases (MiniLM; SLOW on CPU, cached)
# ---------------------------------------------------------------------------

def _extract_keybert(catalog, lyrics_dict, top_n) -> dict[str, list[str]]:
    try:
        from keybert import KeyBERT
    except ImportError:
        raise ImportError("keybert not installed. Run: pip install keybert sentence-transformers")
    kw = KeyBERT("all-MiniLM-L6-v2")

    raw: dict[str, list[str]] = {}
    n = len(catalog)
    for i, it in enumerate(catalog):
        text = _full_text(it, lyrics_dict.get(it.item_id, ""), lyrics_cap=0)
        if not text.strip():
            raw[it.item_id] = []
            continue
        kws = kw.extract_keywords(text, keyphrase_ngram_range=(1, 2),
                                  stop_words="english", top_n=top_n)
        raw[it.item_id] = [p.lower() for p, _ in kws]
        if (i + 1) % 250 == 0:
            print(f"[extract:keybert] {i + 1}/{n}")
    return raw


# ---------------------------------------------------------------------------
# llm — concrete imagery / cultural motifs (OpenAI, cached per song)
# ---------------------------------------------------------------------------

_LLM_SYSTEM = (
    "You extract creative cues for a music generation system. "
    "Given a song's metadata and lyrics, list concrete imagery words, cultural "
    "allusions, and narrative motifs (e.g. 'train platform', 'neon rain', "
    "'broken phone'). Prefer concrete nouns and short noun phrases. "
    "Avoid generic words (love, baby, music) and verbs."
)


def _extract_llm(catalog, lyrics_dict, top_n) -> dict[str, list[str]]:
    import cue_clients

    raw: dict[str, list[str]] = {}
    n = len(catalog)
    for i, it in enumerate(catalog):
        prompt = _llm_prompt(it, lyrics_dict, top_n)
        resp = cue_clients.chat(prompt, system=_LLM_SYSTEM, temperature=0.2,
                                max_tokens=256)
        raw[it.item_id] = _parse_llm_cues(resp)
        if (i + 1) % 100 == 0:
            print(f"[extract:llm] {i + 1}/{n}")
    return raw


# ---------------------------------------------------------------------------
# llm via OpenAI Batch API — async extraction for large runs
# ---------------------------------------------------------------------------

def submit_llm_batch(
    catalog: list,
    lyrics_dict: Optional[dict[str, str]],
    top_n: int,
    force: bool = False,
    cache_tag: str = "",
) -> dict:
    """Submit LLM extraction as an OpenAI Batch job.

    Returns either:
      {"mode": "cached", "raw": ...}
      {"mode": "batch", "batch_id": ..., "request_ids": [...], ...}
    """
    cached = None if force else _read_cached_raw("llm", len(catalog), cache_tag)
    if cached is not None:
        return {"mode": "cached", "raw": cached}

    import cue_clients

    client = cue_clients._get_openai()
    lyrics_dict = lyrics_dict or {}
    out_dir = os.path.join(_OUTPUT_ROOT, "llm")
    os.makedirs(out_dir, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    input_path = os.path.join(out_dir, f"batch_input_{len(catalog)}_{stamp}.jsonl")
    request_ids: list[str] = []
    with open(input_path, "w", encoding="utf-8") as f:
        for it in catalog:
            custom_id = str(it.item_id)
            request_ids.append(custom_id)
            messages = [
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": _llm_prompt(it, lyrics_dict, top_n)},
            ]
            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": cue_clients.CHAT_MODEL,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 256,
                },
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    with open(input_path, "rb") as f:
        input_file = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    meta = {
        "mode": "batch",
        "batch_id": batch.id,
        "input_file_id": input_file.id,
        "input_path": input_path,
        "request_ids": request_ids,
        "n_items": len(catalog),
        "top_n": top_n,
        "cache_tag": cache_tag,
    }
    meta_path = os.path.join(out_dir, f"batch_meta_{len(catalog)}_{stamp}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    meta["meta_path"] = meta_path
    print(f"[extract:llm:batch] submitted batch {batch.id} ({len(catalog)} songs)")
    return meta


def _file_content_text(client, file_id: str) -> str:
    content = client.files.content(file_id)
    if hasattr(content, "read"):
        data = content.read()
    elif hasattr(content, "text"):
        return content.text
    else:
        data = content
    if isinstance(data, bytes):
        return data.decode("utf-8")
    return str(data)


def collect_llm_batch(
    job: dict,
    poll_seconds: int = 30,
    timeout_seconds: int = 24 * 60 * 60,
) -> dict[str, list[str]]:
    """Wait for an OpenAI Batch LLM extraction job and write raw_cues cache."""
    if job.get("mode") == "cached":
        return job["raw"]

    import cue_clients

    client = cue_clients._get_openai()
    batch_id = job["batch_id"]
    started = time.time()
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        print(f"[extract:llm:batch] {batch_id} status={status}")
        if status == "completed":
            break
        if status in {"failed", "expired", "cancelled", "cancelling"}:
            raise RuntimeError(f"OpenAI batch {batch_id} ended with status={status}: {batch}")
        if time.time() - started > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for OpenAI batch {batch_id}")
        time.sleep(poll_seconds)

    output_file_id = batch.output_file_id
    if not output_file_id:
        raise RuntimeError(f"OpenAI batch {batch_id} completed without output_file_id")

    text = _file_content_text(client, output_file_id)
    out_dir = os.path.join(_OUTPUT_ROOT, "llm")
    output_path = os.path.join(out_dir, f"batch_output_{job['n_items']}_{batch_id}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    raw: dict[str, list[str]] = {iid: [] for iid in job.get("request_ids", [])}
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        item_id = str(row.get("custom_id", ""))
        if row.get("error"):
            print(f"[extract:llm:batch] item {item_id} error: {row['error']}")
            continue
        try:
            body = row["response"]["body"]
            resp = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            print(f"[extract:llm:batch] item {item_id} malformed response: {exc}")
            continue
        raw[item_id] = _parse_llm_cues(resp)

    _write_raw("llm", int(job["n_items"]), raw, job.get("cache_tag", ""))
    print(f"[extract:llm:batch] downloaded output -> {output_path}")
    return raw
