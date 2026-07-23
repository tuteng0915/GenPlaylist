"""cue_eval.py — WP-B Phase 4: cue evaluation.

Level 1 — intrinsic (no LLM, runs on the full set):
    ROUGE between a song's assigned cues and its real lyrics. Are the cues
    grounded in the song's actual content?

Level 2 — semantic retrieval (no LLM):
    assigned cues --embed--> query, artist-free song texts --embed--> candidates
    Reports whether the source song appears in the top K / at what rank.

Level 3 — reconstruction ablation (the encode -> decode -> score loop):
    real lyrics --encode--> 6 cues --decode(LLM)--> regenerated lyrics
                            real lyrics <--BLEU/ROUGE/BERTScore-- regenerated
    Run with several decoder inputs (none / each method's cues / oracle) and
    compare: if cues carry information, scores rise above the metadata-only floor.

Heavy/optional deps (rouge-score, sacrebleu, bert-score, openai) are imported
lazily so this module loads without them.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_data_schema"))
from schema import CatalogItem, CueMappingEntry  # noqa: E402

import cue_clients  # noqa: E402
import cue_normalize  # noqa: E402


# ---------------------------------------------------------------------------
# Level 1 — intrinsic: cues vs real lyrics
# ---------------------------------------------------------------------------

def _get_rouge():
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError("rouge-score not installed. Run: pip install rouge-score")
    return rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def level1_intrinsic(
    item2cues: dict[str, CueMappingEntry],
    vocab: list[str],
    lyrics_dict: dict[str, str],
) -> dict:
    """Mean ROUGE-1/ROUGE-L recall of (assigned cue strings) against real lyrics.

    Only scores items that have lyrics. Cues are joined into a pseudo-document.
    """
    scorer = _get_rouge()
    r1, rl, n = 0.0, 0.0, 0
    for iid, entry in item2cues.items():
        lyrics = lyrics_dict.get(iid, "")
        if not lyrics.strip():
            continue
        cue_doc = " ".join(vocab[c] for c in entry.cue_ids if c != 0)
        if not cue_doc.strip():
            continue
        s = scorer.score(lyrics, cue_doc)   # (target=lyrics, prediction=cues)
        r1 += s["rouge1"].recall
        rl += s["rougeL"].recall
        n += 1
    if n == 0:
        return {"level1_rouge1_recall": 0.0, "level1_rougeL_recall": 0.0, "n_scored": 0}
    return {
        "level1_rouge1_recall": round(r1 / n, 4),
        "level1_rougeL_recall": round(rl / n, 4),
        "n_scored": n,
    }


def within_item_diversity(item2cues, cue_embeddings) -> dict:
    """Mean within-item pairwise cosine of each song's assigned cue embeddings.

    Lower = more diverse (paper target < 0.7). cue_embeddings[k] aligns with vocab
    index k+1 (vocab[0] is '<unk>'). Pad/zero-vector cues are skipped.
    """
    if cue_embeddings is None or len(cue_embeddings) == 0:
        return {"intra_cos_mean": None, "intra_n_scored": 0}
    emb = np.asarray(cue_embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    unit = emb / np.where(norms == 0, 1.0, norms)

    sims: list[float] = []
    for entry in item2cues.values():
        vecs = []
        for c in entry.cue_ids:
            k = c - 1
            if c != 0 and 0 <= k < len(emb) and norms[k, 0] > 0:
                vecs.append(unit[k])
        if len(vecs) < 2:
            continue
        V = np.stack(vecs)
        S = V @ V.T
        iu = np.triu_indices(len(vecs), k=1)   # upper triangle, exclude diagonal
        sims.append(float(S[iu].mean()))

    if not sims:
        return {"intra_cos_mean": None, "intra_n_scored": 0}
    return {"intra_cos_mean": round(float(np.mean(sims)), 4), "intra_n_scored": len(sims)}


# ---------------------------------------------------------------------------
# Level 2 — semantic retrieval: cues -> source song rank
# ---------------------------------------------------------------------------

def _cue_strings(entry: CueMappingEntry, vocab: list[str]) -> list[str]:
    cues: list[str] = []
    for cue_id in entry.cue_ids:
        if cue_id == 0 or cue_id >= len(vocab):
            continue
        cue = vocab[cue_id]
        if cue and not cue.startswith("<pad_"):
            cues.append(cue)
    return cues


def _retrieval_song_text(item: CatalogItem, lyrics: str, lyrics_cap: int = 0) -> str:
    """Artist-free target text for cue-to-song retrieval.

    lyrics arrive already preprocessed upstream (cue_lyrics); cap=0 = no re-truncation.
    """
    meta = " ".join(p for p in [
        item.title, item.genre, item.mood, item.lyric_excerpt, *item.tags
    ] if p)
    if lyrics:
        return f"{meta} {lyrics[:lyrics_cap] if lyrics_cap else lyrics}"
    return meta


def _make_retrieval_embedder():
    """Return (embed_fn, name, independent) for the retrieval metric.

    Deliberately uses a DIFFERENT encoder than assignment (cue_normalize's T5
    encoder) to avoid circularity: assignment selects cues by cosine in that
    space, so scoring retrieval there too would mark its own homework.
    Preference order:
      1. OpenAI text-embedding-3-small  (independent, if API key present)
      2. sentence-transformers all-mpnet-base-v2  (independent, local, different space)
      3. assignment's encoder as fallback  (SHARED — circularity NOT broken; flagged)
    """
    if cue_clients.is_available():
        try:
            def embed(strs):
                return cue_clients.embed(list(strs))
            embed(["_probe_"])  # surface auth/quota errors now, not mid-loop
            return embed, "openai/text-embedding-3-small", True
        except Exception as exc:
            print(f"[retrieval] OpenAI embedder unavailable ({type(exc).__name__}); trying local.")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-mpnet-base-v2")

        def embed(strs):
            v = model.encode(list(strs), normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(v, dtype=np.float32)
        return embed, "sentence-transformers/all-mpnet-base-v2", True
    except Exception:
        pass
    embed_fn, backend = cue_normalize._make_embedder()
    print("[retrieval] WARNING: falling back to the assignment encoder; "
          "retrieval shares that space with assignment (circularity NOT broken).")
    return embed_fn, backend + " (SHARED w/ assignment)", False


def level2_semantic_retrieval(
    catalog_by_id: dict[str, CatalogItem],
    item2cues: dict[str, CueMappingEntry],
    vocab: list[str],
    lyrics_dict: dict[str, str],
    sample_ids: Optional[list[str]] = None,
) -> dict:
    """Rank each source song from its 6 cue strings using embedding similarity.

    Query: assigned cue strings only.
    Targets: artist-free song text (title, genre, mood, lyric excerpt, lyrics).

    Uses an INDEPENDENT encoder (not assignment's T5 encoder) so the metric is not
    circular — see _make_retrieval_embedder. Measures whether the cue bottleneck
    preserves enough semantic identity to retrieve the correct song.
    """
    candidate_ids = [
        iid for iid in (sample_ids or list(catalog_by_id))
        if iid in catalog_by_id and lyrics_dict.get(iid, "").strip()
    ]
    query_ids: list[str] = []
    query_docs: list[str] = []
    for iid in candidate_ids:
        entry = item2cues.get(iid)
        if not entry:
            continue
        cues = _cue_strings(entry, vocab)
        if not cues:
            continue
        query_ids.append(iid)
        query_docs.append(", ".join(cues))

    if not candidate_ids or not query_ids:
        return {
            "retrieval_n": 0,
            "retrieval_r1": 0.0,
            "retrieval_r5": 0.0,
            "retrieval_r10": 0.0,
            "retrieval_mrr": 0.0,
            "retrieval_mean_rank": 0.0,
            "retrieval_median_rank": 0.0,
            "retrieval_encoder": "n/a",
            "retrieval_independent": False,
        }

    target_docs = [
        _retrieval_song_text(catalog_by_id[iid], lyrics_dict.get(iid, ""))
        for iid in candidate_ids
    ]

    # Independent encoder (different from assignment's T5 encoder) to avoid circularity.
    embed_fn, encoder_name, independent = _make_retrieval_embedder()
    all_emb = embed_fn(query_docs + target_docs)
    query_emb = all_emb[:len(query_docs)]
    target_emb = all_emb[len(query_docs):]

    ranks: list[int] = []
    target_pos = {iid: i for i, iid in enumerate(candidate_ids)}
    sims = query_emb @ target_emb.T
    for row, iid in enumerate(query_ids):
        true_pos = target_pos[iid]
        order = np.argsort(sims[row])[::-1]
        rank = int(np.where(order == true_pos)[0][0]) + 1
        ranks.append(rank)

    ranks_arr = np.asarray(ranks, dtype=np.float32)
    return {
        "retrieval_n": int(len(ranks)),
        "retrieval_r1": round(float(np.mean(ranks_arr <= 1)), 4),
        "retrieval_r5": round(float(np.mean(ranks_arr <= 5)), 4),
        "retrieval_r10": round(float(np.mean(ranks_arr <= 10)), 4),
        "retrieval_mrr": round(float(np.mean(1.0 / ranks_arr)), 4),
        "retrieval_mean_rank": round(float(np.mean(ranks_arr)), 2),
        "retrieval_median_rank": round(float(np.median(ranks_arr)), 2),
        "retrieval_encoder": encoder_name,
        "retrieval_independent": independent,
    }


# Backward-compatible alias for older run scripts.
level3_semantic_retrieval = level2_semantic_retrieval


# ---------------------------------------------------------------------------
# Level 3 — reconstruction: cues -> LLM -> lyrics, scored vs real lyrics
# ---------------------------------------------------------------------------

_DECODE_SYSTEM = (
    "You reconstruct a song's full lyrics from a few creative cues and minimal "
    "metadata. Develop the cues into complete lyrics of roughly the requested "
    "length, with the emotional arc and concrete imagery a real song of this style "
    "would have. Write full verses and a repeated chorus. Output ONLY lyric lines "
    "— no title, no section labels, no commentary."
)
_DECODE_TEMP = 0.4
_TARGET_MIN_LINES = 12
DEFAULT_MAX_TARGET_LINES = 48   # cost guard; 0 = no clamp (match real song length)

# Cross-cutting setting: run_compare lifts the clamp for --lyrics-mode full.
_max_target_lines = DEFAULT_MAX_TARGET_LINES


def set_max_target_lines(n) -> None:
    """Set the upper clamp on decode target length. 0/None = unclamped."""
    global _max_target_lines
    _max_target_lines = n or 0


def _target_lines(ref_lyrics: str) -> int:
    """Line budget for the generation, sized to the real lyric.

    Lower bound _TARGET_MIN_LINES; upper bound _max_target_lines (0 = no clamp).
    """
    n = sum(1 for ln in ref_lyrics.splitlines() if ln.strip())
    n = max(_TARGET_MIN_LINES, n)
    if _max_target_lines:
        n = min(n, _max_target_lines)
    return n


def _decode_maxtok(target_lines: int) -> int:
    """Token budget scaled to the target line count (~22 tokens/line + slack)."""
    return max(300, min(2000, target_lines * 22 + 120))


def _decode_prompt(item: CatalogItem, cue_strings: list[str], target_lines: int = 24) -> str:
    """Build the decode prompt. Shared by sync + batch.

    Title and artist are intentionally WITHHELD so the decoder cannot recall the
    specific real song from memory — the cues (and genre/mood) are the only
    song-specific signal, which is what the ablation measures.
    """
    meta = " | ".join(p for p in [item.genre, item.mood] if p) or "a song"
    cue_part = ", ".join(cue_strings) if cue_strings else "(none)"
    return (
        f"Style: {meta}\n"
        f"Creative cues: {cue_part}\n"
        f"Target length: about {target_lines} lines (write full verses and a chorus).\n\n"
        f"Write the complete song lyric now:"
    )


def _decode_lyrics(item: CatalogItem, cue_strings: list[str], target_lines: int = 24) -> str:
    """Decode cues (+ metadata) back into a lyric draft via the LLM client (cached)."""
    return cue_clients.chat(_decode_prompt(item, cue_strings, target_lines),
                            system=_DECODE_SYSTEM, temperature=_DECODE_TEMP,
                            max_tokens=_decode_maxtok(target_lines))


_STS_MODEL = None  # cached free sentence encoder for document-level cosine


def _sts_cosine(preds: list[str], refs: list[str]):
    """Mean whole-document cosine similarity via a free local sentence encoder.

    Embeds each generated and real lyric as one vector and averages the paired
    cosine. Length-robust (unlike BERTScore/BLEU) since each text is a single
    vector. Returns (mean_cosine, encoder_name) or (None, None) if unavailable.
    """
    global _STS_MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None, None
    name = "all-mpnet-base-v2"
    if _STS_MODEL is None:
        _STS_MODEL = SentenceTransformer(name)
    pe = _STS_MODEL.encode(preds, normalize_embeddings=True, show_progress_bar=False)
    re_ = _STS_MODEL.encode(refs, normalize_embeddings=True, show_progress_bar=False)
    cos = np.sum(np.asarray(pe) * np.asarray(re_), axis=1)  # paired dot = cosine
    return round(float(np.mean(cos)), 4), name


# Common scoring window (chars) applied to BOTH generated and real lyrics before
# every reconstruction metric, so all metrics compare the same-length text. 0 = off.
_SCORE_CHARS = 2000


def set_score_chars(n) -> None:
    """Set the common scoring window (chars) applied to both gen and real lyrics."""
    global _SCORE_CHARS
    _SCORE_CHARS = n or 0


def _score_reconstruction(preds: list[str], refs: list[str]) -> dict:
    """ROUGE-1/2/L + BLEU + BERTScore + sentence-embedding cosine (gen vs real).

    Both generated and real lyrics are truncated to the same _SCORE_CHARS window
    first, so every metric compares equal-length text (no long-reference penalty).
    """
    if _SCORE_CHARS:
        preds = [p[:_SCORE_CHARS] for p in preds]
        refs = [r[:_SCORE_CHARS] for r in refs]

    from rouge_score import rouge_scorer
    import sacrebleu
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    n = len(preds)
    if n == 0:
        return {"n_scored": 0}
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for gen, ref in zip(preds, refs):
        s = scorer.score(ref, gen)
        for k in agg:
            agg[k] += s[k].fmeasure

    out = {
        "n_scored": n,
        "rouge1_f": round(agg["rouge1"] / n, 4),
        "rouge2_f": round(agg["rouge2"] / n, 4),
        "rougeL_f": round(agg["rougeL"] / n, 4),
        "bleu": round(sacrebleu.corpus_bleu(preds, [refs]).score, 4),
    }
    try:
        from bert_score import score as bert_score
        _, _, f1 = bert_score(preds, refs, lang="en", verbose=False, rescale_with_baseline=True)
        out["bertscore_f1"] = round(float(f1.mean()), 4)
    except ImportError:
        out["bertscore_f1"] = None

    sts, sts_encoder = _sts_cosine(preds, refs)
    out["sts_cosine"] = sts
    out["sts_encoder"] = sts_encoder
    return out


def _decode_cue_strings(item2cues, vocab, iid) -> list[str]:
    entry = item2cues.get(iid)
    return [vocab[c] for c in entry.cue_ids if c != 0] if entry else []


# ---------------------------------------------------------------------------
# Cue-source builders — each returns {item_id: [cue strings]} for one ablation
# condition. The reconstruction code path is identical for every condition;
# only the cue content differs (random floor < methods < oracle ceiling).
# ---------------------------------------------------------------------------

def assigned_cues_by_id(item2cues, vocab, sample_ids) -> dict[str, list[str]]:
    """A method's actual assigned cues per song."""
    return {iid: _decode_cue_strings(item2cues, vocab, iid) for iid in sample_ids}


def random_cues_by_id(vocab, sample_ids, n_cues: int = 6, seed: int = 0) -> dict[str, list[str]]:
    """Random floor: 6 random real vocab cues per song (structure held constant)."""
    import random as _random
    real = [c for c in vocab if c != "<unk>" and not c.startswith("<pad_")]
    rng = _random.Random(seed)
    k = min(n_cues, len(real))
    return {iid: rng.sample(real, k) for iid in sample_ids}


def no_cues_by_id(sample_ids) -> dict[str, list[str]]:
    """Metadata-only floor: zero cues. _decode_prompt renders an empty cue list as
    "(none)", so the decoder receives nothing but genre/mood (title/artist are
    already withheld from every condition, see _decode_prompt). This isolates what
    cues of ANY kind — even random ones — add over metadata alone, which the
    `random` floor (wrong-but-present cues) doesn't measure on its own.
    """
    return {iid: [] for iid in sample_ids}


_ORACLE_SYSTEM = (
    "You extract creative cues from song lyrics for a music generation system. "
    "List concrete imagery words, cultural allusions, and narrative motifs actually "
    "present in the lyrics. Prefer concrete nouns and short noun phrases."
)


def oracle_cues_by_id(catalog_by_id, lyrics_dict, sample_ids, n_cues: int = 6) -> dict[str, list[str]]:
    """Oracle ceiling: 6 cues an LLM extracts from each song's REAL lyrics (cached)."""
    out: dict[str, list[str]] = {}
    for iid in sample_ids:
        ref = lyrics_dict.get(iid, "")
        if not ref.strip():
            continue
        prompt = (
            f"Extract exactly {n_cues} creative cues from these lyrics as a "
            f"comma-separated list, lowercase, no numbering:\n\n{ref[:2000]}"
        )
        resp = cue_clients.chat(prompt, system=_ORACLE_SYSTEM, temperature=0.2, max_tokens=120)
        cues = [c.strip().lower() for c in resp.replace("\n", ",").split(",") if c.strip()]
        out[iid] = cues[:n_cues]
    return out


def level3_reconstruction(
    catalog_by_id: dict[str, CatalogItem],
    cues_by_id: dict[str, list[str]],
    lyrics_dict: dict[str, str],
    sample_ids: list[str],
) -> dict:
    """Decode lyrics for sampled songs from the given per-song cues, score vs real.

    cues_by_id supplies each song's cue strings for this condition (random /
    assigned / oracle). Decodes via cue_clients.chat (cached), so a prior batch
    run that populated the cache makes this an API-free scoring pass.
    """
    preds: list[str] = []
    refs: list[str] = []
    for iid in sample_ids:
        item = catalog_by_id.get(iid)
        ref = lyrics_dict.get(iid, "")
        if item is None or not ref.strip():
            continue
        cue_strings = cues_by_id.get(iid, [])
        preds.append(_decode_lyrics(item, cue_strings, _target_lines(ref)))
        refs.append(ref)
    return _score_reconstruction(preds, refs)


def generate_lyrics_for_sample(catalog_by_id, cues_by_id, lyrics_dict, sample_ids):
    """Return [(item_id, cue_strings, generated_lyric)] for sampled songs (cached)."""
    out = []
    for iid in sample_ids:
        item = catalog_by_id.get(iid)
        ref = lyrics_dict.get(iid, "")
        if item is None or not ref.strip():
            continue
        cue_strings = cues_by_id.get(iid, [])
        gen = _decode_lyrics(item, cue_strings, _target_lines(ref))
        out.append((iid, cue_strings, gen))
    return out


# ---------------------------------------------------------------------------
# Level 3 via OpenAI Batch API — async reconstruction for large eval samples
# ---------------------------------------------------------------------------

def _recon_requests(catalog_by_id, cues_by_id, lyrics_dict, sample_ids):
    """Build [(item_id, prompt, max_tokens)] for sampled songs that have lyrics.

    max_tokens is per-song (scaled to target length) and MUST match the sync path
    so batch-populated cache entries are read back by _decode_lyrics.
    """
    reqs = []
    for iid in sample_ids:
        item = catalog_by_id.get(iid)
        ref = lyrics_dict.get(iid, "")
        if item is None or not ref.strip():
            continue
        cue_strings = cues_by_id.get(iid, [])
        tl = _target_lines(ref)
        reqs.append((iid, _decode_prompt(item, cue_strings, tl), _decode_maxtok(tl)))
    return reqs


def submit_reconstruction_batch(catalog_by_id, cues_by_id, lyrics_dict,
                                sample_ids, tag):
    """Submit one reconstruction condition as an OpenAI Batch job.

    Only prompts not already in the chat cache are sent. Returns an in-memory job
    dict carrying everything collect_reconstruction_batch needs to score.
    """
    import json as _json
    import time as _time

    reqs = _recon_requests(catalog_by_id, cues_by_id, lyrics_dict, sample_ids)
    prompts = {iid: p for iid, p, _mt in reqs}
    maxtoks = {iid: mt for iid, _p, mt in reqs}
    uncached = [iid for iid, p, mt in reqs
                if not cue_clients.is_chat_cached(p, _DECODE_SYSTEM, _DECODE_TEMP, mt)]

    score_args = (catalog_by_id, cues_by_id, lyrics_dict, sample_ids)

    if not uncached:
        print(f"[recon:batch:{tag}] all {len(reqs)} prompts cached; no batch needed")
        return {"mode": "cached", "tag": tag, "score_args": score_args}

    client = cue_clients._get_openai()
    out_dir = os.path.join(os.path.dirname(__file__), "outputs", "_cache", "recon")
    os.makedirs(out_dir, exist_ok=True)
    stamp = _time.strftime("%Y%m%d_%H%M%S")
    input_path = os.path.join(out_dir, f"batch_input_{tag}_{stamp}.jsonl")
    with open(input_path, "w", encoding="utf-8") as f:
        for iid in uncached:
            line = {
                "custom_id": str(iid),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": cue_clients.CHAT_MODEL,
                    "messages": [
                        {"role": "system", "content": _DECODE_SYSTEM},
                        {"role": "user", "content": prompts[iid]},
                    ],
                    "temperature": _DECODE_TEMP,
                    "max_tokens": maxtoks[iid],
                },
            }
            f.write(_json.dumps(line, ensure_ascii=False) + "\n")

    with open(input_path, "rb") as f:
        input_file = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=input_file.id, endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"[recon:batch:{tag}] submitted {batch.id} ({len(uncached)} uncached / {len(reqs)} total)")
    return {
        "mode": "batch", "tag": tag, "batch_id": batch.id,
        "prompts": prompts, "maxtoks": maxtoks, "score_args": score_args,
    }


def collect_reconstruction_batch(job, poll_seconds=30, timeout_seconds=24 * 60 * 60):
    """Wait for a reconstruction batch, populate the chat cache, then score."""
    import time as _time
    import json as _json

    if job.get("mode") == "batch":
        client = cue_clients._get_openai()
        batch_id = job["batch_id"]
        started = _time.time()
        while True:
            batch = client.batches.retrieve(batch_id)
            print(f"[recon:batch:{job['tag']}] {batch_id} status={batch.status}")
            if batch.status == "completed":
                break
            if batch.status in {"failed", "expired", "cancelled", "cancelling"}:
                raise RuntimeError(f"recon batch {batch_id} status={batch.status}")
            if _time.time() - started > timeout_seconds:
                raise TimeoutError(f"recon batch {batch_id} timed out")
            _time.sleep(poll_seconds)

        import cue_extractors
        text = cue_extractors._file_content_text(client, batch.output_file_id)
        prompts = job["prompts"]
        maxtoks = job.get("maxtoks", {})
        for line in text.splitlines():
            if not line.strip():
                continue
            row = _json.loads(line)
            iid = str(row.get("custom_id", ""))
            if row.get("error") or iid not in prompts:
                continue
            try:
                resp = row["response"]["body"]["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                continue
            # write into the same cache key the sync decode path reads (per-song max_tokens)
            cue_clients.cache_chat_response(
                prompts[iid], resp, system=_DECODE_SYSTEM,
                temperature=_DECODE_TEMP, max_tokens=maxtoks.get(iid, 900))

    # score: decodes now all hit the cache (no API calls)
    catalog_by_id, cues_by_id, lyrics_dict, sample_ids = job["score_args"]
    return level3_reconstruction(catalog_by_id, cues_by_id, lyrics_dict, sample_ids)


# Backward-compatible alias for older run scripts.
level2_reconstruction = level3_reconstruction
