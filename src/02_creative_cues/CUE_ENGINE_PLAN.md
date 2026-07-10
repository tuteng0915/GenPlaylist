# WP-B: Better Cue Extraction Engine — Build Plan

Goal: replace the raw TF-IDF baseline with a faithful implementation of the
paper's 4-step cue pipeline (§4.2), make the extraction front-end swappable
(TF-IDF / KeyBERT / YAKE / LLM), add an encode→decode→score evaluation loop,
and produce a single report comparing all approaches.

Keep `cue_mining.py` as the strict-contract module (schema-compliant outputs).
All new work lands in sibling files so the contract module stays clean.

---

## Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │  catalog + lyrics (data/dataset, data/lyrics) │
                         └───────────────────────┬─────────────────────┘
                                                 │
                 ┌───────────────────────────────┼───────────────────────────────┐
                 │            EXTRACTION FRONT-ENDS (pluggable)                    │
                 │   tfidf      keybert      yake      llm(Qwen3)                  │
                 │   → each emits RAW candidate cues per song                      │
                 └───────────────────────────────┬───────────────────────────────┘
                                                 │   (raw cues, extractor-agnostic from here)
                 ┌───────────────────────────────▼───────────────────────────────┐
                 │            NORMALIZATION PIPELINE  (paper step 3)               │
                 │   1. df-band filter      5 ≤ df ≤ 0.3·N                         │
                 │   2. POS filter          keep noun / adj+noun phrases           │
                 │   3. embed cues          sentence-transformers MiniLM           │
                 │   4. semantic dedup      merge pairs cosine > 0.92              │
                 │   5. rank by IDF, keep top-2047, prepend "<unk>" → 2048         │
                 └───────────────────────────────┬───────────────────────────────┘
                                                 │   cue_vocab.json
                 ┌───────────────────────────────▼───────────────────────────────┐
                 │            ASSIGNMENT  (paper step 4)                            │
                 │   PMI score per (song, cue) + SEMANTIC diversity regularizer    │
                 │   → 6 cues per song, pad with <unk>                             │
                 └───────────────────────────────┬───────────────────────────────┘
                                                 │   item2cues.json (per method)
                 ┌───────────────────────────────▼───────────────────────────────┐
                 │            EVALUATION                                            │
                 │   Level 1 (intrinsic):  ROUGE(cues, real lyrics)  — no LLM      │
                 │   Level 2 (retrieval): cues → source-song rank                  │
                 │   Level 3 (recon):  cues → LLM decode → lyrics                  │
                 │                     BLEU/ROUGE/BERTScore vs real lyrics         │
                 │                     ablation: none < tfidf < keybert < oracle   │
                 └───────────────────────────────┬───────────────────────────────┘
                                                 │
                 ┌───────────────────────────────▼───────────────────────────────┐
                 │   REPORT: comparison table across all methods + samples         │
                 └─────────────────────────────────────────────────────────────────┘
```

---

## File layout

| File | Role | Status |
|------|------|--------|
| `cue_mining.py` | strict-contract module (vocab/assign/export/stats) | exists — keep |
| `cue_clients.py` | shared OpenAI LLM + embedding clients, disk-memoized | NEW |
| `cue_extractors.py` | the 4 swappable raw-cue front-ends (cached) | NEW |
| `cue_normalize.py` | df-band + POS + embed + dedup + IDF (paper step 3) | ✅ built |
| `cue_eval.py` | decoder (cues→lyrics) + ROUGE/BLEU/BERTScore | NEW |
| `run_compare.py` | orchestrates all methods end-to-end, writes report | NEW |
| `outputs/<method>/` | per-method cue_vocab.json + item2cues.json | NEW subdirs |
| `outputs/comparison_report.md` | the final cross-method comparison | NEW |

---

## Phase 1 — Normalization pipeline  (`cue_normalize.py`)

The highest-value change. Extractor-agnostic; runs on whatever raw cues come in.

Functions:
- `df_band_filter(cue_to_docs, N, min_df=5, max_df_frac=0.3)`
  drop cues with `df < min_df` or `df > max_df_frac * N`.
- `pos_filter(cues)` keep only noun phrases / ADJ+NOUN; drop verb-y fragments
  (`need need`, `wanna make`). Uses spaCy `en_core_web_sm` (or NLTK POS as fallback).
- `embed_cues(cues)` → np.ndarray via `sentence-transformers/all-MiniLM-L6-v2` (CPU OK).
- `semantic_dedup(cues, embeddings, threshold=0.92)` greedily merge near-duplicates,
  keep the higher-IDF representative of each cluster.
- `rank_by_idf(cues, df, N)` → sorted; caller takes top-2047, prepends `"<unk>"`.

Output: a 2048-entry vocab + the cue→embedding map (reused in assignment + eval).

Deps: `pip install sentence-transformers spacy && python -m spacy download en_core_web_sm`

---

## Phase 2 — Swappable extraction front-ends  (`cue_extractors.py`)

Uniform interface so the normalizer/assigner don't care which was used:

```python
def extract_raw_cues(catalog, lyrics_dict, method: str) -> dict[str, list[str]]:
    """item_id -> list of raw candidate cue phrases."""
```

| method | how raw cues are produced | deps |
|--------|---------------------------|------|
| `tfidf`   | top TF-IDF terms per song (current baseline) | scikit-learn |
| `keybert` | KeyBERT keyphrases (semantic, imagery-leaning) | keybert |
| `yake`    | YAKE statistical keyphrases | yake |
| `llm`     | LLM prompt: "concrete imagery, cultural allusions, narrative motifs" | swappable client |

`llm` goes through a `_llm_client(prompt)` stub. Default backend = **OpenAI
`gpt-4o-mini`** (via `OPENAI_API_KEY`); a local Ollama/Qwen3 backend can be swapped
in. The module imports & runs without a key — `llm` raises a clear error only when
actually invoked without one.

All four feed the SAME Phase-1 normalizer, so differences in the final vocab are
purely down to extraction quality.

---

## Phase 3 — Semantic-diversity assignment  (extend in `cue_mining.py`)

Paper step 4: 6 cues per song by PMI + diversity reg on *pairwise semantic distance*.
Current `_assign_pmi` penalizes token overlap only — upgrade to use the cue
embeddings from Phase 1 (MMR with cosine distance) so the 6 cues are semantically
spread, not just lexically distinct. Falls back to `<unk>` padding (contract intact).

---

## Phase 4 — Decoder + evaluation  (`cue_eval.py`)

### Level 1 — intrinsic (cheap, no LLM, run on full set)
For each song: `ROUGE-1 / ROUGE-L recall(assigned_cues, real_lyrics)`.
Fast sanity metric: are the cues grounded in the song's actual content?

### Level 2 — cue-to-song semantic retrieval
For each song, embed its assigned cues as a query, embed artist-free song texts as
retrieval targets, and rank the true source song among the eval sample. Metrics:
R@1/R@5/R@10, MRR, mean rank, and median rank.

### Level 3 — reconstruction ablation (the mentor's encode→decode→score)
On a sampled eval set (200–300 songs, to bound LLM cost):

```
real lyrics ─encode→ 6 cues ─decode(LLM)→ regenerated lyrics ─score→ vs real lyrics
```

- `decode(cues, metadata)` = fixed LLM prompt → lyric draft (swappable client, same stub).
- Metrics: **ROUGE-1 / ROUGE-2 / ROUGE-L** (lead), **BLEU** (secondary),
  **BERTScore** (semantic, handles paraphrase — lyrics get reworded).
- **Ablation conditions** (the comparison that matters):

  | Condition | Decoder input | Role |
  |-----------|---------------|------|
  | `none`   | metadata only | floor baseline |
  | `tfidf`  | metadata + TF-IDF cues | |
  | `keybert`| metadata + KeyBERT cues | |
  | `yake`   | metadata + YAKE cues | |
  | `llm`    | metadata + LLM cues | |
  | `oracle` | 6 cues LLM-extracted from the REAL lyrics | ceiling |

  Target story: `none < tfidf < keybert ≤ llm < oracle`.

Deps: `pip install rouge-score sacrebleu bert-score`

---

## Phase 5 — Comparison report  (`run_compare.py` → `outputs/comparison_report.md`)

Single orchestrator runs every method through extract → normalize → assign → eval,
writes per-method outputs, and emits ONE report containing:

1. **Headline table** — one row per method:

   | Method | Coverage | Vocab util | Entropy | ROUGE-1(L1) | Retrieval R@5(L2) | MRR(L2) | ROUGE-L(L3) |
   |--------|----------|-----------|---------|-------------|-------------|----------|---------------|

2. **Qualitative samples** — same 15 songs across all methods, cues side-by-side,
   for manual interpretability inspection (paper's required eval).
3. **Ablation delta** — Level-3 scores vs the `none` floor and `oracle` ceiling.
4. **Top-10 cues** per method (spot lyric-fragment leakage).

CLI:
```
python run_compare.py --limit 1000 --methods tfidf,yake,keybert,llm \
                      --eval-sample 200 [--force]
```

---

## Dependencies (full — compare all methods)

```
pip install scikit-learn numpy yake nltk rouge-score sacrebleu openai
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers keybert bert-score spacy
python -m nltk.downloader averaged_perceptron_tagger punkt
python -m spacy download en_core_web_sm
```

| Library | Role | Install weight |
|---------|------|----------------|
| scikit-learn, numpy | TF-IDF, char-ngram fallback | light (installed) |
| yake | YAKE extractor | light (installed) |
| nltk / spacy | POS filter (one suffices; spaCy preferred) | light |
| rouge-score, sacrebleu | ROUGE + BLEU metrics | light |
| openai | `llm` extractor, L3 decoder, optional embeddings | light |
| torch (CPU wheel) | backs sentence-transformers/keybert/bert-score | heavy download, installs cleanly |
| sentence-transformers | MiniLM dedup embeddings + semantic-diversity PMI | model ~90MB |
| keybert | KeyBERT extractor | uses MiniLM |
| bert-score | semantic eval metric (paraphrase-tolerant) | model ~500MB |

Set the key once per session: `export OPENAI_API_KEY=...`
(PowerShell: `$env:OPENAI_API_KEY="sk-..."`).

---

## Runtime cost on Windows CPU  (5,119 songs; install cost excluded)

| Stage | Method | Runtime | Mitigation |
|-------|--------|---------|------------|
| extract | tfidf | seconds | — |
| extract | yake | ~1-2 min | — |
| extract | **keybert** | **~15-40 min (heaviest)** | cache raw cues; dev on `--limit 1000` |
| extract | llm (OpenAI) | ~10-20 min, network-bound | cache; subset of songs; cents on gpt-4o-mini |
| normalize | df-band + POS (spaCy) | ~2-4 min | one-time per method |
| normalize | semantic dedup (MiniLM) | <1 min | only ~2K cues embedded, not all songs |
| assign | PMI + semantic diversity | ~2-5 min | — |
| eval L1 | ROUGE(cues, lyrics) | seconds | full set OK |
| eval L2 | retrieval | seconds to minutes | sample-bounded, no LLM |
| eval L3 | decode (OpenAI) | network-bound | runs on ~200-song sample only |
| eval L3 | **BERTScore** | **slow on CPU** | only scores the ~200-song sample |

The only genuinely slow-at-runtime steps are **KeyBERT extraction** and **BERTScore**.
Both are bounded by caching + sampling (see below), so a full comparison run is
minutes-to-tens-of-minutes, not hours.

---

## Caching strategy (so expensive stages run once)

Every costly artifact is written to `outputs/<method>/` and reused if present:

```
outputs/<method>/raw_cues.json       # extractor output (skip re-extraction)
outputs/<method>/cue_vocab.json      # normalized vocab
outputs/<method>/cue_embeddings.npy  # vocab embeddings (reused by assign + dedup)
outputs/<method>/item2cues.json      # assignment
outputs/_cache/llm/<hash>.json       # per-prompt LLM responses (extract + decode)
outputs/_cache/embeddings/<hash>.npy # embedding cache
```

- LLM + embedding calls are memoized by prompt/text hash → re-runs cost nothing.
- `--force` flag re-computes a stage; default reuses cache.
- Dev loop: `--limit 1000` for fast iteration; final report run on full 5,119.

---

## Build order (each step independently runnable)

1. **Phase 1** ✅ built (`cue_normalize.py`) — now wire real spaCy POS + MiniLM embedder.
2. **Phase 2** four extractor front-ends (`cue_extractors.py`) with disk caching +
   shared LLM/embedding client (`cue_clients.py`, OpenAI-backed, memoized).
3. **Phase 3** semantic-diversity assignment (extend `cue_mining.py`).
4. **Level 1** ROUGE eval → first cross-method number (no LLM).
5. **Phase 4 Level 2** semantic retrieval.
6. **Phase 4 Level 3** decoder + reconstruction ablation (OpenAI).
7. **Phase 5** `run_compare.py` orchestrator → `comparison_report.md`.

Contract guarantee: `cue_mining.py` outputs (`cue_vocab.json`, `item2cues.json`,
`cue_report.md`) stay schema-valid (`CueMappingEntry.load_mapping()`); the new
multi-method machinery writes to `outputs/<method>/` subdirs and the comparison report.
```
