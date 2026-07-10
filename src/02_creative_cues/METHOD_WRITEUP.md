# WP-B: Creative Cue Mining - Method Writeup

Builds a 2,048-entry creative-cue vocabulary and assigns 6 cues per song, then
evaluates cue quality. The implementation follows the paper's Section 4.2 cue
pipeline, with a swappable extraction front-end so multiple methods can be
compared.

```text
catalog + lyrics
   |
   v  [0] PREPROCESS LYRICS (cap / dedup / full / summarize)  -> song text for extraction
   v  [1] EXTRACT (tfidf / yake / keybert / llm)              -> raw candidate cues per song
   v  [2] CLEAN (shared normalization + filters)              -> 2,048-entry vocabulary
   v  [3] ASSIGN (semantic relevance + diversity)             -> 6 cues per song
   v  [4] EVALUATE (grounding + retrieval + reconstruction)   -> comparison report
```

A song's final representation is `[BOI, z1, z2, z3, z_conf, c1..c6]`, where
`c1..c6` are the cue IDs this package produces.

The runner keeps two lyric views throughout: **processed** lyrics (shaped by the
lyrics mode) feed extraction, assignment, and retrieval; the **original** lyrics
are always used as the reference for grounding and reconstruction scoring.

---

## Step 0 - Lyrics preprocessing

`--lyrics-mode` shapes the lyric text fed into extraction / assignment / retrieval.
The same mode is applied to every method so the comparison stays fair, and the
cache is keyed by mode so switching never reuses stale cues.

| Mode | Behaviour |
|------|-----------|
| `cap` (default) | first `--lyrics-cap` characters (positional truncation) |
| `full` | entire lyric, no truncation (decode target length is also unclamped) |
| `dedup` | drop repeated lines (choruses), keep distinct content, then cap |
| `summarize` | LLM compresses lyrics to a short summary |

`dedup` is the recommended default: it removes chorus repetition (which inflates
TF-IDF and blurs embeddings) with no LLM and no hallucination. `summarize` is
**not recommended for the method comparison** — inserting an LLM summary upstream
of every extractor contaminates the comparison (all methods then sit on the same
LLM digest) and its prompt primes toward the cue target; treat it only as a
separate experiment.

The original (unprocessed) lyrics are retained separately as the scoring reference.

---

## Step 1 - Extraction

Each method turns artist-free song text into a pool of raw candidate cue
phrases:

```text
title + genre + mood + tags + lyric excerpt + processed-lyrics
```

Artist names are intentionally excluded from extraction and assignment text so
cues describe song-level content rather than performer identity. Artist names
are still used for display and for artist-token blocking during cleaning.
(Note: performer names can still leak via `(feat. X)` in titles or via corrupted
lyric files that contain track-listings — see Known limitations.)

| Method | How candidates are produced |
|--------|------------------------------|
| `tfidf` | Corpus-wide TF-IDF; take each song's top terms by weight, using unigrams and bigrams. |
| `yake` | YAKE statistical keyphrase extraction per song. |
| `keybert` | MiniLM embeds song text and selects phrases semantically close to it. |
| `llm` | LLM prompted for concrete imagery, cultural allusions, and narrative motifs. |

Each extractor keeps `top_n` raw candidates per song before cleaning. The
comparison runner defaults to `top_n=40`.

For large LLM runs, `run_compare.py --llm-batch` submits the LLM extraction as an
OpenAI Batch job before running the other selected extraction methods, then
collects the LLM raw cues last and feeds them into the same cleaning,
assignment, and evaluation stages.

All methods emit the same shape:

```text
{item_id: [phrase, ...]}
```

Then all methods feed the same cleaning pipeline, so downstream comparison is
primarily about extraction quality.

---

## Step 2 - Cleaning To A Controlled Vocabulary

The pooled candidates from all songs pass through shared normalization and
filters. `N` is the number of songs and `df(c)` is the number of songs containing
cue `c`.

| # | Filter | Rule | Removes |
|---|--------|------|---------|
| 0 | Normalize + token-identity dedup | Lowercase, normalize punctuation/apostrophes, collapse colloquial variants (`talkin` -> `talking`); tokenize each cue with the **T5 tokenizer** and dedup on token-id equality within each song. | Spelling variants, repeated phrases. |
| 1 | df-band | Keep `min_df <= df(c) <= 0.3 * N`. `min_df` scales with corpus size (5 on the full catalog, lower on small dev samples). | One-off noise and corpus-dominant terms. |
| 2 | POS | Keep noun-ish / adj+noun phrases; drop verb-led fragments. Uses spaCy, then NLTK, then heuristic fallback. | Verb fragments such as `wanna make`, `know let`. |
| 3 | Blocklist + fragment filter | Drop boilerplate tokens, artist-only phrases, exact junk phrases, and multi-word phrases starting/ending with fragment tokens. | `artist`, `album`, `title`, `bpm key`, `love don`, `ain got`, `baby girl`, `rock intense`. |
| 4 | Semantic dedup | Embed cues with the **sentence encoder (MiniLM)** and greedily merge near-duplicates with cosine > 0.92. | Near-duplicates such as `running` / `runnin`. |
| 5 | IDF rank + cut | Rank survivors by IDF and keep top 2,047, then prepend `<unk>`. | Low-discriminative leftovers beyond the fixed vocab budget. |

Then `<unk>` is prepended at index 0, producing exactly 2,048 entries. If fewer
than 2,047 real cues survive, remaining slots are `<pad_*>` placeholders and are
not assigned to songs.

**Encoders used here matter.** Two different roles:
- **T5 tokenizer** (step 0) is used *only* to canonicalize/dedup cues by token
  identity — a text-normalization step, not a similarity metric.
- **Sentence encoder — `all-MiniLM-L6-v2`** (step 4, and Step 3 assignment) produces
  the semantic embeddings whose cosine reflects meaning. This is the paper's
  "multilingual sentence encoder" role. The raw T5 *encoder* is deliberately NOT
  used for embeddings: it is not trained for sentence similarity, so its cosines are
  uncalibrated and collapse cue-to-song relevance (which previously caused rare cues
  to be sprayed across unrelated songs).

---

## Step 3 - Assignment

Assignment selects exactly 6 cues per song from the normalized vocabulary.

- **Relevance:** cosine similarity between the embedded song text and cue
  embedding, both from the MiniLM sentence encoder.
- **Selection:** greedy MMR over the top relevant candidates:

```text
argmax_c lambda * relevance(c) - (1 - lambda) * max_{s in selected} cos(c, s)
```

This favors cues that are relevant to the song while discouraging near-duplicate
cues within the same six-token set.

- **Fallback:** if fewer than 6 cues are available, remaining slots are padded
  with `<unk>` (index 0).

Output:

```text
item2cues.json = {item_id: [c1, c2, c3, c4, c5, c6]}
```

---

## Step 4 - Evaluation

### Coverage / Distribution / Cue quality

These are cheap intrinsic stats (no model calls):

- **Coverage:** fraction of songs with at least one non-`<unk>` cue (target >=40%).
- **UNK rate:** fraction of all cue slots that are `<unk>` (target <=60%).
- **Vocabulary utilization:** fraction of non-`<unk>` vocab entries assigned to
  at least one song (target >=50%; scales with corpus size, so low on small samples).
- **Entropy:** Shannon entropy of assigned non-`<unk>` cue usage; higher means
  less collapse onto a few generic cues.
- **Intra-item cosine (diversity):** mean within-item pairwise cosine of each song's
  6 cue embeddings (target <0.7; lower = more diverse, less redundant cue sets).

### Level 1 - Lexical Grounding

`level1_intrinsic()` computes ROUGE-1 and ROUGE-L recall between assigned cue
text and real lyrics.

This is a sanity check for lexical grounding. It is not the main semantic metric,
because good cues can paraphrase the lyric idea without sharing exact words.

### Level 2 - Cue-To-Song Semantic Retrieval

`level2_semantic_retrieval()` embeds each song's assigned cue strings as a query,
then ranks artist-free song texts by cosine similarity.

Targets use:

```text
title + genre + mood + tags + lyric excerpt + lyrics
```

Reported metrics:

- **R@1 / R@5 / R@10:** fraction of cue queries whose true song appears in the
  top K retrieved songs.
- **MRR:** mean reciprocal rank of the true song.
- **Mean / median rank:** lower is better.

This is the main cheap semantic metric because it asks whether the six cues
preserve enough meaning to identify their source song without relying on exact
lyric overlap.

**Independent encoder (avoiding circularity).** Assignment (Step 3) selects cues
by MiniLM cosine similarity to the song. If retrieval scored cues in the *same*
MiniLM space, it would reward exactly what assignment optimized — the metric would
be circular and inflated by construction. Retrieval therefore uses a **different**
encoder, chosen in order: OpenAI `text-embedding-3-small` (if an API key is set),
else `all-mpnet-base-v2` (a different local sentence encoder), else MiniLM as a
last resort (flagged in the report as circular). The chosen encoder and an
`independent` flag are recorded in the report. Even with an independent encoder,
retrieval is best read as a *cheap sanity check*; Level 3 reconstruction is the
primary downstream-usefulness metric because it does not depend on any embedding
similarity at all.

### Level 3 - Reconstruction Comparison (Optional)

`level3_reconstruction()` is enabled with `--level3` (add `--recon-batch` to run the
decoding through the OpenAI Batch API).

```text
real lyrics -> encode to 6 cues -> LLM decode to regenerated lyrics -> score vs real lyrics
```

The decoder receives the 6 cues plus **genre/mood only** — **title and artist are
withheld** so the model cannot recall the specific song from memory; the cues are
the only song-specific signal, which is what the ablation measures. It writes a
lyric draft sized to the real song's length (unclamped in `full` mode). For length
consistency, both the generated and the real lyric are truncated to the same
`--score-chars` window before every metric.

Metrics span lexical to semantic:
- **ROUGE-1/2/L, BLEU:** lexical n-gram overlap.
- **BERTScore-F1:** token-level contextual-embedding matching (baseline-rescaled, so
  values near/below 0 are expected — read rankings, not signs).
- **Cosine (STS):** whole-document semantic similarity via the free
  `all-mpnet-base-v2` encoder; **length-robust** (one vector per text) and the
  cleanest semantic signal.

Bracketed ablation — all conditions share the same decoder, only the cues differ:

| Condition | The 6 cues are... | Role |
|-----------|-------------------|------|
| `random` | 6 random vocab cues | **floor** — controls for the LLM writing plausible lyrics regardless of input |
| `tfidf` / `yake` / `keybert` / `llm` | that method's assigned cues | the methods under test |
| `oracle` | 6 cues an LLM extracts from the **real** lyrics | **ceiling** — best-possible cues |

Read each method as a delta between the random floor and the oracle ceiling
(`random < methods < oracle`). Absolute scores are low because the decoder
paraphrases; the bracket position is the signal. This stage is optional because it
costs LLM time / API usage; all decodes are cached.

---

## Outputs

| File | Contents |
|------|----------|
| `outputs/<method>/cue_vocab.json` | 2,048 cue strings; index = cue ID. |
| `outputs/<method>/item2cues.json` | `{item_id: [6 cue IDs]}`. |
| `outputs/comparison_report_<stamp>.md` (+ `comparison_report.md`) | TL;DR, vocabulary health, cue quality, grounding + retrieval, reconstruction ablation, qualitative samples, appendix. Timestamped archive plus a `latest` copy. |
| `outputs/reconstruction_report_<stamp>.md` (+ `reconstruction_report.md`) | Side-by-side original vs regenerated lyrics per method. |

---

## Design Choices Worth Noting

- **Flat cue vocabulary:** cues are not typed yet; type-aware cues can be a
  separate experiment.
- **Artist-free extraction and assignment:** artist names are excluded from cue
  evidence to avoid performer identity leakage.
- **Genre/mood retained:** genre and mood can still be useful generation
  controls, unlike artist identity.
- **Extractor-agnostic cleaning/assignment:** every method goes through the same
  cleaning and assignment code.
- **Sentence encoder for semantics, T5 only for tokenization:** MiniLM produces the
  cosine-meaningful embeddings; T5 is used solely for step-0 token canonicalization.
- **Independent retrieval encoder:** Level 2 scores with a different encoder than
  assignment, so retrieval is not circular.
- **Bracketed Level 3:** random floor and oracle ceiling bracket the methods; the
  decoder withholds title/artist to avoid memorization leakage.
- **Toggleable LLM batch:** `--llm-batch` (extraction) and `--recon-batch`
  (reconstruction) use the asynchronous OpenAI Batch API.
- **`top_n=40` by default:** cleaning is strict, so each extractor supplies a
  larger raw candidate pool before filtering.
- **`min_df` scales with corpus size:** it is an absolute count, so the right value
  depends on `N` — `5` for the full catalog, lower for small dev runs.
- **Caching:** extraction (keyed by method + song count + lyrics-mode), LLM calls,
  and embeddings are cached. Use `--force` after changing extraction/cleaning rules.

---

## Known limitations

- **Corrupted lyric files:** a small number of lyric files are not lyrics — some are
  Christmas-carol wrong-track matches, some are release-calendar / track-listing
  scrapes (hundreds of `Artist - "Song" ft. X` lines). These inject artist names and
  off-topic terms into the vocabulary out of proportion to their count. A
  junk-lyrics filter and a lyrics-fetch audit are open follow-ups.
- **`(feat. X)` in titles:** featured-performer names leak via titles even though the
  artist field is excluded.
- **Genre coverage is sparse:** genre is missing for most songs, so the genre/mood
  cue category rides mainly on a coarse 6-value heuristic mood label.
- **Metric bias:** grounding and retrieval reward lexical overlap with the lyrics,
  which favors extractive methods (tfidf) over abstractive imagery (llm). Concreteness
  and novelty-over-metadata are candidate additions to measure imagery quality.
- **`summarize` lyrics mode is not a fair comparison mode:** it inserts an LLM digest
  upstream of every extractor, contaminating the method comparison.
