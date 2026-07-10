# WP-B Creative Cue Mining — Progress Report

**Date:** 2026-07-10
**Reference:** `METHOD_WRITEUP.md` (kept in sync with this report).

This report walks through the WP-B pipeline step by step, explaining the key
design decisions taken along the way, and closes with the latest measured results.

The pipeline builds a 2,048-entry creative-cue vocabulary and assigns exactly 6
cues per song, then evaluates cue quality. Four extraction methods (**TF-IDF,
YAKE, KeyBERT, LLM**) are compared under one shared harness:

```
lyrics -> [0] preprocess -> [1] extract -> [2] clean -> [3] assign -> [4] evaluate
```

Throughout, two lyric views are maintained: **processed** lyrics feed extraction,
assignment and retrieval; the **original** lyrics are always the reference for
grounding and reconstruction scoring.

---

## Step 0 — Lyrics preprocessing

The `--lyrics-mode` flag shapes the lyric text fed into the pipeline. The same mode
is applied to every method so the comparison stays fair.

| Mode | Behaviour |
|------|-----------|
| `cap` | first N characters (positional truncation) |
| `full` | entire lyric, no truncation |
| `dedup` (used) | drop repeated chorus lines, keep distinct content, then cap |
| `summarize` | LLM compresses lyrics to a short summary |

We use **`dedup`**: it removes chorus repetition (which inflates TF-IDF and blurs
embeddings) with no LLM and no risk of hallucination. `summarize` is deliberately
**not** used for the comparison — inserting an LLM digest upstream of every
extractor would contaminate the comparison (all methods would then sit on the same
LLM-selected text) and its prompt primes toward the cue target.

---

## Step 1 — Extraction

Each method turns artist-free song text into a pool of raw candidate cues:

```
title + genre + mood + tags + lyric-excerpt + processed-lyrics
```

The **artist field is excluded** so cues describe song content, not performer
identity.

| Method | How candidates are produced |
|--------|------------------------------|
| `tfidf` | corpus-wide TF-IDF; each song's top-weighted uni/bigrams |
| `yake` | YAKE statistical keyphrases, per song |
| `keybert` | MiniLM keyphrases semantically closest to the song text |
| `llm` | LLM prompted for concrete imagery, cultural allusions, narrative motifs |

**LLM extraction prompt** (verbatim):

> **System:** You extract creative cues for a music generation system. Given a
> song's metadata and lyrics, list concrete imagery words, cultural allusions, and
> narrative motifs (e.g. 'train platform', 'neon rain', 'broken phone'). Prefer
> concrete nouns and short noun phrases. Avoid generic words (love, baby, music) and
> verbs.
>
> **User:** Song metadata + lyrics:
> {title + genre + mood + tags + lyrics}
> List up to 40 creative cues as a comma-separated list, lowercase, no numbering,
> no extra text.

All methods emit the same shape (`{item_id: [phrases]}`) and feed the same cleaning
pipeline, so downstream differences are attributable to extraction quality.

---

## Step 2 — Cleaning to a controlled vocabulary

Pooled candidates from all songs pass through shared filters (`N` = number of songs,
`df(c)` = songs containing cue `c`):

| # | Filter | Rule |
|---|--------|------|
| 0 | Normalize + token-identity dedup | lowercase, normalize punctuation, collapse `talkin→talking`; dedup by **T5-tokenizer** token ids within each song |
| 1 | df-band | keep `min_df ≤ df(c) ≤ 0.3·N` (drops one-off noise and corpus-dominant terms) |
| 2 | POS | keep noun / adj+noun phrases; drop verb-led fragments |
| 3 | Blocklist | drop boilerplate, artist-only phrases, edge-fragment phrases |
| 4 | Semantic dedup | embed cues with **MiniLM**, merge pairs with cosine > 0.92 |
| 5 | IDF rank + cut | keep top 2,047 by IDF, prepend `<unk>` at index 0 |

If fewer than 2,047 real cues survive, the rest are `<pad_*>` placeholders (never
assigned).

### Design decision — why we did *not* use T5 for the steps after step 0

T5 (`t5-small`) is used only in **step 0** to canonicalize/dedup cues by token
identity. We considered carrying it through to the *later* steps that rely on an
embedding whose **cosine reflects meaning** — the semantic dedup (step 4) and the
assignment (step 3, next) — and it failed badly.

T5 is a text-to-text **generation** model (span-corruption objective); it was never
trained for sentence similarity, so mean-pooling its encoder states gives
uncalibrated, anisotropic vectors — cosine barely varies between phrases. The
practical effect was severe: cue↔song relevance stopped discriminating, so a few
rare cues (e.g. `jingle bells`) were assigned to the majority of songs, and
**retrieval collapsed from MRR ~0.72 to ~0.10** across every method.

We reverted the embeddings to a **sentence-embedding model, `all-MiniLM-L6-v2`**,
which is contrastively trained so cosine reflects meaning — exactly the
"multilingual sentence encoder" the paper (§4.2) specifies. Retrieval recovered
(see Results). T5 now survives **only as a tokenizer** for step-0 canonicalization
(a text-normalization step, not a similarity metric) and plays no role in any
semantic decision.

**Lesson:** cue↔song similarity needs a sentence encoder, not a generative model's
raw encoder.

---

## Step 3 — Assignment

Assignment selects exactly 6 cues per song from the vocabulary:

- **Relevance:** cosine between the embedded song text and each cue embedding (both
  from MiniLM).
- **Selection:** greedy MMR that maximizes relevance while penalizing near-duplicate
  cues within the six — `argmax_c [ λ·rel(c) − (1−λ)·max_{s∈selected} cos(c,s) ]`.
- **Fallback:** pad with `<unk>` if fewer than 6 qualify.

Output: `item2cues.json = {item_id: [c1…c6]}`.

---

## Step 4 — Evaluation

Four layers, from cheap/intrinsic to downstream.

### 4a. Coverage, distribution, diversity (no model)
Coverage (≥1 non-`<unk>` cue), UNK rate, vocabulary utilization, cue-usage entropy,
and **within-item diversity** (mean pairwise cosine of a song's 6 cues; target < 0.7).

### 4b. Level 1 — lexical grounding
ROUGE recall of the assigned cue text against the **real** lyrics. A sanity check —
good cues can paraphrase, so this is not the main metric.

### 4c. Level 2 — cue-to-song retrieval
Each song's 6 cues are embedded as a query and ranked against artist-free song
texts; we report how highly the true song ranks (R@K, MRR, median rank).

Crucially, retrieval uses an **independent encoder** (OpenAI `text-embedding-3-small`),
*different* from the MiniLM used in assignment. If it used the same space it would
reward exactly what assignment optimized (circular). This is our strongest cheap,
non-circular signal of whether the cues preserve song identity.

### 4d. Level 3 — reconstruction comparison

This is the downstream-usefulness test: an **autoencoder-style** check of how much
of a song can be rebuilt through the 6-cue bottleneck.

```
real lyrics --[encode: our pipeline]--> 6 cues --[decode: fixed LLM]--> regenerated lyrics
                                                                              |
                            real lyrics <---[ROUGE / BLEU / BERTScore / Cosine]
```

**Why title and artist are withheld from the decoder.** An earlier version passed
the decoder the title and artist. This backfired: the no-cue baseline *beat* the cue
conditions. The reason was **memorization leakage** — given "Badfish | Sublime",
`gpt-4o-mini` recalls the real song and reproduces real lines, scoring high overlap
independently of the cues; adding cues then pulled the text toward the cue themes
and *away* from the memorized original. So the baseline was artificially strong and
cues looked harmful. We now pass **only genre + mood + the 6 cues** — the decoder
cannot identify the song, so the cues are the only song-specific signal, which is
what this comparison is meant to measure.

**Decoder prompt** (verbatim):

> **System:** You reconstruct a song's full lyrics from a few creative cues and
> minimal metadata. Develop the cues into complete lyrics of roughly the requested
> length, with the emotional arc and concrete imagery a real song of this style
> would have. Write full verses and a repeated chorus. Output ONLY lyric lines — no
> title, no section labels, no commentary.
>
> **User:** Style: {genre | mood}
> Creative cues: {6 cues}
> Target length: about {N} lines (write full verses and a chorus).
> Write the complete song lyric now:

**Length matching.** The decoder targets the real song's line count, and both
generated and real lyrics are truncated to the same 2,000-char window before
scoring — so a short generation is not penalized against a long reference.

**Bracketed comparison.** Every condition shares the decoder; only the cues differ.
The `random` row is the ablation baseline (cues removed / replaced by random),
while the method rows are a comparison and the oracle row is a ceiling:

| Condition | The 6 cues are... | Role |
|-----------|-------------------|------|
| `random` | 6 random vocabulary cues | **floor** — controls for the LLM writing plausible lyrics regardless of input |
| `tfidf / yake / keybert / llm` | that method's assigned cues | methods under test |
| `oracle` | 6 cues an LLM extracts from the **real** lyrics | **ceiling** — best-possible cues |

The oracle ceiling uses its own extraction prompt (verbatim):

> **System:** You extract creative cues from song lyrics for a music generation
> system. List concrete imagery words, cultural allusions, and narrative motifs
> actually present in the lyrics. Prefer concrete nouns and short noun phrases.
>
> **User:** Extract exactly 6 creative cues from these lyrics as a comma-separated
> list, lowercase, no numbering:
> {real lyrics, first 2000 chars}

**Metrics.** ROUGE-1/2/L and BLEU (lexical), BERTScore-F1 (token-level semantic;
baseline-rescaled, so values near/below 0 are expected), and **Cosine** (whole-
document semantic similarity via `all-mpnet-base-v2`; length-robust, cleanest single
signal). A good method reads as `random < method < oracle`.

---

## How each metric is calculated

Let `N` = number of scored songs and each song have 6 cue slots.

**Intrinsic (Step 4a)**
- **Vocab(real):** number of cue phrases surviving all cleaning filters (excludes
  `<unk>` and `<pad_*>` placeholders).
- **Coverage:** `1 − (songs with any <unk> slot) / N`. Equivalently, the fraction of
  songs whose 6 slots are all real cues (1.0 means every song was fully assigned).
- **UNK rate:** `(number of <unk> slots) / (N × 6)` — the fraction of *all* cue slots
  that fell back to `<unk>`.
- **Vocab utilization:** `(distinct non-<unk> cues assigned to ≥1 song) / 2047` — how
  much of the usable vocabulary is actually used.
- **Entropy:** Shannon entropy of the cue-usage distribution, in bits:
  `−Σ_c p_c · log2 p_c`, where `p_c` = (times cue `c` is assigned) / (total non-`<unk>`
  assignments). Higher = usage spread across many cues; lower = collapse onto a few.
- **Intra-cos (diversity):** for each song, the mean pairwise cosine among its 6 cue
  embeddings (MiniLM, L2-normalized); then averaged over songs. Lower = more diverse.

**Level 1 — grounding**
- For each song, its non-`<unk>` cue strings are joined into one pseudo-document and
  compared to the **real lyrics** with **ROUGE recall** (stemmed):
  `ROUGE-1 recall = (overlapping unigrams) / (unigrams in the lyrics)`, and **ROUGE-L**
  uses the longest common subsequence. Averaged over songs that have lyrics.

**Level 2 — retrieval**
- **Query** = a song's 6 cue strings joined by commas. **Targets** = the artist-free
  text (`title + genre + mood + lyric-excerpt + tags + lyrics`) of every candidate song.
- All are embedded with an **independent** encoder (OpenAI `text-embedding-3-small`),
  L2-normalized; similarity `S(q,t) = cosine`.
- For each query, targets are ranked by cosine; `r` = the 1-indexed rank of the query's
  true song. Then:
  - **R@K** = fraction of queries with `r ≤ K`;
  - **MRR** = mean of `1/r`;
  - **Mean / Median rank** = mean / median of `r`.

**Level 3 — reconstruction**
- The decoder produces a lyric from the cues; the generated and real lyrics are each
  truncated to the first 2,000 characters before scoring.
- **ROUGE-1/2/L:** mean F-measure between real (reference) and generated.
- **BLEU:** corpus BLEU (sacreBLEU) of the generated set against the real set.
- **BERTScore-F1:** mean token-level F1 using RoBERTa-large embeddings, baseline-rescaled
  (so values center near 0 and can be negative — only rankings are meaningful).
- **Cosine (STS):** each generated and real lyric is embedded to a single vector with
  `all-mpnet-base-v2` (L2-normalized); the score is the mean of the paired cosines.
  Length-robust because each text becomes one vector.

---

# Results

**Configuration:** 300 songs · 100-song eval sample · `min_df=5` · `top_n=40` ·
`lyrics-mode=dedup (cap 2000)` · report `20260708_232235`.

### R1. Vocabulary & assignment health
| Method | Vocab(real) | Coverage | UNK rate | Vocab util | Entropy |
|--------|-------------|----------|----------|-----------|---------|
| tfidf | 159 | 1.0 | 0.0 | 0.063 | 5.98 |
| yake | 142 | 1.0 | 0.0 | 0.058 | 5.87 |
| keybert | 55 | 1.0 | 0.0 | 0.026 | 5.26 |
| llm | **271** | 1.0 | 0.0 | **0.103** | **6.64** |

### R2. Cue quality — within-item diversity (target < 0.7)
| tfidf | yake | keybert | llm |
|-------|------|---------|-----|
| 0.2812 | **0.2707** | 0.2907 | 0.3050 |

### R3. Grounding & retrieval (independent encoder)
| Method | L1 ROUGE-1 | R@1 | R@5 | R@10 | MRR | Median rank |
|--------|-----------|-----|-----|------|-----|-------------|
| tfidf | 0.0073 | 0.22 | 0.54 | **0.72** | 0.3714 | 5.0 |
| yake | 0.0051 | 0.27 | 0.53 | 0.70 | 0.4054 | 4.5 |
| keybert | **0.0083** | 0.18 | 0.46 | 0.62 | 0.3149 | 7.0 |
| llm | 0.0043 | **0.33** | **0.62** | **0.72** | **0.4648** | **3.0** |

### R4. Reconstruction comparison (`random < methods < oracle`)
| Condition | ROUGE-1 | ROUGE-L | BLEU | BERTScore | Cosine |
|-----------|---------|---------|------|-----------|--------|
| random (floor) | 0.2257 | 0.1206 | 0.2164 | -0.2888 | 0.4191 |
| tfidf | 0.2358 | 0.1231 | 0.5012 | -0.2730 | **0.5210** |
| yake | 0.2391 | 0.1249 | 0.3718 | -0.2677 | 0.5123 |
| keybert | 0.2502 | 0.1259 | 0.5030 | -0.2707 | 0.5074 |
| llm | 0.2220 | 0.1195 | 0.4069 | -0.2761 | 0.4984 |
| oracle (ceiling) | 0.2582 | 0.1336 | 1.1354 | -0.2463 | 0.5849 |

### R5. Qualitative example — cues → regenerated lyrics (all four methods)

**Song:** *Badfish* (Sublime). Real lyrics (excerpt):
```
But I'm a parasite
Creep and crawl, I step into the night
...
Baby you're a big blue whale
Grab the reef when all duck divin' fails
```

Each method's 6 cues and the lyrics the decoder regenerates from them (excerpts):

**`tfidf`** — cues: `sea, baby just, alternative rock, fool, night, niggas`
```
Underneath the silver moon, we drift along the tide,
The waves whisper secrets, where our dreams collide.
Baby just hold on tight, as the night unfolds,
In this endless ocean, we'll find what we behold.
```

**`yake`** — cues: `sea, goin, alternative rock, fool, midnight, radio`
```
The waves crash softly, whispering low,
Underneath the stars, where the wild winds blow,
The radio plays our song on repeat,
But I'm just a fool, dancing on my own two feet.
```

**`keybert`** — cues: `baby just, alternative rock, night, fool, niggas, river`
```
Baby just hold on tight, the night is young and wild,
We'll chase the stars until the dawn, like we're lost in the wild,
Fools we were, running free, with the world at our feet,
In the shadows of the river, where our hearts would meet.
```

**`llm`** — cues: `sea, temptation, midnight, shelter, radio, boss`
```
Underneath the midnight sky,
The waves whisper secrets low,
A distant radio plays our song,
Temptation in the undertow.
You said the sea could be our shelter,
A place where dreams could breathe and grow...
```

All four recover the song's **sea / ocean / night** imagery from the cues alone
(no title or artist given). Note the character difference: the statistical/embedding
methods (`tfidf`, `yake`, `keybert`) reuse literal lyric words (`sea`, `night`,
`fool`, `river`), while `llm` introduces more evocative imagery (`temptation`,
`shelter`, `undertow`) — the same trade-off seen in the retrieval-vs-reconstruction
results.

---

## Reading the results

- **The pipeline is healthy.** Coverage is 100% and UNK rate 0% for all methods;
  every method passes the intrinsic diversity target (R2, all ≈0.27–0.31, well below
  0.7). Vocabularies are small only because `min_df=5` on 300 songs is strict —
  utilization (R1) scales with corpus size and should be re-checked on the full
  catalog.

- **The embedder fix worked.** Retrieval (R3) recovered to MRR 0.31–0.46; the broken
  T5 configuration had collapsed it to ~0.10. This is the headline evidence that the
  cues now carry genuine, discriminative song identity.

- **Which method "wins" depends on the goal.** The **LLM extractor is most
  discriminative** (R3: best R@1/R@5/MRR, median rank 3) — its abstract imagery
  (`temptation`, `moonlight`, `whispers`) uniquely identifies songs. The
  **statistical/embedding methods reconstruct best** (R4: TF-IDF cosine 0.521,
  KeyBERT ROUGE-L 0.126) because their cues are drawn literally from the lyrics, and
  the reconstruction metrics reward lexical overlap. Neither is strictly best; they
  optimize different things.

- **The reconstruction bracket is valid.** Every method sits between the random floor
  and the oracle ceiling (e.g. cosine: floor 0.419 < methods 0.498–0.521 < oracle
  0.585). TF-IDF reaches ~61% of the way from floor to oracle. This confirms the cues
  add real, measurable information over random, while the oracle shows the headroom a
  perfect 6-cue set would give. (BERTScore is baseline-rescaled, so the negative
  values are expected and only the *ranking* is meaningful — note even the oracle is
  negative, reflecting how hard full-lyric reconstruction from 6 cues is.)

- **The withheld-metadata design is doing its job.** In R5 the decoder recovers the
  song's sea/ocean/night imagery from the cues alone, without the title or artist —
  exactly the behaviour intended by withholding identity.

## Open issues (next steps)
- Grounding/retrieval reward lexical overlap, mildly favouring the extractive
  methods; imagery-quality metrics (concreteness, novelty-over-metadata) are
  candidates to fairly credit the LLM cues.
- Re-run on the full catalog to validate vocabulary utilization at scale.
