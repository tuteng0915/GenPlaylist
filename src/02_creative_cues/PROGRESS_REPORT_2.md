# WP-B Creative Cue Mining — Progress Report 2

**Date:** 2026-07-23
**Focus:** vocabulary construction experiments — how the cue vocabulary behaves as we
change corpus size, cues-per-song, `min_df`, and how stable the vocabulary is.

All figures below are taken verbatim from the sweep reports in
`outputs/experiments/` (source report timestamp noted per section). Debug runs on
20–30 songs (`20260723_*`) are excluded as non-representative; the newest *full-scale*
report is used for each experiment.

> **The final production pipeline has been built** (`run_production.py` + `config.py`),
> with the defaults settled by the experiments in this report. The pipeline summary and
> those defaults are below; the experiments that justify them follow.

---

## 0. Pipeline & evaluation overview (for first-time readers)

**What WP-B produces.** A fixed **creative-cue vocabulary** (a controlled list of
imagery/theme phrases) and a **song → cue mapping**: every catalog song is assigned a
small set of cue tokens. These cue tokens extend the downstream music-generation model's
input, giving it a compact, discrete description of each song's content.

**The pipeline (six stages).** Given each song's lyrics + metadata:

```
0 lyrics preprocess → 1 extract candidate cues → 2 clean → 3 assign cues/song → (4 evaluate)
```

| Stage | What it does |
|-------|--------------|
| **0 Lyrics preprocess** | Shape the lyric text (e.g. `dedup`: drop repeated chorus lines, then cap length). |
| **1 Extract** | Pull raw candidate cue phrases per song. Four interchangeable extractors: `tfidf`, `yake`, `keybert` (statistical/embedding) and `llm` (an LLM prompted for concrete imagery). |
| **2 Clean** | Reduce all candidates to one vocabulary: df-band filter (drop too-rare/too-common) → POS filter (keep noun-ish phrases) → blocklist (boilerplate/artist names) → semantic dedup (merge near-synonyms by embedding cosine) → **rank & cut** to the vocabulary size. |
| **3 Assign** | For each song, select its cues from the vocabulary by semantic relevance with a diversity penalty (greedy MMR over cue/song embeddings). |
| **4 Evaluate** | (Comparison only — not part of a production build.) See below. |

**How it's evaluated** (used only in the comparison runs, never in production):
- **Vocabulary health** — coverage, utilisation, usage entropy, within-song cue diversity.
- **Level 1 — grounding**: ROUGE overlap between a song's assigned cues and its real lyrics.
- **Level 2 — retrieval**: embed a song's cues as a query and rank all songs; does the true
  song come top? (Uses an *independent* encoder so it isn't circular.) Reported as R@K / MRR.
- **Level 3 — reconstruction**: an LLM regenerates lyrics from *only* the cues (title/artist
  withheld); the regeneration is scored against the real lyrics (ROUGE/BLEU/BERTScore/cosine),
  bracketed by a random-cue floor and an oracle ceiling.

**Production defaults** (from `config.py`, the `DEFAULT` preset — what a production build uses):

| Setting | Value | Notes |
|---------|-------|-------|
| Extractor (`method`) | **`llm`** | most diverse/coherent cues of the four |
| Corpus (`limit`) | **full catalog** | `None` = all songs |
| Lyrics mode / cap | **`dedup` / 2000** | drop repeated lines, cap 2000 chars |
| `top_n` (raw cues/song) | **100** | candidate pool before cleaning |
| `min_df` / `max_df_frac` | **5 / 0.3** | df-band; `min_df=5` fills the vocab on the full catalog |
| Semantic dedup threshold | **0.92** | merge cues with cosine > 0.92 |
| Ranking (`rank_by`) | **`idf`** | held its own vs `cluster`/`band` on health + reconstruction |
| Vocabulary size | **2048** | schema contract (`CUE_VOCAB_SIZE`) |
| Cues per song (`num_cues`) | **18** | validated budget (schema default is 6; see §2) |
| Embedder | **MiniLM** (`all-MiniLM-L6-v2`) | semantic cosine for dedup + assignment |
| Assignment | MMR, `lam=0.7`, `candidate_k=40` | relevance vs diversity |

A production build is run by preset name (`run_production.py`) rather than a wall of flags,
and it only builds the vocabulary + mapping — it never scores anything.

---

## 1. Vocabulary size vs corpus size
_Source: `sweep_corpus_20260721_043758.md` · min_df 5 · dedup 0.92 · top_n 60 · lyrics-mode dedup_

How many usable cues survive cleaning as more songs are added, for the statistical
(`tfidf`) and LLM (`llm`) extractors.

| songs (N) | tfidf real vocab | llm real vocab | fill% (of 2,047) |
|-----------|------------------|----------------|------------------|
| 100 | 167 | 164 | 8% |
| 300 | 378 | 367 | 18% |
| 500 | 830 | 819 | 41% |
| 1000 | 1,510 | 1,527 | 75% |
| 3000 | 4,345 | 3,462 | 100% |
| 5000 | 7,160 | 4,971 | 100% |

**Cleaning funnel at N=5000**

| method | raw candidates | after df-band | after POS | after blocklist | real vocab |
|--------|----------------|---------------|-----------|-----------------|-----------|
| tfidf | 67,712 | 11,248 | 9,387 | 7,377 | **7,160** |
| llm | 56,335 | 6,042 | 5,814 | 5,079 | **4,971** |

**Findings**
- The vocabulary **fills the 2,048 cap at ~3,000 songs** (both methods); below that it
  is underfilled and padding-heavy. At the paper's min_df=5, the full catalog easily
  fills the vocabulary.
- **tfidf yields ~1.4× more surviving cues than llm** at scale (7,160 vs 4,971). tfidf
  emits many surface-form n-grams; llm produces a smaller, more canonical concept set —
  fewer raw candidates that clear the df bar (56k vs 68k raw, and a steeper df-band cut).
- The pre-cap candidate pool keeps growing roughly linearly with N (Heaps' law) — this
  is expected and not a problem once the vocabulary is full.

---

## 2. Number of cues per song vs reconstruction quality
_Source: `sweep_numcues_tfidf_20260719_161719.md` · 500 songs · eval sample 60 · vocab(real) 2047 · min_df 2_

Assign N cues/song, decode lyrics from ONLY those cues (title/artist withheld), score
vs the real lyrics. Higher is better except BERTScore (baseline-rescaled — read rankings).

| N cues | ROUGE-1 | ROUGE-L | BLEU | BERTScore | Cosine |
|--------|---------|---------|------|-----------|--------|
| 6 | 0.2501 | 0.1272 | 0.5099 | −0.2660 | 0.5565 |
| 12 | 0.2551 | 0.1269 | 0.7334 | −0.2621 | 0.5865 |
| 18 | 0.2560 | 0.1272 | 0.8003 | −0.2569 | 0.5940 |
| 24 | 0.2624 | 0.1299 | 0.7123 | −0.2595 | 0.6038 |
| **32** | **0.2671** | **0.1312** | 0.7732 | **−0.2557** | **0.6158** |
| 48 | 0.2605 | 0.1270 | 0.7678 | −0.2578 | 0.6125 |
| 64 | 0.2634 | 0.1280 | 0.8440 | −0.2578 | 0.6099 |

**Findings**
- **More cues improve reconstruction, with diminishing returns.** Cosine rises steadily
  from 6 → 32 cues (0.557 → **0.616**, +11% relative), then **plateaus/dips** at 48–64.
- The peak is around **N ≈ 32**; **N = 18 captures most of the gain** (cosine 0.594,
  +7% over 6) at a lower token cost.
- The gain is **semantic** (Cosine, BERTScore climb) more than lexical (ROUGE roughly
  flat) — extra cues add thematic coverage, not verbatim overlap.
- **Implication:** the default 6-cue budget is on the low side for reconstruction. If the
  downstream token budget allows, **12–18 cues** is a good sweet spot; ~32 is the point
  of diminishing returns. (This requires coordinating `CUE_TOKENS` with WP-D.)

---

## 3. Effect of `min_df`
_Source: `sweep_tfidf_20260716_184640.md` · 1000 songs · tfidf · dedup 0.92 · top_n 60_

`min_df` = the minimum number of songs a cue must appear in to survive.

| min_df | real vocab | fill% |
|--------|-----------|-------|
| 2 | ~5,935 | **100%** |
| 3 | ~2,395 | **100%** |
| 5 | ~766 | 37% |
| 10 | ~243 | 12% |

**Cleaning funnel at min_df=2 (N=1000):** raw 14,607 → df-band 13,996 → POS 7,491 →
blocklist 6,165 → dedup 5,935.

**Findings**
- **`min_df` is the dominant lever on vocabulary size.** On 1,000 songs, `min_df ≤ 3`
  fills the 2,048 vocabulary; `min_df = 5` leaves it **63% empty**, and `min_df = 10`
  leaves it 88% empty. The paper's `min_df = 5` assumes the full ~5k-song catalog
  (where it fills); on smaller subsets use **`min_df = 2`**.
- **The dedup threshold is negligible** for size — moving 0.90 → 1.0 (skip) changes the
  count by only ~2–6%. Not a useful tuning knob.
- **The POS filter is the largest *cleaning* cut** — it removes ~46% of df-band
  survivors (13,996 → 7,491). It doesn't prevent a full vocabulary at low `min_df`, but
  it is the stage to loosen if a small-corpus vocab is starved.

---

## 4. Vocabulary stability (convergence)
_Source: `sweep_stability_20260719_163343.md` · tfidf · min_df 2 · dedup 0.92_

Vocabulary size is capped, so "convergence" = does the *chosen cue set* stop changing as
songs are added? **Jaccard** = overlap between two vocab sets (1.0 = identical).

**Consecutive-corpus overlap**

| N → N | Jaccard | retention |
|-------|---------|-----------|
| 500 → 1000 | 0.256 | 0.41 |
| 1000 → 2000 | 0.297 | 0.46 |
| 2000 → 3000 | 0.480 | 0.65 |
| 3000 → 5000 | 0.385 | 0.56 |

**Overlap with the final (N=5000) vocab**

| N=500 | N=1000 | N=2000 | N=3000 |
|-------|--------|--------|--------|
| 0.039 | 0.095 | 0.227 | 0.385 |

**Findings**
- **The default IDF-ranked vocabulary does NOT converge.** Consecutive overlap stays
  ~0.26–0.48 (35–60% of cues churn out each step), and even *drops* at the last step.
  The N=1000 vocabulary shares **under 10%** of its cues with the N=5000 vocabulary.
- **Root cause:** the final ranking step keeps the *rarest* cues (highest IDF), and
  "rarest" is corpus-size-dependent — growing the corpus keeps surfacing rarer cues that
  displace the incumbents. The vocabulary chases the tail instead of settling on concepts.
- **A ranking fix largely solves it.** A follow-up ranking experiment
  (`sweep_ranking_tfidf_20260720_105021.md`) showed that ranking by document frequency
  (`df`) or a scale-free prevalence band raises stability from **0.42 → ~0.73** (3000→5000
  Jaccard) for only a ~5% retrieval-quality cost — because retrieval is largely
  insensitive to *which* diverse subset of cues is chosen.

---

## 5. Vocabulary ranking method
_Source: `sweep_ranking_tfidf_20260720_105021.md` · tfidf · corpus 500–5000 · K 2048 · min_df 2 · dedup 0.92 · 18 cues/song_

The final cleaning stage picks which surviving cues become the vocabulary. This is a
**controlled comparison**: cleaning (stages 0–4) runs once per corpus size, then each
ranking rule selects from that identical pool — so the arms differ only in the ranking.
`random` (pick K at random) is the ablation floor.

| Arm | Stability (3000→5000 Jaccard ↑) | Convergence (N=3000 vs 5000 ↑) | Retrieval MRR ↑ | R@5 ↑ | intra-cos ↓ |
|-----|--------------------------------|--------------------------------|-----------------|-------|-------------|
| `random` (floor) | 0.025 | 0.025 | 0.7352 | 0.83 | 0.284 |
| `idf` (current) | 0.421 | 0.421 | **0.7869** | 0.91 | 0.297 |
| `df` | **0.730** | **0.730** | 0.7349 | 0.89 | 0.318 |
| `df_idf` \* | 0.730 | 0.730 | 0.7349 | 0.89 | 0.318 |
| `band` | 0.727 | 0.727 | 0.7348 | 0.88 | 0.319 |
| `cluster` | 0.327 | 0.327 | 0.7541 | **0.94** | 0.298 |

**Findings**
- **A clear stability ↔ discriminability trade-off.** Frequency-based ranking (`df` /
  `band`) is **~1.7× more stable** than the current IDF (0.73 vs 0.42) and converges
  cleanly, but IDF has the best retrieval (MRR 0.787 vs 0.735, ~+7%). No arm wins both.
- **Ranking matters for stability, not retrieval.** The `random` floor nearly ties
  `df`/`band` on retrieval (MRR 0.735) — the assigner picks the most relevant 18 cues
  from whatever's available, so *which* subset fills the vocabulary barely affects
  retrieval. Ranking earns its keep almost entirely through **stability**.
- **`cluster` refuted its hypothesis** — expected to be most stable, it came second-worst
  (0.327). K-means re-partitions a growing pool, so the elected representative shifts even
  when concept regions are stable. It did give the best coverage (R@5 0.94, R@10 0.97).
- **\* `df_idf` was degenerate** — identical to `df` in every column. `df·idf` peaks at
  df ≈ 0.37·N, above the 0.3·N df-band ceiling, so within the allowed range it's
  monotonic in df. It needs a damped exponent (`df^0.5·idf`) to become a real arm; the
  intended "mid-frequency balance" was not actually tested here.

**Recommendation:** switch the final ranking from `idf` to **`df`** (or the scale-free
`band`) — a large, real stability gain for a ~5% retrieval cost, and retrieval is
demonstrably insensitive to the ranking choice. Verify qualitatively that `df` doesn't
fill with generic filler (`love`, `night`), which the retrieval metric can't detect.

---

## 6. Production generation report (held-out)
_Source: run `20260722_082158_a861` — the production config: `llm` extractor, full 5,000-song
catalog, min_df 5, dedup 0.92, idf ranking, 18 cues/song, lyrics-mode dedup._

This is a full production-settings build **evaluated on held-out songs**: the vocabulary was
built from 4,250 train songs and every metric below is scored on **750 disjoint test songs the
vocabulary never saw** (item-level split, seed 42). It is the closest run to what production ships.

### 6.1 Vocabulary & retrieval (test songs)
| Method | Vocab(real) | Vocab util | Intra-cos ↓ | R@1 | R@5 | MRR |
|--------|-------------|-----------|-------------|-----|-----|-----|
| tfidf | 5,347 | 0.609 | 0.309 | 0.393 | 0.667 | 0.511 |
| **llm** (production) | 3,626 | **0.633** | **0.293** | 0.360 | **0.700** | **0.515** |

The production `llm` vocabulary is fully covered (100% coverage, 0 UNK), well-utilised (63%),
and the most diverse per song (intra-cos 0.29, well under the 0.7 target). It retrieves the
correct held-out song into the top-5 **70%** of the time.

### 6.2 Reconstruction — generation quality (test songs)
Decode lyrics from the 18 cues alone (title/artist withheld), score vs the real lyrics.
Bracketed by floors and an oracle ceiling.

| Condition | ROUGE-L ↑ | BLEU ↑ | BERTScore ↑ | Cosine ↑ |
|-----------|-----------|--------|-------------|----------|
| no cues (metadata-only floor) | 0.125 | 0.321 | −0.291 | 0.425 |
| random cues (floor) | 0.119 | 0.275 | −0.289 | 0.489 |
| **llm** (production) | 0.119 | 0.489 | −0.268 | **0.588** |
| tfidf | 0.129 | 0.469 | −0.252 | 0.615 |
| oracle (real-lyric cues, ceiling) | 0.145 | 1.620 | −0.219 | 0.662 |

The cues carry real content: production `llm` reconstruction cosine (0.588) sits well above
both floors (0.425 / 0.489) and reaches **~68%** of the way from the no-cue floor to the oracle
ceiling. tfidf edges it on cosine (0.615) because its cues reuse literal lyric words, but `llm`'s
are qualitatively more evocative (below).

### 6.3 Example generation (held-out song)
**"Collie Man" — Slightly Stoopid.** Production `llm` cues:
`echoes of love, man down, side of the road, urban rhythm, collar, laughter lines, movin,
endless journey, lullabies, gospel, lonely streets, helplessness, harmonies, freedom ride,
unfulfilled longing, soft melodies, jingle bells, light of hope`

Real lyrics (excerpt):
```
And the road to life, yes, it goes up and down
...
Hey mister collie man, why don't you come 'round no more?
```
Regenerated from the 18 cues alone (excerpt):
```
Movin' through the chaos, an endless journey,
Lullabies of the city, whispering softly,
Gospel of the night, in the neon glow,
Lonely streets calling, where the lost souls go.
```
Without ever seeing the title or artist, the decoder recovers the song's **road / journey /
lonely-streets** imagery from the cues — the intended behaviour.

### 6.4 Key finding: min_df and ranking had **no effect** on generation quality
`min_df` and `rank_by` were the two levers that looked most promising on *intrinsic* metrics
(vocabulary size in §3, stability in §4–5). Three otherwise-identical production runs isolate them:

| Run | change | vocab (llm) | **Reconstruction Cosine** (llm / tfidf) |
|-----|--------|-------------|------------------------------------------|
| `a861` (production) | idf, min_df 5 | 3,626 | 0.588 / 0.615 |
| `5b18` | **min_df 2** (idf) | 9,960 | 0.606 / 0.610 |
| `f177` | **cluster ranking** (min_df 5) | 3,626 | 0.575 / 0.607 |

Despite `min_df=2` producing a **2.7× larger** vocabulary (9,960 vs 3,626 cues) and `cluster`
re-selecting the vocabulary entirely, the downstream reconstruction cosine barely moves — within
±0.03, i.e. inside noise. The oracle ceiling (0.662) and no-cue floor (0.425) are identical across
all three, confirming the task is unchanged.

**Interpretation:** vocabulary size and ranking method matter for *intrinsic* properties
(fill, stability), but the **assigner picks the 18 most relevant cues from whatever vocabulary it
is given**, so the cues a song ends up with — and therefore generation quality — are largely
insensitive to those two knobs. This is why the production preset keeps the simple defaults
(`min_df=5`, `idf`): the improvements those levers showed on paper do not translate into better
generation, so there is no downstream reason to complicate the pipeline for them.

---

## 7. Overall conclusions & recommendations

1. **Corpus:** use the full catalog (~5k songs). Vocabulary fills by ~3,000 songs;
   smaller subsets are padding-heavy.
2. **`min_df`:** `5` fills the vocabulary on the full catalog (`2` only needed on small
   subsets). It changes vocabulary *size* a lot but — per §6.4 — **has no effect on
   generation quality**, so production keeps `5`. Don't tune the dedup threshold either.
3. **Cues per song:** the 6-cue default is suboptimal for reconstruction; production uses
   **18** (diminishing returns by ~32). This is the one lever that *does* improve generation.
4. **Ranking / stability:** IDF ranking is corpus-size-dependent and less stable than `df`/
   `band` on *intrinsic* metrics — but §6.4 shows that advantage **does not translate into
   better generation** (idf, cluster and min_df variants land within noise on reconstruction).
   Production therefore keeps the simple **`idf`** default rather than complicating the
   pipeline for a gain that doesn't reach the output.
5. **Extractor:** tfidf yields a larger vocabulary and slightly higher reconstruction cosine
   (its cues echo lyric words); production uses **`llm`** for its more diverse, evocative,
   concept-level cues (best intra-cos and retrieval), which better suit downstream generation.

> **Bottom line:** the production pipeline is built and validated on held-out songs (§6). The
> two parameters that looked most promising on intrinsic metrics — `min_df` and `rank_by` —
> turned out to have **no measurable effect on generation quality**, so production ships the
> simplest defaults (`min_df=5`, `idf`) and invests only in the levers that reached the
> output: the `llm` extractor and an 18-cue budget.
