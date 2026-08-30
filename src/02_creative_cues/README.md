# WP-B — Creative Cue Mining

Builds a creative-cue vocabulary (2,048 entries by default — adjustable via
`--vocab-size`) and assigns cues per song:

```
song metadata + lyrics  →  cue_vocab.json + item2cues.json
```

This is the practical "how do I run this" reference. For *why* the pipeline is
shaped the way it is (cleaning stages, ranking rules, evaluation design,
known limitations), see [METHOD_WRITEUP.md](METHOD_WRITEUP.md).

```
catalog + lyrics
   |
   v  [0] PREPROCESS LYRICS (cap / dedup / full / summarize)
   v  [1] EXTRACT (tfidf / yake / keybert / llm)          -> raw candidate cues
   v  [2] CLEAN (df-band, POS, blocklist, semantic dedup) -> 2,048-entry vocabulary
   v  [3] ASSIGN (semantic relevance + diversity)         -> N cues per song
   v  [4] EVALUATE (grounding + retrieval + reconstruction, comparison only)
```

Two entry points share stages 0-3 via `pipeline.py`:

| Script | Does stages | Use it for |
|---|---|---|
| **`run_production.py`** | 0-3 only, one method | Building the actual deliverable — no evaluation, no LLM judging calls beyond extraction itself |
| **`run_compare.py`** | 0-4, one or more methods | Deciding *which* extraction method / settings to use, via the full grounding + retrieval + reconstruction evaluation |

Setup (once): create a venv and `pip install -r requirements.txt` into it, and set
`OPENAI_API_KEY` in the environment or a repo-root `.env` file if you're using
the `llm` method, Level 3 reconstruction, or the independent retrieval encoder.

The commands below assume the venv is activated (`.venv\Scripts\activate` on
Windows) so `python` resolves to the venv's interpreter.

---

## Production run

```bash
python src/02_creative_cues/run_production.py
```

Runs extract → clean → assign → export for **one method**, with **no
evaluation** (no held-out split, no grounding/retrieval/reconstruction, no LLM
judging calls beyond the extraction step itself). Settings come from a named
preset in [`config.py`](config.py) instead of a long flag list — pick a
preset, don't reach for a flag.

### CLI

| Flag | Default | Meaning |
|---|---|---|
| `--config NAME` | `default` | Named preset from `config.py` (see below) |
| `--limit N` | preset's value (`None` = full catalog) | Override song count — use this for a quick smoke test |
| `--force` | off | Bypass extraction/cleaning caches, re-run from scratch |
| `--vocab-size N` | preset's value (`2048`) | Override total vocab entries incl. `<unk>` — a non-default value needs a matching `vocab_size` passed to `CueMappingEntry.validate()`/`load_mapping()` to read the result back correctly |
| `--fixed-vocab PATH` | off | Freeze an existing `cue_vocab.json` and regenerate only the ranked per-song table |
| `--output-root PATH` | legacy production directory | Write a dataset-specific timestamped run and `latest/` mirror without replacing another dataset's cue artifacts; `CUE_OUTPUT_ROOT` is the environment equivalent |
| `--skip-health-check` | off | Skip the free coverage/diversity sanity stats (they cost no API calls; on by default) |

### Presets (`config.py`)

| Preset | method | rank_by | num_cues | min_df | top_n | lyrics_mode |
|---|---|---|---|---|---|---|
| `default` | `llm` | `idf` | **16 stored / 8 active** | 5 | 100 | `dedup` |
| `tfidf` | `tfidf` | `idf` | **16 stored / 8 active** | 5 | 100 | `dedup` |
| `research-18-cues` | `llm` | `idf` | 18 | 5 | 100 | `dedup` |

`default` is the WP-C-compatible production contract: a 2,048-entry cue
vocabulary and a relevance-ranked **16-candidate table per song**. WP-C consumes
the first 8 candidates, so 4/8/12/16-cue ablations share one frozen table.
`tfidf` keeps the same interface as an API-free baseline. `research-18-cues` is
an ablation only and does not match the frozen 16-candidate artifact contract.

Add a new preset in `config.py` (as a `replace(DEFAULT, ...)` entry in
`PRESETS`) when a setting changes, rather than adding a new CLI flag.

### Output

```
outputs/production/<timestamp>/
    cue_vocab.json      # 2,048 cue strings; index = cue ID
    item2cues.json      # {"item_id": [16 relevance-ranked cue IDs]}
    item2cue_scores.json # relevance scores aligned with item2cues.json
    item_cues.tsv       # human-readable long-form table
    cue_manifest.json   # schema, assignment settings, and artifact hashes
    run_config.json     # exact resolved settings for this run
    health_report.md    # coverage / UNK rate / vocab utilization / entropy (unless --skip-health-check)
outputs/production/latest/
    ... same files, overwritten every run — the current production vocab
```

#### Output schema

Exact shape of each file in `outputs/production/<timestamp>/` (and its
`latest/` mirror):

**`cue_vocab.json`** — JSON list of `vocab_size` strings (`2048` unless
`--vocab-size` overrides it). A cue's **index in this list is its cue ID**
everywhere else in the pipeline.

```json
["<unk>", "heartbreak", "neon nights", "..."]
```

> **Index 0 is reserved.** `vocab[0]` is always the literal string `"<unk>"`
> — it is prepended ahead of the ranked cues when the vocab is built, not a
> cue mined from the catalog. Cue ID `0` in `item2cues.json` means "no cue
> assigned" (missing coverage fallback), not an actual cue. Any tooling that
> reads `cue_vocab.json` should treat index 0 as this sentinel, not as
> vocabulary entry #1 — `cue_export.load_vocab()` asserts `vocab[0] ==
> "<unk>"` and rejects the file otherwise.

**`item2cues.json`** — JSON object `{"<item_id>": [cue_id, cue_id, ...]}`.

- `item_id` : string song ID (`"0"`, `"1"`, ...), matching the catalog's item IDs.
- value     : relevance-ranked cue IDs (ints, each `0 <= id < vocab_size`,
  indexing into `cue_vocab.json`). Length is **16** for production; WP-C uses
  `cue_ids[:8]`. `<unk>` padding, if needed, appears only at the end.

```json
{"0": [842, 12, 5, 1090, 3, 77, 0, 15, ...], "1": [3, 77, 900, ...]}
```

**`run_config.json`** — every resolved `ProductionConfig` field (`method`,
`limit`, `top_n`, `lyrics_mode`, `lyrics_cap`, `min_df`, `max_df_frac`,
`dedup_threshold`, `rank_by`, `vocab_size`, `num_cues`, `active_cues`,
`assignment_strategy`, `candidate_k`, `embedder`, `force`) plus
`preset`, `generated` (timestamp), `n_items`, `n_with_lyrics`. This is the
source of truth for the `num_cues`/`vocab_size` a given `item2cues.json`/
`cue_vocab.json` was built with.

**`health_report.md`** — human-readable only, not a stable machine schema.
Coverage rate, UNK rate, vocab utilization, cue entropy, top-10 cues table.

### Examples

```bash
# Standard production build (full catalog, the `default` preset)
python src/02_creative_cues/run_production.py

# Smoke test on a small sample before committing to a full run
python src/02_creative_cues/run_production.py --limit 200

# Build with tfidf instead of llm (no API cost)
python src/02_creative_cues/run_production.py --config tfidf

# Build the 18-cue research ablation (not WP-C compatible)
python src/02_creative_cues/run_production.py --config research-18-cues

# Re-run from scratch, ignoring caches
python src/02_creative_cues/run_production.py --force

# Build a larger 4096-entry vocab instead of the schema-default 2048
python src/02_creative_cues/run_production.py --vocab-size 4096
```

---

## Comparison / evaluation run (`run_compare.py`)

Use this instead of `run_production.py` when you're deciding *between*
methods or settings, not just building the deliverable. Runs one or more
extraction methods through the same cleaning/assignment pipeline, then scores
them: vocabulary health, cue diversity, Level 1 lexical grounding, Level 2
semantic retrieval, and (optionally) Level 3 LLM reconstruction.

```bash
python src/02_creative_cues/run_compare.py --limit 1000 --methods tfidf,yake
```

### CLI

| Flag | Default | Meaning |
|---|---|---|
| `--limit N` | `1000` | Number of catalog songs |
| `--methods` | `tfidf,yake` | Comma-separated: `tfidf`, `yake`, `keybert`, `llm` |
| `--top-n N` | `40` | Raw candidate cues kept per song before cleaning |
| `--min-df N` | `5` | Min songs a cue must appear in (df-band floor) |
| `--dedup-threshold F` | `0.92` | Cosine above which near-duplicate cues merge (higher = larger vocab) |
| `--no-semantic-dedup` | off | Skip semantic dedup entirely |
| `--rank-by` | `idf` | Stage-5 selection rule: `idf`, `df`, `df_idf`, `band`, `random`, `cluster` |
| `--lyrics-mode` | `cap` | `cap` \| `full` \| `dedup` \| `summarize` |
| `--lyrics-cap N` | 2000 | Char cap for `cap`/`dedup` modes |
| `--score-chars N` | `2000` | Common char window for Level 3 scoring (0 = full) |
| `--eval-sample N` | `150` | Songs used for retrieval/reconstruction evaluation |
| `--level3` | off | Run the LLM reconstruction ablation (costs API calls) |
| `--llm-batch` | off | Submit `llm` extraction via the OpenAI Batch API |
| `--recon-batch` | off | Run Level 3 reconstruction via the OpenAI Batch API |
| `--recon-report-samples N` | `15` | Songs shown in the original-vs-regenerated report |
| `--recon-report-lyric-chars N` | `0` | Chars per lyric shown in that report (0 = full) |
| `--llm-batch-poll-seconds N` | `30` | Seconds between OpenAI Batch status checks |
| `--llm-batch-timeout-seconds N` | `86400` | Max seconds to wait for a Batch job |
| `--force` | off | Bypass extraction/cleaning caches |
| `--note TEXT` | — | Free-text note stamped at the top of the report |
| `--held-out-eval` | off | Build vocab from a train split, score on a disjoint test split (generalization, not in-sample fit) |
| `--test-frac F` | `0.15` | Fraction held out for `--held-out-eval` |
| `--split-seed N` | `42` | Seed for the `--held-out-eval` split |
| `--num-cues N` | `8` (`CUE_TOKENS`) | Cues assigned per song — a non-default value needs a matching `n_cues` to read `item2cues.json` back |
| `--vocab-size N` | `2048` (`CUE_VOCAB_SIZE`) | Total vocab entries incl. `<unk>` — a non-default value needs a matching `vocab_size` to validate/read `cue_vocab.json`/`item2cues.json` back |

### Examples

```bash
# Full comparison across all four methods, with reconstruction
python src/02_creative_cues/run_compare.py --methods tfidf,yake,keybert,llm \
    --eval-sample 200 --level3

# LLM extraction via the async Batch API (large runs)
python src/02_creative_cues/run_compare.py --methods tfidf,yake,keybert,llm --llm-batch

# Generalization check: vocab never sees the songs it's scored on
python src/02_creative_cues/run_compare.py --methods tfidf,yake --held-out-eval --test-frac 0.15

# Compare a ranking rule against the default
python src/02_creative_cues/run_compare.py --methods tfidf --rank-by cluster --level3

# Try a larger vocabulary
python src/02_creative_cues/run_compare.py --methods tfidf --vocab-size 4096
```

### Output

```
outputs/runs/<run_id>/
    comparison_report.md        # TL;DR, vocab health, cue quality, grounding+retrieval,
                                 # reconstruction bracket, qualitative samples, appendix
    reconstruction_report.md    # side-by-side original vs regenerated lyrics (Level 3 only)
    run_config.json             # every CLI arg for this run
    methods/<method>/
        cue_vocab.json
        item2cues.json
```

`methods/<method>/cue_vocab.json` and `item2cues.json` follow the exact same
schema as the production output — see [Output schema](#output-schema) above.
`run_config.json` here holds the CLI args instead of a `ProductionConfig`, but
the fields that matter for reading the files back are the same: `--num-cues`
(item length in `item2cues.json`) and `--vocab-size` (bound on cue IDs /
length of `cue_vocab.json`).

---

## Experiment scripts (`sweeps/`)

Narrower, single-question experiments. Each is additive/read-only against the
pipeline — they only import `cue_extractors` / `cue_normalize` / etc., never
modify them. All write timestamped reports to `outputs/experiments/`.

| Script | Question it answers |
|---|---|
| `sweep_cleaning.py` | How do `min_df` and the dedup threshold affect vocabulary size? |
| `sweep_corpus_size.py` | How does real vocabulary size grow as the corpus grows? |
| `sweep_num_cues.py` | Does reconstruction quality improve with more cues/song, and where does it plateau? |
| `sweep_ranking.py` | Which stage-5 ranking rule (`idf`/`df`/`df_idf`/`band`/`random`/`cluster`) gives the most stable, well-utilized, downstream-useful vocabulary? |
| `sweep_vocab_stability.py` | Does the chosen cue set converge as the corpus grows, or keep churning? |
| `sweep_embedder_dedup.py` | Does some `--dedup-threshold` recover a non-`minilm` embedder's (e.g. `qwen3-0.6b`) cue diversity, instead of reusing MiniLM's tuned `0.92`? |

### `sweep_cleaning.py`

```bash
python src/02_creative_cues/sweeps/sweep_cleaning.py --method tfidf --limit 1000 \
    --lyrics-mode dedup --lyrics-cap 2000 --top-n 60 \
    --min-df 2,3,5,10 --dedup-threshold 0.90,0.92,0.95,1.0
```
`--method` (`tfidf`) · `--limit` (`1000`) · `--lyrics-mode` (`dedup`) · `--lyrics-cap` (`2000`) ·
`--top-n` (`60`) · `--min-df` comma-separated grid (`2,3,5,10`) · `--dedup-threshold` comma-separated
grid (`0.90,0.92,0.95,1.0`; `1.0` skips dedup) · `--vocab-size` (`2048`) · `--force`

### `sweep_corpus_size.py`

```bash
python src/02_creative_cues/sweeps/sweep_corpus_size.py --methods tfidf,yake \
    --limits 100,300,500,1000 --min-df 5 --dedup-threshold 0.92 \
    --lyrics-mode dedup --lyrics-cap 2000 --top-n 60
```
`--methods` (`tfidf,yake`) · `--limits` comma-separated song counts (`100,300,500,1000`) ·
`--min-df` (`5`) · `--dedup-threshold` (`0.92`) · `--vocab-size` (`2048`) · `--lyrics-mode` (`dedup`) ·
`--lyrics-cap` (`2000`) · `--top-n` (`60`) · `--force`. Uncached limits trigger extraction —
keybert/llm cost per song, so only sweep those over limits you already have cached, or expect the cost.

### `sweep_num_cues.py`

Requires `OPENAI_API_KEY` (the reconstruction decoder).

```bash
python src/02_creative_cues/sweeps/sweep_num_cues.py --method tfidf --limit 300 \
    --eval-sample 60 --num-cues 0,3,6,9,12 \
    --min-df 5 --dedup-threshold 0.92 --lyrics-mode dedup --lyrics-cap 2000 --top-n 60
```
`--method` (`tfidf`) · `--limit` (`300`) · `--eval-sample` (`60`) ·
`--num-cues` comma-separated grid (`0,3,6,9,12`; `0` = metadata-only floor) · `--min-df` (`5`) ·
`--dedup-threshold` (`0.92`) · `--vocab-size` (`2048`) · `--lyrics-mode` (`dedup`) ·
`--lyrics-cap` (`2000`) · `--top-n` (`60`) ·
`--candidate-k` (`40`, MMR candidate-pool floor) · `--score-chars` (`2000`) · `--force`

### `sweep_ranking.py`

Health/stability are free; retrieval needs `OPENAI_API_KEY` (skip with `--skip-retrieval`).

```bash
python src/02_creative_cues/sweeps/sweep_ranking.py --method tfidf \
    --limits 500,1000,2000,3000,5000 --retrieval-limit 1000 \
    --vocab-size 2048 --min-df 2 --dedup-threshold 0.92 --num-cues 18 \
    --lyrics-mode dedup --lyrics-cap 2000 --top-n 60 --eval-sample 100
```
`--method` (`tfidf`) · `--arms` (`random,df,idf,df_idf,band,cluster`) ·
`--limits` comma-separated corpus sizes (`500,1000,2000,3000,5000`) ·
`--retrieval-limit` (`1000`, which corpus size health/retrieval are measured at) ·
`--vocab-size` (`2048`) · `--min-df` (`2`) · `--dedup-threshold` (`0.92`) · `--num-cues` (`18`) ·
`--eval-sample` (`100`) · `--lyrics-mode` (`dedup`) · `--lyrics-cap` (`2000`) · `--top-n` (`60`) ·
`--seed` (`0`) · `--skip-retrieval` · `--force`

> **Known caveat:** with the default `--dedup-threshold`/df-band settings, `df_idf` selects the
> *identical* set as `df` (its score is monotonic in df across the whole df-band range at
> `max_df_frac=0.3`) — see the note in the script's generated report before reading `df_idf` as a
> distinct arm.

### `sweep_vocab_stability.py`

```bash
python src/02_creative_cues/sweeps/sweep_vocab_stability.py --methods tfidf,yake \
    --limits 500,1000,2000,3000,5000 --min-df 2 --lyrics-mode dedup --lyrics-cap 2000 --top-n 60
```
`--methods` (`tfidf,yake`) · `--limits` comma-separated corpus sizes (`500,1000,2000,3000,5000`) ·
`--min-df` (`2`) · `--dedup-threshold` (`0.92`) · `--vocab-size` (`2048`) · `--lyrics-mode` (`dedup`) ·
`--lyrics-cap` (`2000`) · `--top-n` (`60`) · `--force`

### `sweep_embedder_dedup.py`

Runs extract → clean → assign → health-check per (embedder, dedup-threshold) grid
point and reports `intra_cos_mean` (within-item cue diversity; paper collapse
ceiling `0.7`) side by side across embedders — see
[PROGRESS_REPORT_2.md §6.5](PROGRESS_REPORT_2.md) for why this needs checking at
all: swapping `minilm` for `qwen3-0.6b` at the same `0.92` threshold collapsed
diversity (`0.30` → `0.66`) because Qwen's cosine similarities run more compressed
for short phrases, so a threshold tuned against MiniLM's distribution doesn't
transfer as-is.

```bash
python src/02_creative_cues/sweeps/sweep_embedder_dedup.py \
    --embedders minilm,qwen3-0.6b --dedup-threshold 0.90,0.92,0.94,0.96,0.98 \
    --limit 800 --num-cues 18
```
`--embedders` comma-separated `cue_normalize.EMBEDDER_MODELS` keys (`minilm,qwen3-0.6b`) ·
`--dedup-threshold` comma-separated grid (`0.90,0.92,0.94,0.96,0.98`) ·
`--method` (`tfidf`) · `--limit` (`800`) · `--min-df` (`5`) · `--max-df-frac` (`0.3`) ·
`--rank-by` (`idf`) · `--num-cues` (`18`) · `--vocab-size` (`2048`) · `--lyrics-mode` (`dedup`) ·
`--lyrics-cap` (`2000`) · `--top-n` (`100`) · `--force` ·
`--keep-artifacts` (off — keeps each grid point's `cue_vocab.json`/`item2cues.json` instead of
discarding them after scoring)

The embedder model is loaded once per embedder, not once per grid point (loading
`qwen3-0.6b`/`4b` repeatedly is the expensive part); each grid point still re-embeds
its own threshold-dependent cue set and the song corpus. Report:
`outputs/experiments/sweep_embedder_dedup_<timestamp>/report.md`.

---

## Code layout

```
config.py            named ProductionConfig presets — the production settings live here
data_loading.py       shared catalog/lyrics loading (used by every entry point)
pipeline.py           shared extract -> clean -> assign -> export orchestration
cue_extractors.py     Step 1 — tfidf / yake / keybert / llm raw-cue extraction
cue_normalize.py       Step 2 — cleaning pipeline + stage-5 ranking rules
cue_assign.py          Step 3 — MMR cue assignment
cue_eval.py             Step 4 — Level 1/2/3 evaluation (run_compare.py / sweeps only)
cue_export.py           export_outputs / compute_coverage_stats / write_report
cue_clients.py          shared OpenAI chat + embedding client (disk-memoized)
cue_lyrics.py            lyrics preprocessing modes (cap / full / dedup / summarize)
cue_io.py                 atomic, concurrency-safe file writes for shared caches

run_production.py    production entry point (this doc's main focus)
run_compare.py         comparison + evaluation entry point
sweeps/                narrow single-question experiments (see above)
legacy/                superseded code — do not add new callers (see legacy/cue_mining_legacy.py)

outputs/
    production/<timestamp>/, production/latest/    run_production.py output
    runs/<run_id>/                                   run_compare.py output
    experiments/                                      sweeps/ output
    legacy_reports/                                    pre-restructure stale reports, archived
    _cache/                                             shared LLM + embedding cache (content-hash keyed)
    <method>/                                           shared raw-cue extraction cache
```

For the design rationale behind the cleaning stages, the ranking-rule tradeoffs, the evaluation
levels, and known limitations, see [METHOD_WRITEUP.md](METHOD_WRITEUP.md).
