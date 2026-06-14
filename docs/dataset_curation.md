# Dataset Curation

This document describes the complete data cleaning and curation pipeline used to produce the GenPlaylist training dataset from the Spotify Million Playlist Dataset (MPD) backbone.

---

## Source Data

The starting point is the DDBC-preprocessed subset of the **Spotify Million Playlist Dataset (MPD)** (Spotify RecSys Challenge 2018):

| Field | Value |
|-------|-------|
| Total playlists | 20,000 |
| Total unique songs | 254,155 |
| Backbone files | `bi_full.txt`, `train.txt`, `valid.txt`, `test.txt`, `metadata.json`, `clhe_token.json`, `clhe_weight.npy` |
| Embedding dim | 64 (CLHE) |
| RVQ codebooks | 3 × 256 entries |

Each item has a unique integer `item_id` (0–254,154) and a metadata string of the form `'Title' by Artist in album'Album'`.

---

## Playlist Filtering — Iterative Convergence

The goal is a subset of **mainstream playlists containing only sufficiently popular songs**, suitable for training a next-item generation model. The pipeline applies a length filter, a frequency filter, and an audio-availability filter, iterating until convergence.

### Design Rationale

An earlier v1 pipeline used a playlist length range of 10–30 songs, which produced a dataset dominated by short niche playlists. Analysis showed the MPD median playlist length is 47 songs and mean is 63.6 — the v1 upper bound of 30 was selecting the bottom 31.5% of playlists by length. Longer playlists ("Summer Hits", "Party Mix", etc.) tend to contain more mainstream songs with better lyrics coverage. The v2 pipeline addresses this by raising the length range.

### Round 1 — Length Filter

Keep playlists whose **original length** (before any item filtering) falls within [30, 90] songs. This targets mainstream-length playlists while excluding very long playlists (> 90) whose songs may be too diverse to provide a coherent training signal.

| Metric | Value |
|--------|-------|
| Input playlists | 20,000 |
| Kept playlists | 9,104 |

### Rounds 2–N — Popularity + Audio Filter (Iterative)

Within the R1 playlists, repeatedly apply:
1. **Frequency filter**: keep only songs that appear in ≥ 10 playlists (within the current working set)
2. **Audio filter**: keep only songs that have a downloaded MP3 file
3. **Length recheck**: drop playlists where the filtered song count falls outside [10, 60]

Iterate until the playlist set no longer changes. This converges in 8 rounds.

Frequency threshold of 10 (vs. v1's threshold of 3) ensures every retained song appears in at least 10 playlists, meaningfully improving mainstream representativeness and lyrics API coverage.

| Iteration | Playlists | Unique Songs |
|-----------|-----------|--------------|
| R1 (length filter) | 9,104 | — |
| After iter 1 | 6,850 | 6,946 |
| After iter 2 | 6,628 | 5,208 |
| After iter 3 | 6,600 | 5,138 |
| After iter 4 | 6,592 | 5,128 |
| After iter 5 | 6,587 | 5,125 |
| After iter 6 | 6,586 | 5,122 |
| After iter 7 | 6,585 | 5,119 |
| **After iter 8 (converged)** | **6,585** | **5,119** |

**Converged output is the canonical v2 dataset** stored in `data/playlists/mpd_subset/`.

---

## Asset Acquisition

### Audio (yt-dlp + YouTube)

Songs were downloaded as MP3 via `yt-dlp` using the search query `"{title} {artist}"`. All 5,119 v2 songs were covered by the earlier broader download (which targeted 7,655 R3-era songs):

```
v2 audio coverage: 5,119 / 5,119 (100%)
```

Audio files stored in `data/audio/spotify/{item_id}.mp3`.

### Lyrics (LRCLIB + Genius)

Fetched via two sources in sequence:
1. **[LRCLIB](https://lrclib.net)** — free public API, ~6M tracks, no key required
2. **Genius** — fallback for LRCLIB misses, requires `GENIUS_ACCESS_TOKEN`

```bash
python scripts/fetch_lyrics.py \
    --metadata /path/to/metadata.json \
    --ids-file data/playlists/mpd_subset/item_ids.txt \
    --output   data/lyrics/spotify/ \
    --resume
```

The v2 freq ≥ 10 filter selects more mainstream songs, yielding meaningfully higher lyrics coverage than the v1 dataset (~42–46% vs ~21%):

```
Coverage (in progress): ~2,100 / 5,119  (~41%)
```

Output: `data/lyrics/spotify/{item_id}.txt` (plain text, LRC timestamps stripped).

### Audio Features (librosa — local extraction)

Extracted directly from downloaded MP3 files using `librosa`:

| Feature | Method |
|---------|--------|
| Tempo (BPM) | `librosa.beat.beat_track` |
| Key | Chromagram (CQT) + Krumhansl–Schmuckler key profiles |
| Energy | 75th-percentile RMS |
| Mood | Heuristic from tempo × energy × spectral centroid |
| Duration | `librosa.get_duration` |

```
Coverage: 5,119 / 5,119  (100% — all v2 songs analyzed in prior R3 run)
```

Output: `data/tags/audio_features.json`.

### Genre Tags (MusicBrainz)

Fetched via the MusicBrainz public JSON API (no key, 1 req/s rate limit):

```bash
python scripts/fetch_musicbrainz_tags.py \
    --metadata /path/to/metadata.json \
    --ids-file data/playlists/mpd_subset/item_ids.txt \
    --output   data/tags/musicbrainz_tags.json
```

Output: `data/tags/musicbrainz_tags.json` — per-item genre/style folksonomy tags.

---

## Dataset Freeze

Once lyrics and MusicBrainz fetches complete, run the freeze pipeline:

```bash
# Step 1 — Cross-validate assets, produce catalog.json
python scripts/build_final_catalog.py \
    --ids-file   data/playlists/mpd_subset/item_ids.txt \
    --metadata   /path/to/metadata.json \
    --audio-dir  data/audio/spotify/ \
    --lyrics-dir data/lyrics/spotify/ \
    --output     data/dataset/

# Step 2 — Filter playlists to complete items, split train/val/test
python scripts/freeze_dataset.py \
    --catalog   data/dataset/catalog.json \
    --playlists data/playlists/mpd_subset/playlists.txt \
    --output    data/dataset/ \
    --min-complete 5

# Step 3 — Build catalog_metadata.json for pipeline use
python src/00_data_schema/build_catalog_metadata.py \
    --catalog    data/dataset/catalog.json \
    --output     data/dataset/catalog_metadata.json \
    --lyrics-dir data/lyrics/spotify/
```

### Expected Final Statistics

| Metric | Expected |
|--------|----------|
| Items with audio | 5,119 (100%) |
| Items with lyrics | ~2,200–2,400 (~43–47%) |
| Clean playlists (≥ 5 songs with audio+lyrics) | ~4,000–5,000 |
| Train / Val / Test split | 80 / 10 / 10 |

---

## Output Layout

```
data/
  playlists/
    mpd_subset/
      playlists.txt       — 6,585 playlists (backbone format)
      item_ids.txt        — 5,119 unique item IDs
      item_freq.json      — per-item subset-internal frequency
      stats.json          — filtering summary

  audio/spotify/
      {item_id}.mp3       — downloaded audio (5,119 files)
      download_log.jsonl  — per-item download status + query

  lyrics/spotify/
      {item_id}.txt       — plain-text lyrics (~2,200+ files)
      fetch_log.jsonl     — per-item fetch status + source

  tags/
      audio_features.json     — librosa features per item (5,119)
      musicbrainz_tags.json   — genre/style tags per item

  final/                  — produced by freeze pipeline
      catalog.json        — complete per-item record
      complete_ids.txt    — items with audio + lyrics
      playlists_clean.txt — playlists filtered to complete items
      splits/
          train.txt / val.txt / test.txt
      dataset_card.json   — frozen dataset statistics
      catalog_metadata.json — pipeline-ready format
```

---

## Filtering Summary

| Stage | Playlists | Songs | Note |
|-------|-----------|-------|------|
| Raw MPD backbone | 20,000 | 254,155 | DDBC preprocessed |
| R1: original length 30–90 | 9,104 | — | targets mainstream-length playlists |
| After convergence (8 iters) | 6,585 | 5,119 | freq ≥ 10, audio-verified, filtered len 10–60 |
| Audio coverage | 6,585 | 5,119 (100%) | all songs have MP3 |
| Lyrics coverage (est.) | — | ~2,300 (~45%) | LRCLIB + Genius |
| **Final freeze (expected)** | **~4,500** | **~5,119 audio / ~2,300 lyrics** | **after freeze pipeline** |
