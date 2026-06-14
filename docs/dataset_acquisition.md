# Dataset Acquisition Guide

GenPlaylist uses the **Spotify Million Playlist Dataset (MPD)** as its single evaluation dataset.

Four categories of data are needed:

| Category | Used by | Status |
|----------|---------|--------|
| **Playlist structure** (item sequences, CLHE embeddings) | backbone diffusion model | ✅ in backbone |
| **Raw audio** (.mp3) | ACE-Step style reference, FAD evaluation | ⬜ download needed |
| **Song tags** (genre, mood, BPM, key) | `CatalogItem`, verbalization prompts | ⬜ fetch needed |
| **Lyrics** | `CatalogItem.lyric_excerpt`, WP-B cue mining | ⬜ fetch needed |

---

## Dataset — Spotify Million Playlist Dataset (MPD)

**Origin:** Spotify RecSys Challenge 2018.
**Scale:** 1M playlists, ~2.2M unique tracks.
**Backbone split:** 254,155 songs, 20,000 playlists (already preprocessed).

### What we already have

All backbone files are in `/Users/tut/migration_buffer/model/DDBC_seq/datasets/spotify/`:

```
count.json          # {'#I': 254155, '#B': 20000, ...}
metadata.json       # {"0": "'Title' by Artist in album'Album'", ...}
clhe_token.json     # {"0": [76, 296, 745, 769], ...}  — token IDs per item
clhe_weight.npy     # shape (768, 64) — merged RVQ codebook embeddings
train.txt / test.txt / bi_full.txt
```

### Getting the raw MPD (if a larger split is needed)

1. Register at https://www.aicrowd.com and accept MPD Terms of Use
2. Download from https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/dataset_files
3. Pre-processed version also available: https://huggingface.co/datasets/xhLiu/BundleConstruction
4. Preprocessing scripts: `databuild_utils/` in https://github.com/Rsalganik1123/LARP

---

## Part 1 — Raw audio

### Approach A — Spotify API 30-second previews (preferred)

Spotify provides official 30s preview clips per track via its Web API.
Match is exact; no search ambiguity.

**Setup:**
1. Register a free app: https://developer.spotify.com/dashboard
2. Store `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` in `.env`
3. `pip install spotipy requests`

```python
import os, requests, spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
))

def resolve_uri(title: str, artist: str) -> str | None:
    results = sp.search(q=f"track:{title} artist:{artist}", type="track", limit=1)
    items = results["tracks"]["items"]
    return items[0]["uri"] if items else None

def download_preview(uri: str, out_path: str) -> bool:
    url = sp.track(uri).get("preview_url")
    if not url:
        return False
    with open(out_path, "wb") as f:
        f.write(requests.get(url).content)
    return True
```

Limitations: ~30% of tracks have no `preview_url`; clips are 30s starting mid-track.
**TODO:** write `scripts/download_audio_spotify.py` with resume/log/retry like `download_audio.py`.

### Approach B — YouTube full audio (fallback for missing previews)

```bash
pip install yt-dlp && brew install ffmpeg   # macOS
pip install yt-dlp && apt install ffmpeg    # Linux
```

```bash
# Full test split (88k songs, ~700 MB mp3)
python scripts/download_audio.py \
    --metadata /Users/tut/migration_buffer/model/DDBC_seq/datasets/spotify/metadata.json \
    --ids-file /Users/tut/migration_buffer/model/DDBC_seq/datasets/spotify/test.txt \
    --output   data/audio/spotify/ \
    --format mp3 --sleep 1.5

# Retry failures
python scripts/download_audio.py \
    --metadata /Users/tut/migration_buffer/model/DDBC_seq/datasets/spotify/metadata.json \
    --ids-file data/audio/spotify/failed.txt \
    --output   data/audio/spotify/ --format mp3 --sleep 2.0
```

Both `--resume` and `--retry` are supported; safe to interrupt at any time.

**Storage estimates:**

| Format | Per song (avg 3.5 min) | 254k songs |
|--------|----------------------|------------|
| WAV 44.1 kHz stereo | ~40 MB | ~10 TB |
| MP3 320 kbps | ~8 MB | ~2 TB |
| Opus/WebM (no ffmpeg) | ~3 MB | ~760 GB |

Use `--format mp3` for large-scale runs.

---

## Part 2 — Song tags

### Spotify Web API audio features

```python
# Batch audio features — 100 URIs per call
features = sp.audio_features(["spotify:track:aaa", ...])
# Returns: tempo, key (0–11), mode (0=minor/1=major), danceability, energy, valence, ...

# Genre — on the artist object, not the track
artist = sp.artist(track["artists"][0]["id"])
genre  = artist["genres"][0] if artist["genres"] else ""
```

Key → string:
```python
KEY_NAMES  = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
def key_string(key: int, mode: int) -> str:
    return f"{KEY_NAMES[key]} {'major' if mode else 'minor'}"
```

Mood from `valence` + `energy`:
```python
def infer_mood(valence: float, energy: float) -> str:
    if valence > 0.6 and energy > 0.6:  return "euphoric"
    if valence > 0.6 and energy <= 0.6: return "calm"
    if valence <= 0.4 and energy > 0.6: return "aggressive"
    if valence <= 0.4 and energy <= 0.6: return "melancholic"
    return "neutral"
```

**TODO:** write `scripts/fetch_spotify_tags.py`:
1. Read `metadata.json`; resolve each title+artist to a Spotify URI via search
2. Batch-call `audio_features()` (100/call) and `artist()` for genre
3. Output `data/tags/spotify_tags.json`: `{"item_id": {"tempo":…, "key":…, "genre":…, "mood":…}}`
4. Merge into `catalog_metadata.json` via `build_catalog_metadata.py`

### Last.fm tags (supplement)

Useful as a cross-reference or for items where Spotify audio features are unavailable.

```python
import pylast  # pip install pylast
# Free key: https://www.last.fm/api/account/create
network = pylast.LastFMNetwork(api_key=os.getenv("LASTFM_API_KEY"))
tags = network.get_track("Artist", "Title").get_top_tags(limit=5)
# e.g. [('indie rock', 100), ('melancholic', 70), ...]
```

---

## Part 3 — Lyrics

### LRCLIB (primary — free, no API key)

```python
import requests

def get_lyrics_lrclib(title: str, artist: str) -> str | None:
    r = requests.get("https://lrclib.net/api/search",
                     params={"track_name": title, "artist_name": artist}, timeout=10)
    results = r.json()
    if results:
        return results[0].get("plainLyrics") or results[0].get("syncedLyrics")
    return None
```

### Genius API (fallback)

```bash
pip install lyricsgenius
# Token: https://genius.com/api-clients
```

```python
import lyricsgenius
genius = lyricsgenius.Genius(os.getenv("GENIUS_ACCESS_TOKEN"), quiet=True)
song = genius.search_song("Title", "Artist")
lyrics = song.lyrics if song else None
```

```bash
# Fetch test split lyrics (LRCLIB primary, Genius fallback)
python scripts/fetch_lyrics.py \
    --metadata /Users/tut/migration_buffer/model/DDBC_seq/datasets/spotify/metadata.json \
    --ids-file /Users/tut/migration_buffer/model/DDBC_seq/datasets/spotify/test.txt \
    --output   data/lyrics/spotify/
```

Observed coverage on first 10 items: **9/10 (90%)** via LRCLIB alone.

---

## Output directory layout

```
GenPlaylist_Code/
    data/
        audio/
            spotify/
                {item_id}.mp3       # downloaded audio, named by item_id
                download_log.jsonl  # status / query / path per attempt
                failed.txt          # item_ids that failed; re-feed to --ids-file for retry
        lyrics/
            spotify/
                {item_id}.txt       # plain-text lyrics (LRC timestamps stripped)
                coverage.json       # {"total": N, "found": M, "coverage_rate": ...}
                fetch_log.jsonl     # status / source (lrclib/genius/miss) per attempt
        tags/
            spotify_tags.json       # {"item_id": {"tempo":…, "key":…, "genre":…, "mood":…}}
```

---

## Recommended acquisition order

1. **Playlist structure** — already in backbone ✅
2. **Test-split audio + lyrics** — run in parallel; validates pipeline before committing to full catalog:
   ```bash
   cd /Users/tut/Documents/Research/GenPlaylist/GenPlaylist_Code

   # Terminal 1 — audio
   python scripts/download_audio.py \
       --metadata /Users/tut/migration_buffer/model/DDBC_seq/datasets/spotify/metadata.json \
       --ids-file /Users/tut/migration_buffer/model/DDBC_seq/datasets/spotify/test.txt \
       --output   data/audio/spotify/ --format mp3 --sleep 1.5

   # Terminal 2 — lyrics
   python scripts/fetch_lyrics.py \
       --metadata /Users/tut/migration_buffer/model/DDBC_seq/datasets/spotify/metadata.json \
       --ids-file /Users/tut/migration_buffer/model/DDBC_seq/datasets/spotify/test.txt \
       --output   data/lyrics/spotify/
   ```
3. **Song tags** — write and run `scripts/fetch_spotify_tags.py`; verify `CatalogItem` fields populate correctly
4. **Full catalog audio + tags + lyrics** — scale up after test split is validated
