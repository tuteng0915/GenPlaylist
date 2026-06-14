#!/usr/bin/env python3
"""scripts/fetch_spotify_tags.py — Fetch audio features and genre per track from Spotify API.

For each item:
  1. Search Spotify by title + artist → resolve to a Spotify track URI
  2. Batch-call audio_features() (100 URIs/call) → tempo, key, mode, energy, valence, danceability
  3. Call artist() → genres (first genre string)

Derives human-readable fields:
  key_str  : e.g. "C# minor"
  mood     : "euphoric" | "calm" | "aggressive" | "melancholic" | "neutral"

Writes:
  data/tags/spotify_tags.json     — {"item_id": {tempo, key, key_str, mode, energy,
                                                  valence, danceability, genre, mood}}
  data/tags/spotify_coverage.json

Usage
-----
export SPOTIFY_CLIENT_ID=...
export SPOTIFY_CLIENT_SECRET=...

python scripts/fetch_spotify_tags.py \\
    --metadata  /path/to/metadata.json \\
    --ids-file  data/playlists/r3_10_30_freq3/item_ids.txt \\
    --output    data/tags/spotify_tags.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

_META_RE = re.compile(r"^'(.+)'\s+by\s+(.+?)\s+in\s+album'(.+)'$")
_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _parse_meta(raw: str) -> tuple[str, str]:
    m = _META_RE.match(raw.strip())
    if m:
        return m.group(1), m.group(2)
    return raw.strip(), ""


def _key_string(key: int, mode: int) -> str:
    if key < 0:
        return ""
    return f"{_KEY_NAMES[key % 12]} {'major' if mode else 'minor'}"


def _infer_mood(valence: float, energy: float) -> str:
    if valence > 0.6 and energy > 0.6:
        return "euphoric"
    if valence > 0.6 and energy <= 0.6:
        return "calm"
    if valence <= 0.4 and energy > 0.6:
        return "aggressive"
    if valence <= 0.4 and energy <= 0.6:
        return "melancholic"
    return "neutral"


class SpotifyClient:
    _TOKEN_URL = "https://accounts.spotify.com/api/token"
    _API_BASE  = "https://api.spotify.com/v1"

    def __init__(self, client_id: str, client_secret: str):
        self._id     = client_id
        self._secret = client_secret
        self._token  = ""
        self._expiry = 0.0
        self._refresh_token()

    def _refresh_token(self) -> None:
        r = requests.post(
            self._TOKEN_URL,
            auth=HTTPBasicAuth(self._id, self._secret),
            data={"grant_type": "client_credentials"},
            timeout=10,
        )
        r.raise_for_status()
        d = r.json()
        self._token  = d["access_token"]
        self._expiry = time.time() + d["expires_in"] - 30

    def _headers(self) -> dict:
        if time.time() >= self._expiry:
            self._refresh_token()
        return {"Authorization": f"Bearer {self._token}"}

    def search_track(self, title: str, artist: str) -> dict | None:
        q = f"track:{title} artist:{artist}" if artist else f"track:{title}"
        r = requests.get(f"{self._API_BASE}/search",
                         headers=self._headers(),
                         params={"q": q, "type": "track", "limit": 1},
                         timeout=10)
        if r.status_code == 429:
            retry = int(r.headers.get("Retry-After", 5))
            time.sleep(retry)
            return self.search_track(title, artist)
        r.raise_for_status()
        items = r.json().get("tracks", {}).get("items", [])
        return items[0] if items else None

    def audio_features_batch(self, track_ids: list[str]) -> list[dict | None]:
        """Fetch audio features for up to 100 track IDs."""
        r = requests.get(f"{self._API_BASE}/audio-features",
                         headers=self._headers(),
                         params={"ids": ",".join(track_ids)},
                         timeout=15)
        if r.status_code == 429:
            retry = int(r.headers.get("Retry-After", 5))
            time.sleep(retry)
            return self.audio_features_batch(track_ids)
        r.raise_for_status()
        return r.json().get("audio_features", [])

    def artist_genre(self, artist_id: str) -> str:
        r = requests.get(f"{self._API_BASE}/artists/{artist_id}",
                         headers=self._headers(), timeout=10)
        if r.status_code == 429:
            retry = int(r.headers.get("Retry-After", 5))
            time.sleep(retry)
            return self.artist_genre(artist_id)
        r.raise_for_status()
        genres = r.json().get("genres", [])
        return genres[0] if genres else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata",      required=True)
    parser.add_argument("--ids-file",      required=True,
                        help="One item_id per line")
    parser.add_argument("--output",        default="data/tags/spotify_tags.json")
    parser.add_argument("--sleep",         type=float, default=0.1,
                        help="Sleep between search calls (default: 0.1)")
    parser.add_argument("--client-id",     default=None)
    parser.add_argument("--client-secret", default=None)
    args = parser.parse_args()

    client_id     = args.client_id     or os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = args.client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("[spotify] ERROR: set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET (or pass --client-id/--client-secret)")
        raise SystemExit(1)

    raw_meta: dict = json.load(open(args.metadata))
    ids = [l.strip() for l in open(args.ids_file) if l.strip()]
    print(f"[spotify] Items to fetch : {len(ids):,}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if out_path.is_file():
        existing = json.load(open(out_path))
        print(f"[spotify] Resuming — {len(existing):,} already fetched")

    results = dict(existing)
    sp = SpotifyClient(client_id, client_secret)

    todo = [iid for iid in ids if iid not in results]
    print(f"[spotify] Remaining      : {len(todo):,}")

    # Phase 1: resolve track IDs + artist IDs via search
    track_map: dict[str, dict] = {}   # item_id → {track_id, artist_id}
    n_found = n_miss = 0

    for i, item_id in enumerate(todo, 1):
        raw = raw_meta.get(item_id, "")
        title, artist = _parse_meta(raw)
        track = sp.search_track(title, artist)
        if track:
            track_map[item_id] = {
                "track_id":  track["id"],
                "artist_id": track["artists"][0]["id"] if track.get("artists") else "",
            }
            n_found += 1
        else:
            n_miss += 1

        if i % 200 == 0 or i == len(todo):
            print(f"  [search {i:>6}/{len(todo)}]  found={n_found}  miss={n_miss}")
            # Save intermediate results
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

        time.sleep(args.sleep)

    # Phase 2: batch audio features (100 at a time)
    items_with_track = list(track_map.items())
    for batch_start in range(0, len(items_with_track), 100):
        batch = items_with_track[batch_start:batch_start + 100]
        track_ids = [v["track_id"] for _, v in batch]
        features  = sp.audio_features_batch(track_ids)

        for (item_id, meta), feat in zip(batch, features):
            if not feat:
                results[item_id] = {}
                continue
            key     = feat.get("key", -1)
            mode    = feat.get("mode", 1)
            valence = feat.get("valence", 0.5)
            energy  = feat.get("energy", 0.5)
            results[item_id] = {
                "tempo":       round(feat.get("tempo", 0), 1),
                "key":         key,
                "mode":        mode,
                "key_str":     _key_string(key, mode),
                "energy":      round(energy, 3),
                "valence":     round(valence, 3),
                "danceability": round(feat.get("danceability", 0), 3),
                "genre":       "",   # filled in phase 3
                "mood":        _infer_mood(valence, energy),
                "_track_id":   meta["track_id"],
                "_artist_id":  meta["artist_id"],
            }

        if batch_start % 1000 == 0:
            print(f"  [features {batch_start:>6}/{len(items_with_track)}]")
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Phase 3: artist genres (one call per unique artist)
    artist_genre_cache: dict[str, str] = {}
    for item_id, entry in results.items():
        if not entry or not entry.get("_artist_id"):
            continue
        aid = entry["_artist_id"]
        if aid not in artist_genre_cache:
            artist_genre_cache[aid] = sp.artist_genre(aid)
            time.sleep(0.05)
        entry["genre"] = artist_genre_cache[aid]

    # Clean internal fields
    for entry in results.values():
        entry.pop("_track_id",  None)
        entry.pop("_artist_id", None)

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    fetched = sum(1 for v in results.values() if v)
    cov_path = out_path.with_name("spotify_coverage.json")
    cov_path.write_text(json.dumps({
        "total": len(ids),
        "search_hit": n_found,
        "search_miss": n_miss,
        "coverage_rate": round(fetched / len(ids), 4),
    }, indent=2))

    print(f"\n[spotify] Done. Coverage: {fetched}/{len(ids)} ({fetched/len(ids)*100:.1f}%)")
    print(f"[spotify] Written to {out_path}")


if __name__ == "__main__":
    main()
