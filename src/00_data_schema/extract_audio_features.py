#!/usr/bin/env python3
"""scripts/extract_audio_features.py — Extract audio features locally using librosa.

For each downloaded audio file, extracts:
  tempo      : BPM (beat tracker)
  key        : e.g. "C# minor" (chromagram-based)
  energy     : RMS energy (0–1 normalized)
  mood       : heuristic from tempo + energy + spectral centroid
  duration_s : track length in seconds

No API key or external service needed — runs fully offline on local audio files.

Writes:
  data/tags/audio_features.json    — {"item_id": {tempo, key, energy, mood, duration_s}}
  data/tags/audio_features_coverage.json

Usage
-----
pip install librosa numpy

python scripts/extract_audio_features.py \\
    --audio-dir  data/audio/spotify/ \\
    --ids-file   data/playlists/r3_10_30_freq3/item_ids.txt \\
    --output     data/tags/audio_features.json \\
    --workers    4
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".webm", ".opus", ".ogg"}

_KEY_NAMES   = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_MINOR_PROFILE = [5.0, 2.0, 3.5, 4.5, 4.0, 4.0, 2.5, 4.5, 4.0, 3.5, 1.5, 4.0]
_MAJOR_PROFILE = [6.4, 2.2, 3.9, 2.9, 5.2, 5.4, 2.1, 6.2, 3.4, 4.3, 2.5, 3.3]


def _find_audio(audio_dir: Path, item_id: str) -> Path | None:
    for p in audio_dir.glob(f"{item_id}.*"):
        if p.suffix.lower() in _AUDIO_EXTS:
            return p
    return None


def _estimate_key(chroma_mean) -> tuple[str, str]:
    """Return (key_name, mode_str) from mean chroma vector."""
    import numpy as np
    best_score = -999.0
    best_key   = 0
    best_mode  = "major"
    for root in range(12):
        rotated = np.roll(chroma_mean, -root)
        score_major = float(np.dot(rotated, _MAJOR_PROFILE))
        score_minor = float(np.dot(rotated, _MINOR_PROFILE))
        if score_major > best_score:
            best_score = score_major
            best_key   = root
            best_mode  = "major"
        if score_minor > best_score:
            best_score = score_minor
            best_key   = root
            best_mode  = "minor"
    return _KEY_NAMES[best_key], best_mode


def _infer_mood(tempo: float, energy: float, spectral_centroid: float) -> str:
    """Simple heuristic mood from audio features."""
    bright = spectral_centroid > 2500   # Hz
    fast   = tempo > 120
    loud   = energy > 0.15

    if fast and loud and bright:   return "euphoric"
    if fast and loud and not bright: return "aggressive"
    if not fast and loud:          return "intense"
    if not fast and not loud and not bright: return "melancholic"
    if not fast and not loud and bright:     return "calm"
    return "neutral"


def analyze_one(args_tuple: tuple) -> tuple[str, dict | None]:
    item_id, audio_path_str = args_tuple
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(audio_path_str, sr=22050, mono=True, duration=60)

        tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo_arr.item() if hasattr(tempo_arr, "item") else tempo_arr)

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_name, mode_str = _estimate_key(chroma.mean(axis=1))

        rms = librosa.feature.rms(y=y)
        energy = float(np.percentile(rms, 75))   # 75th pct more robust than mean

        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_hz = float(sc.mean())

        duration_s = librosa.get_duration(y=y, sr=sr)

        mood = _infer_mood(tempo, energy, centroid_hz)

        return item_id, {
            "tempo":        round(tempo, 1),
            "key":          f"{key_name} {mode_str}",
            "energy":       round(energy, 4),
            "spectral_centroid_hz": round(centroid_hz, 1),
            "mood":         mood,
            "duration_s":   round(duration_s, 1),
        }
    except Exception as exc:
        return item_id, {"error": str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--ids-file",  required=True,
                        help="One item_id per line")
    parser.add_argument("--output",    default="data/tags/audio_features.json")
    parser.add_argument("--workers",   type=int, default=2,
                        help="Parallel workers (default: 2; set to CPU count for speed)")
    args = parser.parse_args()

    # Check librosa is installed
    try:
        import librosa  # noqa: F401
    except ImportError:
        print("[features] ERROR: install librosa first:  pip install librosa")
        raise SystemExit(1)

    audio_dir = Path(args.audio_dir)
    ids = [l.strip() for l in open(args.ids_file) if l.strip()]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if out_path.is_file():
        existing = json.load(open(out_path))
        print(f"[features] Resuming — {len(existing):,} already done")

    results = dict(existing)

    # Build work list: only items with an audio file not yet processed
    todo: list[tuple[str, str]] = []
    n_no_audio = 0
    for iid in ids:
        if iid in results:
            continue
        p = _find_audio(audio_dir, iid)
        if p:
            todo.append((iid, str(p)))
        else:
            n_no_audio += 1

    print(f"[features] Target        : {len(ids):,}")
    print(f"[features] No audio file : {n_no_audio:,}")
    print(f"[features] To analyze    : {len(todo):,}  (workers={args.workers})")

    done = 0
    save_every = 100

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(analyze_one, t): t[0] for t in todo}
        for fut in as_completed(futures):
            item_id, feat = fut.result()
            results[item_id] = feat
            done += 1
            if done % save_every == 0 or done == len(todo):
                out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
                pct = (len(existing) + done) / len(ids) * 100
                n_err = sum(1 for v in results.values() if v and "error" in v)
                print(f"  [{len(existing)+done:>6}/{len(ids)}] {pct:4.1f}%  errors={n_err}")

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    n_ok  = sum(1 for v in results.values() if v and "error" not in v)
    n_err = sum(1 for v in results.values() if v and "error" in v)
    cov_path = out_path.with_name("audio_features_coverage.json")
    cov_path.write_text(json.dumps({
        "total": len(ids), "analyzed": n_ok,
        "errors": n_err, "no_audio": n_no_audio,
        "coverage_rate": round(n_ok / len(ids), 4),
    }, indent=2))

    print(f"\n[features] Done. {n_ok:,} analyzed, {n_err} errors, {n_no_audio} missing audio")
    print(f"[features] Written to {out_path}")


if __name__ == "__main__":
    main()
