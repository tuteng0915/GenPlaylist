"""pipeline/mock_run.py — End-to-end mock run of the full GenPlaylist pipeline.

Purpose
-------
Verify that every module boundary is correctly wired BEFORE real data or
real models are available.  All stubs (backbone, Qwen3, ACE-Step) are
replaced by local mock implementations that produce structurally correct
outputs without any GPU or API dependency.

What this tests
---------------
  ContextPrefix   → validate()         [00_data_schema]
  ContextPrefix   → _run_backbone()    [mock: random GeneratedItems]
  GeneratedItem   → verbalize()        [04_synthesis: real kNN + stub Qwen3]
  music_attrs     → synthesize()       [mock: writes a tiny silent .wav]
  audio_path      → SynthesisResult    [00_data_schema]

Run from repo root:
    python src/pipeline/mock_run.py

Expected output (no GPU, no API key required):
    [mock_run] === GenPlaylist end-to-end mock run ===
    [mock_run] catalog : 20 items, emb shape (20, 64)
    [mock_run] context : ['3', '7', '12', '1', '18'] (song_only)
    [mock_run] running 3 backbone samples ...
    [verbalization] _call_qwen3 is a stub ...
    [mock_run] sample 0 | attrs: STUB OUTPUT | audio: .../mock_0.wav
    [mock_run] sample 1 | attrs: STUB OUTPUT | audio: .../mock_1.wav
    [mock_run] sample 2 | attrs: STUB OUTPUT | audio: .../mock_2.wav
    [mock_run] 3/3 SynthesisResult objects passed validate()
    [mock_run] === PASS ===
"""

from __future__ import annotations

import os
import sys
import struct
import tempfile

_SRC = os.path.dirname(os.path.dirname(__file__))   # .../src/
sys.path.insert(0, os.path.join(_SRC, '00_data_schema'))
from schema import (                                # noqa: E402
    CatalogItem, ContextPrefix, GeneratedItem, SynthesisResult,
    CLHE_EMB_DIM, RQ_N_CODEBOOKS, RQ_CODEBOOK_SIZE,
)

import numpy as np

# Import production modules (already grounded in schema)
sys.path.insert(0, os.path.join(_SRC, '04_synthesis'))
import verbalization  # noqa: E402


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mock_catalog(n: int = 20) -> tuple[np.ndarray, list[CatalogItem]]:
    """Create n fake CatalogItems with random unit-normalized embeddings."""
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((n, CLHE_EMB_DIM)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9

    genres  = ["indie pop", "metal", "jazz", "electronic", "folk",
               "hip-hop", "classical", "ambient", "r&b", "punk"]
    moods   = ["melancholic", "energetic", "calm", "aggressive", "nostalgic",
               "euphoric", "dark", "playful", "romantic", "intense"]
    artists = [f"Artist_{i}" for i in range(n)]

    items = []
    for i in range(n):
        items.append(CatalogItem(
            item_id=str(i),
            title=f"Track {i}",
            artist=artists[i],
            genre=genres[i % len(genres)],
            mood=moods[i % len(moods)],
            tempo=80.0 + (i * 3) % 80,
            key=["C major", "A minor", "G major", "E minor"][i % 4],
            language="en",
            lyric_excerpt=f"Sample lyric excerpt for track {i}",
        ))
    return embs, items


def _mock_run_backbone(
    context_prefix: ContextPrefix,
    n_samples: int,
    catalog_embs: np.ndarray,
    catalog_metadata: list[CatalogItem],
) -> list[GeneratedItem]:
    """Return n_samples GeneratedItems with random (but valid) embeddings.

    μ_C and σ²_C are derived from the actual context items' embeddings,
    matching what the real backbone would compute via compute_playlist_structure().
    """
    rng = np.random.default_rng(seed=sum(int(x) for x in context_prefix.item_ids))

    # Compute real μ_C and σ²_C from context embedding rows
    seed_rows = [int(iid) for iid in context_prefix.item_ids if iid.isdigit()]
    seed_embs = catalog_embs[seed_rows]                          # (K, d)
    mu_c      = seed_embs.mean(axis=0)                          # (d,)
    diffs     = seed_embs - mu_c                                 # (K, d)
    sigma_c2  = float(np.mean(np.sum(diffs ** 2, axis=1)))      # scalar

    items = []
    for s in range(n_samples):
        # Random z_hat near μ_C (simulates diffusion output)
        z_hat = mu_c + rng.standard_normal(CLHE_EMB_DIM).astype(np.float32) * 0.3
        z_hat /= np.linalg.norm(z_hat) + 1e-9

        # Random RVQ codes
        rvq_codes     = tuple(int(rng.integers(0, RQ_CODEBOOK_SIZE)) for _ in range(RQ_N_CODEBOOKS))
        conflict_code = int(rng.integers(0, 10))

        items.append(GeneratedItem(
            rvq_codes=rvq_codes,
            conflict_code=conflict_code,
            z_hat_emb=z_hat,
            mu_c_emb=mu_c.copy(),
            sigma_c2=sigma_c2,
            cue_ids=[],
            sample_idx=s,
            context_prefix=context_prefix,
        ))
    return items


def _mock_synthesize(
    music_attributes: str,
    lyric_draft: str,
    audio_duration: int = 30,
    style_ref_audio_path=None,
    output_dir: str = "",
    filename: str | None = None,
) -> str:
    """Write a minimal valid .wav file and return its path."""
    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="genplaylist_mock_")
    os.makedirs(output_dir, exist_ok=True)

    fname = (filename or f"mock_{_mock_synthesize._counter}") + ".wav"
    _mock_synthesize._counter += 1
    out_path = os.path.join(output_dir, fname)

    # Write a minimal 1-second silent PCM WAV (44100 Hz, mono, 16-bit)
    n_samples_wav = 44100
    data_size     = n_samples_wav * 2        # 16-bit = 2 bytes/sample
    with open(out_path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))  # chunk size
        f.write(b"WAVE")
        # fmt sub-chunk
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH",
            16,       # sub-chunk size
            1,        # PCM
            1,        # mono
            44100,    # sample rate
            88200,    # byte rate
            2,        # block align
            16,       # bits per sample
        ))
        # data sub-chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)   # silence

    return os.path.abspath(out_path)

_mock_synthesize._counter = 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[mock_run] === GenPlaylist end-to-end mock run ===\n")

    # 1. Build mock catalog
    catalog_embs, catalog_metadata = _make_mock_catalog(n=20)
    print(f"[mock_run] catalog : {len(catalog_metadata)} items, "
          f"emb shape {catalog_embs.shape}")

    # 2. Build a ContextPrefix (5 songs from the mock catalog)
    ctx = ContextPrefix(
        item_ids=["3", "7", "12", "1", "18"],
        source="song_only",
        raw_input="mock context: 5 songs",
        items=[catalog_metadata[int(i)] for i in ["3", "7", "12", "1", "18"]],
    )
    ctx.validate()
    print(f"[mock_run] context : {ctx.item_ids} ({ctx.source})")

    # 3. Patch the synthesis module with the mock implementation
    #    (real verbalize() uses kNN + stub Qwen3, which is fine)
    import importlib.util
    _synthesis_path = os.path.join(_SRC, "04_synthesis", "synthesis.py")
    _synthesis_spec = importlib.util.spec_from_file_location("synthesis", _synthesis_path)
    synthesis_mod   = importlib.util.module_from_spec(_synthesis_spec)
    _synthesis_spec.loader.exec_module(synthesis_mod)
    synthesis_mod.synthesize = _mock_synthesize  # patch ACE-Step with our mock

    # 4. Run backbone mock
    n_samples = 3
    print(f"[mock_run] running {n_samples} backbone samples ...")
    generated_items = _mock_run_backbone(ctx, n_samples, catalog_embs, catalog_metadata)

    # 5. Verbalize + synthesize each candidate
    results: list[SynthesisResult] = []
    for item in generated_items:
        verb       = verbalization.verbalize(item, catalog_embs, catalog_metadata, k=5)
        audio_path = synthesis_mod.synthesize(
            music_attributes=verb["music_attributes"],
            lyric_draft=verb["lyric_draft"],
            audio_duration=30,
        )
        r = SynthesisResult(
            audio_path=audio_path,
            music_attributes=verb["music_attributes"],
            lyric_draft=verb["lyric_draft"],
            neighbors=verb["neighbors"],
            style_summary=verb["style_summary"],
            generated_item=item,
        )
        results.append(r)

    # 6. Print summary
    print()
    for r in results:
        s = r.generated_item.sample_idx
        print(f"[mock_run] sample {s}")
        print(f"           rvq_codes    : {r.generated_item.rvq_codes}")
        print(f"           sigma_c2     : {r.generated_item.sigma_c2:.4f}")
        print(f"           music_attrs  : {r.music_attributes!r}")
        print(f"           lyric_draft  : {r.lyric_draft!r}")
        print(f"           neighbors    : {[n.to_prompt_line() for n in r.neighbors[:2]]} ...")
        print(f"           audio_path   : {r.audio_path}")
        print()

    # 7. Validate all results
    passed = 0
    for r in results:
        try:
            r.validate()
            passed += 1
        except AssertionError as e:
            print(f"[mock_run] FAIL validate(): {e}")

    print(f"[mock_run] {passed}/{len(results)} SynthesisResult objects passed validate()")

    if passed == len(results):
        print("[mock_run] === PASS ===")
        return 0
    else:
        print("[mock_run] === FAIL ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
