"""encoders.py — Text encoder wrappers with a unified .encode() interface.

Every encoder here implements:
    encode(text_or_texts, normalize_embeddings=True) -> np.ndarray
        single str  → (dim,)  float32, L2-normalised
        list[str]   → (N, dim) float32, L2-normalised

This lets normalizer.py call any encoder through the same embed_text_query() path
without knowing which backend is in use.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# SentenceTransformer wrapper
# ---------------------------------------------------------------------------

class SentenceTransformerEncoder:
    """Wraps SentenceTransformer with explicit naming for comparison logging."""

    NAME = "sentence_transformer"
    MODEL_ID = "all-MiniLM-L6-v2"
    DIM = 384

    def __init__(self, model_id: str = MODEL_ID):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_id)
        self.model_id = model_id

    def encode(
        self,
        texts,
        normalize_embeddings: bool = True,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        **_,
    ) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# CLAP text encoder wrapper
# ---------------------------------------------------------------------------

class CLAPAudioEncoder:
    """CLAP audio encoder (laion/clap-htsat-unfused) — encodes audio files into 512-dim vectors.

    Output is in the SAME embedding space as CLAPTextEncoder, enabling direct cosine
    comparison between a text query and audio embeddings without any projection.

    Requires: librosa (pip install librosa)
    """

    NAME = "clap_audio"
    MODEL_ID = "laion/clap-htsat-unfused"
    DIM = 512
    SAMPLE_RATE = 48_000  # CLAP was trained at 48 kHz

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        import torch
        from transformers import ClapAudioModelWithProjection, AutoFeatureExtractor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClapAudioModelWithProjection.from_pretrained(model_id).to(self.device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model.eval()
        self.model_id = model_id

    def encode_files(
        self,
        audio_paths: list[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> tuple[np.ndarray, list[str]]:
        """Encode a list of audio file paths.

        Returns
        -------
        embs       : (N, 512) float32 array for successfully loaded files
        good_paths : list[str] of paths that encoded without error (same order as embs rows)
        """
        import torch
        import librosa

        embs_list: list[np.ndarray] = []
        good_paths: list[str] = []

        iterable = audio_paths
        if show_progress_bar:
            from tqdm import tqdm
            iterable = tqdm(audio_paths, desc="CLAP audio encoding")

        for path in iterable:
            try:
                audio, _ = librosa.load(path, sr=self.SAMPLE_RATE, mono=True)
                inputs = self.feature_extractor(
                    raw_speech=audio,
                    sampling_rate=self.SAMPLE_RATE,
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    emb = self.model(**inputs).audio_embeds.cpu().numpy()  # (1, 512)
                embs_list.append(emb[0])
                good_paths.append(path)
            except Exception as exc:
                print(f"  [WARN] skipping {path}: {exc}")

        if not embs_list:
            return np.zeros((0, self.DIM), dtype=np.float32), []

        embs = np.stack(embs_list).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs / (norms + 1e-9)
        return embs, good_paths


class CLAPTextEncoder:
    """CLAP text encoder (laion/clap-htsat-unfused) — text-only, no audio files needed.

    Encodes text into a 512-dim music-aware joint space trained on (audio, caption) pairs.
    Better music-domain alignment than general SentenceTransformer because the model
    learned to associate natural-language descriptions with audio content.

    Uses only the text branch — ClapTextModelWithProjection + tokenizer.
    Does NOT require audio files or the audio encoder branch.
    """

    NAME = "clap_text"
    MODEL_ID = "laion/clap-htsat-unfused"
    DIM = 512
    MAX_LENGTH = 77  # CLAP text encoder token limit

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        import torch
        from transformers import ClapTextModelWithProjection, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClapTextModelWithProjection.from_pretrained(model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model.eval()
        self.model_id = model_id

    def encode(
        self,
        texts,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **_,
    ) -> np.ndarray:
        import torch

        single = isinstance(texts, str)
        if single:
            texts = [texts]

        all_embs: list[np.ndarray] = []
        indices = range(0, len(texts), batch_size)

        if show_progress_bar:
            from tqdm import tqdm
            indices = tqdm(indices, desc="CLAP encoding")

        for i in indices:
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.MAX_LENGTH,
            ).to(self.device)

            with torch.no_grad():
                embs = self.model(**inputs).text_embeds.cpu().numpy()  # (B, 512)

            if normalize_embeddings:
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                embs = embs / (norms + 1e-9)

            all_embs.append(embs.astype(np.float32))

        result = np.concatenate(all_embs, axis=0)
        return result[0] if single else result
