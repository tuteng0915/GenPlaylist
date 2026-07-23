"""cue_clients.py — WP-B shared LLM + embedding clients (disk-memoized).

Both the `llm` extractor (cue_extractors.py) and the Level-2 decoder (cue_eval.py)
route through here, so every model call is:
  - configured in ONE place (swap OpenAI <-> local Ollama/Qwen3 here only),
  - memoized to disk by content hash (re-runs cost nothing, survive restarts).

Backends are imported lazily: this module imports fine with no API key and no
`openai` installed. A clear, actionable error is raised only when a call is
actually made without the backend configured.

Default backend: OpenAI
  - chat   : gpt-4o-mini      (cue extraction, lyric decoding)
  - embed  : text-embedding-3-small
Set OPENAI_API_KEY in the environment to enable.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# .env loader (no dependency): populate os.environ from project .env on import
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """Read KEY=VALUE lines from the project-root .env into os.environ.

    Walks up from this file to find a .env. Existing env vars are NOT overwritten,
    so a real shell export always wins over the file.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):  # walk up to the repo root
        candidate = os.path.join(here, ".env")
        if os.path.isfile(candidate):
            with open(candidate, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key, val = key.strip(), val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
            return
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent


_load_dotenv()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHAT_MODEL  = os.environ.get("CUE_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.environ.get("CUE_EMBED_MODEL", "text-embedding-3-small")

_CACHE_ROOT = os.path.join(os.path.dirname(__file__), "outputs", "_cache")
_LLM_CACHE   = os.path.join(_CACHE_ROOT, "llm")
_EMBED_CACHE = os.path.join(_CACHE_ROOT, "embeddings")

_client = None  # lazily-initialized OpenAI client


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:24]


def _get_openai():
    """Return a configured OpenAI client or raise an actionable error."""
    global _client
    if _client is not None:
        return _client
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it to use the 'llm' method:\n"
            "  PowerShell: $env:OPENAI_API_KEY=\"sk-...\"\n"
            "  bash      : export OPENAI_API_KEY=sk-...\n"
            "Or swap the backend in cue_clients.py to a local model."
        )
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai not installed. Run: pip install openai")
    _client = OpenAI()
    return _client


# ---------------------------------------------------------------------------
# Chat completion (memoized)
# ---------------------------------------------------------------------------

def chat(prompt: str, system: str = "", temperature: float = 0.3,
         max_tokens: int = 512, model: str = CHAT_MODEL) -> str:
    """Single-turn chat completion, cached on (model, system, prompt, temp)."""
    os.makedirs(_LLM_CACHE, exist_ok=True)
    key = _hash(model, system, prompt, f"{temperature}", f"{max_tokens}")
    path = os.path.join(_LLM_CACHE, f"{key}.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)["response"]

    client = _get_openai()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""

    import cue_io
    cue_io.atomic_write_json(path, {"prompt": prompt, "system": system, "response": text})
    return text


def _chat_cache_path(prompt: str, system: str, temperature: float,
                     max_tokens: int, model: str) -> str:
    key = _hash(model, system, prompt, f"{temperature}", f"{max_tokens}")
    return os.path.join(_LLM_CACHE, f"{key}.json")


def is_chat_cached(prompt: str, system: str = "", temperature: float = 0.3,
                   max_tokens: int = 512, model: str = CHAT_MODEL) -> bool:
    """True if chat() would hit the disk cache for these exact args (no API call)."""
    return os.path.isfile(_chat_cache_path(prompt, system, temperature, max_tokens, model))


def cache_chat_response(prompt: str, response: str, system: str = "",
                        temperature: float = 0.3, max_tokens: int = 512,
                        model: str = CHAT_MODEL) -> None:
    """Write a response into the chat cache under the same key chat() uses.

    Lets the Batch API populate the cache so a later chat() call is free.
    """
    path = _chat_cache_path(prompt, system, temperature, max_tokens, model)
    import cue_io
    cue_io.atomic_write_json(path, {"prompt": prompt, "system": system, "response": response})


# ---------------------------------------------------------------------------
# Embeddings (memoized, batched)
# ---------------------------------------------------------------------------

# OpenAI embedding models cap input at 8192 tokens. Truncate defensively by chars
# (~3 chars/token is conservative) so a long lyric never exceeds the limit.
_MAX_EMBED_CHARS = 20000


def embed(texts: list[str], model: str = EMBED_MODEL) -> np.ndarray:
    """Embed texts via OpenAI, L2-normalized. Per-text disk cache; batches misses.

    Inputs longer than the model's token limit are truncated by character length.
    """
    os.makedirs(_EMBED_CACHE, exist_ok=True)
    texts = [t[:_MAX_EMBED_CHARS] for t in texts]
    out: list[np.ndarray | None] = [None] * len(texts)
    misses: list[int] = []

    for i, t in enumerate(texts):
        path = os.path.join(_EMBED_CACHE, f"{_hash(model, t)}.npy")
        if os.path.isfile(path):
            out[i] = np.load(path)
        else:
            misses.append(i)

    if misses:
        client = _get_openai()
        BATCH = 256
        for s in range(0, len(misses), BATCH):
            chunk = misses[s:s + BATCH]
            resp = client.embeddings.create(
                model=model, input=[texts[i] for i in chunk]
            )
            import cue_io
            for j, i in enumerate(chunk):
                v = np.asarray(resp.data[j].embedding, dtype=np.float32)
                v /= (np.linalg.norm(v) or 1.0)
                out[i] = v
                cue_io.atomic_save_npy(os.path.join(_EMBED_CACHE, f"{_hash(model, texts[i])}.npy"), v)

    return np.vstack(out) if out else np.zeros((0, 1), dtype=np.float32)


def is_available() -> bool:
    """True if the LLM backend can be used (key present)."""
    return bool(os.environ.get("OPENAI_API_KEY"))
