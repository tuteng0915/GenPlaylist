"""cue_io.py — atomic file writes for concurrency-safe caching.

Writing to a temp file and then os.replace()-ing it into place is atomic on the
same filesystem, so a concurrent reader never sees a half-written file. This lets
multiple run_compare.py processes share the extraction / LLM / embedding caches
without corrupting each other.

Windows note: unlike POSIX rename(), Windows' os.replace() (MoveFileEx) can
transiently raise PermissionError (WinError 5) when two processes race to
replace the SAME destination path at nearly the same instant — e.g. two parallel
run_compare.py runs both cache-missing on the same song's LLM prompt and both
trying to write the same content-hashed cache file. Cache entries are always
written whole (temp file, then replace) and keyed by content hash, so if `path`
already exists after a failed replace, another process's write already landed
there with equivalent content — _replace_or_yield treats that as success rather
than erroring, and only retries for the genuinely transient case (e.g. a
momentary AV/indexer lock).
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import numpy as np


def _safe_rm(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _tmp(dirpath: str, suffix: str):
    os.makedirs(dirpath or ".", exist_ok=True)
    return tempfile.mkstemp(dir=dirpath or ".", suffix=suffix)


def _replace_or_yield(tmp: str, path: str, attempts: int = 5, base_delay: float = 0.05) -> None:
    """os.replace(tmp, path), tolerant of concurrent same-destination writers.

    If a PermissionError hits and `path` already exists, another process's
    equivalent write already succeeded — drop our temp file and return instead
    of erroring. Otherwise retry with backoff a few times before giving up.
    """
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:
            last_exc = exc
            if os.path.isfile(path):
                _safe_rm(tmp)
                return
            time.sleep(base_delay * (2 ** attempt))
    _safe_rm(tmp)
    raise last_exc


def atomic_write_json(path: str, obj) -> None:
    d = os.path.dirname(path) or "."
    fd, tmp = _tmp(d, ".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        _replace_or_yield(tmp, path)
    except BaseException:
        _safe_rm(tmp)
        raise


def atomic_write_text(path: str, text: str) -> None:
    d = os.path.dirname(path) or "."
    fd, tmp = _tmp(d, ".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        _replace_or_yield(tmp, path)
    except BaseException:
        _safe_rm(tmp)
        raise


def atomic_save_npy(path: str, arr: np.ndarray) -> None:
    d = os.path.dirname(path) or "."
    fd, tmp = _tmp(d, ".npy")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, arr)          # file-object form: no extension munging
        _replace_or_yield(tmp, path)
    except BaseException:
        _safe_rm(tmp)
        raise
