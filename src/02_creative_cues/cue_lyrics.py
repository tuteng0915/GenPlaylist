"""cue_lyrics.py — lyrics preprocessing modes shared across the pipeline.

The text a song contributes to extraction / assignment / retrieval can be shaped
in several ways. This is a single control point so every stage sees the same
processed lyrics, and so the choice can be A/B compared:

  mode = "cap"        first `cap` characters (positional truncation; the old default)
  mode = "full"       entire lyric, no truncation
  mode = "dedup"      drop repeated lines (choruses), keep distinct content, then cap
  mode = "summarize"  LLM compresses lyrics to a faithful thematic summary (cached)

'dedup' is usually the best cheap option: it removes chorus repetition (which
inflates TF-IDF and blurs KeyBERT's document embedding) and packs more *distinct*
imagery into the same budget, with no hallucination. 'summarize' concentrates
theme most but costs an API call per song and can introduce words not in the
lyrics, which can hurt grounding — use it as an experiment, not the default.

Note: this shapes the text fed INTO cue extraction/assignment. Level 3
reconstruction always scores against the ORIGINAL lyrics, not the processed text.
"""

from __future__ import annotations

import re

DEFAULT_CAP = 2000
MODES = ("cap", "full", "dedup", "summarize")

_SUMMARY_SYSTEM = "You compress song lyrics into a short, faithful thematic summary."


def _dedup_lines(text: str) -> str:
    """Keep the first occurrence of each distinct (normalized) line, in order."""
    seen: set[str] = set()
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        key = re.sub(r"\s+", " ", stripped.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(stripped)
    return "\n".join(out)


def preprocess_lyrics(text: str, mode: str = "cap", cap: int = DEFAULT_CAP) -> str:
    """Return the processed lyric text for the given mode. Empty in -> empty out."""
    if not text:
        return ""
    if mode == "full":
        return text
    if mode == "cap":
        return text[:cap]
    if mode == "dedup":
        return _dedup_lines(text)[:cap]
    if mode == "summarize":
        import cue_clients
        prompt = (
            "Summarize the concrete imagery, themes, cultural references, and "
            "narrative motifs of these song lyrics in 3-5 sentences. Use only "
            "content present in the lyrics; do not invent details.\n\n"
            f"{text[:4000]}"
        )
        return cue_clients.chat(prompt, system=_SUMMARY_SYSTEM,
                                temperature=0.2, max_tokens=220)
    raise ValueError(f"Unknown lyrics mode '{mode}'. Choose from {MODES}.")


def cache_tag(mode: str, cap: int) -> str:
    """Short tag identifying the mode, for cache filenames (keeps modes separate)."""
    return f"{mode}{cap}" if mode in ("cap", "dedup") else mode
