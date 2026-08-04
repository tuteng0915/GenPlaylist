"""synthesis.py — ACE-Step music synthesis wrapper for GenPlaylist.

Adapted from VibeMus/pipeline.py and VibeMus/tools.py.
Exposes a pure-function interface: given music_attributes (str)
and lyric_draft (str), synthesize a waveform and return the path.
Pipeline loaded once as module-level singleton.
"""

import os
import sys
import re
from typing import Optional

# Add VibeMus ace-step to path
"""sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', 'reference', 'VibeMus', 'src', 'ace-step'
))"""

if os.environ.get("ACE_STEP_PATH"):
    sys.path.insert(0, os.environ["ACE_STEP_PATH"])

# ---------------------------------------------------------------------------
# Pipeline singleton (loaded once on import)
# ---------------------------------------------------------------------------

_pipe = None


def _get_pipeline():
    """Load ACE-Step on first synthesis call, not while importing this module."""
    global _pipe
    if _pipe is not None:
        return _pipe
    try:
        from acestep.pipeline_ace_step import ACEStepPipeline
    except ImportError as exc:
        raise RuntimeError(
            "ACE-Step is unavailable. Install it or set ACE_STEP_PATH to its source tree.") from exc
    _pipe = ACEStepPipeline(
        device_id=int(os.environ.get("ACE_STEP_DEVICE", "0")),
        dtype=os.environ.get("ACE_STEP_DTYPE", "bfloat16"),
        torch_compile=False,
    )
    return _pipe


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize(
    music_attributes: str,
    lyric_draft: str,
    audio_duration: float = 30.0,
    style_ref_audio_path: Optional[str] = None,
    output_dir: str = "outputs",
    filename: Optional[str] = None,
) -> str:
    """Synthesize a music clip from attributes and lyrics via ACE-Step.

    Parameters
    ----------
    music_attributes:
        Comma-separated style tags from verbalization.generate_music_attributes().
        e.g. "synth-pop, nostalgic, 98 BPM, retro synths, F minor, English"
    lyric_draft:
        ACE-Step markup lyrics from verbalization.generate_lyrics().
        e.g. "[verse]\\nLine one\\nLine two\\n[chorus]\\n..."
    audio_duration:
        Target clip length in seconds. Default 30s for fast testing.
    style_ref_audio_path:
        Optional path to nearest-neighbor catalog song for acoustic reference.
        When provided, uses ACE-Step edit task for style transfer.
    output_dir:
        Directory to write the .wav file.
    filename:
        Output filename without extension. Auto-generated if None.

    Returns
    -------
    str: absolute path to generated .wav file.
    """
    if not music_attributes.strip() or not lyric_draft.strip():
        raise ValueError("music_attributes and lyric_draft must not be empty")
    if not 1.0 <= audio_duration <= 600.0:
        raise ValueError(f"audio_duration must be in [1, 600] seconds, got {audio_duration}")

    pipe = _get_pipeline()

    os.makedirs(output_dir, exist_ok=True)

    if style_ref_audio_path and os.path.isfile(style_ref_audio_path):
        # Use edit task with acoustic style reference (nearest catalog neighbor)
        outputs = pipe(
            task='edit',
            src_audio_path=style_ref_audio_path,
            edit_target_prompt=music_attributes,
            edit_target_lyrics=lyric_draft,
            audio_duration=audio_duration,
        )
    else:
        # Standard text-to-music generation
        outputs = pipe(
            prompt=music_attributes,
            lyrics=lyric_draft,
            audio_duration=audio_duration,
        )

    out_path = outputs[0]

    if filename is not None:
        import shutil
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", filename):
            raise ValueError("filename may contain only letters, digits, dot, underscore, and dash")
        dest = os.path.join(output_dir, filename + ".wav")
        shutil.copy2(out_path, dest)
        out_path = dest

    return os.path.abspath(out_path)
