"""synthesis/synthesis.py — ACE-Step music synthesis wrapper for GenPlaylist.

Owned by WP-C.  Receives music_attributes + lyric_draft from verbalization.py
and produces audio_path via ACE-Step.

Adapted from VibeMus/pipeline.py and VibeMus/tools.py.

VibeMus used ACEStepPipeline directly in a Gradio chat loop.
Here we expose a pure-function interface: given music_attributes
(str) and a lyric_draft (str), synthesize a waveform and return
the output path.  The pipeline is loaded once as a module-level
singleton, matching VibeMus's pattern.

TODOs
-----
- [ ] Confirm ACE-Step installation path and import (pip install acestep or local clone)
- [ ] Tune dtype / torch_compile flag for the target GPU
- [ ] Decide default audio_duration for generated playlist items
- [ ] Add style_ref_audio support (nearest-neighbor acoustic reference from verbalization)
- [ ] Batch synthesis: when S candidates exist, call pipe S times and collect paths
- [ ] Add output directory config (currently writes to cwd)
"""

import os
from typing import Optional

# ---------------------------------------------------------------------------
# Pipeline singleton  (mirrors VibeMus/pipeline.py)
# ---------------------------------------------------------------------------
# TODO: install acestep before importing
#   pip install git+https://github.com/ace-step/ACE-Step.git
try:
    from acestep.pipeline_ace_step import ACEStepPipeline
    _pipe = ACEStepPipeline(
        device_id=0,
        dtype="bfloat16",
        torch_compile=False,  # TODO: set True after confirming torch>=2.3
    )
except ImportError:
    _pipe = None
    print("[synthesis] ACEStepPipeline not found — synthesis will be unavailable. "
          "Install: pip install git+https://github.com/ace-step/ACE-Step.git")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize(
    music_attributes: str,
    lyric_draft: str,
    audio_duration: int = 30,
    style_ref_audio_path: Optional[str] = None,
    output_dir: str = "outputs",
    filename: Optional[str] = None,
) -> str:
    """Synthesize a music clip from attributes and lyrics via ACE-Step.

    Parameters
    ----------
    music_attributes:
        Comma-separated style tags, e.g. "pop, energetic, piano, 120bpm, C major, English".
        Produced by verbalization.generate_music_attributes().
    lyric_draft:
        ACE-Step markup lyrics with section labels, e.g.:
            [verse]
            Staring at the neon rain...
            [chorus]
            ...
        Produced by verbalization.generate_lyrics().
    audio_duration:
        Target clip length in seconds.
    style_ref_audio_path:
        Optional path to a reference audio file (nearest catalog neighbor).
        When provided, ACE-Step uses it as acoustic style reference.
        # TODO: wire up repaint/edit tasks for style transfer
    output_dir:
        Directory to write the generated .wav file.
    filename:
        Output filename (without extension). Auto-generated if None.

    Returns
    -------
    str
        Absolute path to the generated .wav file.
    """
    if _pipe is None:
        raise RuntimeError(
            "ACEStepPipeline is not available. "
            "Install acestep: pip install git+https://github.com/ace-step/ACE-Step.git"
        )

    os.makedirs(output_dir, exist_ok=True)

    # TODO: wire style_ref_audio_path into pipe call via task='edit' or repaint
    outputs = _pipe(
        format="wav",
        audio_duration=audio_duration,
        prompt=music_attributes,
        lyrics=lyric_draft,
    )

    out_path = outputs[0]

    if filename is not None:
        import shutil
        dest = os.path.join(output_dir, filename + ".wav")
        shutil.move(out_path, dest)
        out_path = dest

    return os.path.abspath(out_path)
