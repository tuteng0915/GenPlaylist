"""synthesis/app.py — Interactive demo and user study interface for GenPlaylist.

Owned by WP-C.  Calls pipeline.genplaylist.generate() to produce candidates
and displays them with audio playback, attributes, lyrics, and rating forms.

Adapted from VibeMus/main.py (Gradio UI pattern).

Architecture
------------
User input (text / songs)
  ↓  input_normalization.normalizer.normalize()   [WP-A]
ContextPrefix
  ↓  pipeline.genplaylist.generate()              [WP-D + WP-C]
list[SynthesisResult]
  ↓  Gradio UI (this file)                        [WP-C]
Audio playback + user ratings → evaluation_log.jsonl

TODOs (WP-C)
------------
- [ ] Week 1: set up Gradio Blocks layout; input component; display mock examples
- [ ] Week 2: audio playback; show attributes + lyric drafts; connect pipeline
- [ ] Week 3: user study mode (5-point Likert + ranking);
              export logs as CSV/JSON; write user study protocol
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import gradio as gr
from shared.schema import ContextPrefix, SynthesisResult


# ---------------------------------------------------------------------------
# Pipeline stub — replace with real pipeline.genplaylist.generate() call
# ---------------------------------------------------------------------------

def _generate_candidates(context_prefix: ContextPrefix, n_samples: int = 3):
    """Call the full GenPlaylist pipeline and return SynthesisResult list.

    TODO (WP-C Week 2): import and call pipeline.genplaylist.generate()
    """
    # from pipeline.genplaylist import generate
    # return generate(context_prefix, n_samples=n_samples)
    raise NotImplementedError("Connect to pipeline.genplaylist.generate() (WP-C Week 2).")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
# TODO (WP-C Week 1): build full layout

with gr.Blocks(title="GenPlaylist Demo") as demo:
    gr.Markdown("# GenPlaylist — Structure-Aware Playlist Generation")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Describe a playlist or enter song titles",
                placeholder="e.g. late-night study mix with indie and lo-fi tracks",
            )
            generate_btn = gr.Button("Generate Next Song")

        with gr.Column():
            # TODO: display generated candidates with audio playback
            audio_out = gr.Audio(label="Generated candidate", interactive=False)
            attrs_out = gr.Textbox(label="Style attributes", interactive=False)
            lyrics_out = gr.TextArea(label="Lyric draft", interactive=False)

    # TODO (WP-C Week 3): user study rating panel
    with gr.Accordion("Rate this candidate", open=False):
        coherence = gr.Slider(1, 5, step=1, label="Coherence")
        quality   = gr.Slider(1, 5, step=1, label="Music Quality")
        overall   = gr.Slider(1, 5, step=1, label="Overall Satisfaction")
        submit_btn = gr.Button("Submit Rating")


if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")
