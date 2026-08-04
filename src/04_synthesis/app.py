"""synthesis/app.py — Interactive demo and user study interface for GenPlaylist.
WP-C Owner: Aanya Yaduvanshi
"""

import sys
import os
import json
import csv
import random
import datetime
import subprocess
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("GENPLAYLIST_DATA_DIR", REPO_ROOT / "data"))
AUDIO_DIR = DATA_DIR / "audio" / "spotify"
CATALOG_PATH = DATA_DIR / "dataset" / "catalog_metadata.json"
CURATED_DIR = Path(os.environ.get(
    "GENPLAYLIST_CURATED_DIR", Path(__file__).resolve().parent / "data" / "curated"))
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "audio"
LOG_PATH = Path(__file__).resolve().parent / "outputs" / "evaluation_logs" / "user_study.csv"
FULL_SONG_DURATION_SECONDS = 240.0
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ---------------------------------------------------------------------------
# Load catalog
# ---------------------------------------------------------------------------
print("[app] Loading catalog...")
with open(CATALOG_PATH, encoding="utf-8") as f:
    _raw = json.load(f)
CATALOG = _raw if isinstance(_raw, dict) else {str(i['item_id']): i for i in _raw}

def get_audio_path(item_id):
    return str(AUDIO_DIR / f"{item_id}.mp3")

def get_item_label(item_id):
    item = CATALOG.get(str(item_id), {})
    title  = item.get('title', f'Song {item_id}')
    artist = item.get('artist', 'Unknown')
    return f"{title} — {artist}"

# ---------------------------------------------------------------------------
# Load curated playlists (group by playlist_id, hide variants from user)
# ---------------------------------------------------------------------------
print("[app] Loading curated playlists...")
_all_curated = {}
for fname in os.listdir(CURATED_DIR) if CURATED_DIR.is_dir() else []:
    if fname.endswith('.json'):
        with open(os.path.join(CURATED_DIR, fname)) as f:
            d = json.load(f)
        pid = str(d['playlist_id'])
        if pid not in _all_curated:
            _all_curated[pid] = []
        _all_curated[pid].append(d)

# Human-readable genre labels per playlist
_GENRE_LABELS = {
    '13867': '🎤 Hip-Hop & Trap',
    '3122':  '🎀 Pop Hits',
    '3657':  '🌊 Modern Rap & R&B',
    '6334':  '🎸 Indie & Alternative Pop',
    '8280':  '🎧 Alternative & Chill',
}

# Build all 25 variant choices with readable labels
# Format: "Hip-Hop & Trap — Variant 1"
VARIANT_CHOICES = []
VARIANT_MAP = {}  # label -> (pid, variant_data)

for pid, variants in sorted(_all_curated.items()):
    genre = _GENRE_LABELS.get(str(pid), f'Playlist {pid}')
    for i, variant in enumerate(variants, 1):
        label = f"{genre} — Variant {i}"
        VARIANT_CHOICES.append(label)
        VARIANT_MAP[label] = (str(pid), variant)

# Shuffle so playlists are interleaved (balanced exposure)
random.shuffle(VARIANT_CHOICES)

def label_to_variant(label):
    """Return (pid, variant_data) for a given dropdown label."""
    return VARIANT_MAP.get(label, (None, None))

def get_reference_display_from_label(label):
    pid, variant = label_to_variant(label)
    if not variant:
        return "No songs found."
    lines = []
    # ``seed_songs`` is the legacy curated-data field name; semantically these
    # are now the ordered reference music.
    for s in variant['seed_songs']:
        lines.append(f"• {get_item_label(s['id'])}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Auto-detect free GPU
# ---------------------------------------------------------------------------
def get_free_gpu():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')
        for line in lines:
            idx, mem = line.strip().split(', ')
            if int(mem) < 1000:  # less than 1GB used = free
                return int(idx)
        return 2  # fallback
    except Exception:
        return 2

FREE_GPU = get_free_gpu()
print(f"[app] Using GPU {FREE_GPU}")
os.environ['CUDA_VISIBLE_DEVICES'] = str(FREE_GPU)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
_pipeline = None

def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return
    from pipeline import GenPlaylistPipeline

    dataset_dir = DATA_DIR / "dataset"
    emb_path = Path(os.environ.get(
        "GENPLAYLIST_CATALOG_EMBEDDINGS", dataset_dir / "catalog_item_embeddings.npy"))
    mapping_path = Path(os.environ.get(
        "GENPLAYLIST_ITEM_ID_TO_ROW", dataset_dir / "item_id_to_row.json"))
    cue_vocab_path = Path(os.environ.get(
        "GENPLAYLIST_CUE_VOCAB",
        SRC_ROOT / "02_creative_cues" / "outputs" / "production" / "latest" / "cue_vocab.json"))
    if not cue_vocab_path.is_file():
        raise FileNotFoundError(f"Missing production cue vocabulary: {cue_vocab_path}")
    _pipeline = GenPlaylistPipeline.from_files(
        catalog_metadata_path=CATALOG_PATH,
        catalog_embeddings_path=emb_path,
        item_id_to_row_path=mapping_path,
        cue_vocab_path=cue_vocab_path,
        synthesis_output_dir=str(OUTPUT_DIR),
    )

def run_pipeline(text_instruction, pid, variant=None):
    _load_pipeline()

    if variant is None:
        variant = _all_curated.get(str(pid), [{}])[0]

    reference_ids = [str(s['id']) for s in variant['seed_songs']]
    result = _pipeline.generate(
        reference_ids,
        reference_count=len(reference_ids),
        user_instruction=text_instruction,
        audio_duration=30,
    )

    verb_result = _verb_module.verbalize(
        generated=generated,
        catalog_embs=catalog_embs,
        catalog_metadata=catalog_items,
        k=5
    )

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    audio_path = _synth_module.synthesize(
        music_attributes=verb_result['music_attributes'],
        lyric_draft=verb_result['lyric_draft'],
        audio_duration=FULL_SONG_DURATION_SECONDS,
        output_dir=OUTPUT_DIR,
        filename=f"generated_{pid}_{ts}"
    )

    return verb_result['music_attributes'], verb_result['lyric_draft'], audio_path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_rating(playlist_id, song_a_path, song_b_path, song_a_is_generated,
               fit_a, fit_b, quality_a, quality_b, novelty_a, novelty_b,
               preference, listening_freq, musical_training, notes):
    row = {
        'timestamp': datetime.datetime.now().isoformat(),
        'session_id': uuid.uuid4().hex,
        'playlist_id': playlist_id,
        'song_a_path': os.path.basename(song_a_path) if song_a_path else '',
        'song_b_path': os.path.basename(song_b_path) if song_b_path else '',
        'song_a_is_generated': song_a_is_generated,
        'fit_a': fit_a,
        'fit_b': fit_b,
        'quality_a': quality_a,
        'quality_b': quality_b,
        'novelty_a': novelty_a,
        'novelty_b': novelty_b,
        'preference': preference,
        'listening_freq': listening_freq,
        'musical_training': musical_training,
        'notes': notes,
    }
    with open(LOG_PATH, 'a+', newline='', encoding="utf-8") as f:
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass  # Windows deployments should replace CSV with a transactional store.
        f.seek(0, os.SEEK_END)
        file_exists = f.tell() > 0
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (ImportError, NameError):
            pass
    return row

# ---------------------------------------------------------------------------
# UI state for user study (stored per Gradio session; never module-global)
# ---------------------------------------------------------------------------
# Update the displayed ordered reference set.
def update_reference_display(label):
    if not label:
        return "Select a playlist to see its songs."
    return get_reference_display_from_label(label)

# Replace demo_generate:
def demo_generate(playlist_label, text_instruction):
    if not playlist_label:
        return None, "Please select a playlist first.", ""
    pid, variant = label_to_variant(playlist_label)
    if not pid or not variant:
        return None, "Playlist not found.", ""
    try:
        attributes, lyrics, audio_path = run_pipeline(
            text_instruction=text_instruction or "one next song that fits these references",
            pid=pid,
            variant=variant
        )
        return audio_path, attributes, lyrics
    except Exception as e:
        import traceback
        return None, f"Error: {e}\n{traceback.format_exc()}", ""

# Replace load_study_pair:
def load_study_pair(playlist_label):
    if not playlist_label:
        return None, None, "*Select a playlist to begin.*", "", "", {}
    
    pid, variant = label_to_variant(playlist_label)
    if not pid or not variant:
        return None, None, "Playlist not found.", "", "", {}

    reference_ids = [s['id'] for s in variant['seed_songs']]
    reference_labels = "\n".join(
        [f"• {get_item_label(item_id)}" for item_id in reference_ids])

    # Next-one contract: compare only with the immediate held-out next song.
    ground_truth = variant.get('ground_truth_songs', [])
    if not ground_truth:
        return None, None, "⚠️ This example has no ground-truth next song.", reference_labels, "", {}
    gt_path = get_audio_path(ground_truth[0]['id'])
    if not os.path.exists(gt_path) or os.path.getsize(gt_path) <= 0:
        return None, None, (
            "⚠️ The immediate ground-truth next-song audio is missing; "
            "this example cannot be substituted with a later song."
        ), reference_labels, "", {}

    # Generate anonymously — user never sees this happening
    try:
        _, _, gen_path = run_pipeline(
            text_instruction="one next song that fits these references",
            pid=str(pid),
            variant=variant
        )
    except Exception as e:
        return None, None, f"⚠️ Generation failed: {e}", reference_labels, "", {}

    # Randomize A/B so user doesn't know which is generated
    gen_is_a = random.random() > 0.5
    path_a, path_b = (gen_path, gt_path) if gen_is_a else (gt_path, gen_path)

    # Copy both files to /tmp so Gradio can always serve them
    import shutil
    tmp_a = f'/tmp/song_a_{os.path.basename(path_a)}'
    tmp_b = f'/tmp/song_b_{os.path.basename(path_b)}'
    shutil.copy(path_a, tmp_a)
    shutil.copy(path_b, tmp_b)

    study_state = {
        'playlist_id': pid,
        'variant_id': variant.get('variant_id', ''),
        'path_a': path_a,   # keep original paths for logging
        'path_b': path_b,
        'gen_is_a': gen_is_a,
    }

    context = f"""
**🎵 Your task:** You are a music listener who enjoys this kind of playlist.
You are looking for the **next song** to add to it.

**Reference music (in order):**
{reference_labels}

Two candidate songs have been prepared for you.
Listen to both Song A and Song B, then rate them.
The songs are in **random order** — you will not be told which is AI-generated and which is real until after you submit.
    """
    return (tmp_a, tmp_b, context, reference_labels,
            "✅ Songs loaded. Please listen to both before rating.", study_state)


def submit_rating(study_state, fit_a, fit_b, quality_a, quality_b, novelty_a, novelty_b,
                  preference, listening_freq, musical_training, notes):
    if not study_state:
        return "⚠️ Please load a playlist pair first."
    try:
        row = log_rating(
            playlist_id=study_state.get('playlist_id', ''),
            song_a_path=study_state.get('path_a', ''),
            song_b_path=study_state.get('path_b', ''),
            song_a_is_generated=study_state.get('gen_is_a', False),
            fit_a=fit_a, fit_b=fit_b,
            quality_a=quality_a, quality_b=quality_b,
            novelty_a=novelty_a, novelty_b=novelty_b,
            preference=preference,
            listening_freq=listening_freq,
            musical_training=musical_training,
            notes=notes,
        )
        # Reveal which was generated after submission
        gen_was = "Song A" if study_state.get('gen_is_a') else "Song B"
        return (
            f"✅ Thank you! Your rating has been recorded.\n\n"
            f"📊 Reveal: The AI-generated song was **{gen_was}**.\n\n"
            f"🆔 Session ID: {row['session_id']}\n"
            f"💾 Your response has been saved to our research database."
        )
    except Exception as e:
        return f"❌ Error saving rating: {e}"

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="GenPlaylist",
    theme=gr.themes.Soft(),
    css="footer { display: none !important; }",
) as demo:

    study_state = gr.State({})

    gr.Markdown("""
    # 🎵 GenPlaylist
    ### Reference-Conditioned Next-Song Generation
    *Give us multiple reference songs, and we'll generate exactly one song to play next.*
    """)

    with gr.Tabs():

        # ── Tab 1: Generation Demo ──────────────────────────────────────────
        with gr.Tab("🎼 Generate Music"):

            gr.Markdown("""
            ### How it works
            1. **Select multiple reference songs** — these define the musical context
            2. **Optionally describe** what kind of song you want
            3. Click **Generate** and the AI will create a new song that fits your playlist
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    playlist_dropdown_gen = gr.Dropdown(
                        choices=VARIANT_CHOICES,
                        label="🎶 Select a Playlist Style",
                        info="The AI will generate a song that fits this playlist's style"
                    )
                    text_instruction = gr.Textbox(
                        label="✏️ Optional: Describe what you want",
                        placeholder="e.g. something energetic and upbeat, or a chill late-night vibe...",
                        lines=2
                    )
                    generate_btn = gr.Button(
                        "🎵 Generate Song",
                        variant="primary",
                        size="lg"
                    )
                    gr.Markdown(
                        "*Generation takes about 1–2 minutes.*",
                    )

                with gr.Column(scale=1):
                    reference_display_gen = gr.Textbox(
                        label="📋 Ordered reference music",
                        info="The model predicts one song to follow these references",
                        interactive=False,
                        lines=7
                    )

            with gr.Row():
                with gr.Column():
                    attrs_out = gr.Textbox(
                        label="🎼 Generated Music Style",
                        info="The AI's description of the generated song",
                        interactive=False,
                        lines=2
                    )
                    lyrics_out = gr.TextArea(
                        label="📝 Generated Lyrics",
                        interactive=False,
                        lines=14
                    )
                with gr.Column():
                    audio_out = gr.Audio(
                        label="🔊 Generated Song",
                        interactive=False
                    )

            playlist_dropdown_gen.change(
                fn=update_reference_display,
                inputs=[playlist_dropdown_gen],
                outputs=[reference_display_gen]
            )
            generate_btn.click(
                fn=demo_generate,
                inputs=[playlist_dropdown_gen, text_instruction],
                outputs=[audio_out, attrs_out, lyrics_out]
            )

        # ── Tab 2: User Study ───────────────────────────────────────────────
        with gr.Tab("👥 Listener Study"):

            gr.Markdown("""
            ### Help Us Evaluate AI-Generated Music
            We are researchers studying AI music generation. In this study, you will listen to
            **two songs** and rate how well each one fits a given playlist.

            One song is AI-generated, the other is the real held-out next song — they are presented in
            **random order** so you won't know which is which. Please rate them independently
            based on how well they fit the playlist.

            *This study takes about 5 minutes. No musical expertise required.*
            """)

            with gr.Row():
                playlist_dropdown_study = gr.Dropdown(
                    choices=VARIANT_CHOICES,
                    label="🎶 Select a Playlist",
                    info="You will evaluate songs for this playlist style"
                )
                load_btn = gr.Button("▶️ Load Songs (this may take 1-2 minutes)", variant="secondary")

            load_status = gr.Textbox(label="", interactive=False, visible=True)
            context_display = gr.Markdown("*Select a playlist and click Load Songs to begin.*")

            gr.Markdown("---")
            gr.Markdown("### 🎧 Listen to both songs carefully before rating")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Song A")
                    audio_a = gr.Audio(label="Song A", interactive=False)
                with gr.Column():
                    gr.Markdown("## Song B")
                    audio_b = gr.Audio(label="Song B", interactive=False)

            gr.Markdown("---")
            gr.Markdown("""
            ### 📊 Rate Each Song
            *For each dimension, 1 = very poor, 5 = excellent*
            """)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Song A Ratings")
                    fit_a = gr.Slider(1, 5, step=1, value=3,
                        label="🎯 Playlist Fit — How well does this song fit the playlist?")
                    quality_a = gr.Slider(1, 5, step=1, value=3,
                        label="🎵 Audio Quality — How good does this song sound?")
                    novelty_a = gr.Slider(1, 5, step=1, value=3,
                        label="✨ Novelty — How fresh and interesting is this song?")

                with gr.Column():
                    gr.Markdown("#### Song B Ratings")
                    fit_b = gr.Slider(1, 5, step=1, value=3,
                        label="🎯 Playlist Fit — How well does this song fit the playlist?")
                    quality_b = gr.Slider(1, 5, step=1, value=3,
                        label="🎵 Audio Quality — How good does this song sound?")
                    novelty_b = gr.Slider(1, 5, step=1, value=3,
                        label="✨ Novelty — How fresh and interesting is this song?")

            gr.Markdown("---")
            gr.Markdown("### 🏆 Overall Preference")

            preference = gr.Radio(
                choices=["Song A", "Song B", "No preference"],
                label="If this playlist was playing and you heard both songs, which would you rather listen to next?",
                value="No preference"
            )

            gr.Markdown("---")
            gr.Markdown("### 👤 A Little About You")
            gr.Markdown("*This helps us understand our evaluators. All responses are anonymous.*")

            with gr.Row():
                listening_freq = gr.Radio(
                    choices=["Daily", "Several times a week", "Weekly", "Occasionally", "Rarely"],
                    label="How often do you listen to music?",
                    value="Daily"
                )
                musical_training = gr.Radio(
                    choices=["Yes", "No"],
                    label="Do you have formal musical training (lessons, music school, etc.)?",
                    value="No"
                )

            notes = gr.Textbox(
                label="💬 Any comments? (optional)",
                placeholder="e.g. Song A felt too slow, Song B had better energy for this playlist...",
                lines=3
            )

            submit_btn = gr.Button("✅ Submit My Ratings", variant="primary", size="lg")
            submit_status = gr.Textbox(label="", interactive=False)

            load_btn.click(
                fn=load_study_pair,
                inputs=[playlist_dropdown_study],
                outputs=[audio_a, audio_b, context_display, gr.Textbox(visible=False),
                         load_status, study_state]
            )

            submit_btn.click(
                fn=submit_rating,
                inputs=[study_state, fit_a, fit_b, quality_a, quality_b,
                        novelty_a, novelty_b,
                        preference, listening_freq, musical_training, notes],
                outputs=[submit_status]
            )

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        allowed_paths=[str(AUDIO_DIR), str(OUTPUT_DIR)],
    )
