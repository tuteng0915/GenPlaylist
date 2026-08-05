# GenPlaylist: Next-Song Generation from Multiple Reference Tracks

GenPlaylist keeps DDBC as its discrete-diffusion backbone. Given an ordered set
of at least two reference tracks, it predicts exactly one next-item latent,
verbalizes that latent, and synthesizes one original next song. Inference
appends `[BOI, MASK×12, EOS]` and jointly denoises that single target payload;
the legacy DDBC next-block sampler is not used by the production path.

## Frozen experimental protocol

- Training uses chronological windows of at most 16 songs: up to 15 references
  and exactly one next-song target. All usable prefixes are expanded.
- Evaluation keeps test playlists with at least 20 songs and takes their first
  20: songs 1–15 are references and songs 16–20 are ground truth.
- WP-C independently samples the same next-one slot five times from the unchanged
  15-song context, then scores the five predictions against the five targets with
  order-free 5×5 Hungarian matching.
- WP-D synthesis/demo remains a singular next-song experience and is intentionally
  outside this evaluation-protocol change.

The machine-readable constants live in `shared/protocol.py`; the full frozen
contract is documented in [`docs/WP_C_TRAIN_EVAL_PROTOCOL.md`](docs/WP_C_TRAIN_EVAL_PROTOCOL.md).

## Repository Structure

Each Work Package lives in its own top-level directory.  The `pipeline/`
directory coordinates all modules end-to-end.

```
GenPlaylist_Code/
└── src/
    ├── shared/                        # Data schemas and cross-WP interface contracts
    │   ├── schema.py                  # ContextPrefix · CueMappingEntry · GeneratedItem · SynthesisResult
    │   └── protocol.py                # Frozen 16 / first-20 / 15→5 / five-draw contract
    │
    ├── 01_input_normalization/        # WP-A: Reference Input Construction
    │   └── normalizer.py              # raw user input → ContextPrefix(item_ids=[m1,...,mK])
    │
    ├── 02_creative_cues/              # WP-B: Creative Cue Mining and Vocabulary Construction
    │   └── cue_mining.py              # lyrics/metadata → cue_vocab.json + item2cues.json
    │
    ├── 03_backbone_recommender/       # WP-C: DDBC Backbone and Evaluation
    │   ├── diffusion.py               # Dispersion-conditioned masked discrete diffusion
    │   ├── genplaylist_tokenizer.py   # RVQ + eight-cue contract (13-token stride)
    │   ├── tokenizer.py               # Legacy DISCO tokenizer (migration baseline)
    │   ├── playlist_structure.py      # μ_C and σ²_C preference structure computation
    │   ├── dataset.py / dataloader.py
    │   ├── evaluator.py
    │   ├── models/                    # DIT backbone (AdaLN dispersion conditioning)
    │   ├── configs/
    │   └── main.py                    # train / rec_eval / ppl_eval / generate modes
    │
    ├── 04_synthesis/                  # WP-D: Synthesis and Interactive Demo
    │   ├── verbalization.py           # z_hat_emb → music_attributes + lyric_draft  (from VibeMus)
    │   ├── synthesis.py               # attributes + lyrics → audio via ACE-Step     (from VibeMus)
    │   └── app.py                     # Gradio demo + user study UI
    │
    └── pipeline/
        ├── genplaylist.py             # Reusable WP-A → WP-C → WP-D coordinator
        └── run.py                     # Preflight / server generation CLI
```

## Data Flow

```
User Input
  ↓ input_normalization/normalizer.py          [WP-A]
ContextPrefix
  ↓ backbone_recommender/playlist_structure.py  [WP-C]
(μ_C, σ²_C)  +  RVQ tokens  (from creative_cues item2cues.json [WP-B])
  ↓ backbone_recommender/diffusion.py           [WP-C]
GeneratedItem  (z_hat_emb, cue_ids, ...)
  ↓ synthesis/verbalization.py                  [WP-D]
music_attributes + lyric_draft
  ↓ synthesis/synthesis.py                      [WP-D]
SynthesisResult  (audio_path, ...)
  ↓ synthesis/app.py                            [WP-D demo; unchanged for now]
User ratings + metric scores
```

## Running the pipeline

```bash
# Run after configuring artifacts and a newly trained WP-C checkpoint:
# export GENPLAYLIST_BACKBONE_CKPT='/path/to/genplaylist-v1.ckpt'
python -c "
import sys; sys.path.insert(0, 'src')
from pipeline import GenPlaylistPipeline
pipeline = GenPlaylistPipeline.from_environment()
result = pipeline.generate(
    ['18996', '48262'],
    user_instruction='a nocturnal but hopeful transition',
)
print(result.audio_path)
"

# Validate catalog/cue startup artifacts without loading model weights
python src/pipeline/run.py --preflight-only

# Full server request (DDBC checkpoint + OpenAI + ACE-Step must be configured)
python src/pipeline/run.py \
  --references 18996 48262 \
  --instruction 'a nocturnal but hopeful transition'

# Train backbone only
cd src/03_backbone_recommender
python main.py mode=train

# Launch demo
cd src
python 04_synthesis/app.py
```

## Interface Files (shared/)

Each WP should import from `shared/schema.py` when communicating
with another WP.  Direct cross-WP imports are routed through
`pipeline/genplaylist.py`.

The repository currently lacks the generated per-item CLHE/RVQ artifacts and a
new trained checkpoint. Missing runtime inputs raise an explicit setup error;
the demo no longer substitutes random embeddings or candidates.

For the exact server-side conversion, preflight, training, and runtime handoff,
see [docs/SERVER_MIGRATION.md](docs/SERVER_MIGRATION.md).
