# GenPlaylist: Structure-Aware Playlist Generation via Latent Expansion

## Repository Structure

Each Work Package lives in its own top-level directory.  The `pipeline/`
directory coordinates all modules end-to-end.

```
GenPlaylist_Code/
└── src/
    ├── shared/                        # Data schemas and cross-WP interface contracts
    │   └── schema.py                  # ContextPrefix · CueMappingEntry · GeneratedItem · SynthesisResult
    │
    ├── 01_input_normalization/        # WP-A: Context Retrieval and Prefix Construction
    │   └── normalizer.py              # raw user input → ContextPrefix(item_ids=[m1,...,mK])
    │
    ├── 02_creative_cues/              # WP-B: Creative Cue Mining and Vocabulary Construction
    │   └── cue_mining.py              # lyrics/metadata → cue_vocab.json + item2cues.json
    │
    ├── 03_backbone_recommender/       # WP-D: Core Diffusion Model (mentor)
    │   ├── diffusion.py               # Dispersion-conditioned masked discrete diffusion
    │   ├── tokenizer.py               # Joint RVQ + Creative Cue tokenization (12-token stride)
    │   ├── playlist_structure.py      # μ_C and σ²_C computation
    │   ├── dataset.py / dataloader.py
    │   ├── evaluator.py
    │   ├── models/                    # DIT backbone (AdaLN dispersion conditioning)
    │   ├── configs/
    │   └── main.py                    # train / rec_eval / ppl_eval / generate modes
    │
    ├── 04_synthesis/                  # WP-C: Verbalization, Synthesis, and Demo
    │   ├── verbalization.py           # z_hat_emb → music_attributes + lyric_draft  (from VibeMus)
    │   ├── synthesis.py               # attributes + lyrics → audio via ACE-Step     (from VibeMus)
    │   └── app.py                     # Gradio demo + user study UI
    │
    └── pipeline/
        └── genplaylist.py             # End-to-end coordinator: ContextPrefix → list[SynthesisResult]
```

## Data Flow

```
User Input
  ↓ input_normalization/normalizer.py          [WP-A]
ContextPrefix
  ↓ backbone_recommender/playlist_structure.py  [WP-D]
(μ_C, σ²_C)  +  RVQ tokens  (from creative_cues item2cues.json [WP-B])
  ↓ backbone_recommender/diffusion.py           [WP-D]
GeneratedItem  (z_hat_emb, cue_ids, ...)
  ↓ synthesis/verbalization.py                  [WP-C]
music_attributes + lyric_draft
  ↓ synthesis/synthesis.py                      [WP-C]
SynthesisResult  (audio_path, ...)
  ↓ synthesis/app.py + evaluation               [WP-C / WP-D]
User ratings + metric scores
```

## Running the pipeline

```bash
# Run end-to-end (once all WPs are connected)
python -c "
import sys; sys.path.insert(0, 'src')
from pipeline.genplaylist import generate
from shared.schema import ContextPrefix
results = generate(ContextPrefix(item_ids=[42, 17, 83, 5, 11]), n_samples=3)
print(results[0].audio_path)
"

# Train backbone only
cd src/03_backbone_recommender
python main.py mode=train

# Launch demo
cd src
python 04_synthesis/app.py
```

## Interface Files (shared/)

Each WP should only import from `shared/schema.py` when communicating
with another WP.  Direct cross-WP imports are routed through
`pipeline/genplaylist.py`.
