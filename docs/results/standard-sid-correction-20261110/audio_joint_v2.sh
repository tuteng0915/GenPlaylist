#!/usr/bin/env bash
set -euo pipefail
R=/storage_fast/ttu/GenPlaylist
E=$R/experiments/tiger-joint-20261108
O=$R/outputs/tiger-joint-20261108
DS=${1:?}; ARM=${2:?}; START=${3:?}; COUNT=${4:?}
CODE=$R/v2-recovery-20261106/code
RUN=$O/$DS/audio
PLAN=$O/$DS/joint/audio/plan.json
SID=$R/outputs/two-dataset-fair-20261108/$DS/tiger/audio-plan.json
if [[ $DS == mpd ]]; then
 PREP=$R/original-ncl/data/processed/genplaylist-v4-8cue-20item-joint-15to5
 DATA=/storage/ttu/GenPlaylist/runtime/mpd/data/dataset
 ART=$DATA
 CUES=$R/original-ncl/src/02_creative_cues/outputs/production/20260804_172128
else
 PREP=$R/experiments/two-dataset-fair-20261108/m4a-audio-prepared
 DATA=$R/original-ncl/data/dataset-music4all-onion-v3-u500
 ART=$R/original-ncl/data/artifacts-music4all-onion-v3-u500
 CUES=$R/original-ncl/data/processed/cues-music4all-onion-v3-u500/latest
fi
[[ $ARM == full ]] || exit 2
SYSTEM=GenPlaylist
source "$R/environment/env.sh"
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$RUN/$ARM"
cd "$CODE"
python scripts/run_end_to_end_verbalization.py --prepared-dir "$PREP" --ddbc-sft-result "$SID" --genplaylist-result "$PLAN" \
 --catalog-metadata "$DATA/catalog_metadata.json" --catalog-embeddings "$ART/catalog_item_embeddings.npy" --item-id-to-row "$ART/item_id_to_row.json" --codebook-weights "$ART/rvq_codebook_weights.npy" \
 --item2cues "$CUES/item2cues.json" --cue-vocab "$CUES/cue_vocab.json" --output-dir "$RUN/$ARM/verbalization" \
 --model-name Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 --device cuda --seed 42 --start-index "$START" --max-examples "$COUNT" --system "$SYSTEM" --cue-intervention none > "$RUN/$ARM/verbalization-$START.log" 2>&1
source "$R/environment/ace-env.sh"
python scripts/run_end_to_end_synthesis.py --verbalization-dir "$RUN/$ARM/verbalization" --output-dir "$RUN/$ARM/synthesis" --ace-step-path "$ACE_STEP_SOURCE" --checkpoint-dir "$ACE_STEP_SNAPSHOT" --device-id 0 --duration 30 --inference-steps 60 --seed-base 42000 --start-index "$START" --max-examples "$COUNT" --system "$SYSTEM" > "$RUN/$ARM/synthesis-$START.log" 2>&1
