#!/usr/bin/env bash
set -euo pipefail
R=/storage_fast/ttu/GenPlaylist
E=$R/experiments/cue-v2-r31e-simple-reference-audio-20261117
DS=${1:?dataset}; ARM=${2:?arm}; START=${3:?start}; COUNT=${4:?count}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "$R/environment/env.sh"
python "$E/verbalize.py" --dataset "$DS" --arm "$ARM" --start "$START" --count "$COUNT"
source "$R/environment/ace-env.sh"
python "$E/synthesize.py" --dataset "$DS" --arm "$ARM" --start "$START" --count "$COUNT"
python "$E/validate.py" --dataset "$DS" --arm "$ARM" --start "$START" --count "$COUNT"
