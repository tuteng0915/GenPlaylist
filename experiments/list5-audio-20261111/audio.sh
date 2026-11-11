#!/usr/bin/env bash
set -euo pipefail
R=/storage_fast/ttu/GenPlaylist
E=$R/experiments/list5-audio-20261111
DS=${1:?};ARM=${2:?};START=${3:?};COUNT=${4:?}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "$R/environment/env.sh"
python "$E/verbalize.py" --dataset "$DS" --arm "$ARM" --start "$START" --count "$COUNT"
source "$R/environment/ace-env.sh"
python "$E/synthesize.py" --dataset "$DS" --arm "$ARM" --start "$START" --count "$COUNT"
python "$E/validate.py" --dataset "$DS" --arm "$ARM" --start "$START" --count "$COUNT"
