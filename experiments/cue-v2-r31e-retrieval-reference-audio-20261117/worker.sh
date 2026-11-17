#!/usr/bin/env bash
set -euo pipefail
R=/storage_fast/ttu/GenPlaylist
E=$R/experiments/cue-v2-r31e-retrieval-reference-audio-20261117
O=$R/outputs/cue-v2-r31e-retrieval-reference-audio-20261117
GPU=${1:?gpu}; shift
mkdir -p "$O/logs" "$O/locks"
exec 9>"$O/locks/gpu${GPU}.lock"; flock -n 9
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for spec in "$@"; do
  dataset=${spec%%:*}; arm=${spec##*:}
  exec 8>"$O/locks/${dataset}-${arm}.lock"; flock -n 8
  echo "$(date -Is) START gpu=$GPU $dataset $arm"
  bash "$E/audio.sh" "$dataset" "$arm" 0 500
  echo "$(date -Is) DONE gpu=$GPU $dataset $arm"
done
