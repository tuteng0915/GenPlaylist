#!/usr/bin/env bash
# Resumable, exclusive worker for one checked-free GPU.  It never overwrites a
# completed clip and runs MERT/FAD only after the arm's 500 clips validate.
set -euo pipefail
R=/storage_fast/ttu/GenPlaylist
E=$R/experiments/cue-v2-r31e-list5-audio-20261117
O=$R/outputs/cue-v2-r31e-list5-audio-20261117
GPU=${1:?GPU index}; shift
mkdir -p "$O/logs" "$O/locks"
exec 9>"$O/locks/gpu${GPU}.lock"; flock -n 9
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for SPEC in "$@"; do
  DS=${SPEC%%:*}; ARM=${SPEC##*:}
  exec 8>"$O/locks/${DS}-${ARM}.lock"; flock -n 8
  echo "$(date -Is) START gpu=$GPU $DS $ARM"
  bash "$E/audio.sh" "$DS" "$ARM" 0 500
  source "$R/environment/env.sh"
  source "$R/envs/audio-eval-20261107/bin/activate"
  python "$E/metrics.py" --dataset "$DS" --arm "$ARM" --metric mert
  flock "$O/$DS/fad-reference.lock" python "$E/metrics.py" --dataset "$DS" --arm "$ARM" --metric fad
  echo "$(date -Is) DONE gpu=$GPU $DS $ARM"
done
