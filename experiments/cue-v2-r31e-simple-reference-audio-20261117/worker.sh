#!/usr/bin/env bash
# Wait for one assigned card to become genuinely idle, then render one finite arm.
set -euo pipefail
R=/storage_fast/ttu/GenPlaylist
E=$R/experiments/cue-v2-r31e-simple-reference-audio-20261117
O=$R/outputs/cue-v2-r31e-simple-reference-audio-20261117
GPU=${1:?gpu}; DATASET=${2:?dataset}; ARM=${3:?arm}; DEADLINE=${4:?deadline_epoch}
mkdir -p "$O/logs" "$O/locks"
exec 9>"$O/locks/${DATASET}-${ARM}.lock"
flock -n 9
LOG="$O/logs/${DATASET}-${ARM}-gpu${GPU}.log"
while (( $(date +%s) < DEADLINE )); do
  # A card is safe only if no compute app owns it and it is not holding model memory.
  APPS=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '[0-9]' || true)
  MEM=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')
  if [[ -z "$APPS" && "${MEM:-999999}" -lt 1000 ]]; then
    echo "$(date -Is) START gpu=$GPU $DATASET $ARM" | tee -a "$LOG"
    export CUDA_VISIBLE_DEVICES=$GPU
    bash "$E/audio.sh" "$DATASET" "$ARM" 0 100 2>&1 | tee -a "$LOG"
    echo "$(date -Is) DONE gpu=$GPU $DATASET $ARM" | tee -a "$LOG"
    exit 0
  fi
  echo "$(date -Is) WAIT gpu=$GPU mem=${MEM:-unknown} apps=${APPS:-none}" >> "$LOG"
  sleep 60
done
echo "$(date -Is) DEADLINE gpu=$GPU $DATASET $ARM" | tee -a "$LOG"
exit 3
