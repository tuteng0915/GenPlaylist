#!/bin/bash
set -euo pipefail
test "$(hostname)" = next-asus-02
R=/storage_fast/ttu/GenPlaylist
E=$R/experiments/ddbc-preserve-20261111
O=$R/outputs/ddbc-preserve-20261111
mkdir -p "$O"
exec 9>"$O/launch.lock"
flock -n 9 || exit 0
# Dedicated ASUS02 GPU0 only. Do not touch ASUS01 workers or other processes.
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0)
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$uuid"; then
  echo 'GPU0 occupied; declining launch';exit 3
fi
remaining=$($R/envs/repro-torch26/bin/python -c 'import datetime,time; print(int(datetime.datetime.fromisoformat("2026-11-12T01:20:30+08:00").timestamp()-time.time()))')
test "$remaining" -gt 600
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
exec timeout --signal=TERM --kill-after=20s "${remaining}s" nice -n 10 "$R/envs/repro-torch26/bin/python" -u "$E/worker.py"
