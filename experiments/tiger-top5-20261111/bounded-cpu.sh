#!/usr/bin/env bash
set -euo pipefail
R=/storage_fast/ttu/GenPlaylist
[[ $(hostname -s) == next-asus-02 ]] || exit 2
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export HF_HOME=$R/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
END=$(date --date='2026-11-12T01:20:30+08:00' +%s)
SECONDS_LEFT=$((END-$(date +%s)))
(( SECONDS_LEFT > 600 )) || exit 3
cd "$R"
exec timeout --signal=TERM --kill-after=20s "${SECONDS_LEFT}s" nice -n 19 "$R/envs/repro-torch26/bin/python" -u "$R/experiments/tiger-top5-20261111/worker.py"
