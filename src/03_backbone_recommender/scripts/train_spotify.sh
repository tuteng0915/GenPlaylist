#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$WP_ROOT:${PYTHONPATH:-}"

# Generate unique run name with timestamp to avoid conflicts
RUN_NAME="genplaylist-v1-spotify30-$(date +%Y%m%d-%H%M%S)"

cd "$WP_ROOT"
python main.py \
  loader.batch_size=300 \
  loader.eval_batch_size=128 \
  model=small \
  data=spotify \
  run_name=${RUN_NAME} \
  parameterization=subs \
  eval.compute_generative_perplexity=False \
  sampling.steps=25



