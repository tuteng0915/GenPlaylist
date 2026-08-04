#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$WP_ROOT:${PYTHONPATH:-}"

REPO_ROOT="$(cd "$WP_ROOT/../.." && pwd)"
TRAIN_MODE="${GENPLAYLIST_TRAIN_MODE:-warmstart}"
DATA_ROOT="${GENPLAYLIST_DATA_ROOT:-$REPO_ROOT/data/dataset}"
ARTIFACT_ROOT="${GENPLAYLIST_ARTIFACT_ROOT:-$REPO_ROOT/data/dataset}"
WARMSTART_CKPT="${GENPLAYLIST_WARMSTART_CKPT:-$REPO_ROOT/checkpoints/pretrained/ddbc/spotify30.ckpt}"

case "$TRAIN_MODE" in
  warmstart)
    if [[ ! -f "$WARMSTART_CKPT" ]]; then
      echo "Missing DDBC warm-start checkpoint: $WARMSTART_CKPT" >&2
      exit 1
    fi
    CHECKPOINT_ARGS=(
      checkpointing.resume_from_ckpt=false
      checkpointing.warmstart_path="$WARMSTART_CKPT"
    )
    ;;
  resume)
    CHECKPOINT_ARGS=(
      checkpointing.resume_from_ckpt=true
      checkpointing.warmstart_path=null
    )
    ;;
  scratch)
    CHECKPOINT_ARGS=(
      checkpointing.resume_from_ckpt=false
      checkpointing.warmstart_path=null
    )
    ;;
  *)
    echo "GENPLAYLIST_TRAIN_MODE must be warmstart, resume, or scratch" >&2
    exit 2
    ;;
esac

# Generate unique run name with timestamp to avoid conflicts
RUN_NAME="genplaylist-v1-spotify16-$(date +%Y%m%d-%H%M%S)"

cd "$WP_ROOT"
python main.py \
  loader.batch_size=300 \
  loader.eval_batch_size=128 \
  model=small \
  data=spotify \
  data_root="$DATA_ROOT" \
  catalog_embeddings_path="$ARTIFACT_ROOT/catalog_item_embeddings.npy" \
  item_id_to_row_path="$ARTIFACT_ROOT/item_id_to_row.json" \
  semantic_tokens_path="$ARTIFACT_ROOT/semantic_tokens.json" \
  codebook_weights_path="$ARTIFACT_ROOT/rvq_codebook_weights.npy" \
  run_name=${RUN_NAME} \
  parameterization=subs \
  eval.compute_generative_perplexity=False \
  sampling.steps=25 \
  "${CHECKPOINT_ARGS[@]}"
