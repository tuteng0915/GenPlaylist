#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$WP_ROOT/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$WP_ROOT:${PYTHONPATH:-}"

DATA_ROOT="${GENPLAYLIST_DATA_ROOT:-$REPO_ROOT/data/dataset}"
ARTIFACT_ROOT="${GENPLAYLIST_ARTIFACT_ROOT:-$REPO_ROOT/data/dataset}"
DEFAULT_PREPARED_ROOT="${DATA_ROOT%/dataset}/processed/genplaylist-v3-20item-joint-15to5"
PREPARED_DATA_ROOT="${GENPLAYLIST_PREPARED_DATA_ROOT:-$DEFAULT_PREPARED_ROOT}"
EVAL_CKPT="${GENPLAYLIST_EVAL_CKPT:-}"
MODEL_SIZE="${GENPLAYLIST_MODEL_SIZE:-small}"
EVAL_BATCH_SIZE="${GENPLAYLIST_EVAL_BATCH_SIZE:-32}"
EVAL_SEED="${GENPLAYLIST_EVAL_SEED:-1}"
ALLOW_PROTOCOL_OVERRIDE="${GENPLAYLIST_EVAL_ALLOW_PROTOCOL_OVERRIDE:-false}"

# The official protocol uses 256 reverse-diffusion steps. Shorter settings are
# smoke tests and must write to a separately named result file.
SAMPLING_STEPS="${GENPLAYLIST_EVAL_SAMPLING_STEPS:-256}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_ROOT="${GENPLAYLIST_EVAL_RESULTS_ROOT:-$WP_ROOT/outputs/evaluation}"

if [[ -z "$EVAL_CKPT" ]]; then
  echo "GENPLAYLIST_EVAL_CKPT must point to the checkpoint to evaluate." >&2
  exit 2
fi

CKPT_LABEL="$(basename "$EVAL_CKPT" .ckpt)"
RESULTS_PATH="${GENPLAYLIST_EVAL_RESULTS_PATH:-$RESULTS_ROOT/wp-c-${CKPT_LABEL}-steps${SAMPLING_STEPS}-seed${EVAL_SEED}-${STAMP}.json}"

for required in \
  "$EVAL_CKPT" \
  "$PREPARED_DATA_ROOT/prepared_manifest.json" \
  "$DATA_ROOT/catalog_metadata.json" \
  "$ARTIFACT_ROOT/catalog_item_embeddings.npy" \
  "$ARTIFACT_ROOT/item_id_to_row.json" \
  "$ARTIFACT_ROOT/semantic_tokens.json" \
  "$ARTIFACT_ROOT/rvq_codebook_weights.npy" \
  "$REPO_ROOT/src/02_creative_cues/outputs/production/latest/item2cues.json" \
  "$REPO_ROOT/src/02_creative_cues/outputs/production/latest/cue_vocab.json" \
  "$REPO_ROOT/src/02_creative_cues/outputs/production/latest/cue_manifest.json"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing evaluation input: $required" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$RESULTS_PATH")"
GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"

cd "$WP_ROOT"
python main.py \
  mode=rec_eval \
  model="$MODEL_SIZE" \
  data=spotify \
  data_root="$DATA_ROOT" \
  catalog_embeddings_path="$ARTIFACT_ROOT/catalog_item_embeddings.npy" \
  item_id_to_row_path="$ARTIFACT_ROOT/item_id_to_row.json" \
  semantic_tokens_path="$ARTIFACT_ROOT/semantic_tokens.json" \
  codebook_weights_path="$ARTIFACT_ROOT/rvq_codebook_weights.npy" \
  item2cues_path="$REPO_ROOT/src/02_creative_cues/outputs/production/latest/item2cues.json" \
  cue_vocab_path="$REPO_ROOT/src/02_creative_cues/outputs/production/latest/cue_vocab.json" \
  cue_manifest_path="$REPO_ROOT/src/02_creative_cues/outputs/production/latest/cue_manifest.json" \
  prepared_dataset_path="$PREPARED_DATA_ROOT" \
  eval.checkpoint_path="$EVAL_CKPT" \
  eval.results_path="$RESULTS_PATH" \
  eval.git_commit="$GIT_COMMIT" \
  eval.disable_ema=false \
  eval.allow_protocol_override="$ALLOW_PROTOCOL_OVERRIDE" \
  eval_batch_size="$EVAL_BATCH_SIZE" \
  seed="$EVAL_SEED" \
  sampling.steps="$SAMPLING_STEPS" \
  parameterization=subs \
  eval.compute_generative_perplexity=false \
  +run_name="genplaylist-v3-joint15to5-official-eval-${STAMP}"

echo "Official WP-C result: $RESULTS_PATH"
