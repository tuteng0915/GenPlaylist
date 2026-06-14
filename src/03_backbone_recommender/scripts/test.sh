export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/home/sjj/wenhao/DISCO:$PYTHONPATH

# 三个 lr sweep 训练结果的 checkpoint 路径
# CKPTS=(
#   "/home/sjj/wenhao/DISCO/outputs/yelp/2026.03.26/144527/checkpoints/best.ckpt"  # lr=5e-4
#   "/home/sjj/wenhao/DISCO/outputs/yelp/2026.03.26/155821/checkpoints/best.ckpt"  # lr=1e-4
#   "/home/sjj/wenhao/DISCO/outputs/yelp/2026.03.26/171018/checkpoints/best.ckpt"  # lr=5e-5
# )
CKPTS=(
  "/home/sjj/wenhao/DISCO/outputs/yelp/2026.03.29/110818/checkpoints/best.ckpt"  # lr=3e-4, no CFG
  "/home/sjj/wenhao/DISCO/outputs/yelp/2026.03.30/002348/checkpoints/best.ckpt"  # lr=3e-4, CFG mean-pool
  "/home/sjj/wenhao/DISCO/outputs/yelp/2026.03.31/121517/checkpoints/best.ckpt"  # lr=3e-4, CFG encoder
)
NAMES=("3e-4-noCFG" "3e-4-CFG" "3e-4-CFG-enc")
CFG_FLAGS=("false" "true" "true")
CFG_ENC_FLAGS=("false" "false" "true")

for i in "${!CKPTS[@]}"; do
  CKPT="${CKPTS[$i]}"
  NAME="${NAMES[$i]}"
  CFG="${CFG_FLAGS[$i]}"
  CFG_ENC="${CFG_ENC_FLAGS[$i]}"
  echo "========================================"
  echo "Testing ${NAME}: ${CKPT}"
  echo "========================================"

  python /home/sjj/wenhao/DISCO/main.py \
    mode=rec_eval \
    evaluator.candidate_multiplier=19 \
    evaluator.allow_duplicate_items=true \
    training.layer_loss_weights.enabled=false \
    loader.batch_size=32 \
    loader.eval_batch_size=1 \
    dataset=Yelp \
    model=small \
    model.hidden_size=64 \
    parameterization=subs \
    backbone=dit \
    rq_n_codebooks=3 \
    rq_codebook_size=256 \
    model.length=52 \
    seq_len=10 \
    swap_ratio=0 \
    evaluator.dataset=Yelp \
    eval.predict_num_items=3 \
    eval.checkpoint_path="${CKPT}" \
    sampling.cfg_enabled=${CFG} \
    sampling.cfg_encoder=${CFG_ENC} \
    sampling.cfg_w=2.0

  echo "Done: ${NAME}"
done

echo "========================================"
echo "全部测试完成"
echo "========================================"
