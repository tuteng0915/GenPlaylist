#!/bin/bash

# DISCO Yelp 学习率搜索脚本
# 依次训练 lr=5e-4, 1e-4, 5e-5 三个模型，每个各使用双 GPU

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/home/sjj/wenhao/DISCO:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
for LR in 3e-4; do
  RUN_NAME="disco-yelp-CFGenc-lr${LR}-$(date +%Y%m%d-%H%M%S)"
  echo "========================================"
  echo "开始训练: lr=${LR}  run=${RUN_NAME}"
  echo "========================================"

  python /home/sjj/wenhao/DISCO/main.py \
    training.layer_loss_weights.enabled=false \
    loader.batch_size=256 \
    loader.eval_batch_size=256 \
    trainer.max_steps=20000 \
    model=small \
    model.hidden_size=64 \
    data=yelp \
    dataset=Yelp \
    run_name=${RUN_NAME} \
    parameterization=subs \
    seq_len=10 \
    rq_n_codebooks=3 \
    rq_codebook_size=256 \
    model.length=52 \
    swap_ratio=0 \
    eval.compute_generative_perplexity=False \
    sampling.steps=25 \
    sampling.cfg_enabled=true \
    sampling.cfg_encoder=true \
    sampling.cfg_p_drop=0.1 \
    sampling.cfg_w=2.0 \
    use_tensorboard=true \
    optim.lr=${LR} \
    checkpointing.resume_from_ckpt=false

  echo "完成: lr=${LR}"
done

echo "========================================"
echo "全部训练完成"
echo "========================================"
