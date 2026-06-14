#!/bin/bash

# DISCO Yelp训练脚本
# 使用时序数据集，保留时序信息（swap_ratio=0）
# Strategy 1: 增加codebook数量（3→4）以提升重建质量

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1  # 双GPU
export PYTHONPATH=/home/sjj/wenhao/DISCO:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
# Generate unique run name with timestamp to avoid conflicts
RUN_NAME="disco-yelp-$(date +%Y%m%d-%H%M%S)"

# 训练参数
# 如果要恢复训练，取消下面的注释并设置正确的checkpoint路径
# RESUME_CKPT="/home/sjj/wenhao/DISCO/outputs/yelp/2026.02.07/222943/checkpoints/last.ckpt"

if [ -f "$RESUME_CKPT" ]; then
  echo "恢复训练从: $RESUME_CKPT"
  python /home/sjj/wenhao/DISCO/main.py \
    training.layer_loss_weights.enabled=false\
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
    checkpointing.resume_ckpt_path="$RESUME_CKPT"
else
  echo "从头开始训练"
  python /home/sjj/wenhao/DISCO/main.py \
    training.layer_loss_weights.enabled=false\
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
    checkpointing.resume_from_ckpt=false
fi
