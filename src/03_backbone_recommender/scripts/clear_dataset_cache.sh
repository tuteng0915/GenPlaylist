#!/bin/bash
# 清理datasets缓存脚本
# 使用场景：改变RVQ配置（codebook数量、大小等）后运行

echo "========================================="
echo "清理Datasets缓存"
echo "========================================="
echo ""

# 1. 清理HuggingFace datasets缓存
if [ -d ~/.cache/huggingface/datasets ]; then
    echo "[1/4] 清理 HuggingFace datasets 缓存..."
    rm -rf ~/.cache/huggingface/datasets/
    echo "  ✓ 已删除 ~/.cache/huggingface/datasets/"
else
    echo "[1/4] HuggingFace datasets 缓存不存在，跳过"
fi

# 2. 清理项目内的arrow缓存（如果有）
echo "[2/4] 清理项目内的arrow缓存..."
find /home/sjj/wenhao/DISCO/datasets -name "*.arrow" -delete 2>/dev/null
echo "  ✓ 已删除 .arrow 文件"

# 3. 清理tokenized数据集缓存目录
echo "[3/5] 清理tokenized数据集缓存..."
if [ -d /home/sjj/wenhao/DISCO/datasets/Yelp/sid_seq ]; then
    rm -rf /home/sjj/wenhao/DISCO/datasets/Yelp/sid_seq
    echo "  ✓ 已删除 Yelp/sid_seq"
fi
if [ -d /home/sjj/wenhao/DISCO/datasets/spotify/sid_seq ]; then
    rm -rf /home/sjj/wenhao/DISCO/datasets/spotify/sid_seq
    echo "  ✓ 已删除 spotify/sid_seq"
fi

# 4. 清理RVQ tokenizer 生成的 clhe 缓存文件（改变 rq_codebook_size/rq_n_codebooks 时必须清理）
echo "[4/5] 清理 RVQ tokenizer 缓存文件（clhe_sid/token/weight）..."
for DATASET_DIR in /home/sjj/wenhao/DISCO/datasets/Yelp /home/sjj/wenhao/DISCO/datasets/spotify; do
    for FILE in clhe_sid.npy clhe_token.json clhe_weight.npy; do
        if [ -f "$DATASET_DIR/$FILE" ]; then
            rm "$DATASET_DIR/$FILE"
            echo "  ✓ 已删除 $DATASET_DIR/$FILE"
        fi
    done
done

# 5. 清理__pycache__（可选，通常不需要）
read -p "[5/5] 是否清理 __pycache__? (通常不需要) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    find /home/sjj/wenhao/DISCO -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    echo "  ✓ 已删除 __pycache__"
else
    echo "  - 跳过 __pycache__ 清理"
fi

echo ""
echo "========================================="
echo "缓存清理完成！"
echo "现在可以重新运行训练/测试，会使用新的配置"
echo "========================================="
