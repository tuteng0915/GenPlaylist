#!/usr/bin/env python3
"""
分析RVQ文件的冲突统计
用法: python scripts/analyze_rvq_conflict.py
"""

import json
import numpy as np
from collections import Counter

# 配置
DATASET = "Yelp"
FEATURE_TYPE = "clhe"
BASE_PATH = f"/home/sjj/wenhao/DISCO/datasets/{DATASET}"

def analyze_conflict():
    # 加载token文件
    token_path = f"{BASE_PATH}/{FEATURE_TYPE}_token.json"
    with open(token_path, 'r') as f:
        tokens = json.load(f)

    total_items = len(tokens)

    # 提取conflict digit (最后一个token)
    conflict_tokens = [v[-1] for v in tokens.values()]
    conflict_counter = Counter(conflict_tokens)

    # 推断n_codebooks
    sample_token = list(tokens.values())[0]
    token_length = len(sample_token)
    n_codebooks = token_length - 1  # 减去conflict digit

    # 计算最小conflict token值
    codebook_size = 256  # 假设默认值
    min_conflict_token = n_codebooks * codebook_size + 1

    print("=" * 60)
    print(f"RVQ Conflict Analysis - {DATASET} Dataset")
    print("=" * 60)
    print(f"Dataset: {DATASET}")
    print(f"Feature type: {FEATURE_TYPE}")
    print(f"Total items: {total_items}")
    print(f"Token length: {token_length} (n_codebooks={n_codebooks} + 1 conflict)")
    print(f"Min conflict token: {min_conflict_token}")
    print()

    # 统计conflict分布
    no_conflict_count = conflict_counter[min_conflict_token]
    conflict_count = total_items - no_conflict_count

    print("=" * 60)
    print("Conflict Distribution Summary")
    print("=" * 60)
    print(f"No conflict (conflict=1):  {no_conflict_count:6d} items ({no_conflict_count/total_items*100:.2f}%)")
    print(f"Has conflict (conflict>1): {conflict_count:6d} items ({conflict_count/total_items*100:.2f}%)")
    print()

    # 详细分布（前10个最常见的conflict值）
    print("=" * 60)
    print("Top 10 Conflict Token Values")
    print("=" * 60)
    print(f"{'Token':<10} {'Count':<10} {'Percentage':<12} {'Conflict Level'}")
    print("-" * 60)

    for token, count in conflict_counter.most_common(10):
        conflict_level = token - min_conflict_token + 1
        percentage = count / total_items * 100
        print(f"{token:<10} {count:<10} {percentage:>6.2f}%       conflict={conflict_level}")

    if len(conflict_counter) > 10:
        remaining = sum(count for token, count in conflict_counter.items() if token not in dict(conflict_counter.most_common(10)))
        print(f"{'...':<10} {remaining:<10} (remaining values)")

    print()

    # 最大冲突级别
    max_conflict_token = max(conflict_tokens)
    max_conflict_level = max_conflict_token - min_conflict_token + 1
    max_conflict_count = conflict_counter[max_conflict_token]

    print("=" * 60)
    print("Maximum Conflict Statistics")
    print("=" * 60)
    print(f"Maximum conflict level: {max_conflict_level}")
    print(f"  (Token value: {max_conflict_token})")
    print(f"  (Number of items with this conflict: {max_conflict_count})")
    print()

    # RVQ空间覆盖率
    rvq_space_size = codebook_size ** n_codebooks
    coverage = total_items / rvq_space_size * 100

    print("=" * 60)
    print("RVQ Space Coverage")
    print("=" * 60)
    print(f"RVQ space size ({codebook_size}^{n_codebooks}):  {rvq_space_size:,}")
    print(f"Items in dataset:       {total_items:,}")
    print(f"Coverage:               {coverage:.4f}%")
    print()

    print("=" * 60)
    print("Interpretation")
    print("=" * 60)
    print(f"✓ {no_conflict_count/total_items*100:.2f}% of items have unique RVQ codes")
    print(f"✓ {conflict_count/total_items*100:.2f}% of items share RVQ codes with other items")
    print(f"✓ Maximum {max_conflict_level} items share the same RVQ codes")
    print("=" * 60)

if __name__ == "__main__":
    analyze_conflict()
