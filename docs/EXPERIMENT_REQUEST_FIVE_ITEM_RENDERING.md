# 实验请求：用全部 5 个预测 item 生成音乐

提出日期：2026-11-10。论文写作方提出，实验方（Codex）执行。

**执行方只跑实验、交付结果，不修改论文仓库（Odyssey-GenPlaylist）里的任何文件。**

## 背景

目前生成音乐只用第一个预测 item：
`scripts/run_end_to_end_verbalization.py` 里写死了
`prediction["semantic_token_ids"][example_index][0]`、`item_ids[...][0]` 和 `cue_ids[...][0]`。
模型其实每个 context 预测 5 个 item（SID + 8 个 cue），后 4 个在生成阶段完全没用上。
这次要补一个实验，让生成阶段用上全部 5 个预测 item。

## 新的渲染方式：five-item

prompt 的规模保持和 first-item 一样（5 首元数据歌 + 8 个 cue），这样差异只来自"用了几个预测 item"。

1. **音乐元数据**：对 5 个预测 item，分别用各自的 3 个 RVQ 索引重建 embedding，各取 1 个最近邻 catalog 歌，一共 5 首，用来替换原来"第一个 item 的 5 个最近邻"。如果近邻重复，按距离顺延到下一个不重复的。
2. **Cue**：把 5 个 item 的 40 个 cue 合在一起，按出现次数排序，平局时按 item 顺序再按 item 内位置，取前 8 个不重复的。
   - predicted cues 组用模型预测的 cue；
   - catalog cues 组和 SID-only 组用各自选中歌曲的 catalog cue，聚合规则相同。
3. 其余全部不变：history-centroid 风格锚点、15 首历史歌元数据、Qwen3-4B-Instruct-2507 greedy、ACE-Step v1 3.5B、60 steps、30 秒、seed `42000+i`、同一批 941 个 context。

建议加一个 `--item-aggregation {first,five}` 参数，默认值保持 `first`，旧结果可以原样复现。

## 实验组

两个数据集（MPD、M4A）各 941 个 context。

| 优先级 | 组 | 说明 |
|---|---|---|
| P0 | GenPlaylist (DDBC) + predicted cues, five-item | 主方法 |
| P0 | DDBC SID-only + catalog cues, five-item | 同 backbone 的对照 |
| P0 | GenPlaylist (DDBC) SIDs + catalog cues, five-item | pipeline 反事实 |
| P1 | GenPlaylist (DDBC) SIDs + no cues, five-item | |
| P1 | Native TIGER + catalog cues, five-item | |
| P1 | TIGER SID-only / TIGER + cues, five-item | 等 TIGER matched 推理完成后再跑 |

History-only 不依赖预测 item，沿用已有音频，不用重跑。

P0 一共 3 × 941 × 2 = 5,646 个 clip。论文截止时间是 2026-11-17 19:59 SGT，P0 请务必在这之前完成，最好 11/15 前出结果。

## 评测和交付

- 指标和现有的一样：MERT history fit、CLAP prompt adherence、VGGish FAD、MERT diversity。
- 每组的 paired bootstrap 95% CI（10,000 次重采样，seed 42）。
- 额外交付：同一组 five-item 减 first-item 的 paired 差值和 CI。这能直接回答"用全部 5 个 item 是否更好"。
- 交付格式和 `docs/results/*-generation-*` 一样（`RESULTS.txt`、`mert-paired.json`、`secondary.json`），外加每组的表格行数值，放到新目录 `docs/results/five-item-generation-<日期>/`。
- 任何组如果跑不完，如实说明完成到哪一步，不要拿 first-item 的结果顶替。
