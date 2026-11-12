# 实验请求：重建 creative cues（cue v2）

提出日期：2026-11-12。论文写作方（Claude）与作者讨论后提出，实验方（Codex）执行。

**执行方只跑实验、交付结果，不修改论文仓库（Odyssey-GenPlaylist）里的任何文件。不得覆盖或改写 v1 cue 产物（见 `docs/CUE_VOCAB_FREEZE.md`），一切新产物另建版本目录并记录哈希。push 仍需作者每次明确授权。**

截稿：2026-11-17 19:59 SGT。本请求分两个阶段，**阶段 A 完成后停下汇报，等作者明确说“继续”再做阶段 B。**

---

## 1. 背景：v1 cue 的问题

v1 的做法（`src/02_creative_cues/cue_assign.py`，`strategy="relevance"`）：

- 把歌名、genre、mood、歌词摘要、tags 和完整歌词拼成一段文本；
- 用 all-MiniLM-L6-v2 编码成一个向量；
- 与 2,048 个公共词表 cue 算余弦相似度；
- 直接取 Top-16，训练使用前 8。

已确认的问题：

1. **跨歌曲集中（hubness）。** `pop rap` 出现在 50.75% 歌曲的 16 个候选中；Music4All 上 75.7% 的歌曲被分到 pop rap，前 10 种 cue 占全部出现次数的 49.1%。因此 Popularity 的 Cue F1 偏高。
2. **歌内冗余。** 8 月 4 日的提交把 Waikei 原先验证过的 MMR 分配改成了纯相似度排序，并把每首歌的 cue 数从 18 改为 8；同一首歌的 8 个 cue 常是同一概念的近义词。
3. **不忠实。** 例：Joshua Radin《Winter》，元数据为 singer-songwriter / folk / acoustic，却被分到 southern hip hop、karaoke、urban rhythm。原因是“全词表硬分配”：歌曲没有对应 cue 时，也会被迫选满。
4. **词表噪声。** 词表混有人名、歌名、歌词碎片和冒犯性用语。Music4All 直接复用 MPD 的词表。

## 2. 设计目标

8 个 cue 是一首歌的**可生成语义摘要**：

- 每个 cue 都有依据；
- 能指导生成；
- 8 个组合起来有辨识度；
- 单个 cue 能跨歌曲复用。

区分单首歌曲是 SID 的任务。cue 不追求罕见词，辨识度来自组合。选择优先级：

**有证据支持 ＞ 提供新信息 ＞ 能指导生成 ＞ 适度区分歌曲**

Waikei 设计的歌词意象/叙事类 creative cue 是本方法的特色，保留并作为主体。

## 3. 类别与软配额

| 类别 | 参考数量 | 内容 | 证据来源 |
|---|---|---|---|
| 声音特征 `sound` | ~2 | 最突出的乐器、律动、唱法、音色 | 音频或可靠元数据（genre、tags），**不能从歌词推断** |
| 情绪与表达 `emotion` | ~2 | 情绪状态、能量、表达姿态 | 歌词与元数据；“歌词表达的情绪”与“听起来的情绪”冲突时，以后者为准，或留空 |
| 主题与意象 `theme` | ~4 | 主题、场景、叙事关系、象征意象 | 歌词；允许概括和转述，不照抄原句，不添加原文没有的情节 |

- 总数固定为 8。2:2:4 是约数，可以浮动，例如 1+2+5、3+2+3。
- 无歌词或纯器乐的歌，把 theme 的位置让给 sound。
- 证据不足的位置填 `<unk>`（ID 0，作为 NONE），不为凑数编造。

## 4. 构造流程

### ① 逐歌提取候选（LLM）

- **模型**：用本地 vLLM 跑 Qwen3-8B-Instruct 或 Qwen3-4B-Instruct-2507，temperature 0，结构化 JSON 输出。在 `run_config.json` 里记录所用模型和提示词全文。
- **输入**：歌名、genre、mood、tags、歌词。**不提供 artist 字段。**
- **输出**：每个类别 3–8 个候选，每个候选包含：
  - `phrase`：2–5 个英文单词的短语；
  - `category`：sound / emotion / theme；
  - `evidence`：歌词或元数据中的依据片段，仅内部保存；
  - `confidence`：0–1。
- **提示词必须明确**：不输出人名、歌名、原歌词片段、冒犯性或露骨表达；声音类只能依据元数据或音频，不能依据歌词。
- **硬过滤**：在 LLM 之后再加一层过滤，覆盖冒犯词表和人名。人名检查可以对照 catalog 的 artist 和 title 字段。
- **歌词质量**：Waikei 的笔记记录过，部分歌词文件其实是错歌、发行列表或曲目列表。这类歌曲视为无歌词处理，并单独计数。

### ② 在两个数据集合并的约 1.6 万首歌上，建立新的公共词表

- **聚类**：把全部候选按类别分别做语义聚类，得到规范化短语。每簇的规范名取最接近簇中心的短语，或由 LLM 命名。
- **规模**：词表仍为 2,048 个 ID（`0=<unk>`，1..2047 为真实 cue），**token 布局和模型结构都不变**。
- **类别预算**：参考值为 sound ≈ 300、emotion ≈ 200、theme ≈ 1,500，可以按候选分布调整。
- **粒度下限**：一个规范 cue 至少被 5 首歌共享。达不到的并入更概括的概念，例如“想重逢却不敢联系”并入“hesitant reunion”。原始自由短语保留在内部记录里。

### ③ 映射与选择

1. **映射**：把每首歌的候选映射到规范词表。映射分数为“相似度 − 该规范词对全体候选的平均相似度”，用来软性压低泛化词。**不机械删除高频词**，例如 `acoustic guitar` 很常见，但对生成有用。
2. **相关性门槛**：先过滤掉证据或置信度不足的候选，罕见度不提供任何加分。
3. **去冗余**：按 MMR 选择（可从 λ=0.7 起），同时参考 2:2:4 软配额。允许类别之间互相补位。
4. **固定顺序**：按 sound → emotion → theme 排列，同类内按分数降序。类别标签只保存在内部记录里，不增加模型 token。

### ④ 产物

新建版本目录，例如 `src/02_creative_cues/outputs/cue_v2/<timestamp>/`，包含：

- `cue_vocab.json`（含类别）；
- `item2cues.json`（每首 8 个）；
- `item2cue_candidates.json`（候选、证据、置信度）；
- `run_config.json`；
- SHA256 记录。

## 5. 阶段 A：试点与诊断（目标 11/12 当天完成）

### A1. 试点

- **抽样**：每个数据集抽 100 首，按曲风、流行度和歌词完整度分层。
- **对照组**（每组 8 个 cue）：
  1. v1：纯相似度 Top-8，即现有产物；
  2. v1 词表 + MMR（`strategy="mmr"`），单独测去冗余的作用；
  3. v2：本请求的新方案。
- 试点阶段的 v2 词表可以只在试点样本加一个更大的候选池上建立，全量阶段再重建。

### A2. 不训练的诊断（三组都算）

| 诊断 | 做法 |
|---|---|
| 真实性 | LLM 逐个判断每个 cue 是否有证据支持，报告支持率；另外导出 40 首（每个数据集 20 首）v1 / v1+MMR / v2 并排的可读表，供作者人工抽查 |
| 冗余度 | 8 个 cue 两两之间的平均和最大语义相似度 |
| 近似歌曲区分度 | 只在同曲风、相近情绪的歌曲子集里，用 8 个 cue 检索原歌，报告 MRR 和 Recall@10 |
| 集中度 | 单个 cue 最多出现在多少比例的歌曲里、前 10 个 cue 的占比、熵 |
| 个性化信号 | 在 941 个评测上下文上，用历史 15 首歌出现最多的 8 个 cue 作为预测，与 Popularity 对比逐首匹配 Cue F1，报告两者差距 |

全量诊断（集中度、个性化信号）可以直接在全目录上用 v2 算，不必等重训。

### A3. 生成门槛测试（决定是否进入阶段 B）

- **样本**：每个数据集抽 200 个上下文。固定条件 item 为**第一首目标歌曲**。
- **三组渲染**：
  1. 该歌的 v2 catalog cues；
  2. 该歌的 v1 catalog cues；
  3. 不加 cues。
- **渲染设置**与现有 first-item 流水线相同：Qwen3-4B-Instruct-2507 greedy、ACE-Step v1 3.5B、60 steps、30 秒、seed `42000+i`。
- **指标**：
  - MERT-OAS：与观察到的 5 首后续歌曲的相似度；
  - Content fit：生成的歌词和属性文本与 5 首后续歌曲 catalog cue 文本之间的相似度（v1 / v2 各算一次）；
  - MERT history fit；
  - FAD（VGGish）。
- **报告**：给出配对差值和 bootstrap 95% 置信区间。共约 1,200 段音频。

### A4. 阶段 A 交付

结果包放在 `docs/results/cue-v2-20261112/`，包含：

- READ-ME（中文）；
- 诊断 JSON；
- 40 首并排对照表；
- 门槛测试结果；
- 运行配置和哈希。

**在对话里用中文汇报要点后停下**，等作者决定是否进入阶段 B。

作者参考的通过条件（最终由作者判断）：

- v2 的真实性明显高于 v1；
- v2 的集中度明显低于 v1；
- 个性化信号的差距不小于 v1；
- 在门槛测试中，v2 cues 相对“不加 cues”提升了 MERT-OAS 或 Content fit，并且不差于 v1。

## 6. 阶段 B：全量重建与重训（作者确认后再做）

1. **全量构造**：对约 1.6 万首歌全量提取、建词表、选 8 个，生成 v2 正式产物。
2. **重训 4 个模型**，配方与当前主表一致，只替换 cue：
   - **DDBC + cues v2**（MPD、M4A）：从官方 DDBC 热启动，先训 5,000 步、再训 1,000 步，batch 512，学习率 3e-4 → 1e-4，seed 1，cue 权重 0.1，三种 mask 等概率；推理 256 步、beam 4。**同时报告 α=1 和 α=0.2。**
   - **TIGER + cues v2**（MPD、M4A）：从头训练 20k 步，batch 256，Adafactor，seed 1，损失为 L_SID + 0.1 L_cue + 0.1 L_EOS，beam 20。
   - SID-only 基线不重训，沿用现有结果。
3. **推荐评测**：941 个上下文，全目录和“100 negatives + targets”两种候选设置，报告：
   - Recall@5；
   - Diversity（catalog coverage）；
   - **逐首最优匹配 Cue F1**（总体，以及 sound / emotion / theme 分别计算）。

   SID-only、Popularity、Random 用所选歌曲的 v2 catalog cue 重新计分。原合并口径的 Cue F1 一并附上，供对照。
4. **音频**：两个数据集各 941 个上下文，first-item 渲染，P0 组为：
   - DDBC + cues v2，predicted cues；
   - DDBC SID-only，所选歌曲的 v2 catalog cues；
   - TIGER + cues v2，predicted cues；
   - TIGER SID-only，所选歌曲的 v2 catalog cues。

   **P1 组**：DDBC + cues v2 的 SIDs 搭配不加 cues。

   指标：MERT-OAS、Content fit、MERT history fit、FAD。**不再跑 CLAP。**
5. **交付**：`docs/results/cue-v2-full-2026111X/`，内容同上，并附一份主表行的建议更新（只给数值和来源，不改论文文件）。

## 7. 需要先确认并在阶段 A 汇报的事项

1. **MPD 是否有可用音频或可靠的乐器类标签。**
   - 如果没有：两个数据集的 sound 类都只依据 genre/tags，以保持对称；Music4All 的音频只用于抽查 sound cue 是否属实。
   - 如果有：说明覆盖率。
2. **v1 的冒犯性用语曾进入 Qwen 的歌词提示词。** 请扫描现有生成歌词（first-item 11,292 条和 five-item 9,000 条），列出命中数量和样例 ID。只报告，不删除。
3. 给出各步骤实际耗时，以及 GPU 占用情况。

## 8. 参考

- v1 冻结记录：`docs/CUE_VOCAB_FREEZE.md`
- v1 分配实现：`src/02_creative_cues/cue_assign.py`（`_song_text`、`assign_all`）
- 提取提示词：`src/02_creative_cues/cue_extractors.py`
- Waikei 的进展报告与核查记录：Codex 本机 `outputs/GenPlaylist-waikei-cue-notes-audit-20261112/`
- 逐首匹配 Cue F1 的重算记录：Codex 本机 `outputs/GenPlaylist-cue-matching-audit-20261112/`
