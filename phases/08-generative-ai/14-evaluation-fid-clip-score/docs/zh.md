# Evaluation — FID, CLIP Score, Human Preference（评估 —— FID、CLIP Score、Human Preference）

> 每一个生成模型排行榜都引用 FID、CLIP score 和来自人类偏好竞技场的胜率。每个数字都有一个坚定的研究者可以游戏的失败模式。如果你不知道失败模式，你就不能区分真正的改进和一个游戏运行。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 01 (Taxonomy)（Phase 8 第 01 课 分类）, Phase 2 · 04 (Evaluation Metrics)（Phase 2 第 04 课 评估指标）
**Time:** ~45 minutes（约 45 分钟）

## The Problem（问题）

一个生成模型在*样本质量*和*条件遵循*上被评判。两者都没有闭式测量。你的模型必须渲染 10,000 张图像；必须有一些东西给它们分配数字；你必须信任跨模型家族、跨分辨率、跨架构的数字。三个指标在 2014-2026 年的考验中幸存下来：

- **FID (Fréchet Inception Distance)。** 两个分布之间的距离 —— 真实和生成 —— 在 Inception 网络的特征空间中。越低越好。
- **CLIP score。** 生成图像的 CLIP 图像嵌入和提示的 CLIP 文本嵌入之间的余弦相似度。越高越好。测量提示遵循。
- **Human preference（人类偏好）。** 在相同提示下将两个模型正面竞争，让人类（或一个 GPT-4 级模型）挑选更好的一个，聚合成 Elo 分数。

你还将看到：IS（inception score，大部分退役）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每一个都修正了前一个的一个失败。

## The Concept（概念）

![FID, CLIP, and preference: three axes, different failure modes（FID、CLIP 和偏好：三个轴，不同的失败模式）](../assets/evaluation.svg)

### FID —— sample quality（样本质量）

Heusel et al.（2017）。步骤：

1. 为 N 个真实图像和 N 个生成图像提取 Inception-v3 特征（2048-D）。
2. 将高斯拟合到每个池：计算均值 `μ_r, μ_g` 和协方差 `Σ_r, Σ_g`。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解释：特征空间中两个多元高斯之间的 Fréchet 距离。越低 = 越相似的分布。

失败模式：
- **Biased on small N（在小 N 上有偏）。** FID 是特征分布上的均方 —— 小 N 低估协方差，给出错误地低 FID。始终使用 N ≥ 10,000。
- **Inception-dependent（依赖 Inception）。** Inception-v3 是在 ImageNet 上训练的。远离 ImageNet 的领域（人脸、艺术、文本图像）产生无意义的 FID。使用领域特定的特征提取器。
- **Gaming（游戏）。** 过度拟合 Inception 先验给出低 FID 而没有视觉质量改进。用 CMMD（见下）击败它。

### CLIP score —— prompt adherence（提示遵循）

Radford et al.（2021）。对于一个生成图像 + 提示：

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

跨 30k 生成图像平均 → 一个可在模型之间比较的标量。

失败模式：
- **CLIP's own blind spots（CLIP 自己的盲点）。** CLIP 在组合推理上很弱（“a red cube on a blue sphere”经常失败）。模型可以在 CLIP score 上排名良好，而无需真正遵循复杂提示。
- **Short prompt bias（短提示偏差）。** 短提示在野外有更多 CLIP 图像匹配。长提示机械地有更低的 CLIP score。
- **Prompt gaming（提示游戏）。** 在提示中包含“high quality, 4k, masterpiece”在没有改进图像-文本绑定的情况下膨胀 CLIP score。

CMMD（Jayasumana et al.，2024）修复了其中一些：使用 CLIP 特征而不是 Inception，最大均值差异而不是 Fréchet。在检测微妙质量差异方面更好。

### Human preference —— the ground truth（人类偏好 —— 真实情况）

挑选一个提示池。用模型 A 和模型 B 生成。将配对展示给人类（或一个强大的 LLM 法官）。将胜利聚合成 Elo 或 Bradley-Terry 分数。基准：

- **PartiPrompts (Google)**：1,600 个多样提示，12 个类别。
- **HPSv2**：107k 人类注释，广泛用作自动化代理。
- **ImageReward**：137k 提示-图像偏好对，MIT 许可。
- **PickScore**：在 Pick-a-Pic 2.6M 偏好上训练。
- **Chatbot-Arena-style image arenas**：https://imagearena.ai/ 和其他。

失败模式：
- **Judge variance（法官差异）。** 非专家与专家有不同的偏好。使用两者。
- **Prompt distribution（提示分布）。** 精选的提示有利于一个家族。始终记录。
- **LLM-judge reward hacking（LLM 法官奖励黑客）。** GPT-4 法官被漂亮但错误的输出愚弄。与人类三角测量。

## Use together（一起使用）

一个生产评估报告应该包括：

1. 针对保留真实分布的 10-30k 样本的 FID（样本质量）。
2. 相同样本与它们的提示的 CLIP score / CMMD（遵循）。
3. 在盲目竞技场中对先前模型的胜率（总体偏好）。
4. 失败模式分析：50 个随机抽样输出，标记为已知问题（手部解剖、文本渲染、一致对象计数）。

任何单一指标都是一个谎言。三个相互支持的指标 + 定性审查是一个主张。

```figure
gx-fid-distributions
```

## Build It（动手实现）

`code/main.py` 在合成“特征向量”（我们使用 4-D 向量作为 Inception 特征的替代品）上实现 FID、CLIP-score-like 和 Elo 聚合。你看到：

- 在小 N 和大 N 上的 FID 计算 —— 偏差。
- 作为特征池之间余弦相似度的“CLIP score”。
- 来自合成偏好流的 Elo 更新规则。

### Step 1: FID in four lines（四行 FID）

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### Step 2: CLIP-style cosine-similarity（CLIP 风格余弦相似度）

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### Step 3: Elo aggregation（Elo 聚合）

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## Pitfalls（陷阱）

- **FID at N=1000（N=1000 时的 FID）。** 启发式在 N<10k 下不可靠。报告低 N FID 的论文是在游戏。
- **Comparing FID across resolutions（跨分辨率比较 FID）。** Inception 的 299×299 调整大小改变了特征分布。仅在匹配分辨率下比较。
- **Reporting one seed（报告一个种子）。** 最少运行 3 个种子。报告 std。
- **CLIP score inflation via negative prompts（通过负面提示膨胀 CLIP score）。** 一些管道通过过度拟合提示来提高 CLIP。检查视觉饱和度。
- **Elo bias from prompt overlap（来自提示重叠的 Elo 偏差）。** 如果两个模型都在训练期间看到了基准提示，Elo 是无意义的。使用保留提示集。
- **Human eval paid-crowd skew（人工评估付费人群偏差）。** Prolific、MTurk 注释者偏向更年轻/更 tech-friendly。与招募的艺术/设计专家混合。

## Use It（实际应用）

2026 年生产评估协议：

| Pillar（支柱） | Minimum（最小） | Recommended（推荐） |
|--------|---------|-------------|
| Sample quality（样本质量） | FID on 10k vs held-out real（与保留真实的 10k FID） | + CMMD on 5k + FID on subset per category |
| Prompt adherence（提示遵循） | CLIP score on 30k | + HPSv2 + ImageReward + VQA-style question answering |
| Preference（偏好） | 200 blinded pairs vs baseline | + 2000 paired human + LLM-judge + Chatbot Arena |
| Failure analysis（失败分析） | 50 hand-flagged（50 手动标记） | 500 hand-flagged + automated safety classifier |

一个报告中的全部四个支柱 = 主张。任何单独一个 = 营销。

## Ship It（交付）

保存为 `outputs/skill-eval-report.md`。该技能接收一个新模型检查点 + 基线，并输出一个完整的评估计划：样本大小、指标、失败模式探测、签署标准。

## Exercises（练习）

1. **Easy（简单）。** 运行 `code/main.py`。在相同的合成分布上比较 N=100 和 N=1000 的 FID。报告偏差大小。
2. **Medium（中等）。** 从合成 CLIP 风格特征实现 CMMD（见 Jayasumana et al.，2024 的公式）。与 FID 相比，对质量差异的敏感性。
3. **Hard（困难）。** 复制 HPSv2 设置：从 Pick-a-Pic 的子集中取 1000 个图像-提示对，在偏好上微调一个小型 CLIP 基础评分器，并测量它与保留集的一致。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| FID | "Fréchet Inception Distance"（Fréchet Inception Distance） | 真实 vs 生成 Inception 特征的高斯拟合之间的 Fréchet 距离。 |
| CLIP score | "Text-image similarity"（文本-图像相似度） | CLIP 图像和文本嵌入之间的余弦相似度。 |
| CMMD | "FID's replacement"（FID 的替代品） | CLIP 特征 MMD；偏差更小，没有高斯假设。 |
| IS | "Inception score"（Inception score） | Exp KL(p(y|x) || p(y))；在现代模型上相关性差，退役。 |
| HPSv2 / ImageReward / PickScore | "Learned preference proxies"（学习的偏好代理） | 在人类偏好上训练的小模型；用作自动法官。 |
| Elo | "Chess rating"（象棋评分） | 配对胜利的 Bradley-Terry 聚合。 |
| PartiPrompts | "The benchmark prompt set"（基准提示集） | 跨 12 个类别的 1,600 个 Google 策划提示。 |
| FD-DINO | "Self-sup replacement"（自监督替代品） | 使用 DINOv2 特征的 FD；对于 ImageNet 外领域更好。 |

## Production note: evaluation is an inference workload too（生产笔记：评估也是一个推理工作负载）

在 10k 样本上运行 FID 意味着生成 10k 张图像。对于在单个 L4 上 1024² 的 50 步 SDXL 基础，这是约 11 小时的单请求推理。评估预算是真实的，框架正好是离线推理场景（最大化吞吐量，忽略 TTFT）：

- **Batch hard, forget latency（硬批处理，忘记延迟）。** 离线评估 = 在适合内存的最大尺寸上的静态批处理。`pipe(...).images`，在 80GB H100 上 `num_images_per_prompt=8` 运行 4-6 倍比单请求快挂钟。
- **Cache the real features（缓存真实特征）。** 对真实参考集的 Inception（FID）或 CLIP（CLIP score、CMMD）特征提取运行*一次*，存储为 `.npz`。不要每次评估重新计算。

对于 CI / 回归门：每个 PR 在 500 样本子集上运行 FID + CLIP score（约 30 分钟）；每夜运行完整 10k FID + HPSv2 + Elo。

## Further Reading（延伸阅读）

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) —— FID 论文。
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) —— CMMD。
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) —— CLIP。
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) —— HPSv2。
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) —— ImageReward。
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) —— PartiPrompts。
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) —— 失败模式调查。
