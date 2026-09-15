# Jamba：混合 SSM-Transformer（Hybrid SSM-Transformer）

> 状态空间模型（state space models，SSM）和 transformer 想要的东西不同。Transformer 用注意力换质量，代价是二次方。SSM 用递推换线性时间推理和常数内存，但质量落后。AI21 的 Jamba（2024 年 3 月）和 Jamba 1.5（2024 年 8 月）把它们放进同一个模型：每 7 个 Mamba 层配 1 个 Transformer 层，每隔一块用 MoE，以及能塞进单张 80GB GPU 的 256k 上下文窗口。Mamba-3（ICLR 2026）用复值状态空间和 MIMO 投影收紧 SSM 一侧。本课把两套架构从头读到尾，并解释为什么混合配方在纯 SSM 和纯 Transformer 长上下文尝试都没站住的三年里活了下来。

**Type:** Learn
**Languages:** Python (stdlib, layer-mix calculator)
**Prerequisites:** Phase 10 · 14 (open-model architectures), Phase 10 · 17 (native sparse attention)
**Time:** ~60 minutes

## 学习目标（Learning Objectives）

- 解释 Jamba 块里的三个原语——Transformer 层、Mamba 层、MoE——以及 1:7:偶数交错配方
- 在高层次陈述 SSM 的递推长什么样，以及为什么它能实现常数内存推理
- 计算 Jamba 模型在 256k 上下文下的 KV 缓存占用，并与纯 Transformer 模型所需比较
- 说出 Mamba-3 的三项创新（指数梯形离散化、复值状态更新、MIMO）以及各自瞄准的问题

## 问题（The Problem）

注意力对序列长度是二次的。状态空间模型是线性的。这个差异会放大：在 256k token 时，Transformer 注意力图每个头有 65B 个条目；SSM 的递推状态无论序列多长都是固定大小。

纯 SSM 模型（Mamba、Mamba-2）在小规模上匹配 Transformer 困惑度，但在状态跟踪任务上落后，并在某些上下文内检索类别上失败。直觉：SSM 把历史压进固定状态，历史一长信息就泄漏。注意力精确记住一切，但付二次代价。

显而易见的修法：两者都用。在精确召回要紧的地方放 Transformer 层。其余地方用 SSM 层。调比例。Jamba 是第一个把这套混合配方做到生产级规模的模型（总计 52B，激活 12B，256k 上下文，单张 80GB GPU）。Jamba 1.5 把家族扩到总计 398B / 激活 94B。Mamba-3（ICLR 2026）是当前最好的纯 SSM 基线，混合模型可以围着它重建。

本课阅读这三篇论文，产出「选对比例」的心智模型。

## 概念（The Concept）

### 一页讲完 SSM（An SSM in one page）

状态空间模型通过固定大小的状态 `h` 处理序列 `x_1, ..., x_N`：

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

每一步状态经线性动力学 `A` 演化，吃进输入 `B x_t`，发出输出 `C h_t`。`A, B, C` 可以学习。注意关键性质：计算 `y_t` 只需要 `h_{t-1}` 和 `x_t`，不需要任何更早的 `x`。内存是常数。推理是每 token O(1)。

建模质量的诀窍在 `A` 的结构。S4（Gu 2021）用高度结构化的矩阵，训练时能作为长卷积高效求值。Mamba（Gu、Dao 2023）把固定的 `A, B, C` 换成数据依赖的（「选择性」部分）。Mamba-2（2024）进一步简化结构。Mamba-3（2026）在特定位置重新加回复杂度。

关键性质：对解码器 LLM，SSM 层是注意力层的即插替换，用固定大小的每层状态代替不断增长的 KV 缓存。

### Jamba 块（The Jamba block）

Jamba 块按两个数字交错层：

- `l`：注意力对 Mamba 的比例。Jamba 用 `l = 8`，意思是每 7 个 Mamba 层配 1 个 Transformer 层（7 Mamba + 1 Attention = 每组 8 层）。
- `e`：MoE 频率。Jamba 用 `e = 2`，意思是每隔一层应用 MoE。

块内层序列：

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

每个 Jamba 块是 8 层。4 块深（总计 32 层）时，你得到 28 个 Mamba 和 4 个 Attention 层。其中 16 个用 MoE。

### 为什么是 1:7 比例（Why the 1:7 ratio）

AI21 做了消融：注意力对 Mamba 的什么比例给出最佳每参数困惑度，以及在他们的长上下文评测上最佳的上下文内召回？

- 注意力太多（1:1）：质量上去，但内存和速度变差。
- 注意力太少（1:15）：内存很好，但上下文内检索失败。
- 甜点：1:7 或 1:8。

直觉：Transformer 层处理精确召回和状态跟踪。Mamba 层处理廉价的主体加工。

### 位置编码（Positional encoding）

Mamba 层本身是位置感知的（经由递推）。最初基于 Mamba 的混合里，注意力层不用 RoPE——SSM 层提供位置信息。Jamba 1.5 给注意力层加上 RoPE，以获得更长上下文泛化，这是基于经验长上下文评测的事后改进。

### 内存预算（The memory budget）

对 Jamba-1 形状（32 层：28 Mamba + 4 Attention，隐层 4096，32 个注意力头）：

- KV 缓存（仅注意力层）：256k BF16 下 `2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`。只有 4 个注意力层贡献。
- SSM 状态：每个 token 前缀 `28 * hidden * state_size`，但这是每层固定大小，不随序列长度缩放。典型 Mamba 状态是每特征 16，隐层 4096：`28 * 4096 * 16 * 2 = 3.7 MB` 总计。

对比纯 Transformer 32 层、同样隐层、32 头全 MHA：256k BF16 下 `2 * 32 * 32 * 128 * 256k * 2 = 128 GB`。KV 缓存降低 8 倍。即使对比 2024 年大多数模型用的 GQA(8) 基线（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`），Jamba 的 1:7 混合在 16 GB 仍小 2 倍。

这就是 AI21 说「单张 80GB GPU 上 256k 上下文」的意思。全 MHA 纯 Transformer 的 KV 缓存塞不进去；即使 GQA 基线也没给权重和激活留空间；Jamba 的能。

### Mamba-3：2026 年的纯 SSM 基线（Mamba-3: the pure-SSM baseline in 2026）

Mamba-3（ICLR 2026，arXiv:2603.15569）在纯 SSM 一侧引入三项创新：

1. **指数梯形离散化（exponential-trapezoidal discretization）。** 用更有表达力的递推替换 Mamba-2 里的欧拉法离散化。卷积式运算作用在核心递推内部的状态-输入上，而不是作为 `x_t` 上的外层卷积。

2. **复值状态更新（complex-valued state update）。** 先前的 Mamba 把状态矩阵从复数（S4）减到实对角（Mamba）再到缩放单位阵（Mamba-2）。Mamba-3 重新加回复数——等价于状态上的数据依赖旋转嵌入。这恢复了先前实值简化所付出的状态跟踪能力。

3. **多输入多输出（multi-input multi-output，MIMO）投影。** 不用每特征标量投影，而用矩阵值投影。提高建模能力与推理时硬件利用率，同时不增加解码延迟。

在 1.5B 参数上，Mamba-3 相对 Gated DeltaNet 把平均下游准确率提高 0.6 点；MIMO 变体再加 1.2，总计 1.8 点增益。在同样状态大小下，Mamba-3 用一半状态匹配 Mamba-2。

Mamba-3 尚未以生产级混合规模出货——但它是下一款 Jamba 级模型 SSM 一侧的显然候选。

### 何时伸手去拿混合（When to reach for a hybrid）

混合赢在：

- 上下文长到纯 Transformer KV 缓存开始疼（64k+）。
- 任务混合短程结构（SSM 擅长）与长程召回（需要 Transformer）。
- 你想部署在单 GPU 内存预算上，纯 Transformer 的 KV 缓存单独就塞不进去。

混合输在：

- 上下文短（低于 16k）。SSM 开销浪费；纯 Transformer 就够。
- 任务需要处处对处处的注意力（深度推理、多文档交叉引用）。混合里注意力层的稀疏会伤。
- 你在扩到万亿参数前沿模型。纯 Transformer + MLA + MoE（DeepSeek-V3 风格）目前在能力竞赛中领先。

### 竞争格局（The competitive landscape）

| 模型 | 家族 | 规模 | 独特主张 |
|-------|--------|------|-------------|
| Mamba-2 | 纯 SSM | 3B | 线性时间，常数内存 |
| Jamba | 混合 | 52B/12B | 80GB 上 256k |
| Jamba 1.5 Large | 混合 | 398B/94B | 企业级长上下文 |
| Mamba-3 | 纯 SSM | 1.5B（论文） | 状态跟踪被恢复 |
| DeepSeek-V3 | 纯 Transformer + MoE | 671B/37B | 前沿能力 |

2026 年格局：纯 Transformer MoE 主导前沿，但混合拥有 256k 以上上下文的利基。Mamba-3 的状态跟踪胜利可能在下一代把混合比例压得更低（更多 SSM，更少注意力）。

```figure
swiglu-ffn
```

## 使用它（Use It）

`code/main.py` 是混合架构的内存计算器。给定 SSM-Transformer 比例以及隐层大小 / 层数配置，它计算：

- 目标上下文下的 KV 缓存。
- SSM 状态内存。
- 一系列模型形状在上下文 N 下的总内存。

计算器支持：

- 纯 Transformer 基线（KV 缓存随 N 增长）。
- Jamba 风格 1:7 混合。
- 纯 SSM（完全没有 KV 缓存）。

数字对已发布形状直接来自 Jamba-1 和 Jamba-1.5 论文，对假设变体做外推。

真实部署的集成考虑：

- 大多数生产推理服务器（vLLM、SGLang）支持 Jamba 和 Mamba。核对具体版本。
- 在 256k 上下文，Jamba 的内存优势体现在并发请求吞吐上。同一块 VRAM 能塞进比 Transformer 更多的 Jamba 序列。
- Mamba-3 作为独立模型尚未进入生产——1.5B 的研究预览。

## 交付（Ship It）

本课产出 `outputs/skill-hybrid-picker.md`。给定工作负载规格（上下文长度画像、任务混合、内存预算），它在纯 Transformer、Jamba 风格混合和纯 SSM 之间推荐，并给出内存与质量权衡的明确推理。

## 练习（Exercises）

1. 运行 `code/main.py`，计算 32 层纯 Transformer（隐层 4096，32 头）以及同样形状的 Jamba-1 混合在 256k 上下文下的 KV 缓存。验证 AI21 论文声称的约 8 倍内存降低。

2. 修改计算器，建模 1:3 混合（4 Mamba : 1 Attention）和 1:15 混合（14 Mamba : 1 Attention）。画出 KV 缓存对比例。在什么比例下 KV 缓存等于 SSM 状态内存？

3. 阅读 Jamba 论文（arXiv:2403.19887）第 3 节。解释为什么 AI21 用 Mamba-1 而不是更快的 Mamba-2。提示：混合消融节记录了这一点。

4. 计算 Jamba 1.5 Large（总计 398B，激活 94B）每隔一层 MoE 的参数开销。把激活比与 DeepSeek-V3（37B/671B）比较，并解释为什么 Jamba 的架构把激活比推得更高。

5. 阅读 Mamba-3 论文（arXiv:2603.15569）第 3 节。用三句话解释为什么复值状态更新等价于数据依赖旋转嵌入。把答案和 Phase 7 · Lesson 04 的 RoPE 推导联系起来。

## 关键术语（Key Terms）

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 状态空间模型（SSM） | 「带固定状态的递推」 | 带可学习递推 `h_t = A h_{t-1} + B x_t` 的层；每 token 常数内存 |
| 选择性 SSM | 「Mamba 的诀窍」 | 数据依赖的 A、B、C 参数，在线性时间给出类似门控的选择性 |
| 注意力对 Mamba 比例 | 「有多少注意力层」 | 在 Jamba 里，`l = 8` 表示每 7 个 Mamba 层配 1 个注意力层 |
| Jamba 块 | 「那组 8 层」 | 一个注意力 + 七个 Mamba + 交替位置上的 MoE |
| SSM 状态 | 「隐藏缓冲区」 | 每层固定大小的状态，替代 Mamba 层的 KV 缓存 |
| 256k 上下文 | 「Jamba 的旗舰数字」 | Jamba-1 能塞进单张 80GB GPU 的序列长度；同样规模纯 Transformer 不能 |
| Mamba-3 | 「2026 纯 SSM」 | 当前最好的纯 SSM 架构，带复值状态 + MIMO；混合围绕它重建的基线 |
| MIMO | 「多输入多输出」 | Mamba-3 创新，用矩阵值投影代替每特征标量 |
| 指数梯形离散化 | 「Mamba-3 的递推」 | 更有表达力的递推，涵盖 Mamba-2 的欧拉法离散化 |
| 混合架构 | 「混合注意力和 SSM」 | 任何交错 Transformer 与 SSM 层的模型；Jamba 是生产原型 |

## 延伸阅读（Further Reading）

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — 最初的 Jamba 论文，比例消融，256k 上下文主张
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — 放大后的家族，398B/94B 与 12B/52B 公开发布
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — Jamba 所基于的选择性 SSM 论文
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — 简化后的结构化状态空间后继
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — 复值状态、MIMO，2026 纯 SSM 前沿
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — S4 论文，LLM 用 SSM 谱系的起点
