# 原生稀疏注意力（Native Sparse Attention，DeepSeek NSA）

> 在 64k token 时，注意力（attention）会吃掉 70–80% 的解码延迟。每个开放模型实验室都有修复方案。DeepSeek 的 NSA（ACL 2025 最佳论文）是真正站住脚的那一个：三条并行注意力分支——压缩后的粗粒度 token、选择性保留的细粒度 token，以及给局部上下文用的滑动窗口——再通过可学习门控（learned gate）组合。它对齐硬件（内核友好），原生可训练（在预训练里生效，而不是推理时再打补丁），在 64k 解码上比 FlashAttention 更快，质量匹配甚至超过全注意力。本课端到端构建这三条分支，并说明为什么这种稀疏性是端到端可微的。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 12 (KV cache, flash-attention), Phase 7 · 15 (attention variants), Phase 10 · 16 (differential attention)
**Time:** ~60 minutes

## 学习目标（Learning Objectives）

- 陈述 NSA 的三条注意力分支，以及各自捕获什么信息
- 解释为什么 NSA 是「原生可训练的」，而先前的稀疏注意力方法往往只能用于推理
- 以压缩块大小和选择 top-k 为函数，计算 64k 上下文下 NSA 相对全注意力的计算节省
- 在短合成序列上用标准库 Python 实现三分支组合，并验证门控权重的行为

## 问题（The Problem）

序列长度为 N 时，全注意力（full attention）的时间代价是 `O(N^2)`，每层 KV 缓存（KV cache）是 `O(N)`。到 64k token 时，算力和内存带宽数字是灾难性的。NSA 论文给出的理论估计：64k 时注意力占总体解码延迟的 70–80%。下游一切——首 token 时间（TTFT）、tokens/sec、每百万 token 成本——都被注意力成本主导。

稀疏注意力（sparse attention）是显而易见的答案。先前尝试分成两类。固定模式稀疏（滑动窗口、跨步、块局部）会丢掉信息，在长程召回任务上失败。推理时稀疏（KV 缓存剪枝、H2O、StreamingLLM）是给已经用稠密注意力预训练好的模型打补丁，只能回收一部分潜在加速，因为模型从未被要求通过稀疏模式路由信息。

原生稀疏注意力（Native Sparse Attention，Yuan 等人，DeepSeek + PKU + UW，ACL 2025 最佳论文，arXiv:2502.11089）两者都做：模型在预训练中学习稀疏模式，同时实现成对齐内核的算法，推理时真能兑现计算节省。两年后，NSA 或其直系后继会成为每个前沿长上下文模型的默认注意力。

## 概念（The Concept）

### 三条并行分支（Three parallel branches）

对每个查询（query），NSA 对 KV 缓存的三种不同视图各跑一次注意力：

1. **压缩分支（compressed branch）。** Token 按大小为 `l` 的块分组（通常 32 或 64）。每个块经一个小型可学习 MLP 压缩成一个摘要 token。查询在这些压缩 token 上做注意力，得到整段序列的粗粒度视图。

2. **选择分支（selected branch）。** 利用压缩分支的注意力分数，找出对当前查询最相关的 top-k 块。读出这些块里未压缩的细粒度 token，查询再对它们全部做注意力。可以把压缩分支注意力看成选择的路由信号。

3. **滑动窗口分支（sliding-window branch）。** 查询对最近 `W` 个 token（通常 512）做注意力，以获取局部上下文。这一支捕获另外两支可能漏掉的、结构密集的短程模式（句法、局部共指）。

三个分支输出通过每个位置上可学习的门控组合：

```
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win` 是查询上一个小型 MLP 产出的门控权重。它们不必和为 1——可以独立加权各分支。

### 为什么这是「原生可训练的」（Why this is "natively trainable"）

选择步骤（top-k 块）是离散的。离散操作会切断梯度流。先前的稀疏注意力要么不把反传穿过选择（限制训练），要么用连续松弛，推理时得不到真正的稀疏。

NSA 绕开了这一点：压缩分支注意力本身就是对整段序列可微的粗粒度注意力。top-k 只是复用压缩分支的最高注意力分数，决定加载哪些细粒度块。梯度穿过压缩分支分数（既影响压缩输出，也影响选择逻辑），被选中块对最终输出的贡献同样可微。不可微的 `top_k` 在前向计算图上相当于空操作——它只控制从内存加载哪些块。

这就是 NSA 能端到端用于预训练的原因。模型联合学习通过三条分支路由信息，得到一种在推理时真能兑现承诺加速的稀疏模式。

### 硬件对齐内核（Hardware-aligned kernel）

NSA 的内核按现代 GPU 内存层次设计。内核按 GQA 组加载查询（外循环），再按组取出对应的稀疏 KV 块（内循环），在 SRAM 上跑注意力。因为每个查询组看到相同的被选块（选择按查询组而不是按查询头），KV 加载在组内摊销。算术强度保持较高。

论文报告 Triton 内核在 64k 解码上比 FlashAttention 快 9 倍，加速比随序列长度增长。前向和反向内核都提供。

### 计算预算（The compute budget）

令 `N` 为序列长度，`l` 为压缩块大小，`k` 为 top-k 选择个数，`w` 为滑动窗口，`b` 为被选块大小（通常等于 `l`）。

- 压缩分支：每个查询 `O(N/l)` 个 key，总计 `O(N * N / l)`。
- 选择分支：每个查询 `O(k * b)` 个 key，总计 `O(N * k * b)`。
- 滑动分支：每个查询 `O(w)` 个 key，总计 `O(N * w)`。

总计：`O(N * (N/l + k*b + w))`。

取 `N = 64k, l = 64, k = 16, b = 64, w = 512`：每查询代价是 `1000 + 1024 + 512 = 2536` 个 key。全注意力是 `64000` 个 key。计算量降低约 25 倍。

取 `N = 128k, l = 64, k = 16, b = 64, w = 512`：每查询代价是 `2000 + 1024 + 512 = 3536` 个 key。全注意力是 `128000` 个 key。降低约 36 倍。收益随序列长度增长，这正是要点。

### 如何比较（How does it compare）

| 方法 | 可微 | 真实推理加速 | 长程召回 |
|--------|---------------|----------------------|-------------------|
| 仅滑动窗口 | 是 | 是 | 失败 |
| 跨步 / 块稀疏 | 是 | 是 | 部分 |
| KV 剪枝（H2O、StreamingLLM） | 不适用（推理时） | 是 | 部分 |
| MoBA（Moonshot） | 部分 | 是 | 好 |
| NSA | 是（原生） | 是（64k 上 9 倍） | 匹配全注意力 |

MoBA（Moonshot，arXiv:2502.13189）同期发表，采用类似的「三个比一个好」思路，把 MoE 原则用到注意力块上。NSA 与 MoBA 是 2026 年长上下文预训练必须知道的两个架构。

```figure
sliding-window-attention
```

## 动手构建（Build It）

`code/main.py` 在短合成序列上实现三条分支，并展示：

- 压缩 MLP（教学上用简单均值池化基线；真正的 NSA 用可学习 MLP）。
- 由压缩分支分数驱动的 top-k 块选择。
- 对最后 `w` 个 token 的滑动窗口注意力。
- 门控组合。
- 与全注意力对比的计算计数打印。

### 第 1 步：把 token 压缩成块（Step 1: compress tokens into blocks）

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### 第 2 步：压缩分支注意力（Step 2: compressed-branch attention）

对压缩后的 key 跑查询的 softmax 注意力。压缩分支分数同时作为 top-k 选择的信号。

### 第 3 步：top-k 块选择（Step 3: top-k block selection）

选出得分最高的 `k` 个压缩块的索引。加载这些块里原始未压缩 token，并对它们跑注意力。

### 第 4 步：滑动窗口注意力（Step 4: sliding-window attention）

取最后 `w` 个 token，对它们跑标准注意力。

### 第 5 步：门控 + 组合（Step 5: gate + combine）

查询上的小型 MLP 产出三个门控权重。最终输出是三个分支输出的加权和。

### 第 6 步：计算计数（Step 6: compute counting）

打印每个分支每查询关注的 key 数量以及总计。与 `N`（全注意力）比较。在 1024 token 的合成序列上，取 `l = 32, k = 4, w = 128`，NSA 每查询看到 `32 + 128 + 128 = 288` 个 key，全注意力是 1024——少约 3.5 倍。

## 使用它（Use It）

NSA 已进入 DeepSeek 自己的长上下文预训练流水线。截至 2026 年 4 月，公开推理栈的集成状态：

- **DeepSeek 内部**：原生支持，已发布权重使用 NSA 或其后继 DSA（Deepseek Sparse Attention）。
- **vLLM**：针对 DeepSeek-V3.x 权重的实验性 NSA 支持正在开发。
- **SGLang**：已发布 NSA 基准；生产路径跟随 vLLM。
- **llama.cpp / CPU**：不支持；内核拆分的开销在 CPU 吞吐下不划算。

何时该用 NSA：

- 以 64k 以上上下文为目标、算力预算认真的预训练或持续训练。
- 推理 DeepSeek 自己的长上下文检查点。这些权重是 NSA 原生的。

何时不该用：

- 服务已有的稠密注意力预训练模型。不继续训练就无法把 NSA 装回去。
- 上下文低于 16k。三分支开销会压过节省。
- 批量为 1 的交互聊天。对延迟敏感的解码有收益，但只在长上下文时。

## 交付（Ship It）

本课产出 `outputs/skill-nsa-integrator.md`。给定一次长上下文预训练运行规格，它产出 NSA 集成计划：压缩块大小、top-k、滑动窗口、门控 MLP 宽度、内核选择，以及能证明这次架构改动值得的具体长上下文评测。

## 练习（Exercises）

1. 在 1024 token 的合成序列上运行 `code/main.py`。对三组 `(l, k, w)` 预设扫一遍并打印计算计数。找出在大海捞针（needle-in-haystack）测试上相对全注意力保持 95% 召回、同时每查询 key 数最低的预设。

2. 把均值池化压缩器换成一个很小的可学习 MLP（2 层，隐层 32）。在「信号是块平均值」的合成任务上训练。在留出数据上测量相对均值池化基线的困惑度差距。

3. 实现门控 MLP。它以查询为输入，输出三个标量。展示门控行为合理：随机查询上接近均匀加权；当查询命中很靠后的块时，选择分支权重大。

4. 计算启用 NSA 的 70B 模型在 128k 上下文下的 KV 缓存内存预算。KV 头为 8，头维度 128，BF16。与全注意力以及 MLA（Phase 10 · 14 给出了 MLA 的数字）比较。找出 NSA 细粒度分支 KV 缓存等于全注意力的序列长度。

5. 阅读 NSA 论文（arXiv:2502.11089）第 4 节，用三句话解释为什么复用压缩分支的注意力分数做 top-k 选择，而不是另算一套路由分数。把答案和梯度流联系起来。

## 关键术语（Key Terms）

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 压缩分支 | 「粗视图」 | 在块平均后的 key 上做注意力，以每查询 O(N/l) 个 key 提供全局上下文 |
| 选择分支 | 「top-k 块」 | 对压缩分支分数最高的 `k` 个块做细粒度注意力 |
| 滑动窗口 | 「局部上下文」 | 对最后 `W` 个 token 做注意力，捕获短程模式 |
| 原生可训练性 | 「带着稀疏性一起预训练」 | 稀疏模式在预训练中学到，而不是推理时再打补丁 |
| 压缩块大小 l | 「粗视图的分组大小」 | 多少个 token 合成一个摘要；典型 32–64 |
| Top-k | 「要保留的块」 | 读取未压缩 token 的压缩块个数；典型 16 |
| 滑动窗口 W | 「局部注意力半径」 | 典型 512；太短伤局部连贯，太长浪费算力 |
| 分支门控 | 「怎么混合三条」 | 每个位置的 MLP 输出，加权三个分支的贡献 |
| 硬件对齐 | 「内核友好的稀疏」 | 选择的稀疏模式让实际 GPU 内核能兑现理论加速 |
| DSA | 「NSA 的后继」 | Deepseek Sparse Attention，DeepSeek 谱系中接在 NSA 之后的架构 |

## 延伸阅读（Further Reading）

- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — 论文本身
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — NSA 所针对的架构家族
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) — 同期工作，块上的 MoE 式注意力
- [Beltagy et al. — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) — 滑动窗口的源头
- [Xiao et al. — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) — NSA 所改进的推理时稀疏基线
- [Dao et al. — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) — NSA 内核在 64k 上击败的全注意力基线
