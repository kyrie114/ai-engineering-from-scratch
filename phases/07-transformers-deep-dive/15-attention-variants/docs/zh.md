# Attention Variants — Sliding Window, Sparse, Differential（注意力变体 —— 滑动窗口、稀疏、差分）

> 完整注意力是一个圆。每个 token 看到每个 token，内存为此付出代价。四个变体改变圆的形状，恢复一半成本。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention)（自注意力）, Phase 7 · 03 (Multi-Head)（多头）, Phase 7 · 12 (KV Cache / Flash Attention)（KV Cache / Flash Attention）
**Time:** ~60 minutes（约 60 分钟）

## The Problem（问题）

完整注意力的内存和计算成本在序列长度上是 `O(N²)`。对于一个 128K 上下文的 Llama 3 70B，每层是 160 亿个注意力条目，乘以 80 层。Flash Attention（Lesson 12）隐藏了 `O(N²)` 激活内存，但没有改变算术成本 —— 每个 token 仍然关注每个其他 token。

三个类的变体改变了注意力矩阵本身的拓扑：

1. **Sliding window attention (SWA)（滑动窗口注意力）。** 每个 token 只关注一个固定窗口的邻居，不是完整前缀。内存和计算降到 `O(N · W)`，其中 `W` 是窗口。Gemma 2/3、Mistral 7B 的第一层、Phi-3-Long。
2. **Sparse / block attention（稀疏/块注意力）。** 只有选定的对 `(i, j)` 被评分；其他的被迫为零权重。Longformer、BigBird、OpenAI sparse transformer。
3. **Differential attention（差分注意力）。** 用单独的 Q/K 投影计算两个注意力图，从一个减去另一个。杀死把权重 bleeding 到前几个 token 的“attention sink”。Microsoft 的 DIFF Transformer（2024）。

这些共存。一个 2026 年前沿模型经常混合它们：大多数层是 SWA-1024，每第五层是全局完整注意力，少数是清理检索的差分头。Gemma 3 的 5:1 SWA-to-global 比率是当前的 textbook default。

## The Concept（概念）

### Sliding Window Attention (SWA)（滑动窗口注意力）

每个位置 `i` 的查询只关注 `[i - W, i]`（causal SWA）或 `[i - W/2, i + W/2]`（bidirectional）中的位置。窗口外的 token 在分数矩阵中得到 `-inf`。

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

对于 `N = 8192` 和 `W = 1024`，分数矩阵预期有 1024 × 8192 个非零行 —— 一个 8× 减少。

**KV cache 随 SWA 缩小。** 每层只需要每个 token 最后 `W` 个 token 的 K 和 V。对于一个 Gemma-3-ish 配置（1024 窗口，128K 上下文），KV cache 下降 128×。

**Quality cost（质量成本）。** 纯 SWA transformer 在长范围检索上挣扎。修复：在 SWA 层之间穿插完整注意力层。Gemma 3 使用 5:1 SWA:global。Mistral 7B 使用一个 causal-SWA 栈，信息通过重叠窗口“forward flow” —— 每层通过 `W` 扩展有效感受野，在 `L` 层之后，模型可以 attend `L × W` token 回来。

### Sparse / Block Attention（稀疏/块注意力）

在之前选一个 `N × N` 稀疏 pattern。三个规范形状：

- **Local + strided (OpenAI sparse transformer)。** Attend 到最后的 `W` token 加上每 `stride`-th token 之前。捕获 local 和 long-range，在 `O(N · sqrt(N))` 计算下。
- **Longformer / BigBird。** Local window + 一小部分 global token（例如 `[CLS]`），attend 到每个人并被每个人 attend + random-sparse links。在匹配质量下经验 2× 上下文。
- **Native Sparse Attention (DeepSeek, 2025)。** 学习哪些 `(Q, K)` 块重要；在 kernel 级别跳过零块。FlashAttention-compatible。

稀疏注意力是一个 kernel-engineering 故事。数学很简单（mask 分数矩阵）；win 来自从不把零条目加载到 SRAM。FlashAttention-3 和 2026 年 FlexAttention API 使自定义稀疏 pattern 在 PyTorch 中成为 first-class。

### Differential Attention (DIFF Transformer, 2024)（差分注意力）

Regular attention 有一个“attention sink”问题：softmax 强制每行 sum 到 1，所以不想 attend 到任何特定东西的 token 把 weight dump 在第一个 token（或前几个）上。这偷走了应该去真实内容的容量。

Differential attention 通过计算 **两个** 注意力图并相减来修复：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是一个学到的 scalar（通常 0.5–0.8）。A1 捕获真实内容权重；A2 捕获 sink。减法取消 sink，把 weight 重新分配到相关 token。

报告结果（Microsoft 2024）：5–10% 更低困惑度，在相同训练长度下 1.5–2× 更长有效上下文，更尖锐的 needle-in-haystack 检索。

### Variant Comparison（变体比较）

| Variant（变体） | Compute（计算） | KV cache | Quality vs full（与完整的质量） | Production use（生产使用） |
|---------|---------|----------|-----------------|--------|
| Full attention（完整注意力） | O(N²) | O(N) per layer（每层） | baseline（基线） | every model's default layer（每个模型的默认层） |
| SWA (window 1024) | O(N·W) | O(W) per layer（每层） | -0.1 ppl，good with global layers（有全局层时好） | Gemma 2/3, Phi-3-Long |
| Local + strided sparse（局部 + 步长稀疏） | O(N·√N) | mixed（混合） | similar to SWA（类似于 SWA） | OpenAI sparse transformer, Longformer |
| BigBird (local + global + random)（局部 + 全局 + 随机） | O(N) approx（大约） | mixed（混合） | matches full at 2× context（在 2× 上下文下匹配完整） | early long-context BERT |
| Native Sparse (DeepSeek-V3.2) | O(N · active fraction)（O(N · 活跃分数)） | O(N) | within 0.05 ppl | DeepSeek-V3.2, 2025 |
| Differential（差分） | O(2·N²) | O(2N) | -5 to -10% ppl | DIFF Transformer, early 2026 models（早期 2026 模型） |

```figure
gqa-kv-sharing
```

## Build It（动手实现）

见 `code/main.py`。我们实现一个 causal mask comparator，在一个 toy 序列上并排显示 full、SWA、local+strided 和 differential attention。

### Step 1: full causal mask (baseline)（完整因果遮罩（基线））

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Lesson 07 的 baseline。下三角；对角线上方零权重。

### Step 2: sliding window causal mask（滑动窗口因果遮罩）

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

一个参数 —— `window`。对于 `window >= n`，你恢复完整因果注意力。对于 `window = 1`，每个 token 只 attend 到自身。

### Step 3: local + strided sparse mask（局部 + 步长稀疏遮罩）

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

Dense local window 加上每 `stride`-th token 回到序列开始。感受野随额外层以 log 步长增长。

### Step 4: differential attention（差分注意力）

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次注意力传递，用一个学到的 mixing coefficient 相减。在代码中，我们比较 single vs differential 的 attention-sink heatmap，并观察 sink collapse。

### Step 5: KV cache sizes（KV cache 大小）

打印每个变体在 `N = 131072` 下每层的 cache 大小。SWA 和稀疏变体下降 10–100×。Differential 翻倍。有意识地为你的内存账单付费。

## Use It（实际应用）

2026 生产 pattern：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 混合 SWA (window=1024) 和全局层在 5:1。
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 中的 FlexAttention 接受一个 mask function：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

这编译为一个自定义 Triton kernel。在常用 pattern 下，速度在 FlashAttention-3 的 10% 以内，mask function 是一个 Python callable。

**何时选择每个：**

- **Pure full attention** —— 每层最多 ~16K 上下文，或当检索质量是 paramount 时。
- **SWA + global mix** —— 长上下文（>32K），训练和推理 memory-bound。2026 年 32K 以上的默认。
- **Sparse block attention** —— 自定义 kernel，自定义 pattern。保留用于专业工作负载（检索、audio）。
- **Differential attention** —— 任何 attention-sink contamination 伤害的工作负载（长上下文 RAG、needle-in-haystack）。

## Ship It（交付）

见 `outputs/skill-attention-variant-picker.md`。这个 skill 在给定目标上下文长度、检索需求和训练/推理计算概况的情况下，为一个新模型选择注意力拓扑。

## Exercises（练习）

1. **Easy（简单）。** 运行 `code/main.py`。验证 `window=4` 的 SWA 把每行最后 4 个 token 之外的所有东西归零。验证 `window=n` bit-identically 重现完整因果注意力。
2. **Medium（中等）。** 在 Lesson 07 毕业设计上实现 causal SWA，`window=1024`。在 tinyshakespeare 上训练 1,000 步。val loss 相比完整注意力退化多少？peak memory 下降多少？
3. **Hard（困难）。** 在毕业设计模型中实现一个 Gemma-3-style 5:1 layer mix（5 SWA，1 global）。在匹配参数下与 pure-SWA 和 pure-global baseline 比较 loss、memory 和 generation quality。
4. **Hard（困难）。** 实现带每个头学到的 `λ` 的 differential attention。在一个合成检索任务（一个 needle，2,000 distractors）上训练。在匹配参数下与 single-attention baseline 比较检索准确率。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Sliding window attention (SWA)（滑动窗口注意力） | "Local attention"（局部注意力） | 每个查询 attend 到它的最后 `W` token；KV cache 缩小到 `O(W)`。 |
| Effective receptive field（有效感受野） | "How far back the model sees"（模型能看到多远） | 在一个 `L`-layer SWA 栈中，窗口为 `W`，最多 `L × W` token。 |
| Longformer / BigBird | "Local + global + random"（局部 + 全局 + 随机） | 带几个总是 attend 所有人的 global token 的稀疏 pattern；早期长上下文方法。 |
| Native Sparse Attention（原生稀疏注意力） | "DeepSeek's kernel trick"（DeepSeek 的 kernel 技巧） | 学习块级稀疏性；在 kernel 级别跳过零块，同时保持质量。 |
| Differential attention（差分注意力） | "Two maps, one subtracts"（两张图，一张减去） | DIFF Transformer：用学到的 `λ` 乘以第二个注意力图从第一个减去，以取消 attention sink。 |
| Attention sink（注意力 sink） | "Weight bleeds to token 0"（权重 bleeding 到 token 0） | Softmax 归一化强制行 sum 到 1；无信息的查询把 weight dump 在位置 0 上。 |
| FlexAttention | "Mask-as-Python"（Mask 即 Python） | PyTorch 2.5+ API，把任意 mask function 编译为 FlashAttention-shape kernels。 |
| Layer type mix（层类型混合） | "5:1 SWA-to-global" | 在栈中交错稀疏和完整注意力层，以更低内存保持质量。 |

## Further Reading（延伸阅读）

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) —— 规范滑动窗口 + global-token 论文。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) —— local + global + random。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) —— OpenAI 的 local+strided pattern。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) —— 1:1 SWA:global mix。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) —— 现在 textbook default 的 5:1 mix，window=1024。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) —— DIFF Transformer 论文。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) —— DeepSeek-V3.2 的 learned-sparsity attention。
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) —— Use It 中引用的 mask-as-callable pattern 的 API reference。
