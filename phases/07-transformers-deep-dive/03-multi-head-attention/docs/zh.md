# Multi-Head Attention（多头注意力）

> 一个注意力头学一种关系。八个头学八种。头是自由的。多要几个。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)（从零实现自注意力）
**Time:** ~75 minutes（约 75 分钟）

## The Problem（问题）

单个自注意力头只计算一个注意力矩阵。这个矩阵只捕获一种关系——通常是那个在给定训练信号下最小化损失的关系。如果你的数据里同时有主谓一致、指代消解、长距离篇章结构和句法块划分，单个头会把它们混入同一个 softmax 分布，丢掉一半信号。

2017 年 Vaswani 论文给出的解法：并行运行多个注意力函数，每个有自己独立的 Q、K、V 投影，然后把输出拼接起来。每个头在 `d_model / n_heads` 的更小子空间里运行。总参数量不变。表达能力提升。

多头注意力是 2026 年每个 transformer 默认带的结构。唯一争论的是*到底用多少个头*，以及键和值是否共享投影（Grouped-Query Attention、Multi-Query Attention、Multi-head Latent Attention）。

## The Concept（概念）

![Multi-head attention splits, attends, concatenates（多头注意力：拆分、并行注意力、拼接）](../assets/multi-head-attention.svg)

**拆分（Split）。** 取形状为 `(N, d_model)` 的 `X`，投影到 Q、K、V，每个形状为 `(N, d_model)`。重塑为 `(N, n_heads, d_head)`，其中 `d_head = d_model / n_heads`。转置为 `(n_heads, N, d_head)`。

**并行注意力（Attend in parallel）。** 在每个头内部运行缩放点积注意力。每个头产生 `(N, d_head)`。这些头在嵌入的不同子空间上操作，在注意力计算期间彼此不通信。

**拼接并投影（Concatenate and project）。** 把头堆叠回 `(N, d_model)`，乘以一个学到的输出矩阵 `W_o`，形状为 `(d_model, d_model)`。`W_o` 是头之间开始混合的地方。

**为什么有效（Why it works）。** 每个头都可以专门化，而不必在表示预算上相互竞争。2019–2024 年的探测研究显示了头有不同的角色：位置头、关注前一个 token 的头、复制头、命名实体头、 induction head（它们是 in-context learning 的基础）。

**2026 年的变体谱系：**

| Variant（变体） | Q heads（Q 头数） | K/V heads（K/V 头数） | Used by（使用者） |
|---------|---------|-----------|---------|
| Multi-head (MHA)（标准多头） | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA)（多查询） | N | 1 | PaLM, Falcon |
| Grouped-query (GQA)（分组查询） | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA)（多头潜空间） | N | compressed to low-rank（压缩为低秩） | DeepSeek-V2, V3 |

GQA 是现代默认，因为它把 KV cache 内存缩小了 `N/G` 倍，同时几乎保持完整质量。MLA 更进一步，把 K/V 压缩进潜空间，在计算时再投影回来——多花 FLOPs，省更多内存。

```figure
multihead-split
```

## Build It（动手实现）

### Step 1: 从已有的单头注意力拆分出头

拿 Lesson 02 的 `SelfAttention`，用 split/concat 对包装它。见 `code/main.py` 的 numpy 实现；逻辑是：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一次 reshape，一次 transpose。没有循环。这正是 PyTorch 在 `nn.MultiheadAttention` 里做的。

### Step 2: 每个头运行缩放点积注意力

每个头拿到自己那一片 Q、K、V。注意力变成批量矩阵乘法：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在真实硬件上 `Qh @ Kh.transpose(...)` 就是一个 `bmm`。GPU 看到的是一个批量矩阵乘法，形状为 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`。加头是免费的。

### Step 3: Grouped-Query Attention 变体

只有键和值投影改变。Q 得到 `n_heads` 组；K 和 V 得到 `n_kv_heads < n_heads` 组，并被重复以匹配：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

在推理时这会节省内存，因为 KV cache 里只有 `n_kv_heads` 个副本，而不是 `n_heads` 个。Llama 3 70B 使用 64 个 query 头搭配 8 个 KV 头——8 倍 cache 缩小。

### Step 4: 探测每个头学到了什么

在一个有 4 个头的短句子上运行 MHA。对每个头，打印 `(N, N)` 注意力矩阵。你会看到不同的头挑出不同的结构，即使权重是随机初始化的——这部分是信号，部分是子空间里的旋转对称。

## Use It（实际应用）

在 PyTorch 中，一行版本：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ 的 GQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention 在 CUDA 上自动调度 Flash Attention。
# 对于 GQA，传入形状为 (B, n_heads, N, d_head) 的 Q 和形状为
# (B, n_kv_heads, N, d_head) 的 K、V。PyTorch 处理重复。
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**多少个头合适？** 2026 年生产模型的经验法则：

| Model size（模型规模） | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M)（小模型） | 768 | 12 | 64 |
| Base (~350M)（基础模型） | 1024 | 16 | 64 |
| Large (~1B)（大模型） | 2048 | 16 | 128 |
| Frontier (~70B)（前沿模型） | 8192 | 64 | 128 |

`d_head` 几乎总是落在 64 或 128。它衡量一个头能“看到”多少。低于 32，头会开始和缩放因子 `sqrt(d_head)` 打架；高于 256，你就失去了“许多小专家”的好处。

## Ship It（交付）

见 `outputs/skill-mha-configurator.md`。这个 skill 会在给定参数预算、序列长度和部署目标的情况下，为一个新 transformer 推荐头数、KV 头数和投影策略。

## Exercises（练习）

1. **Easy（简单）。** 拿 `code/main.py` 里的 MHA，把 `n_heads` 从 1 改成 16，固定 `d_model=64`。在一个小合成复制任务上绘制单层模型的损失曲线。更多头是帮助、 plateau 还是伤害？
2. **Medium（中等）。** 实现 MQA（所有 query 头共享一个 KV 头）。测量相比完整 MHA 参数数量下降多少。计算在 N=2048 时推理的 KV cache 大小缩小了多少。
3. **Hard（困难）。** 实现一个微型 Multi-head Latent Attention：把 K、V 压缩到秩为 `r` 的潜空间，在 KV cache 中存储潜空间，在注意力时解压。在哪个 `r` 下 cache 内存能降到完整 MHA 的 1/8 以下，同时质量保持在验证困惑度 1 bit 以内？

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Head（注意力头） | "A single attention circuit"（单一注意力回路） | 一个 Q/K/V 投影，维度为 `d_head = d_model / n_heads`，有自己独立的注意力矩阵。 |
| d_head（头维度） | "Head dimension"（头的维度） | 每个头的隐藏宽度；生产环境中几乎总是 64 或 128。 |
| Split / combine（拆分/合并） | "Reshape tricks"（重塑技巧） | `(N, d_model) ↔ (n_heads, N, d_head)` 在注意力周围的 reshape+transpose。 |
| W_o（输出投影） | "Output projection"（输出投影） | 在拼接头之后应用的 `(d_model, d_model)` 矩阵；头在这里混合。 |
| MQA（多查询注意力） | "One KV head"（只有一个 KV 头） | Multi-Query Attention：单个共享的 K/V 投影。最小的 KV cache，有一些质量损失。 |
| GQA（分组查询注意力） | "The default since Llama 2"（Llama 2 以来的默认） | Grouped-Query Attention，`n_kv_heads < n_heads`；重复以匹配 Q。 |
| MLA（多头潜空间注意力） | "DeepSeek's trick"（DeepSeek 的技巧） | Multi-head Latent Attention：K、V 压缩到低秩潜空间，在 attend 时解压。 |
| Induction head（诱导头） | "The circuit behind in-context learning"（in-context learning 背后的回路） | 一对检测到之前出现并复制其后内容的头。 |

## Further Reading（延伸阅读）

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 原始多头规格。
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA 论文。
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — 如何在训练后将 MHA 转换为 GQA。
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA 以及它为什么在 cache 内存上优于 MHA/GQA。
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — 从机械角度看待头实际在做什么。
