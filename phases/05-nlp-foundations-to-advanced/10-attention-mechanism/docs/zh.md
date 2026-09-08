# 注意力机制 — 突破（Attention Mechanism — The Breakthrough）

> 解码器停止眯眼查看压缩摘要，开始查看整个源。此后的一切都是注意力加工程。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 09 (Sequence-to-Sequence Models)
**Time:** ~45 minutes

## 问题（The Problem）

第 09 课以有分寸的失败结束。在玩具复制任务上训练的 GRU 编码器-解码器在长度 5 时准确率为 89%，在长度 80 时接近随机。原因是结构性的，不是训练 bug：编码器获得的每一点信息都必须塞进一个固定大小的隐藏状态，解码器看不到其他任何东西。

Bahdanau、Cho 和 Bengio 在 2014 年发表了一个三行修复。不是只给解码器最后编码器状态，保留每个编码器状态。在每个解码器步，计算编码器状态的加权平均，权重表示"解码器现在需要查看编码器位置 `i` 的程度？"这个加权平均就是上下文，它在每个解码器步变化。

这就是全部想法。Transformers 扩展了它。自注意力将其应用于单个序列。多头注意力并行运行它。但 2014 年的版本已经打破了瓶颈，一旦你有了它，pivot 到 transformers 是工程问题，不是概念问题。

## 概念（The Concept）

![Bahdanau 注意力：解码器查询所有编码器状态](../assets/attention.svg)

在每个解码器步 `t`：

1. 使用前一个解码器隐藏状态 `s_{t-1}` 作为查询（query）。
2. 对每个编码器隐藏状态 `h_1, ..., h_T` 打分。每个编码器位置一个标量。
3. 对分数 softmax 得到注意力权重 `α_{t,1}, ..., α_{t,T}`，和为 1。
4. 上下文向量 `c_t = Σ α_{t,i} * h_i`。编码器状态的加权平均。
5. 解码器取 `c_t` 加上前一个输出词元，产生下一个词元。

加权平均是重点。当解码器需要将 "Je" 翻译为 "I" 时，它对 "Je" 上方的编码器状态权重高，其他低。当它需要 "not" 时，它对 "pas" 权重高。上下文向量重塑每一步。

## 形状（每个人都会踩的坑）（Shapes (the thing that bites everyone)）

这是每个注意力实现第一次出错的地方。慢慢读。

| 对象 | 形状 | 说明 |
|-------|-------|-------|
| 编码器隐藏状态 `H` | `(T_enc, d_h)` | 如果是 BiLSTM，`d_h = 2 * d_hidden` |
| 解码器隐藏状态 `s_{t-1}` | `(d_s,)` | 一个向量 |
| 注意力分数 `e_{t,i}` | 标量 | 每个编码器位置一个 |
| 注意力权重 `α_{t,i}` | 标量 | 在所有 `i` 上 softmax 之后 |
| 上下文向量 `c_t` | `(d_h,)` | 与编码器状态相同形状 |

**Bahdanau（加法）分数。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` 的形状是 `(d_s,)`，`h_i` 的形状是 `(d_h,)`。
- `W_a` 的形状是 `(d_attn, d_s)`。`U_a` 的形状是 `(d_attn, d_h)`。
- 它们在 tanh 内的和形状是 `(d_attn,)`。
- `v_α` 的形状是 `(d_attn,)`。与 `v_α` 的内积 collapse 为一个标量。这就是 `v_α` 的作用。它不是魔法。它是将注意力维度向量投影为标量分数的投影。

**Luong（乘法）分数。** 三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`。硬约束。如果你的编码器是双向的，跳过。
- `general`：`e_{t,i} = s_t^T * W * h_i`，`W` 形状 `(d_s, d_h)`。移除等维约束。
- `concat`：本质上是 Bahdanau 形式。因为前两种更便宜，很少使用。

**一个值得命名的 Bahdanau / Luong 陷阱。** Bahdanau 使用 `s_{t-1}`（生成当前词之前的解码器状态）。Luong 使用 `s_t`（之后的解码器状态）。混淆它们会产生微妙错误的梯度，极难调试。选定一篇论文并遵守其约定。

```figure
attention-heatmap
```

## 构建它（Build It）

### 步骤 1：加法注意力（Bahdanau 注意力）（additive (Bahdanau) attention）

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

对照上面的表格检查你的形状。`encoder_states` 的形状是 `(T_enc, d_h)`。`projected_enc` 的形状是 `(T_enc, d_attn)`。`projected_dec` 的形状是 `(d_attn,)` 并广播。`combined` 的形状是 `(T_enc, d_attn)`。`scores` 的形状是 `(T_enc,)`。`weights` 的形状是 `(T_enc,)`。`context` 的形状是 `(d_h,)`。发布它。

### 步骤 2：Luong dot 和 general（Luong dot and general）

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

每段三行。这就是为什么 Luong 的论文能发表。大多数任务上准确率相同，代码少很多。

### 步骤 3：一个 worked 数值示例（a worked numerical example）

给定三个编码器状态（大致是 "cat"、"sat"、"mat"）和一个与第一个最对齐的解码器状态，注意力分布集中在位置 0。如果解码器状态 shift 到与第三个编码器状态更对齐，注意力移动到位置 2。上下文向量跟踪。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一行获胜。然后将解码器状态移近第三个编码器状态，观察权重 shift。这就是全部。注意力是显式对齐。

### 步骤 4：为什么这是通往 transformers 的桥梁（why this is the bridge to transformers）

将上面的语言翻译为 Q/K/V：

- **查询** = 解码器状态 `s_{t-1}`
- **键** = 编码器状态（我们对其打分）
- **值** = 编码器状态（我们权重并求和）

在经典注意力中，键和值是同一个东西。自注意力将它们分离：你可以查询一个序列自身，为 K 和 V 使用不同的 learned 投影。多头注意力用不同的 learned 投影并行运行。Transformers 堆叠整个阶段多次并抛弃 RNNs。

数学相同。形状相同。从 Bahdanau 注意力到缩放点积注意力的教学跳跃主要是符号。

## 使用它（Use It）

PyTorch 和 TensorFlow 直接提供注意力。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

这就是一个 transformer 注意力层。5 个位置的查询批次，10 个位置的键/值批次，每个 128 维，8 个头。`output` 是新的上下文增强查询。`weights` 是 5x10 对齐矩阵，你可以可视化。

### 经典注意力仍然重要的场景（When classical attention still matters）

- 教学。单头、单层、基于 RNN 的版本使每个概念可见。
- 端侧序列任务，transformers 不适合。
- 2014-2017 年的任何论文。不知道 Bahdanau 的约定你会误读它。
- MT 中的细粒度对齐分析。原始注意力权重是可解释性工具，即使在 transformer 模型上，阅读它们需要知道它们是什么。

### 注意力权重作为解释的陷阱（The attention-weight-as-explanation trap）

注意力权重看起来可解释。它们是在位置上和为 1 的权重；你可以绘制它们；高意味着"查看了这个"。审稿人喜欢它们。

它们并不像看起来那样可解释。Jain 和 Wallace（2019）表明注意力分布可以被置换并用任意替代品替换，而不会改变某些任务的模型预测。在没有消融或反事实检查的情况下，永远不要将注意力权重报告为推理的证据。

## 交付物（Ship It）

保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 练习（Exercises）

1. **简单。** 实现 `softmax` 掩码，使编码器中的填充词元获得零注意力权重。在批次长度变化的序列上测试。
2. **中等。** 为 Luong `general` 形式添加多头注意力。将 `d_h` 分成 `n_heads` 组，为每个头运行注意力，拼接。验证单头情况与你之前的实现匹配。
3. **困难。** 在第 09 课的玩具复制任务上训练带 Bahdanau 注意力的 GRU 编码器-解码器。绘制准确率 vs 序列长度。与无注意力基线比较。你应该看到差距随长度增长而扩大，确认注意力提升了瓶颈。

## 关键术语（Key Terms）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Attention | 关注事物 | 值序列的加权平均，权重由查询-键相似度计算。 |
| Query, Key, Value | QKV | 三个投影：Q 询问，K 是匹配对象，V 是返回内容。 |
| Additive attention | Bahdanau | 前馈分数：`v^T tanh(W q + U k)`。 |
| Multiplicative attention | Luong dot / general | 分数是 `q^T k` 或 `q^T W k`。更便宜，大多数任务上准确率相同。 |
| Alignment matrix | 漂亮的图 | 作为 `(T_dec, T_enc)` 网格的注意力权重。阅读它来看模型关注了什么。 |

## 拓展阅读（Further Reading）

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 论文。
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 三种分数变体及其比较。
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 可解释性注意事项。
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — 使用 PyTorch 的可构建演练。
