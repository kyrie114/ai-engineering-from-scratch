# GPT — Causal Language Modeling（GPT — 因果语言模型）

> BERT 看两边。GPT 只看过去。三角形遮罩是 Modern AI 中最有影响力的单行代码。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention)（自注意力）, Phase 7 · 05 (Full Transformer)（完整 Transformer）, Phase 7 · 06 (BERT)（BERT）
**Time:** ~75 minutes（约 75 分钟）

## The Problem（问题）

语言模型回答一个问题：给定前 `t-1` 个 token，token `t` 上的概率分布是什么？在这个信号上训练——next-token prediction——你会得到一个可以逐 token 生成任意文本的模型。

要在一个序列上端到端并行训练，你需要每个位置的预测只依赖于更早的位置。否则模型会作弊看答案。

因果遮罩做到了这一点。它是一个 `-inf` 值的上三角矩阵，在 softmax 之前加到注意力分数上。softmax 之后，这些位置变成 0。每个位置只能关注自己和更早的位置。而且因为你一次性应用到整个序列，你在一次前向传播中得到 N 个并行的 next-token 预测。

GPT-1 (2018)、GPT-2 (2019)、GPT-3 (2020)、GPT-4 (2023)、GPT-5 (2025)、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi——它们都是具有相同核心循环的 decoder-only causal transformer。区分它们的是数据质量、规模、架构细化，以及 post-training（SFT、RLHF、DPO 及其继任者）。

## The Concept（概念）

![Causal mask creates a triangular attention matrix（因果遮罩创建一个三角形注意力矩阵）](../assets/causal-attention.svg)

### 遮罩

给定长度为 `N` 的序列，构建一个 `N × N` 矩阵：

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

在 softmax 之前将 `M` 加到原始注意力分数上。`exp(-inf) = 0`，所以被遮罩的位置贡献零权重。注意力矩阵的每一行是只对先前位置的概率分布。

实现成本：一个 `torch.tril()` 调用。计算时间：纳秒。对领域的影响：一切。

### 三角形从何而来

遮罩通常被呈现为附加到注意力的补丁。反向运行推导，它就不再神秘了：注意力是前缀平均值的第三次细化，而三角形是该平均值的循环边界，写成矩阵形式。

**Stage 1 — 前缀平均值。** 序列最愚蠢的因果摘要：位置 `i` 变成位置 `0…i` 的均值。作为循环，那就是 `out[i] = X[:i+1].mean(0)`。同样的计算是一个矩阵乘法。取一个下三角的 ones 矩阵，每一行除以其计数，然后相乘：

```python
import numpy as np

A = np.tril(np.ones((n, n)))
A = A / A.sum(axis=1, keepdims=True)
out = A @ X
```

`A` 的第 `i` 行是 `[1/(i+1), …, 1/(i+1), 0, …, 0]`。对角线上方的零就是因果性。关于未来没有任何东西被遮罩掉；未来根本不在求和中。

**Stage 2 — 学到的权重。** 均匀平均值把每个过去的 token 视为同样相关。用学到的分数矩阵 `S` 替换 ones。现在行不再按构造和为 1，所以用 softmax 代替除以计数对每行归一化。Softmax 从不会输出确切的零，这打破了因果性——除非未来的分数以 `-inf` 进入，因为 `exp(-inf) = 0`：

```python
def softmax(x, axis):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

S = S + np.triu(np.full((n, n), -np.inf), k=1)
A = softmax(S, axis=1)
out = A @ X
```

同样的三角形，同样的行随机矩阵，同样的一个矩阵乘法。`-inf` 遮罩不是新机制。它是 Stage 1 的零条目，翻译到 softmax 的输入域。

**Stage 3 — 内容相关的权重。** 在 Stage 2 中，`S` 在训练后是固定的：位置 7 总是以同样的方式权衡位置 3，无论 token 说什么。让分数依赖于 token 本身：`S = Q @ K.T / sqrt(d_k)`。其他什么都不变。遮罩、softmax、矩阵乘法——完全相同。

三个阶段，一个不变量：一个下三角行随机矩阵乘以序列。均匀平均值、学到的静态权重、内容相关的权重。遮罩从来不是加到注意力上的。它从平均值中幸存下来。

```figure
mask-derivation
```

### 并行训练，串行推理

训练：对整个 `(N, d_model)` 序列进行一次前向传播，计算 N 个交叉熵损失（每个位置一个），求和，反向传播。沿序列并行。这就是为什么 GPT 训练能扩展——你在一次 GPU 传递中处理一个 batch 里的 100 万个 token。

推理：你逐 token 生成。输入 `[t1, t2, t3]`，得到 `t4`。输入 `[t1, t2, t3, t4]`，得到 `t5`。输入 `[t1, t2, t3, t4, t5]`，得到 `t6`。KV cache（Lesson 12）保存 `t1…tn` 的隐藏状态，这样你就不用每一步重新计算它们。但推理时的串行深度 = 输出长度。这就是自回归税，也是为什么解码是每个 LLM 的延迟瓶颈。

### 损失 —— 右移一位

给定 token `[t1, t2, t3, t4]`：

- 输入：`[t1, t2, t3]`
- 目标：`[t2, t3, t4]`

对于每个位置 `i`，计算 `-log P(target_i | inputs[:i+1])`。求和。这是整个序列的交叉熵。

你听说过的每个 transformer LM 都用这个损失训练。预训练、微调、SFT——同样的损失，不同的数据。

### 解码策略

训练之后，采样选择比人们想象的更重要。

| Method（方法） | What it does（做什么） | When to use（何时使用） |
|--------|--------------|-------------|
| Greedy（贪心） | 每一步 argmax | 确定性任务、代码补全 |
| Temperature（温度） | 将 logits 除以 T，然后采样 | 创造性任务，更高的 T = 更多多样性 |
| Top-k | 只从 top-k token 采样 | 杀死低概率尾部 |
| Top-p (nucleous)（核采样） | 从累积概率 ≥ p 的最小集合中采样 | 2020+ 默认；适应分布形状 |
| Min-p | 保留 `p > min_p * max_p` 的 token | 2024+；比 top-p 更好地拒绝长尾 |
| Speculative decoding（推测解码） | 草稿模型提出 N 个 token，大模型验证 | 相同质量下 2–3 倍延迟降低 |

2026 年，min-p + temperature 0.7 是开放权重模型的合理默认。推测解码是任何生产推理栈的标准配置。

### 让“GPT 配方”起作用的因素

1. **Decoder-only。** 没有编码器开销。每层一次注意力 + FFN 传递。
2. **规模。** 124M → 1.5B → 175B → 万亿。Chinchilla scaling laws（Lesson 13）告诉你如何花费计算。
3. **In-context learning。** 在 6B–13B 左右出现。模型可以在不微调的情况下遵循 few-shot 示例。
4. **RLHF。** 在人类偏好上的 post-training 将原始预训练文本转化为聊天助手。
5. **Pre-norm + RoPE + SwiGLU。** 大规模稳定训练。

核心架构自 GPT-2 以来没有太大变化。所有有趣的事情都发生在数据、规模和 post-training 中。

```figure
causal-mask
```

## Build It（动手实现）

### Step 1: 因果遮罩

见 `code/main.py`。一行代码：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

在 softmax 之前把它加到注意力分数上。这就是整个机制。

### Step 2: 一个 2 层 GPT 风格模型

堆叠两个解码器块（masked self-attention + FFN，没有交叉注意力）。加一个 token 嵌入、一个位置编码和一个 unembedding（与 token 嵌入矩阵绑定——自 GPT-2 以来的标准技巧）。

### Step 3: next-token prediction，端到端

在一个 20 token 的 toy 词表上，在每个位置产生 logits。对右移一位的目标计算交叉熵损失。没有梯度——这是前向传播健全性检查。

### Step 4: 采样

实现 greedy、temperature、top-k、top-p、min-p。在每个固定的提示上运行每个并比较输出。一个采样函数是 10 行代码。

## Use It（实际应用）

PyTorch，2026 年风格：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

在底层，`generate()` 运行前向传播，拉取最后位置的 logits，采样下一个 token，附加它，然后重复。每个生产 LLM 推理栈（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都用大量优化实现相同的循环——批量 prefill、连续批处理、KV cache 分页、推测解码。

**GPT vs BERT，一行 each：** GPT 预测 `P(x_t | x_{<t})`。BERT 预测 `P(x_masked | x_unmasked)`。损失决定了模型是否能生成。

## Ship It（交付）

见 `outputs/skill-sampling-tuner.md`。这个 skill 为一个新的生成任务选择采样参数，并标记何时需要确定性解码。

## Exercises（练习）

1. **Easy（简单）。** 运行 `code/main.py`，验证因果注意力矩阵在 softmax 之后是下三角的。抽查：第 3 行应该只在列 0–3 有权重。
2. **Medium（中等）。** 实现宽度为 4 的 beam search。在 10 个短提示上比较 beam-4 和 greedy 的困惑度。Beam 总是赢吗？（提示：通常在翻译中，不在开放对话中。）
3. **Hard（困难）。** 实现推测解码：用一个微型 2 层模型作为草稿，用一个 6 层模型作为验证器。在 100 个长度为 64 的补全上测量 wall-clock 加速。确认输出与验证器的 greedy 匹配。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Causal mask（因果遮罩） | "The triangle"（三角形） | 加到注意力分数上的上三角 `-inf` 矩阵，使位置 `i` 只能看到位置 `≤ i`。 |
| Next-token prediction（next-token 预测） | "The loss"（损失） | 模型分布与每个位置真实下一个 token 的交叉熵。 |
| Autoregressive（自回归） | "Generate one at a time"（一次生成一个） | 把输出反馈为输入；只有训练时并行，生成时不并行。 |
| Logits | "Pre-softmax scores"（softmax 前的分数） | LM 头在 softmax 之前的原始输出；采样发生在这里。 |
| Temperature（温度） | "Creativity knob"（创造性旋钮） | 将 logits 除以 T；T→0 = greedy，T→∞ = uniform。 |
| Top-p | "Nucleus sampling"（核采样） | 截断分布到累积概率 ≥ p 的最小集合；从剩余部分采样。 |
| Min-p | "Better than top-p"（比 top-p 更好） | 保留 `p ≥ min_p × max_p` 的 token；根据分布的尖锐度调整截断。 |
| Speculative decoding（推测解码） | "Draft + verify"（草稿 + 验证） | 廉价模型提出 N 个 token；大模型并行验证。 |
| Teacher forcing（教师强制） | "Training trick"（训练技巧） | 在训练期间，输入真实的先前 token，而不是模型的预测。每个 seq2seq LM 的标准。 |

## Further Reading（延伸阅读）

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1。
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2。
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 和 in-context learning。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 推测解码论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 规范 causal-LM 参考代码。
