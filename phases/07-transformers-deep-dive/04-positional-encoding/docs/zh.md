# Positional Encoding — Sinusoidal, RoPE, ALiBi（位置编码：正弦、RoPE、ALiBi）

> 注意力是置换不变的。"The cat sat on the mat" 和 "mat the on sat cat the" 在没有位置信号的情况下会产生相同的输出。三种算法修复了这一点——每种算法对“位置”意味着什么有不同的押注。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention)（自注意力）, Phase 7 · 03 (Multi-Head Attention)（多头注意力）
**Time:** ~45 minutes（约 45 分钟）

## The Problem（问题）

缩放点积注意力是顺序无关的。注意力矩阵 `softmax(Q K^T / √d) V` 是从 pairwise similarity 计算出来的。打乱 `X` 的行，输出的行也会以同样的方式被打乱。注意力内部没有任何东西关心位置。

对于词袋模型来说这不是 bug。对于语言、代码、音频、视频——任何顺序承载意义的东西——这是致命的。

修复方法是以某种方式把位置注入嵌入。三个时代的答案：

1. **绝对正弦（Absolute sinusoidal）**（Vaswani 2017）。把位置的 `sin/cos` 加到嵌入上。简单，不需要学习，超出训练长度外推表现差。
2. **RoPE — 旋转位置嵌入（Rotary Position Embeddings）**（Su 2021）。把 Q 和 K 向量旋转一个与位置成比例的角。直接在点积中编码*相对*位置。2026 年的主流。
3. **ALiBi — 带线性偏置的注意力（Attention with Linear Biases）**（Press 2022）。完全跳过嵌入；根据距离给注意力分数加一个每个头独立的线性惩罚。外推长度表现优秀。

截至 2026 年，基本上每个前沿开放模型都使用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数长上下文模型使用 ALiBi 或其现代变体。绝对正弦已经成了历史。

## The Concept（概念）

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias（绝对正弦 vs RoPE 旋转 vs ALiBi 距离偏置）](../assets/positional-encoding.svg)

### Absolute sinusoidal（绝对正弦）

预计算一个固定矩阵 `PE`，形状为 `(max_len, d_model)`：

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

然后在注意力之前做 `X' = X + PE[:N]`。每个维度是一个不同频率的正弦波。模型学会从相位模式中读取位置。在 `max_len` 之外失败：模型没有被告诉在位置 2048 会发生什么，如果它只见过位置 0–2047。

### RoPE

旋转 Q 和 K 向量（不是嵌入）。对于一对维度 `(2i, 2i+1)`：

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default（默认）
```

对键应用相同的旋转，位置为 `pos_k`。点积 `q'_m · k'_n` 变成只依赖于 `(m - n)` 的函数。也就是说：**注意力分数只依赖于相对距离**，即使旋转是基于绝对位置键控的。漂亮的技巧。

扩展 RoPE：`base` 可以被缩放（NTK-aware、YaRN、LongRoPE）以在不重新训练的情况下外推到更长的上下文。Llama 3 用这种方法从 8K 扩展到 128K 上下文。

### ALiBi

跳过嵌入技巧。直接给注意力分数加偏置：

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中 `m_h` 是一个头特定的斜率（例如 `1 / 2^(8·h/H)`）。更近的 token 获得提升；更远的 token 受到惩罚。训练时没有成本。论文显示长度外推 beats 正弦，并在其原始训练长度上与 RoPE 匹配。

### 2026 年选什么

| Variant（变体） | Extrapolation（外推） | Training cost（训练成本） | Used by（使用者） |
|---------|---------------|---------------|---------|
| Absolute sinusoidal（绝对正弦） | poor（差） | free（免费） | original transformer, early BERT（原始 transformer，早期 BERT） |
| Learned absolute（可学习绝对） | none（无） | tiny（极小） | GPT-2, GPT-3 |
| RoPE | good with scaling（有缩放则好） | free（免费） | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | excellent（优秀） | fine-tune stage（微调阶段） | Qwen2-1M, Llama 3.1 128K |
| ALiBi | excellent（优秀） | free（免费） | BLOOM, MPT, Baichuan |

RoPE 赢了，因为它无缝插入注意力而不改变架构，编码相对位置，而且它的 `base` 超参数为长上下文微调提供了一个干净的旋钮。

```figure
rope-explorer
```

## Build It（动手实现）

### Step 1: 正弦编码

见 `code/main.py`。一个 4 行计算：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一个注意力层之前把这个加到嵌入矩阵上。

### Step 2: 应用到 Q、K 的 RoPE

RoPE 在原地操作 Q 和 K。对于每一对维度：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键：对位置 `m` 的 Q 和位置 `n` 的 K 应用相同的函数。它们的点积在每一个坐标对上都获得一个 `cos((m-n)·θ_i)` 因子。注意力免费学习了相对位置。

### Step 3: ALiBi 斜率和偏置

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # 在 softmax 之前加到注意力分数上
```

把 `bias[h]` 加到头 `h` 的 `(seq_len, seq_len)` 注意力分数矩阵上，然后 softmax。

### Step 4: 验证 RoPE 的相对距离属性

选两个随机向量 `a, b`。用 `(pos_a, pos_b)` 旋转。然后用 `(pos_a + k, pos_b + k)` 旋转。两个点积必须在浮点误差内匹配。这个属性是 RoPE 的核心——它对绝对偏移不变，只有相对差距重要。

## Use It（实际应用）

PyTorch 2.5+ 在 `torch.nn.functional` 中内置了 RoPE 工具。大多数生产代码使用 `flash_attn` 或 `xformers`，其中 RoPE 在注意力内核内部应用。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026 年的长上下文技巧：**

- **NTK-aware interpolation（NTK 感知插值）。** 当从 4K 扩展到 16K+ 时，将 `base` 重缩放为 `base * (scale_factor)^(d/(d-2))`。
- **YaRN。** 更智能的插值，在长上下文上保持注意力熵。Llama 3.1 128K 使用它。
- **LongRoPE。** Microsoft 2024 年的方法，使用进化搜索为每个维度选择缩放因子。Phi-3-Long 使用它。
- **Position interpolation + fine-tuning（位置插值 + 微调）。** 只是按扩展因子缩小位置，然后微调 1–50 亿个 token。出奇地有效。

## Ship It（交付）

见 `outputs/skill-positional-encoding-picker.md`。这个 skill 会在给定目标上下文长度、外推需求和训练预算的情况下，为一个新模型选择编码策略。

## Exercises（练习）

1. **Easy（简单）。** 将正弦 `PE` 矩阵绘制为热图，`max_len=512, d=128`。确认“随着维度索引增长，条纹变得更宽”的模式。
2. **Medium（中等）。** 实现 NTK-aware RoPE 缩放。在长度为 256 的序列上训练一个微型 LM，然后在有缩放和没有缩放的情况下测试长度 1024。测量困惑度。
3. **Hard（困难）。** 在同一个注意力模块中实现 ALiBi 和 RoPE。在长度为 512 的复制任务上训练一个 4 层 transformer。在测试时外推到 2048。比较退化程度。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Positional encoding（位置编码） | "Tells attention about order"（告诉注意力顺序） | 任何添加到嵌入或注意力中编码位置的信号。 |
| Sinusoidal（正弦） | "The original one"（原始的那个） | 以几何频率添加到嵌入上的 `sin/cos`；不能外推。 |
| RoPE | "Rotary embeddings"（旋转嵌入） | 按位置相关的角度旋转 Q、K；点积编码相对距离。 |
| ALiBi | "Linear bias trick"（线性偏置技巧） | 给注意力分数加 `-m·|i-j|`；不需要嵌入，外推优秀。 |
| base | "RoPE's knob"（RoPE 的旋钮） | RoPE 中的频率缩放器；增加它以在推理时扩展上下文。 |
| NTK-aware（NTK 感知） | "A RoPE scaling trick"（一种 RoPE 缩放技巧） | 重新缩放 `base`，以便在上下文扩展时高频维度不会被挤压。 |
| YaRN | "The fancy one"（花哨的那个） | 每个维度的插值+外推，保持注意力熵。 |
| Extrapolation（外推） | "Works beyond trained length"（在训练长度之外工作） | 位置方案能否在训练中见过的 `max_len` 之后提供正确的输出？ |

## Further Reading（延伸阅读）

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 原始正弦。
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 论文。
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi。
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 最先进的 RoPE 缩放。
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta 的 Llama 2 长上下文论文。
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — 被 Phi-3-Long 使用的 Microsoft 方法，在 Use It 部分被引用。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 每个 RoPE 缩放方案的生产级实现（default、linear、dynamic、YaRN、LongRoPE、Llama-3）。
