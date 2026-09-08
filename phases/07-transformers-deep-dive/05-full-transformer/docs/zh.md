# The Full Transformer — Encoder + Decoder（完整 Transformer：编码器 + 解码器）

> 注意力是明星。其他一切——残差、归一化、前馈、交叉注意力——都是让你能够把它堆得很深的脚手架。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention)（自注意力）, Phase 7 · 03 (Multi-Head Attention)（多头注意力）, Phase 7 · 04 (Positional Encoding)（位置编码）
**Time:** ~75 minutes（约 75 分钟）

## The Problem（问题）

单个注意力层是特征提取器，不是模型。每层一次矩阵乘法不足以支撑语言。你需要深度——而深度没有正确的管道就会崩溃。

2017 年 Vaswani 论文打包了六个设计决策，把一层注意力变成了可堆叠的块。从那时起的每个 transformer——encoder-only（BERT）、decoder-only（GPT）、encoder-decoder（T5）——都继承了相同的骨架。2026 年的块已经被细化（RMSNorm、SwiGLU、pre-norm、RoPE），但骨架是一样的。

这节课讲骨架。接下来的课专门化它——06 讲编码器，07 讲解码器，08 讲编码器-解码器。

## The Concept（概念）

![Encoder and decoder block internals, wired（编码器和解码器块内部，已连线）](../assets/full-transformer.svg)

### 六个组件

1. **嵌入 + 位置信号。** Token → 向量。通过 RoPE（现代）或正弦（经典）注入位置。
2. **自注意力。** 每个位置关注每个其他位置。在解码器中被遮罩。
3. **前馈网络（FFN）。** 逐位置两层 MLP：`W_2 · activation(W_1 · x)`。默认扩展比例 4 倍。
4. **残差连接。** `x + sublayer(x)`。没有这个，梯度在第 6 层左右就会消失。
5. **层归一化。** `LayerNorm` 或 `RMSNorm`（现代）。稳定残差流。
6. **交叉注意力（仅解码器）。** 查询来自解码器，键和值来自编码器输出。

观察一个向量在一个块中的流动：注意力混合跨位置，残差把它向前传递，FFN 变换它，norm 保持流稳定。

```figure
transformer-block
```

### 编码器块（BERT、T5 编码器使用）

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

编码器是双向的。没有遮罩。所有位置看到所有位置。

### 解码器块（GPT、T5 解码器使用）

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

解码器每个块有三个子层。中间那个——交叉注意力——是信息从编码器流向解码器的唯一地方。在纯 decoder-only 架构（GPT）中，交叉注意力被省略，你只有 masked self-attention + FFN。

### Pre-norm vs post-norm

原始论文：`x + sublayer(LN(x))` vs `LN(x + sublayer(x))`。Post-norm 在 2019 年左右失宠——没有精心预热，很难深度训练。Pre-norm（*在* 子层之前 `LN`）是 2026 年的默认：Llama、Qwen、GPT-3+、Mistral 都使用它。

### 2026 年现代化块

Vaswani 2017  shipped LayerNorm + ReLU。现代栈替换了两者。生产块实际看起来是什么样：

| Component（组件） | 2017 | 2026 |
|-----------|------|------|
| Normalization（归一化） | LayerNorm | RMSNorm |
| FFN activation（FFN 激活） | ReLU | SwiGLU |
| FFN expansion（FFN 扩展） | 4× | 2.6×（SwiGLU 使用三个矩阵，总参数匹配） |
| Position（位置） | Sinusoidal absolute（绝对正弦） | RoPE |
| Attention（注意力） | Full MHA（完整多头） | GQA（或 MLA） |
| Bias terms（偏置项） | Yes（有） | No（无） |

RMSNorm 去掉了 LayerNorm 的均值中心化（少一次减法），节省计算且经验上至少同样稳定。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在 Llama、PaLM 和 Qwen 论文中一致比 ReLU/GELU FFN 好约 0.5 点困惑度。

### 参数数量

对于一个块，`d_model = d`，FFN 扩展 `r`：

- MHA：`4 · d²`（Q、K、V、O 投影）
- FFN (SwiGLU)：`3 · d · (r · d)` ≈ `3rd²`
- Norms：可忽略

在 `d = 4096, r = 2.6, layers = 32`（大致 Llama 3 8B），总计：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B 参数每层 × 32 ≈ 7B`（加上嵌入和头）。匹配已发布的计数。

## Build It（动手实现）

### Step 1: 基础组件

使用 Lesson 03 的小 `Matrix` 类（复制到这个文件以保持独立）：

- `layer_norm(x, eps=1e-5)` — 减去均值，除以标准差。
- `rms_norm(x, eps=1e-6)` — 除以 RMS。不做均值减法。
- `gelu(x)` 和 `silu(x) * W3 x`（SwiGLU）。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 和 `decoder_block(x, enc_out, params)`。

见 `code/main.py` 获取完整连线。

### Step 2: 连接一个 2 层编码器和一个 2 层解码器

堆叠它们。把编码器输出传入每个解码器交叉注意力。在输出投影之前加一个最终 LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### Step 3: 在 toy 示例上运行前向传播

传入一个 6 token 的源和一个 5 token 的目标。验证输出形状是 `(5, vocab)`。没有训练——这节课讲架构，不是损失。

### Step 4: 换成 RMSNorm + SwiGLU

用 RMSNorm 和 SwiGLU 替换 LayerNorm 和 ReLU-FFN。确认形状仍然匹配。这是用一个函数替换完成的 2026 年现代化。

## Use It（实际应用）

PyTorch/TF 参考实现：`nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。但大多数 2026 年生产代码自己实现块，因为：

- Flash Attention 在注意力内部调用，而不是通过 `nn.MultiheadAttention`。
- GQA / MLA 不在 stdlib 参考中。
- RoPE、RMSNorm、SwiGLU 不是 PyTorch 默认。

HF `transformers` 有干净的参考块值得一读：`modeling_llama.py` 是 2026 年 decoder-only 块的规范。它大约 500 行，值得走一遍。

**编码器 vs 解码器 vs 编码器-解码器——何时选择：**

| Need（需求） | Pick（选择） | Example（示例） |
|------|------|---------|
| 分类、嵌入、文本问答 | Encoder-only（仅编码器） | BERT, DeBERTa, ModernBERT |
| 文本生成、对话、代码、推理 | Decoder-only（仅解码器） | GPT, Llama, Claude, Qwen |
| 结构化输入 → 结构化输出（翻译、摘要） | Encoder-decoder（编码器-解码器） | T5, BART, Whisper |

Decoder-only 在语言领域获胜，因为它扩展最干净，同时处理理解和生成。当输入有一个明确的“源序列”身份（翻译、语音识别、结构化任务）时，Encoder-decoder 仍然是最佳选择。

## Ship It（交付）

见 `outputs/skill-transformer-block-reviewer.md`。这个 skill 对照 2026 年默认值审查一个新的 transformer 块实现，并标记缺失的部分（pre-norm、RoPE、RMSNorm、GQA、FFN 扩展比例）。

## Exercises（练习）

1. **Easy（简单）。** 在你的 `encoder_block` 中计算参数数量，`d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`。通过实现块并使用 `sum(p.numel() for p in block.parameters())` 验证。
2. **Medium（中等）。** 从 post-norm 切换到 pre-norm。初始化两者，在随机输入上测量 12 个堆叠层后的激活范数。Post-norm 的激活应该爆炸；pre-norm 的应该保持有界。
3. **Hard（困难）。** 在一个 toy 复制任务（复制 `x` 的反转）上实现一个 4 层编码器-解码器。训练 100 步。报告损失。换成 RMSNorm + SwiGLU + RoPE——损失下降了吗？

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Block（块） | "One transformer layer"（一个 transformer 层） | norm + attention + norm + FFN 的堆栈，用残差连接包裹。 |
| Residual（残差） | "Skip connection"（跳跃连接） | `x + f(x)` 输出；使梯度能够流过深层堆栈。 |
| Pre-norm | "Normalize before, not after"（在前面归一化，而不是后面） | 现代：`x + sublayer(LN(x))`。无需预热体操即可深度训练。 |
| RMSNorm | "LayerNorm without the mean"（没有均值的 LayerNorm） | 除以 RMS；少一个操作，同样经验稳定。 |
| SwiGLU | "The FFN everyone switched to"（所有人都切换到的 FFN） | `Swish(W1 x) ⊙ W3 x → W2`。在 LM 困惑度上比 ReLU/GELU 好 ~0.5 点。 |
| Cross-attention（交叉注意力） | "How the decoder sees the encoder"（解码器如何看到编码器） | Q 来自解码器、K/V 来自编码器输出的 MHA。 |
| FFN expansion（FFN 扩展） | "How wide the middle MLP is"（中间 MLP 有多宽） | 隐藏大小与 d_model 的比率，通常是 4（LayerNorm）或 2.6（SwiGLU）。 |
| Bias-free（无偏置） | "Drop the +b terms"（去掉 +b 项） | 现代栈在线性层中省略偏置；轻微困惑度改进，模型更小。 |

## Further Reading（延伸阅读）

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始块规范。
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — 为什么 pre-norm 深度上优于 post-norm。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 2026 年 decoder-only 块的规范。
