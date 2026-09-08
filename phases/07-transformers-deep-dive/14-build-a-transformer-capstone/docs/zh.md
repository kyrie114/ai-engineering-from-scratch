# Build a Transformer from Scratch — The Capstone（从零构建 Transformer —— 毕业设计）

> 十三节课。一个模型。没有捷径。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 01 through 13（第 01 到 13 课）。不要跳过。
**Time:** ~120 minutes（约 120 分钟）

## The Problem（问题）

你读了每篇论文。你实现了 attention、multi-head split、positional encoding、encoder 和 decoder block、BERT 和 GPT loss、MoE、KV cache。现在让它们在一个真实任务上一起工作。

毕业设计：在一个 character-level language modeling 任务上端到端训练一个小 decoder-only transformer。它读莎士比亚。它生成新的莎士比亚。它足够小，可以在笔记本电脑上不到 10 分钟内训练。它足够正确，交换一个更大的数据集和更长的训练会得到一个真正的 LM。

这是课程的“nanoGPT”。它不是原创的 —— Karpathy 2023 年的 nanoGPT 教程是每个学生至少写一次的参考实现。我们采用形状并重新工具化，以覆盖我们学过的东西。

## The Concept（概念）

![Transformer-from-scratch block diagram（从零构建 Transformer 块图）](../assets/capstone.svg)

架构，带注释：

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)（第 04 课，RoPE 选项）
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05（第 05 课）
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)（第 03 + 07 课，因果遮罩）
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05（第 05 课）
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07（第 07 课）
```

### 我们 ship 什么

- `GPTConfig` —— 一个地方配置所有超参数。
- `MultiHeadAttention` —— causal、batched，带可选 Flash-style pathway（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN` —— 现代 FFN。
- `Block` —— pre-norm、residual-wrapped attention + FFN。
- `GPT` —— embeddings、stacked blocks、LM head、generate()。
- 带 AdamW、cosine LR、gradient clipping 的训练循环。
- 莎士比亚文本的 char-level tokenizer。

### 我们不 ship 什么

- RoPE —— 在 Lesson 04 中概念实现。这里我们使用 learned positional embeddings 以保持简单。练习要求你换成 RoPE。
- KV cache during generation —— 每个生成步骤重新计算整个前缀的注意力。更慢但更简单。练习要求你添加一个 KV cache。
- Flash Attention —— PyTorch 2.0+ 在输入匹配时自动调度；我们使用 `F.scaled_dot_product_attention`。
- MoE —— 每个块单个 FFN。你在 Lesson 11 中看到了 MoE。

### 目标 metric

在一个 Mac M2 笔记本电脑上，一个 4-layer、4-head、d_model=128 GPT 在 `tinyshakespeare.txt` 上训练 2,000 步：

- Training loss 从 ~4.2（random）收敛到 ~1.5，大约 6 分钟。
- Sampled output 看起来像莎士比亚-shaped：archaic words、line breaks、proper names like "ROMEO:" emerge。
- Val loss（held-out final 10% of text）紧密跟踪 training loss；在这个 size/budget 下没有 overfitting。

```figure
n5-block-stack
```

## Build It（动手实现）

这节课使用 PyTorch。安装 `torch`（CPU build 就够了）。见 `code/main.py`。脚本处理：

- 如果缺失，下载 `tinyshakespeare.txt`（或读取本地副本）。
- Byte-level char tokenizer。
- 90/10 train/val split。
- 带 bf16 autocast 在支持硬件上的训练循环。
- 训练完成后采样。

### Step 1: data（数据）

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 个唯一字符。Tiny 词表。Fits a 4-byte vocab_size。没有 BPE，没有 tokenizer drama。

### Step 2: model（模型）

见 `code/main.py`。块是 Lesson 05 的 textbook —— pre-norm、RMSNorm、SwiGLU、causal MHA。4/4/128 的参数计数：~800K。

### Step 3: training loop（训练循环）

获取一个 length-256 token 窗口的随机 batch。Forward。Shift-by-one cross-entropy。Backward。AdamW step。Log。Repeat。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### Step 4: sample（采样）

给定一个 prompt，重复 forward，从 top-p logits 采样，追加，然后继续。在 500 token 后停止。

### Step 5: read the output（读取输出）

2,000 步之后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亚。但莎士比亚-shaped。~800K 参数和笔记本电脑上 6 分钟的一个 clear win。

## Use It（实际应用）

这个毕业设计是一个参考架构。三个 extension 把它 ship 到真实的东西：

1. **Swap the tokenizer。** 使用 BPE（例如 `tiktoken.get_encoding("cl100k_base")`）。词表大小从 65 跳到 ~50,000。模型容量需要扩展以补偿。
2. **Train on a bigger corpus。** 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace）。10B token 在单个 A100 上需要 ~24 小时，用于一个 125M-param GPT。
3. **Add RoPE + KV cache + Flash Attention。** 下面的练习带你走过每个。

这最终变成一个 125M-parameter GPT，生成流畅的英语。不是一个前沿模型。但同样的代码路径 —— 只是更大 —— 是 Karpathy、EleutherAI 和 Allen Institute 在 2026 年训练 research checkpoint 使用的。

## Ship It（交付）

见 `outputs/skill-transformer-review.md`。这个 skill 审查一个从零构建的 transformer 实现，对照前面 13 节课检查正确性。

## Exercises（练习）

1. **Easy（简单）。** 运行 `code/main.py`。验证你训练的模型的最后一步验证损失低于 2.0。把 `max_steps` 从 2,000 改成 5,000 —— val loss 继续改进吗？
2. **Medium（中等）。** 用 RoPE 替换 learned positional embeddings。在 `MultiHeadAttention` 内部把旋转应用到 Q 和 K。训练并验证 val loss 至少一样低。
3. **Medium（中等）。** 在采样循环中实现一个 KV cache。有和没有 cache 生成 500 token。Wall-clock 应该在笔记本电脑上提高 5–20×。
4. **Hard（困难）。** 给模型添加第二个头，预测 next-plus-one token（MTP —— DeepSeek-V3 的 Multi-Token Prediction）。联合训练。它有帮助吗？
5. **Hard（困难）。** 用 4-expert MoE 替换每个块的单个 FFN。Router + top-2 routing。在匹配的活跃参数下看 val loss 如何变化。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| nanoGPT | "Karpathy's tutorial repo"（Karpathy 的教程 repo） | 最小 decoder-only transformer 训练代码，~300 LOC；规范参考。 |
| tinyshakespeare | "The standard toy corpus"（标准 toy 语料库） | ~1.1 MB 文本；每个 character-LM 教程自 2015 年以来都使用它。 |
| Tied embeddings（绑定嵌入） | "Share input/output matrix"（共享输入/输出矩阵） | LM head weight = token embedding matrix 的 transpose；节省参数，改进质量。 |
| bf16 autocast | "Training precision trick"（训练精度技巧） | 前向/反向在 bf16 中运行，optimizer 状态在 fp32 中保持；自 2021 年以来的标准。 |
| Gradient clipping（梯度裁剪） | "Stops spikes"（停止 spikes） | 把 global grad norm 限制在 1.0；防止训练 blowups。 |
| Cosine LR schedule（余弦 LR schedule） | "The 2020+ default"（2020+ 默认） | LR 线性 ramp up（warmup）然后 cosine-shaped decay 到峰值的 10%。 |
| MFU | "Model FLOP Utilization"（模型 FLOP 利用率） |  Achieved FLOPs / theoretical peak（实现的 FLOPs / 理论峰值）；2026 年 dense 40%，MoE 30% 是强的。 |
| Val loss | "Held-out loss"（held-out 损失） | 模型从未见过的数据上的 cross-entropy；overfit detector。 |

## Further Reading（延伸阅读）

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — 经典带注释的实现。
