# Why Transformers — The Problems with RNNs（为什么用 Transformer——RNN 的问题）

> RNNs process tokens one at a time. Transformers process all tokens at once. That single architectural bet changed every scaling curve in deep learning after 2017.
> RNN 一次处理一个 token。Transformer 一次处理所有 token。这个单一的架构赌注改变了 2017 年之后深度学习的每一条扩展曲线。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**Time:** ~45 minutes

## The Problem（问题）

Before 2017, every state-of-the-art sequence model on the planet — language, translation, speech — was a recurrent neural network. LSTMs and GRUs won ImageNet-equivalent translation benchmarks for half a decade. They were the only tool anyone had.
2017 年之前，地球上每个最先进的序列模型——语言、翻译、语音——都是循环神经网络。LSTM 和 GRU 在相当于 ImageNet 的翻译基准上赢了五年。它们是任何人唯一有的工具。

They had three fatal weaknesses. Sequential computation meant you could not parallelize along the time axis: token `t+1` needs the hidden state from token `t`. A 1,024-token sequence meant 1,024 serial steps on a GPU that can do 1,000,000 floating-point ops per cycle. Training wall-clock time scaled linearly with sequence length on hardware designed for parallelism.
它们有三个致命弱点。顺序计算意味着你无法沿着时间轴并行化：token `t+1` 需要来自 token `t` 的隐藏状态。一个 1024 token 的序列意味着在每周期可以做 1,000,000 次浮点运算的 GPU 上做 1024 个串行步骤。训练挂钟时间在设计用于并行化的硬件上随序列长度线性扩展。

Vanishing gradients meant information 50 tokens back was already compressed through 50 non-linearities. Gated recurrent units (LSTM, GRU) softened the crush but never eliminated it. Long-range dependencies — "the book I read last summer on a plane to Kyoto was…" — routinely failed.
梯度消失意味着 50 个 token 之前的信息已经通过 50 个非线性压缩了。门控循环单元（LSTM、GRU）减轻了挤压但从未消除它。长程依赖——"the book I read last summer on a plane to Kyoto was…"—— routinely failed。

Fixed-width hidden states meant the encoder squeezed the entire source sequence into a single vector before the decoder saw anything. Doesn't matter if the source is 5 tokens or 500; the bottleneck is the same shape.
固定宽度的隐藏状态意味着编码器在解码器看到任何东西之前把整个源序列压缩成一个单一向量。源是 5 个 token 还是 500 个都不重要；瓶颈是相同的形状。

The 2017 paper "Attention Is All You Need" proposed something radical: drop recurrence entirely. Let every position attend to every other position in parallel. Train in one big matrix multiplication instead of 1,024 sequential ones.
2017 年的论文"Attention Is All You Need"提出了一个激进的想法：完全放弃循环。让每个位置并行地关注每个其他位置。用一个大的矩阵乘法训练，而不是 1024 个串行的。

The result dominates every modality by 2026. Language (GPT-5, Claude 4, Llama 4), vision (ViT, DINOv2, SAM 3), audio (Whisper), biology (AlphaFold 3), robotics (RT-2). Same block, different inputs.
结果在 2026 年主导每个模态。语言（GPT-5、Claude 4、Llama 4）、视觉（ViT、DINOv2、SAM 3）、音频（Whisper）、生物学（AlphaFold 3）、机器人（RT-2）。相同的块，不同的输入。

## The Concept（概念）

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**Recurrence as a bottleneck（循环作为瓶颈）. ** An RNN computes `h_t = f(h_{t-1}, x_t)`. Each step depends on the previous. You cannot compute `h_5` before `h_4`. On modern GPUs with 10,000+ parallel cores, this wastes 99% of the silicon on a long sequence.
  RNN 计算 `h_t = f(h_{t-1}, x_t)`。每一步依赖前一步。你不能在 `h_4` 之前计算 `h_5`。在现代有 10,000+ 并行核心的 GPU 上，这在长序列上浪费了 99% 的硅。

**Attention as a broadcast（注意力作为广播）. ** Self-attention computes `output_i = sum_j(a_ij * v_j)` for every pair `(i, j)` simultaneously. The whole N×N attention matrix fills in one batched matmul. No step depends on another. GPUs love it.
  自注意力同时为每对 `(i, j)` 计算 `output_i = sum_j(a_ij * v_j)`。整个 N×N 注意力矩阵在一个批处理 matmul 中填充。没有一步依赖另一步。GPU 喜欢它。

**The speedup is not a constant（加速不是常数）. ** It is the difference between `O(N)` serial depth and `O(1)` serial depth. In practice, transformers train 5–10× faster per epoch on matched hardware at N=512, and the gap widens with sequence length until you hit the `O(N²)` memory wall of attention (which Flash Attention later fixed — see Lesson 12).
  这是 `O(N)` 串行深度和 `O(1)` 串行深度之间的差异。实际上，在 N=512 的匹配硬件上，transformer 每轮训练快 5-10 倍，并且差距随序列长度扩大，直到你碰到注意力的 `O(N²)` 内存墙（Flash Attention 后来修复了——见第 12 课）。

**What transformers cost（Transformer 的成本）. ** Attention memory scales as `O(N²)`. For 2K context, fine. For 128K context, you need sliding windows, RoPE extrapolation, Flash Attention tiling, or linear attention variants. Recurrence was `O(N)` in both time and memory; transformers trade time for memory and then win the time back through parallelism.
  注意力内存随 `O(N²)` 扩展。对于 2K 上下文，没问题。对于 128K 上下文，你需要滑动窗口、RoPE 外推、Flash Attention 分块或线性注意力变体。循环在时间和内存上都是 `O(N)`；transformer 用时间换内存，然后通过并行化把时间赢回来。

**The inductive bias shift（归纳偏置转变）. ** RNNs assume locality and recency. Transformers assume nothing — every pair is a candidate for attention. That is why transformers need more data to train well but scale further once they have it. Chinchilla (2022) formalized this: given enough tokens, a transformer always beats an RNN of equal parameter count.
  RNN 假设局部性和新近性。Transformer 什么都不假设——每对都是注意力的候选。这就是为什么 transformer 需要更多数据才能训练好，但一旦有了数据就能进一步扩展。Chinchilla (2022) 形式化了这一点：给定足够的 token，transformer 总是击败相同参数数量的 RNN。

```figure
rnn-vs-parallel
```

## Build It（动手实现）

No neural network here — we simulate the core bottleneck numerically so you feel the gap on your laptop.
这里没有神经网络——我们数值模拟核心瓶颈，因此你可以在你的笔记本上感受到差距。

### Step 1: measure serial depth（测量串行深度）

See `code/main.py`. We build two functions. One encodes a sequence as a chain of additions (serial, like an RNN). One encodes it as a parallel reduction (broadcast, like attention). Same math, different dependency graph.
参见 `code/main.py`。我们构建两个函数。一个把序列编码为加法链（串行，像 RNN）。一个把它编码为并行归约（广播，像注意力）。相同的数学，不同的依赖图。

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

We time both on sequences up to 100,000 elements. The RNN version is O(N) and a single CPU pipeline. Even in pure Python, the attention-style reduction beats it at length ≥ 1,000 because Python's `sum()` is implemented in C and iterates without interpreter overhead per step.
我们在最多 100,000 个元素的序列上对两者计时。RNN 版本是 O(N) 和一个单一 CPU 流水线。即使在纯 Python 中，注意力风格的归约在长度 ≥ 1000 时也击败它，因为 Python 的 `sum()` 用 C 实现并且迭代时没有每步的解释器开销。

### Step 2: count theoretical operations（计算理论操作数）

Both algorithms do N adds. The difference is *dependency depth*: how many operations must happen sequentially before the next can start. RNN depth = N. Attention depth = log(N) with a tree reduction, or 1 with a parallel scan. Depth, not op count, decides GPU time.
两种算法都做 N 次加法。区别在于*依赖深度*：在下一个可以开始之前必须顺序发生多少操作。RNN 深度 = N。注意力深度 = log(N)（树归约），或 1（并行扫描）。深度，而非操作数，决定 GPU 时间。

### Step 3: empirical scaling on long sequences（长序列的经验扩展）

We print a timing table that makes the O(N) gap visible. On a 2026 Mac laptop, sequences under 1,000 elements are too fast to measure. Sequences of 100,000 show a clean linear scan. Scale that to a 16,384-token transformer with a 12-layer LSTM equivalent and you see why training wall-clock was a blocker in 2016.
我们打印一个计时表，使 O(N) 差距可见。在 2026 年的 Mac 笔记本上，1000 个元素以下的序列太快无法测量。100,000 的序列显示一个干净的线性扫描。把它放大到一个 16,384 token 的 transformer，相当于一个 12 层 LSTM，你就会明白为什么训练挂钟时间在 2016 年是一个障碍。

## Use It（实际应用）

When to still pick an RNN in 2026:
2026 年什么时候仍然选择 RNN：

| Situation | Pick |
|-----------|------|
| Streaming inference, one token at a time, constant memory | RNN or state-space model (Mamba, RWKV) |
|                                   | 流式推理，一次一个 token，恒定内存 |
| Very long sequences (>1M tokens) where attention memory explodes | Linear attention, Mamba 2, Hyena |
|                                      | 注意力内存爆炸的非常长序列 (>1M token) |
| Edge device with no matmul accelerator | Depthwise-separable RNN still wins on FLOPs/watt |
|                         | 没有 matmul 加速器的边缘设备 |
| Anything else (training, batched inference, context up to 128K) | Transformer |
|                                  | 其他任何（训练、批处理推理、上下文到 128K） |

State-space models (SSMs) like Mamba are essentially RNNs with structured parameterization that gives them the best of both: `O(N)` scan memory, parallel training via selective scan. They recover 90% of transformer quality with better long-context scaling. In 2026 most frontier labs train hybrid SSM+transformer models (e.g. Jamba, Samba) — recurrence is not dead, it is a component.
像 Mamba 这样的状态空间模型（SSM）本质上是带结构化参数化的 RNN，给了它们两者最好的部分：`O(N)` 扫描内存，通过选择性扫描的并行训练。它们在更好的长上下文扩展下恢复 90% 的 transformer 质量。2026 年大多数前沿实验室训练混合 SSM+transformer 模型（例如 Jamba、Samba）——循环没有死，它是一个组件。

## Ship It（交付成果）

See `outputs/skill-architecture-picker.md`. The skill picks an architecture for a new sequence problem given length, throughput, and training-budget constraints. It should always refuse to recommend a pure RNN for training runs above 1B tokens without stating the trade-off.
参见 `outputs/skill-architecture-picker.md`。该技能为给定的长度、吞吐量和训练预算约束的新序列问题选择架构。它应该始终拒绝推荐纯 RNN 用于 10 亿 token 以上的训练运行，除非说明权衡。

## Exercises（练习）

1. **Easy（简单）.** Take `rnn_style` from `code/main.py` and replace the scalar hidden state with a length-64 vector of hidden states. Re-measure. How much does the serial overhead grow with hidden-state dimension?
   从 `code/main.py` 取 `rnn_style`，把标量隐藏状态替换为长度 64 的隐藏状态向量。重新测量。串行开销随隐藏状态维度增长多少？
2. **Medium（中等）.** Implement a parallel prefix-sum (Hillis-Steele scan) in pure Python. Verify it produces the same numerical output as a serial scan on length 1024. Count the depth.
   用纯 Python 实现并行前缀和（Hillis-Steele 扫描）。验证它在长度 1024 上产生与串行扫描相同的数值输出。计算深度。
3. **Hard（困难）.** Port the attention-style reduction to PyTorch on GPU. Time both as you sweep sequence length from 64 to 65,536. Plot and explain the curve shape.
   把注意力风格的归约移植到 GPU 上的 PyTorch。当你把序列长度从 64 扫到 65,536 时对两者计时。绘制并解释曲线形状。

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Recurrence | "RNNs are sequential" | Computation where step `t` depends on step `t-1`, forcing serial execution along the time axis. |
|            | "RNN 是顺序的" | 计算中步骤 `t` 依赖步骤 `t-1`，强制沿时间轴的串行执行。 |
| Serial depth | "How deep the graph is" | Longest chain of dependent ops; bounds wall-clock even on infinite hardware. |
|               | "图有多深" | 依赖操作的最长链；即使在无限硬件上也限制挂钟时间。 |
| Attention | "Let tokens look at each other" | Weighted sum `sum_j a_ij v_j` where `a_ij` comes from a similarity score between positions i and j. |
|           | "让 token 互相看" | 加权和 `sum_j a_ij v_j`，其中 `a_ij` 来自位置 i 和 j 之间的相似度分数。 |
| Context window | "How much the model sees" | Number of positions an attention layer can take as input; quadratic memory cost scales here. |
|                | "模型看到多少" | 注意力层可以作为输入的位置数；二次内存成本在这里扩展。 |
| Inductive bias | "Assumptions baked into the architecture" | Prior about what the data looks like; CNNs assume translation invariance, RNNs assume recency. |
|                | "烘焙在架构中的假设" | 关于数据样子的先验；CNN 假设平移不变性，RNN 假设新近性。 |
| State-space model | "RNN with algebra behind it" | Recurrence parameterized for parallel training via structured state-space matrices. |
|                    | "背后有代数的 RNN" | 通过结构化状态空间矩阵参数化的循环，用于并行训练。 |
| Quadratic bottleneck | "Why context costs so much" | Attention memory = `O(N²)` in sequence length; Flash Attention hides the constants, not the scaling. |
|                       | "为什么上下文这么贵" | 注意力内存 = 序列长度的 `O(N²)`；Flash Attention 隐藏常数，而非扩展。 |

## Further Reading（延伸阅读）

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the paper that killed recurrence in mainstream NLP.
  Vaswani 等 (2017).《Attention Is All You Need》——在主流 NLP 中杀死循环的论文。
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — where attention was born, bolted onto an RNN.
  Bahdanau、Cho、Bengio (2014).《通过联合学习对齐和翻译进行神经机器翻译》——注意力的诞生地， bolt 到 RNN 上。
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — the original LSTM paper, for the record.
  Hochreiter、Schmidhuber (1997).《长短期记忆》——原始的 LSTM 论文，存档用。
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — modern recurrent answer to transformers.
  Gu、Dao (2023).《Mamba：用选择性状态空间做线性时间序列建模》——对 transformer 的现代循环回答。
