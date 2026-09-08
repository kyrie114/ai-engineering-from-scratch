# KV Cache, Flash Attention & Inference Optimization（KV Cache、Flash Attention 与推理优化）

> 训练是并行且 FLOP-bound 的。推理是串行且 memory-bound 的。不同的瓶颈，不同的技巧。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention)（自注意力）, Phase 7 · 05 (Full Transformer)（完整 Transformer）, Phase 7 · 07 (GPT)（GPT）
**Time:** ~75 minutes（约 75 分钟）

## The Problem（问题）

一个 naive 自回归解码器做 `O(N²)` 工作来生成 `N` 个 token：在每一步它重新计算整个前缀的注意力。对于一个 4K token 的响应，那是 16M 注意力操作，其中大部分是冗余的。一个前缀 token 的每个隐藏状态一旦计算出来就是确定性的 —— 你只需要运行新 token 的查询，针对之前所有内容的缓存键和值。

在注意力之上，它移动大量数据。标准注意力实现一个 N×N 分数矩阵，N×d softmax 输出，N×d 最终输出 —— 太多的 HBM 读写。对于 N≥2K，注意力在成为 FLOP-bound 之前先成为 memory-bound。经典注意力内核对现代 GPU 的使用不足 4–10 倍。

两个优化，都来自 Dao et al.，把前沿推理从“慢”推到“快”：

1. **KV cache。** 存储每个前缀 token 的 K 和 V 向量。每个新 token 的注意力是针对缓存键的一次查询。推理从每个生成步骤的 `O(N²)` 减少到 `O(N)`。
2. **Flash Attention。** Tile 注意力计算，使完整的 N×N 矩阵从不命中 HBM。所有 softmax + 矩阵乘法都在 SRAM 中发生。在 A100 上 2–4× wall-clock 加速；在 H100 上用 FP8 上 5–10×。

到 2026 年，两者都是通用的。每个生产推理栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）都假设它们。每个前沿模型都带着启用的 Flash Attention 发货。

## The Concept（概念）

![KV cache growth and Flash Attention tiling（KV cache 增长和 Flash Attention 分块）](../assets/kv-cache-flash-attn.svg)

### KV cache math（KV cache 数学）

每个解码器层，每个 token，每个头：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

对于一个 7B 模型，32 层，32 个头，d_head=128，fp16：

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

对于 Llama 3 70B（80 层，d_head=128，GQA 带 8 KV 头）：

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

那个 10 GB 是为什么 Llama 3 70B 在 128K 上下文下 batch size 为 1 时需要大部分 40 GB A100 仅用于 KV cache。

**GQA 是 KV-cache win。** 64 个头的 MHA 将是 32 GB。MLA 压缩更多。

拖动维度并观察 cache 大小移动。把序列长度或 batch 推上去，看看它多快超过单个 GPU：

```figure
kv-cache-sizer
```

### Flash Attention — tiling trick（分块技巧）

标准注意力：

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

三次 HBM round trips。在 H100 上，HBM 带宽是 3 TB/s；SRAM 是 30 TB/s。每次 HBM trip 与保持在片上相比是一个 10 倍的减速。

Flash Attention：

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

每个 tile 一次 HBM trip。总内存占用从 `O(N²)` 降到 `O(N)`。后向传播从 forward pass 重新计算一些值而不是存储它们 —— 另一个内存 win。

**数值技巧。** 运行 softmax 跨 tile 维护 `(max, sum)`，使最终归一化是精确的。不是近似 —— Flash Attention 计算与标准注意力 bit-identical 的输出（模 fp16 非结合性）。

**版本演进：**

| Version（版本） | Year（年份） | Key change（关键变化） | Speedup on reference hardware（参考硬件上的加速） |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | Tiled SRAM kernel | 2× on A100 |
| Flash 2 | 2023 | Better parallelism, causal-first ordering（更好的并行性，因果优先顺序） | 3× on A100 |
| Flash 3 | 2024 | Hopper asynchrony, FP8 | 1.5–2× on H100 (~740 TFLOPs FP16) |
| Flash 4 | 2026 | Blackwell 5-stage pipeline, software exp2（Blackwell 5 级流水线，软件 exp2） | Inference-first（推理优先）（最初仅前向传播） |

Flash 4 在发布时仅限前向传播。训练仍然使用 Flash 3。Flash 4 的 GQA 和 varlen 支持 pending（2026 年中期）。

### Speculative decoding — the other latency win（推测解码 —— 另一个延迟 win）

廉价模型提出 N 个 token。大模型并行验证所有 N 个。如果验证接受 k 个 token，你为 k 次生成支付了 1 次大模型前向传播。典型 k=3–5 在代码和散文上。

2026 默认：

- **EAGLE 2 / Medusa。** 集成草稿头，共享验证器的隐藏状态。2–3× 加速，无质量损失。
- **Speculative decoding with draft model。** 在消费者硬件上 2–4× 加速。
- **Lookahead decoding。** Jacobi iteration；不需要草稿模型。利基但免费。

### Continuous batching（连续批处理）

经典批处理推理：等待最慢的序列完成，然后开始一个新的 batch。当短响应提前完成时浪费 GPU。

连续批处理（首次在 Orca 中 shipped，现在在 vLLM、TensorRT-LLM、SGLang 中）：一旦旧序列完成，就把新请求 swap 进 batch。对于典型聊天工作负载，5–10× 吞吐量增益。

### PagedAttention — KV cache as virtual memory（KV cache 作为虚拟内存）

vLLM 的 headline 特性。KV cache 以 16 token 块分配；一个页表把逻辑位置映射到物理块。让你跨并行样本共享 KV（beam search、parallel sampling），热 swap 前缀用于 prompt caching，和 defragment 内存。与 naive contiguous allocation 相比 4× 吞吐量改进。

```figure
flash-attention-memory
```

## Build It（动手实现）

见 `code/main.py`。我们实现：

1. 一个 naive `O(N²)` 增量解码器。
2. 一个 `O(N)` KV-cached 解码器。
3. 一个模拟 Flash Attention 的 running-max 算法的 tiled softmax。

### Step 1: KV cache

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

简单：在 per-layer、per-head 列表中持续增长 per-token K、V 向量。

### Step 2: tiled softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

与一次性的 `softmax(qK) V` bit-identical 输出，但在任何时间工作集是一个 `tile × d_head` 块，不是完整的 `N × d_head`。

### Step 3: 在 100-token 生成上比较 naive vs cached decoding

计算注意力操作。Naive：`O(N²)` = 5050。Cached：`O(N)` = 100。代码打印两者。

## Use It（实际应用）

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically（generate() 自动使用 KV cache）
```

vLLM 生产：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求的 prefix caching 是一个巨大的 2026 win —— 相同的系统提示、few-shot examples 或长上下文文档跨调用重用 KV。对于带有重复工具提示的 agent 工作负载，prefix caching  routinely 是 5× 吞吐量增益。

## Ship It（交付）

见 `outputs/skill-inference-optimizer.md`。这个 skill 为一个新的推理部署选择注意力实现、KV cache 策略、量化和推测解码。

## Exercises（练习）

1. **Easy（简单）。** 运行 `code/main.py`。确认 naive 和 cached 解码器产生相同的输出；注意操作计数差异。
2. **Medium（中等）。** 实现 prefix caching：给定一个提示 P 和几个补全，对 P 运行一次前向传播来填充 KV cache，然后 per-completion 分支。与为每个补全重新编码 P 相比测量加速。
3. **Hard（困难）。** 实现一个 toy PagedAttention：KV cache 在固定 16 token 块中带一个 free-list。当一个序列完成时，把它的块返回池。模拟 1,000 个聊天补全，长度不同。与 contiguous allocation 比较内存碎片。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| KV cache | "The trick that makes decoding fast"（让解码变快的技巧） | 存储的每个前缀 token 的 K 和 V；新查询关注它们而不是重新计算。 |
| HBM | "GPU main memory"（GPU 主内存） | 高带宽内存；H100 上 80 GB，B200 上 192 GB。~3 TB/s 带宽。 |
| SRAM | "On-chip memory"（片上内存） | 每个 SM 快速内存，H100 上每个 SM ~256 KB。~30 TB/s 带宽。 |
| Flash Attention | "Tiled attention kernel"（分块注意力内核） | 在 HBM 中不实现 N×N 矩阵的情况下计算注意力。 |
| Continuous batching（连续批处理） | "No-wait batching"（无等待批处理） | 把完成的序列 swap out，新序列 in，不用排空 batch。 |
| PagedAttention | "vLLM's headline"（vLLM 的标题） | KV cache 在固定块中分配，带一个页表；消除碎片。 |
| Prefix caching（前缀缓存） | "Reuse long prompts"（重用长提示） | 跨请求缓存共享前缀的 KV；对 agent 的主要成本削减。 |
| Speculative decoding（推测解码） | "Draft + verify"（草稿 + 验证） | 廉价草稿模型提出 token；大模型并行验证 k 个。 |

## Further Reading（延伸阅读）

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1。
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2。
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3。
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5-stage pipeline 和 software-exp2 技巧；阅读 repo README 了解本课提到的 forward-only launch caveats。
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM 论文。
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 推测解码。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — 本课引用的集成草稿方法 EAGLE-1/2 论文。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — 与 EAGLE 一起引用的 Medusa 方法。
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 关于 16 token 块和页表设计的规范深度 dive。
