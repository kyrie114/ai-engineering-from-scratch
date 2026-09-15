# 梯度检查点与激活重计算（Gradient Checkpointing and Activation Recomputation）

> 反向传播保留每一个中间激活。在 70B 参数和 128K 上下文下，那是每个 rank 3 TB 激活。检查点用 FLOPs 换内存：重算而不是保存。问题是丢掉哪些段，答案不是「全部丢掉」。

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 minutes

## 问题（The Problem）

训练 transformer 时，每一层都会存储反向中被微分的每个算子的输入：注意力输入、Q/K/V 投影、softmax 输出、FFN 输入、归一化输出，以及残差流。对隐层大小 `d`、序列长度 `L`、批量 `B` 的一层，量级是每层 `12 * B * L * d` 个浮点。

对 `d=8192, L=8192, B=1`，那是 BF16 下每层 800 MB。64 层模型是 51 GB 激活——而这还没乘微批量大小，还没加注意力 softmax 中间量（每头 `L^2`），还没计入张量并行的部分拷贝。

两边账单：BF16 权重加优化器状态也许能塞进 80GB，但激活把你推过去。梯度检查点（gradient checkpointing，又称激活重计算 activation recomputation）是标准修法。丢掉大多数激活；反向时重做前向把它们拿回来。代价：额外 FLOPs。收益：内存按检查点段数对总层数的比例下降。

做得朴素，检查点大约让每步多 33% 前向 FLOPs。做得好——按 Korthikanti 等人的「聪明选择」做选择性检查点——你用不到 5% 的 FLOP 开销省下 5 倍内存。再加上 FP8 矩阵乘、FSDP 卸载和专家并行 MoE，这真的要紧：内存和浪费的算力你都负担不起。

## 概念（The Concept）

### 反向实际需要什么（What Backward Actually Needs）

`output = layer(input)`。反向想要 `grad_input` 和 `grad_params`。要算它们，它需要：

- `input`（对线性层计算 `grad_params = input.T @ grad_output`）
- 一些激活导数中间量（ReLU/GELU/softmax 的导数依赖激活值）

前向在自动微分图里自动存储这些。每一个 `tensor.retain_grad()` 和每一个需要其输入的算子都保留一份引用。

### 朴素全检查点（Naive Full Checkpointing）

把网络拆成 `N` 段。前向期间只存储每段的*输入*。反向需要中间量时，重跑该段前向把它们物化，再微分。

例子：32 层 transformer 拆成 32 段，每段 1 层。

- 内存：32 个层输入（小）对 32 *（每层激活体积）（巨大）。
- 额外计算：每段 1 次额外前向，即总共约多 33% 前向 FLOPs（因为反向是 2 倍前向，整步从 1 + 2 = 3 个单位变成 1 + 1 + 2 = 4）。

这是 Chen 等人 2016 的原始配方：每 `sqrt(L)` 层一个检查点，以平衡内存与计算。对 L=64，那是 8 个检查点。

### 选择性检查点（Selective Checkpointing，Korthikanti 2022）

不是所有激活代价相同。注意力 softmax 输出是 `B*L*L*heads`，随序列长度*二次*增长。FFN 隐激活是 `B*L*4d`，线性增长。长序列时 softmax 占主导。

选择性检查点保留便宜存储的激活（线性投影、残差），只重算昂贵的那些（注意力）。你付极少 FLOPs 去重算，却省下 O(L^2) 内存。

Megatron-Core 把它实现成「选择性」激活重计算。用在大多数 2024+ 前沿训练运行里。

### 卸载（Offload）

重算的替代：在前向与反向之间把激活运到 CPU RAM。需要 PCIe 带宽；当空闲带宽超过重物化代价时有益。混合策略常见：检查点某些层，卸载另一些。

FSDP2 把卸载做成一等选项。当 GPU 卡在内存、但 CPU-GPU 传输还有余量时，卸载发光。

### 重算成本模型（Recompute Cost Model）

每 `k` 层做一次朴素检查点、共 `L` 层时的每步 FLOPs：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

选择性检查点只重算注意力内核，不是整层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型（Memory Savings Model）

每层激活体积：`A`。对 `L` 层，总激活内存：`L * A`。

全检查点（段大小 1）：只存 `L * input_volume`（标准 transformer 大约 `L * 1/10 A`）。节省约 `9 * L * A * 1/10`。

每 `k` 层检查点：存 `L/k * A`，再加上活动段内 `k-1` 层的量。

在 `k = sqrt(L)`，内存和重算成本都随 `sqrt(L)` 缩放——对均匀代价层的最优权衡。

### 何时不要检查点（When Not to Checkpoint）

- 流水线阶段里已经在飞的最内层。它们反正得做完。
- 第一层和最后一层，若它们主导该阶段的计算（transformer 里少见）。
- 已经用 FlashAttention 的注意力内核——Flash 已经很快地重算 softmax，再加一层级检查点收益很小。

### 实现模式（Implementation Patterns）

1. **函数包装：** 把一段包进 `torch.utils.checkpoint.checkpoint(fn, input)`。PyTorch 只存 `input`，反向时重算其他一切。

2. **基于装饰器：** 把层标成可检查点；训练器在配置时决定哪些段被包装。

3. **手工显式重算：** 自己写反向，调用自定义 `recompute_forward`，用存下的输入复制前向。

三者功能结果相同。包装器是标准惯用法。

### 与 TP / PP / FP8 的交互（Interaction with TP / PP / FP8）

- **张量并行：** 检查点输入在重算时必须 gather 或再 scatter；处理通信成本。
- **流水线并行：** 典型模式是检查点每个流水线阶段的前向，好让逆序微批次复用激活内存。
- **FP8 重算：** 重算期间更新的 amax 历史必须匹配原始前向，否则 FP8 缩放会漂。大多数框架会快照缩放。

```figure
activation-recompute
```

## 动手构建（Build It）

### 第 1 步：带分段的玩具模型（Step 1: A Toy Model With Segments）

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 第 2 步：需要全部激活的朴素反向（Step 2: Naive Backward Needing All Activations）

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 第 3 步：每 k 检查点的内存（Step 3: Checkpoint-Every-k Memory）

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 第 4 步：成本模型（Step 4: Cost Model）

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 第 5 步：内存估计器（Step 5: Memory Estimator）

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 第 6 步：最优段大小（Step 6: Optimal Segment Size）

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 第 7 步：选择性检查点决策（Step 7: Selective Checkpoint Decision）

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 使用它（Use It）

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint` —— PyTorch 里的规范包装器。包装一个函数；只存输入，反向时重算。
- **Megatron-Core 激活重计算**：支持 `selective`、`full` 和 `block` 模式。2024+ 前沿训练的标准。
- **FSDP2 卸载**：`module.to_empty(device="cpu")` 配合 FSDP2 的 `offload_policy`，把激活分片到 CPU 而不是重算。
- **DeepSpeed ZeRO-Offload**：优化器状态和激活的 CPU 卸载，补充检查点。

## 交付（Ship It）

本课产出 `outputs/prompt-activation-recompute-policy.md`——一份提示，接受你的模型配置（层数、隐层、序列、批量）和可用 GPU 内存，发出逐层重算策略（无 / 选择性 / 全量 / 卸载）。

## 练习（Exercises）

1. 验证正确性。跑 `model_forward` + `model_backward`（全部激活）对比 `model_forward_checkpointed` + `model_backward_checkpointed`（分段）。参数梯度必须在机器精度内相同。

2. 把段大小 `k` 从 1 扫到 `L`。画出 FLOP 开销和内存。找出曲线的拐点。

3. 实现选择性检查点：存注意力模块输入但不存其中间量。对 32 层模型在 seq=8192 测量相对整层检查点的 FLOP 开销。

4. 加卸载。把段输入存进模拟「CPU 缓冲」（另一个列表）。把「PCIe 带宽」测成 bytes/time，找出卸载与重算的盈亏平衡点。

5. 给真实 PyTorch transformer 做有无 `torch.utils.checkpoint` 的基准。测量内存（经 `torch.cuda.max_memory_allocated`）和步时间。

## 关键术语（Key Terms）

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 梯度检查点 | 「重做前向以省内存」 | 只存段输入；反向时重算中间量以得到支持梯度的张量 |
| 激活重计算 | 「和检查点一样」 | 同一技术的 HPC 风味名字 |
| 段大小（k） | 「每个检查点多少层」 | 中间量被丢掉并一起再物化的层数 |
| 选择性检查点 | 「Korthikanti 的诀窍」 | 只重算存储昂贵的激活（注意力 softmax）；保留便宜的 |
| 全检查点 | 「朴素版本」 | 每段重算每一层的中间量 |
| 块检查点 | 「粗粒度」 | 检查点整个 transformer 块；最大粒度 |
| FLOP 开销 | 「计算税」 | 每步额外 FLOPs = (重算 FLOPs) / (前向 + 反向 FLOPs)；朴素 33%，选择性 5% |
| 激活卸载 | 「运到 CPU」 | 跨前向->反向把激活搬到 CPU RAM；重算的替代 |
| sqrt-L 规则 | 「经典最优」 | 对均匀代价层，最优检查点间距是 sqrt(L) 层 |
| 注意力 softmax 体积 | 「O(L^2) 问题」 | L^2 * heads * batch 个浮点；长上下文时主导激活内存 |

## 延伸阅读（Further Reading）

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 形式化梯度检查点的原始论文
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- 选择性激活重计算与形式化成本分析
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- 经反向模式再物化的常数内存替代
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- 规模上的激活卸载
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) -- 标准 API
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- selective、full 和 block 模式
