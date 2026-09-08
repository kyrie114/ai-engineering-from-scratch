# Scaling Laws（Scaling Laws）

> 2020 年的 Kaplan 论文说：更大的模型，更低的损失。2022 年的 Hoffmann 论文说：你训练不足。计算进入两个桶 —— 参数和 token —— 而拆分并不明显。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer)（完整 Transformer）, Phase 7 · 07 (GPT)（GPT）
**Time:** ~45 minutes（约 45 分钟）

## The Problem（问题）

当你有 C FLOPs 的训练计算，想要最好的模型时，你面临两个旋钮：

1. **多少参数 (N)？** 更大的模型，更高的容量。
2. **多少训练 token (D)？** 更多数据，更好地利用容量。

FLOPs 大约为 `6 × N × D`。你可以推高 N 降低 D，或者推高 D 降低 N。哪个更好？

2022 年之前，答案是“硬推 N”。GPT-3 (2020) 是 175B 参数，训练在 ~300B token 上。一个大约 1.7 个 token  per 参数的比率。Kaplan scaling laws 支持这一点。

Hoffmann et al. (2022)，训练一个叫做 Chinchilla 的小模型家族，发现了不同的东西：最佳比率更接近 **20 token  per 参数**。GPT-3 被训练不足 10 倍。Chinchilla（70B params，1.4T tokens）在每个基准上都击败 GPT-3（175B，300B tokens），同时推理成本低 2.5 倍。

2026 年是 Chinchilla 的世界 —— 带一个重要 twist。Llama 3 8B 训练在 15 trillion token 上，一个 1,875 token  per 参数的比率。Chinchilla-optimal 的 94 倍过去。对于大规模使用的模型，推理成本比训练成本更重要，所以为更小的可部署足迹过度训练（past Chinchilla）是 2026 年默认。

## The Concept（概念）

![Chinchilla curves: loss vs compute at various N/D ratios（Chinchilla 曲线：不同 N/D 比率下的损失 vs 计算）](../assets/scaling-laws.svg)

### Hoffmann law（Hoffmann 定律）

从 Chinchilla 论文，损失遵循：

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = 参数（非嵌入）。
- `D` = 训练 token。
- `α ≈ 0.34`，`β ≈ 0.28`（大致对称）。
- `E ≈ 1.69`，不可约损失天花板。
- `A ≈ 406`，`B ≈ 411`。

两个项在扩展时相互权衡。在固定计算（C = 6ND）下对 `N` 求导并求解：

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

计算最优：每个参数 20 个 token。

### 为什么还要过度训练

Chinchilla-optimal 最小化每训练 FLOP 的训练损失。但你支付训练成本一次；推理成本永远。

对于一个每月服务一万亿 token 的 chatbot，推理占主导总成本。Llama 的方法：训练更小，更长。8B 在 15T token 上是深度推理优化的：

- 适合消费者 GPU。
- 延迟是 70B Chinchilla-optimal 的一小部分。
- 质量对大多数任务来说足够接近。

DeepMind 2024 论文（"Over-training is the new optimal"）形式化了这一点。对于推理主导的工作负载，正确的比率更接近每个参数 100–500 个 token，取决于服务量。

### Emergence vs smoothness（emergence 与平滑）

声称：某些能力（arithmetic、multi-step reasoning、chain-of-thought following）在某个规模上突然“emerges”。

Schaeffer et al. (2023) 认为这是一个测量伪影：emergent metric 使用不连续评分（exact match、at threshold 的 accuracy），隐藏了底层 logits 中的平滑改进。连续 metric（cross-entropy）显示平滑曲线。

2026 年共识是：通过连续损失的预测是可靠的。基准跳跃通常是 scorer artifact。根据连续 metric 计划预算。

### 2026 年图景

Scaling laws 仍然有效，但：

| Factor（因素） | Changed how（改变方式） |
|--------|-------------|
| Data quality（数据质量） | 策划“good” token（Phi-style）把曲线移动 >2× 有效计算 |
| MoE | 总参数与活跃 FLOPs 解耦；per-active-FLOP scaling laws |
| Post-training | 某些能力（instruction following、code）随 SFT+RLHF 的变化比预训练更多 |
| Multimodality（多模态） | Image + text token 一起扩展；每个模态单独的曲线 |
| Synthetic data（合成数据） | 模型生成训练数据；有效计算可以复合 |

Muon 优化器（Kimi Moonlight，2024）显示在匹配数据上 ~2× 有效计算增益 over AdamW。一些 2026 年训练运行默认使用 Muon。改变 scaling law 中的绝对常数，不是它的形状。

```figure
scaling-laws
```

## Build It（动手实现）

见 `code/main.py`。我们实现 Chinchilla 损失方程，并在每个几个计算预算下求解计算最优的 `(N, D)`。

### Step 1: Chinchilla loss

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

在固定 `C = 6ND` 的 `(N, D)` 上绘制 `L` 作为 contour。找到最小值。

### Step 2: compute-optimal frontier（计算最优前沿）

对于从 `1e17` 到 `1e25` FLOPs 的计算预算，找到在 `6ND = C` 约束下最小化损失的 `(N, D)`。验证比率 `D/N ≈ 20`。

### Step 3: over-training cost（过度训练成本）

计算训练一个 10× 更小的模型（1/10 最优 N，10× 最优 D）所支付的额外损失。报告推理 FLOP 节省（与 N 成比例）作为交换。

### Step 4: compare to real models（与真实模型比较）

为 GPT-3、Chinchilla、Llama 3 8B、DeepSeek-V3（active params）drop in 已知的 `(N, D)` 对，并比较预测 vs 报告的损失。

## Use It（实际应用）

你不太可能自己训练一个前沿模型。但 scaling laws 告诉你：

1. **你的微调是否有足够的数据。** 如果你的任务特定数据低于基础模型每个参数 20 个 token，预期会在某个损失 floor 饱和。
2. **是否选择更大的基础模型。** 如果你把所有预算都花在推理上，更喜欢一个更小、更长训练的模型。
3. **回报何时递减。** 超过 1000× Chinchilla-optimal，log-loss 变化成为噪声。

**2026 年研究轨迹：**

- **Data-constrained regime（数据受限制度）。** 网络有有限数量的高质量 token（过滤后 ~5–10 trillion English）。前沿预训练接近这个天花板。合成数据、多语言、多模态和 RLHF-scaled 微调是下一个杠杆。
- **Compute-multiplier tricks（计算乘数技巧）。** Muon 优化器、MoE、更好的数据 curation —— 每个移动绝对常数，不是渐近线。
- **RL 的 Scaling laws。** 开放问题。早期证据表明 RL sample 的幂律，但与预训练非常不同的指数。

## Ship It（交付）

见 `outputs/skill-training-budget-estimator.md`。这个 skill 在给定计算预算、部署约束和目标损失的情况下，为一个新的训练运行选择 `(N, D, hours, GPU)`。

## Exercises（练习）

1. **Easy（简单）。** 运行 `code/main.py`。打印计算预算 `1e20`、`1e22`、`1e24` 的 Chinchilla-optimal `(N, D)`。与真实模型表比较。
2. **Medium（中等）。** 实现 Hoffmann loss-as-function-of-compute 曲线。绘制 loss vs `log10(C)` 对于计算最优前沿。识别当定律预测我们需要 `>10^28` FLOPs 来获得下一个 0.1 cross-entropy 减少时。
3. **Hard（困难）。** 在相同数据集上训练 5 个 tiny 模型（100K 到 10M params）。估计 `α` 和 `E`。你的指数与已发布的有多少匹配？

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Parameters (N)（参数） | "Model size"（模型大小） | 非嵌入权重计数；决定容量。 |
| Tokens (D)（token） | "Training data"（训练数据） | 看到的训练 token 数量；决定参数得到多好利用。 |
| Compute (C)（计算） | "FLOPs spent"（花费的 FLOPs） | 对于标准 transformer 大约 `6 × N × D`。 |
| Chinchilla-optimal（Chinchilla 最优） | "D/N ≈ 20" | 最小化每训练 FLOP 损失的比率。 |
| Over-training（过度训练） | "Past Chinchilla"（超过 Chinchilla） | 花费额外的训练 FLOPs 来节省推理 FLOPs；D/N >> 20。 |
| Irreducible loss（不可约损失） | "The floor"（地板） | scaling law 中的 `E` 项；数据本身的熵。 |
| Emergent capability（涌现能力） | "Sudden jumps at scale"（在规模上突然跳跃） | 通常是一个 scorer artifact；连续损失是平滑的。 |
| Effective compute（有效计算） | "Training-efficiency multiplier"（训练效率乘数） | 更好的数据 / 优化器 / 架构乘以一个 FLOP 能走多远。 |

## Further Reading（延伸阅读）

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — 第一 scaling law 论文；训练不足。
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla。
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — emergence 作为测量伪影。
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — 为什么 Llama 的过度训练对其工作负载是正确的。
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — 2× 计算乘数。
