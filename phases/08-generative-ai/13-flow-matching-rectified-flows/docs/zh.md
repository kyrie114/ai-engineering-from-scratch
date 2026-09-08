# Flow Matching & Rectified Flows（流匹配与 Rectified Flows）

> 扩散模型需要 20-50 个采样步骤，因为它们沿着一条弯曲的路径从噪声走到数据。流匹配（Lipman et al.，2023）和 rectified flow（Liu et al.，2022）训练直的路径。更直的路径意味着更少的步骤意味着更快的推理。Stable Diffusion 3、Flux.1 和 AudioCraft 2 都在 2024 年切换到流匹配。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 06 (DDPM)（Phase 8 第 06 课 DDPM）, Phase 1 · Calculus（Phase 1 微积分）
**Time:** ~45 minutes（约 45 分钟）

## The Problem（问题）

DDPM 的反向过程是从 `N(0, I)` 回到数据分布的 1000 步随机游走。DDIM 将其折叠为 20-50 个确定性步骤。你想要更少的步骤 ——  ideally 一个。阻碍因素是反向过程的 ODE 是刚性的；路径是弯曲的。

如果你能以这样的方式训练模型，即从噪声到数据的路径是一条*直线*，那么从 `t=1` 到 `t=0` 的一个单 Euler 步骤就能工作。流匹配直接构建这个：定义从 `x_1 ∼ N(0, I)` 到 `x_0 ∼ data` 的直线插值，训练一个向量场 `v_θ(x, t)` 以匹配其时间导数，在推理时积分。

Rectified flow（Liu 2022）更进一步：用 reflow 程序迭代地直线化路径，该程序产生一个渐进接近线性的 ODE。在两个 reflow 迭代之后，一个 2 步采样器匹配 50 步 DDPM 质量。SDXL-Turbo、SD3-Turbo、LCM 都是从流匹配模型蒸馏的。

## The Concept（概念）

![Flow matching: straight-line interpolation between noise and data（流匹配：噪声和数据之间的直线插值）](../assets/flow-matching.svg)

### Straight-line flow（直线流）

定义：

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

其中 `x_0 ~ data` 和 `x_1 ~ N(0, I)`。沿着这条直线的时间导数是常数：

```
dx_t / dt = x_1 - x_0
```

定义一个神经向量场 `v_θ(x_t, t)` 并训练它匹配这个导数：

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

这是**条件流匹配**损失（Lipman 2023）。训练是无模拟的：你从未展开 ODE。只需采样 `(x_0, x_1, t)` 并回归。

### Sampling（采样）

在推理时，将学习的向量场*向后*随时间积分：

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

从 `x_1 ~ N(0, I)` 开始，Euler 步下到 `t=0`。

### Rectified flow (Liu 2022)

直线流有效，但学习的路径*实际上不是直的* —— 它们弯曲，因为许多 `x_0` 可以映射到同一个 `x_1`。Rectified flow 的 reflow 步骤：

1. 用随机配对训练流模型 v_1。
2. 通过将 v_1 从 `x_1` 积分到其落地 `x_0` 来采样 N 对 `(x_1, x_0)`。
3. 在这些配对示例上训练 v_2。因为配对现在是“ODE 匹配的”，它们之间的直线插值实际上是更平的。
4. 重复。

在实践中，2 个 reflow 迭代让你达到近似线性，启用 2-4 步推理。SDXL-Turbo、SD3-Turbo、LCM 都是从流匹配蒸馏的模型。

### Why this won for images in 2024（为什么这在 2024 年为图像获胜）

三个原因：

1. **Simulation-free training（无模拟训练）** —— 训练期间没有 ODE 展开，容易实现。
2. **Better loss geometry（更好的损失几何）** —— 直的路径有一致的信号到噪声比，而 DDPM ε 损失在调度边缘有较差的 SNR。
3. **Faster inference（更快的推理）** —— 在 SDXL-Turbo 质量下 4-8 步；用一致性蒸馏 1 步。

## Flow matching vs DDPM —— the exact connection（流匹配 vs DDPM —— 确切联系）

具有高斯条件路径的流匹配是具有特定噪声调度的扩散*。选择 `x_t = α(t) x_0 + σ(t) x_1` 调度，流匹配恢复 Stratonovich 重公式化扩散，`v = α'·x_0 - σ'·x_1`。对于高斯路径，两者在代数上是等价的。

流匹配添加了什么：目标的*清晰度*（一个普通速度），一个更干净的损失，以及实验非高斯插值的许可。

```figure
normalizing-flow
```

## Build It（动手实现）

`code/main.py` 在双峰高斯混合上实现了 1-D 流匹配。向量场 `v_θ(x, t)` 是一个用直线目标训练的微型 MLP。在推理时，积分 1、2、4 和 20 Euler 步骤并比较样本质量。

### Step 1: training loss（训练损失）

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### Step 2: multi-step inference（多步推理）

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### Step 3: compare step counts（比较步骤数）

期望 4 步采样器已经匹配 20 步质量 —— 对延迟来说是一个大问题。

## Pitfalls（陷阱）

- **Time parameterization（时间参数化）。** 流匹配使用 `t ∈ [0, 1]`，`t=0` 在数据，`t=1` 在噪声。DDPM 使用 `t ∈ [0, T]`，`t=0` 在数据，`t=T` 在噪声。相同的方向，不同的尺度。论文 constantly 弄错这个。
- **Schedule choice（调度选择）。** Rectified flow 的直线是“the”流匹配调度，但你可以使用余弦或 logit-normal t 采样（SD3 这样做）以获得更好的尺度覆盖。
- **Reflow cost（Reflow 成本）。** 为 reflow 生成配对数据集是每样本一次完整推理传递。只在你真的需要 1-2 步推理时做 reflow。
- **Classifier-free guidance still applies（无分类器引导仍然适用）。** 只需在线性组合中用 v 替换 ε：`v_cfg = (1+w) v_cond - w v_uncond`。

## Use It（实际应用）

| Use case（用例） | 2026 stack（2026 栈） |
|----------|-----------|
| Text-to-image, best quality（文本到图像，最佳质量） | Flow matching: SD3, Flux.1-dev |
| Text-to-image, 1-4 steps（文本到图像，1-4 步） | Distilled flow matching: Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| Real-time inference（实时推理） | Consistency distillation from a flow-matched base (LCM, PCM) |
| Audio generation（音频生成） | Flow matching: Stable Audio 2.5, AudioCraft 2 |
| Video generation（视频生成） | Flow matching mixed with diffusion (Sora, Veo, Stable Video) |
| Science / physics (particle trajectories, molecules)（科学/物理（粒子轨迹、分子）） | Flow matching + equivariant vector field |

每当一个论文在 2025-2026 年说“比扩散更快”时，它几乎总是流匹配 + 蒸馏。

## Ship It（交付）

保存为 `outputs/skill-fm-tuner.md`。该技能接收一个扩散风格模型规范并将其转换为流匹配训练配置：调度选择、时间采样分布（均匀/logit-normal）、优化器、reflow 计划、目标步骤数、评估协议。

## Exercises（练习）

1. **Easy（简单）。** 运行 `code/main.py` 并比较 1 步 vs 20 步 MSE 与真实数据分布。
2. **Medium（中等）。** 从均匀 `t` 采样切换到 logit-normal（将采样集中在中间 t）。模型质量提高了吗？
3. **Hard（困难）。** 实现一个 reflow 迭代：通过积分第一个模型生成配对 `(x_0, x_1)`，在这些对上训练第二个模型，并比较 1 步样本质量。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Flow matching（流匹配） | "Straight-line diffusion"（直线扩散） | 训练 `v_θ(x, t)` 以匹配插值上的 `x_1 - x_0`。 |
| Rectified flow（整流流） | "Reflow"（回流） | 直线化学习流的迭代过程。 |
| Velocity field（速度场） | "v_θ" | 模型的输出 —— 移动 `x_t` 的方向。 |
| Straight-line interpolant（直线插值） | "The path"（路径） | `x_t = (1-t)·x_0 + t·x_1`；平凡的靶导数。 |
| Euler sampler（Euler 采样器） | "1st order ODE solver"（一阶 ODE 求解器） | 最简单的积分器；当路径直时工作良好。 |
| Logit-normal t（Logit-normal t） | "SD3 sampling"（SD3 采样） | 将 `t` 采样集中在梯度最强的中间值。 |
| Consistency distillation（一致性蒸馏） | "1-step sampler"（1 步采样器） | 训练一个学生将任何 `x_t` 直接映射到 `x_0`。 |
| CFG with velocity（带速度的 CFG） | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`；相同的技巧，新的变量。 |

## Production note: Flux.1-schnell is flow matching at its fastest（生产笔记：Flux.1-schnell 是流匹配最快的）

流匹配的生产胜利是 Flux.1-schnell —— 一个流匹配 DiT 蒸馏到 1-4 推理步骤，同时保持 Flux-dev 级质量。Niels 的“在 8GB 机器上运行 Flux”笔记本是参考部署配方：T5 + CLIP 编码，量化 MMDiT 去噪（schnell 4 步 vs dev 50 步），VAE 解码。成本计算：

| Variant（变体） | Steps（步数） | Latency at 1024² on L4（L4 上 1024² 的延迟） | Total FLOPs (relative)（总 FLOPs（相对）） |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (raw)（Flux.1-dev（原始）） | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08× (12× faster)（12 倍更快） |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3 s | 0.03× |

生产规则：**流匹配基础 + 蒸馏 = 2026 年快速文本到图像的默认。** 每个主要供应商都发布这个组合：SD3-Turbo（SD3 + 流 + 蒸馏）、Flux-schnell（Flux-dev + rectified-flow 直线化）、CogView-4-Flash。纯扩散基础仅存在于遗留检查点。

## Further Reading（延伸阅读）

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) —— rectified flow。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) —— 流匹配。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) —— SD3，规模的 rectified flow。
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) —— 涵盖 FM + 扩散的通用框架。
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) —— 扩散/流的 1 步蒸馏。
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) —— turbo 变体。
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) —— 生产中的流匹配。
