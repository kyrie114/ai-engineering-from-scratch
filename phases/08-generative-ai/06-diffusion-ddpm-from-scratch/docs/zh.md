# Diffusion Models — DDPM from Scratch（扩散模型 —— 从零实现 DDPM）

> Ho、Jain、Abbeel（2020）给领域提供了一个无法放弃的配方。在一千个小步骤中用噪声摧毁数据。训练一个神经网络来预测噪声。在推理时反转这个过程。今天每一个主流图像、视频、3D 和音乐模型都运行在这个循环上，可能在其上还有流匹配或一致性技巧。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop)（Phase 3 第 02 课 反向传播）, Phase 8 · 02 (VAE)（Phase 8 第 02 课 VAE）
**Time:** ~75 minutes（约 75 分钟）

## The Problem（问题）

你想要 `p_data(x)` 的一个采样器。GAN 玩一个经常发散的极小极大博弈。VAE 从高斯解码器产生模糊的样本。你真正想要的是一个训练目标，它是 (a) 一个单一稳定的损失（没有鞍点，没有极小极大），(b) `log p(x)` 的一个下界（所以你有似然），以及 (c) 匹配 SOTA 质量的样本。

Sohl-Dickstein et al.（2015）有一个理论答案：定义一个马尔可夫链 `q(x_t | x_{t-1})`，它逐渐添加高斯噪声，并训练一个反向链 `p_θ(x_{t-1} | x_t)` 去噪。Ho、Jain、Abbeel（2020）表明损失可以被简化成一行 —— 预测噪声 —— 并清理了数学。在 2020 年这是一个好奇。在 2021 年它产生了最先进的样本。在 2022 年它成为 Stable Diffusion。在 2026 年它是基质。

## The Concept（概念）

![DDPM: forward noise, reverse denoise（DDPM：前向噪声，反向去噪）](../assets/ddpm.svg)

**Forward process `q`（前向过程 `q`）。** 在 `T` 个小步骤中添加高斯噪声。闭式解 —— 数学可行的原因 —— 是累积步骤也是高斯的：

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)` 对于一个 `β_t` 调度。在 T=1000 步上从 1e-4 到 0.02 线性选择 `β_t`，`x_T` 大约是 `N(0, I)`。

**Reverse process `p_θ`（反向过程 `p_θ`）。** 学习一个神经网络 `ε_θ(x_t, t)`，它预测被添加的噪声。给定 `x_t`，通过以下方式去噪：

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

其中 `σ_t` 是 `sqrt(β_t)` 或一个学习的方差。这个表达式很丑，但它只是代数 —— 给定后验 `q(x_{t-1} | x_t, x_0)` 求解 `x_{t-1}`，并用其噪声预测估计替换 `x_0`。

**Training loss（训练损失）。**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从数据中采样 `x_0`，挑选一个随机 `t`，采样 `ε ~ N(0, I)`，通过闭式解一次性计算噪声 `x_t`，并对噪声回归。一个损失，没有极小极大，没有 KL，没有重参数化技巧。

**Sampling（采样）。** 从 `x_T ~ N(0, I)` 开始。从 `t = T` 到 `1` 迭代反向步骤。完成。

## Why it works（为什么它有效）

三个直觉：

1. **Denoising is easy; generating is hard（去噪容易；生成难）。** 在 `t=T` 时，数据是纯噪声 —— 网络必须解决一个平凡问题。在 `t=0` 时，网络只需要清理几个像素。在中间 `t` 时，问题很难，但网络通过相同的权重从每一个噪声水平获得许多梯度流动。
2. **Score matching in disguise（伪装的分数匹配）。** Vincent（2011）证明预测噪声等价于估计 `∇_x log q(x_t | x_0)`，即*分数*。反向 SDE 使用这个分数沿着密度梯度向上行走 —— 一个向高概率区域引导的随机游走。
3. **The ELBO reduces to simple MSE（ELBO 简化为简单 MSE）。** 完整的变分下界在每个时间步长有一个 KL 项。使用 DDPM 的参数化，这些 KL 项简化为对噪声预测的 MSE，具有特定系数；Ho 丢弃了系数（称之为“simple”损失）并且质量*提高*了。

```figure
diffusion-denoise
```

## Build It（动手实现）

`code/main.py` 实现了一个 1-D DDPM。数据是一个双峰混合。“网络”是一个微型 MLP，它接收 `(x_t, t)` 并输出预测的噪声。训练是单行损失。采样迭代反向链。

### Step 1: the forward schedule (closed form)（前向调度（闭式解））

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### Step 2: sample `x_t` in one shot（一次性采样 `x_t`）

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### Step 3: one training step（一个训练步骤）

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### Step 4: reverse sampling（反向采样）

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

对于一个 1-D 问题，40 个时间步长和一个 24 单元 MLP，这在大约 200 个 epoch 内学习双峰混合。

## Time conditioning（时间条件化）

网络需要知道它正在去噪哪个时间步长。两个标准选项：

- **Sinusoidal embedding（正弦嵌入）。** 像 Transformer 位置编码。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。通过 MLP 传递，广播到网络中。
- **Film / group-norm conditioning（Film / 组归一化条件化）。** 将嵌入投影到每个块的每个通道缩放/偏置（FiLM）。

我们的玩具代码使用正弦 → 连接。生产级 U-Net 使用 FiLM。

## Pitfalls（陷阱）

- **Schedule matters a lot（调度非常重要）。** 线性 `β` 是 DDPM 默认值，但余弦调度（Nichol & Dhariwal，2021）在相同计算下给出更好的 FID。如果质量停滞，切换调度。
- **Timestep embedding is fragile（时间步长嵌入是脆弱的）。** 传递原始 `t` 作为浮点数对于玩具 1-D 有效，但对于图像失败；始终使用适当的嵌入。
- **V-prediction vs ε-prediction（V 预测 vs ε 预测）。** 对于狭窄范围（非常小或非常大的 t），`ε` 有较差的信噪比。V 预测（`v = α·ε - σ·x`）更稳定；SDXL、SD3 和 Flux 使用它。
- **Classifier-free guidance（无分类器引导）。** 在推理时，计算条件和无条件 `ε`，然后 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`，`w ≈ 3-7`。在第 08 课中涵盖。
- **1000 steps is a lot（1000 步很多）。** 生产使用 DDIM（20-50 步）、DPM-Solver（10-20 步）或蒸馏（1-4 步）。见第 12 课。

## Use It（实际应用）

| Role（角色） | Typical stack in 2026（2026 年的典型栈） |
|------|-------|
| Image pixel-space diffusion (small, toy)（图像像素空间扩散（小，玩具）） | DDPM + U-Net |
| Image latent diffusion（图像潜空间扩散） | VAE encoder + U-Net or DiT (Lesson 07)（VAE 编码器 + U-Net 或 DiT（第 07 课）） |
| Video latent diffusion（视频潜空间扩散） | Spatiotemporal DiT (Sora, Veo, WAN)（时空 DiT（Sora、Veo、WAN）） |
| Audio latent diffusion（音频潜空间扩散） | Encodec + diffusion transformer |
| Science (molecules, proteins, physics)（科学（分子、蛋白质、物理）） | Equivariant diffusion (EDM, RFdiffusion, AlphaFold3)（等变扩散（EDM、RFdiffusion、AlphaFold3）） |

扩散是通用生成主干。流匹配（第 13 课）是 2024-2026 年的竞争者，对于相同质量通常在推理速度上获胜。

## Ship It（交付）

保存为 `outputs/skill-diffusion-trainer.md`。该技能接收一个数据集 + 计算预算，并输出：调度（线性/余弦/S 型）、预测目标（ε/v/x）、步数、引导尺度、采样器族和评估协议。

## Exercises（练习）

1. **Easy（简单）。** 将 `code/main.py` 中的 `T` 从 40 更改为 10。样本质量（输出的视觉直方图）如何退化？在什么 `T` 时双峰结构崩溃？
2. **Medium（中等）。** 从 ε 预测切换到 v 预测。重新推导反向步骤。比较最终样本质量。
3. **Hard（困难）。** 添加无分类器引导。以一个类标签 `c ∈ {0, 1}` 为条件，在训练期间 10% 的时间丢弃它，在采样时间使用 `ε = (1+w)·ε_cond - w·ε_uncond`。在 `w = 0, 1, 3, 7` 时测量条件模式命中率。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Forward process（前向过程） | "Adding noise"（添加噪声） | 固定马尔可夫链 `q(x_t \| x_{t-1})`，它摧毁数据。 |
| Reverse process（反向过程） | "Denoising"（去噪） | 学习的链 `p_θ(x_{t-1} \| x_t)`，它重建数据。 |
| β schedule（β 调度） | "The noise ladder"（噪声梯子） | 每步方差；线性、余弦或 S 型。 |
| α̅ | "Alpha bar" | 累积乘积 `∏(1 - β)`；从 `x_0` 给出闭式解 `x_t`。 |
| Simple loss（简单损失） | "MSE on noise"（噪声上的 MSE） | `\|\|ε - ε_θ(x_t, t)\|\|²`；所有变分推导都简化为这个。 |
| ε-prediction（ε 预测） | "Predict noise"（预测噪声） | 输出是添加的噪声；标准 DDPM。 |
| V-prediction（V 预测） | "Predict velocity"（预测速度） | 输出是 `α·ε - σ·x`；跨 t 的更好条件化。 |
| DDPM | "The paper"（论文） | Ho et al. 2020；线性 β，1000 步，U-Net。 |
| DDIM | "Deterministic sampler"（确定性采样器） | 非马尔可夫采样器，20-50 步，相同的训练目标。 |
| Classifier-free guidance（无分类器引导） | "CFG" | 混合条件和无条件噪声预测以放大条件化。 |

## Production note: diffusion inference is a step-count problem（生产笔记：扩散推理是一个步数问题）

DDPM 论文运行 T=1000 个反向步骤。没有人以那个数量发布。每一个真实的推理栈选择三种策略之一 —— 每一种都干净地映射到生产框架中的“延迟来自哪里”：

1. **Faster sampler, same model（更快的采样器，相同的模型）。** DDIM（20-50 步）、DPM-Solver++（10-20）、UniPC（8-16）。反向循环的即插即用替换；训练的 `ε_θ` 权重 untouched。将延迟降低 20-50 倍。
2. **Distillation（蒸馏）。** 训练一个学生在更少的步骤中匹配教师：渐进式蒸馏（2 → 1）、一致性模型（任意 → 1-4）、LCM、SDXL-Turbo、SD3-Turbo。将延迟再降低 5-10 倍，需要重新训练。
3. **Caching and compilation（缓存和编译）。** `torch.compile(unet, mode="reduce-overhead")`、TensorRT-LLM 的扩散后端、`xformers`/SDPA 注意力、bf16 权重。将每步延迟降低约 2 倍。与 (1) 和 (2) 堆叠。

对于一个生产扩散服务器，预算对话与生产文献对 LLM 的描述相同：延迟是 `num_steps × step_cost + VAE_decode`，吞吐量是 `batch_size × (num_steps × step_cost)^-1`。TTFT 很小（一步）；TPOT 等效物是完整响应时间，因为从用户的角度来看图像生成是“一次性”的。

## Further Reading（延伸阅读）

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) —— 扩散论文，超前于时代。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) —— DDPM。
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) —— DDIM，更少的步数。
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) —— 余弦调度，学习的方差。
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) —— 分类器引导。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) —— CFG。
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) —— 统一的符号，最干净的配方。
