# Autoencoders & Variational Autoencoders (VAE)（自编码器与变分自编码器 (VAE)）

> 一个普通自编码器压缩然后重建。它记住了。它不生成。加一个技巧 —— 迫使代码看起来像高斯分布 —— 你就得到了一个采样器。那一个技巧，`z = μ + σ·ε` 的重参数化，就是为什么你在 2026 年使用的每一个潜空间扩散和流匹配图像模型在输入端都有一个 VAE。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop)（Phase 3 第 02 课 反向传播）, Phase 3 · 07 (CNNs)（Phase 3 第 07 课 卷积神经网络）, Phase 8 · 01 (Taxonomy)（Phase 8 第 01 课 分类）
**Time:** ~75 minutes（约 75 分钟）

## The Problem（问题）

把一个 784 像素的 MNIST 数字压缩成一个 16 个数的代码，然后重建。一个普通自编码器会在重建 MSE 上表现出色，但代码空间是一个凹凸不平的混乱。在代码空间里挑一个随机点，解码它，你会得到噪声。它没有采样器。它是一个穿着马甲的压缩模型。

你真正想要的是：(a) 代码空间是一个干净、平滑的分布，你可以从中采样 —— 比如各向同性高斯分布 `N(0, I)`，(b) 解码任何样本都能产生一个合理的数字，以及 (c) 编码器和解码器仍然能很好地压缩。三个目标，一个架构，一个损失函数。

Kingma 2013 年的 VAE 通过训练编码器输出一个*分布* `q(z|x) = N(μ(x), σ(x)²)` 来解决这个问题，通过 KL 惩罚将该分布拉向先验分布 `N(0, I)`，然后在解码之前从 `q(z|x)` 中采样 `z`。在推理时，丢弃编码器，采样 `z ~ N(0, I)`，解码。KL 惩罚是迫使代码空间结构化的原因。

在 2026 年，VAE 很少独立部署 —— 它们在原始图像质量方面已经被扩散模型超越 —— 但它们是每个潜空间扩散模型（SD 1/2/XL/3、Flux、AudioCraft）的首选编码器。学会 VAE，你就学会了你使用的每一个图像管道中看不见的第一层。

## The Concept（概念）

![Autoencoder vs VAE: the reparameterization trick（自编码器与 VAE：重参数化技巧）](../assets/vae.svg)

**Autoencoder（自编码器）。** `z = encoder(x)`，`x̂ = decoder(z)`，损失 = `||x - x̂||²`。代码空间无结构。

**VAE encoder（VAE 编码器）。** 输出两个向量：`μ(x)` 和 `log σ²(x)`。这些定义了 `q(z|x) = N(μ, diag(σ²))`。

**Reparameterization trick（重参数化技巧）。** 从 `q(z|x)` 中采样是不可微的。将样本重写为 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。现在 `z` 是 `(μ, σ)` 加一个非参数噪声的确定性函数 —— 梯度流经 `μ` 和 `σ`。

**Loss（损失）。** 证据下界 (Evidence Lower BOund, ELBO)，两项：

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重建推动 `x̂` 接近 `x`。KL 推动 `q(z|x)` 接近先验。它们进行权衡。小的 β (<1) = 更锐利的样本，代码空间不那么高斯。大的 β (>1) = 更干净的代码空间，更模糊的样本。β-VAE（Higgins 2017）使这个旋钮闻名，并开启了解耦研究。

**Sampling（采样）。** 在推理时：抽取 `z ~ N(0, I)`，前向通过解码器。一次前向传递 —— 不像扩散那样需要迭代采样。

```figure
vae-latent-grid
```

## Build It（动手实现）

`code/main.py` 实现了一个没有 numpy 或 torch 的小型 VAE。输入是从 8 维空间中的两个分量高斯混合分布中抽取的 8 维合成数据。编码器和解码器是单隐藏层 MLP。我们实现了 tanh 激活、前向传递、损失和一个手写的反向传递。不是生产级 —— 是教学用的。

### Step 1: encoder forward（编码器前向传递）

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

使用 `log σ²` 而不是 `σ`，这样网络输出是无约束的（σ 的 softplus 是一个陷阱 —— 梯度在 σ ≈ 0 时死亡）。

### Step 2: reparameterize and decode（重参数化与解码）

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### Step 3: the ELBO（ELBO）

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

因为两个分布都是高斯分布，所以有精确的闭式 KL。不要数值积分。人们在 2026 年仍然会提交带有蒙特卡洛 KL 估计的代码 —— 它慢 3 倍，没有任何理由。

### Step 4: generate（生成）

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型。五行代码。

## Pitfalls（陷阱）

- **Posterior collapse（后验崩溃）。** KL 项过于激进地将 `q(z|x)` 驱动到 `N(0, I)`，以至于 `z` 不携带关于 `x` 的任何信息。修复方法：β-annealing（从 β=0 开始，斜坡到 1）、free bits，或者在非活跃维度上跳过 KL。
- **Blurry samples（模糊样本）。** 高斯解码器似然隐含 MSE 重建，这对于 L2（均值）是贝叶斯最优的 —— 一组合理数字的均值是一个模糊的数字。修复方法：离散解码器（VQ-VAE、NVAE），或者仅将 VAE 用作编码器，并将扩散模型堆叠在潜空间上（这就是 Stable Diffusion 所做的）。
- **β too large, too early（β 太大，太早）。** 见后验崩溃。从 β≈0.01 开始并斜坡上升。
- **Latent dim too small（潜空间维度太小）。** MNIST 用 16-D，ImageNet 256² 用 256-D，ImageNet 1024² 用 2048-D。Stable Diffusion 的 VAE 将 512×512×3 压缩为 64×64×4（空间面积 32x 下采样，通道 32x）。

## Use It（实际应用）

2026 年的 VAE 栈：

| Situation（场景） | Pick（选择） |
|-----------|------|
| Image-latent encoder for diffusion（扩散模型的图像潜空间编码器） | Stable Diffusion VAE (`sd-vae-ft-ema`) or Flux VAE |
| Audio-latent encoder（音频潜空间编码器） | Encodec (Meta), SoundStream, or DAC (Descript) |
| Video latents（视频潜空间） | Sora's spatiotemporal patches, Latte VAE, WAN VAE |
| Disentangled representation learning（解耦表示学习） | β-VAE, FactorVAE, TCVAE |
| Discrete latents (for transformer modelling)（离散潜空间（用于 Transformer 建模）） | VQ-VAE, RVQ (ResidualVQ) |
| Continuous latents for generation（用于生成的连续潜空间） | Plain VAE, then condition a flow/diffusion model in that latent space |

潜空间扩散模型是在编码器和解码器之间有一个扩散模型的 VAE。VAE 做粗压缩，扩散模型做繁重的工作。视频（VAE + 视频扩散 DiT）和音频（Encodec + MusicGen transformer）采用相同的模式。

## Ship It（交付）

保存为 `outputs/skill-vae-trainer.md`。

该技能接收：数据集概况 + 潜空间维度目标 + 下游用途（重建、采样或潜空间扩散输入），并输出：架构选择（plain/β/VQ/RVQ）、β 调度、潜空间维度、解码器似然（高斯 vs 分类）和评估计划（重建 MSE、每维 KL、`q(z|x)` 与 `N(0, I)` 之间的 Fréchet 距离）。

## Exercises（练习）

1. **Easy（简单）。** 将 `code/main.py` 中的 `β` 更改为 `0.01`、`0.1`、`1.0`、`5.0`。记录最终重建 MSE 和 KL。对于你的合成数据，哪个 β 是 Pareto 最优的？
2. **Medium（中等）。** 用 Bernoulli 似然（交叉熵损失）替换高斯解码器似然。在相同合成数据的二值化版本上比较样本质量。
3. **Hard（困难）。** 将 `code/main.py` 扩展为一个小型 VQ-VAE：用 K=32 个条目的码本中的最近邻查找替换连续 `z`。比较重建 MSE，并报告使用了多少个码本条目（码本崩溃是真实存在的）。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Autoencoder（自编码器） | Encode-decode network（编码-解码网络） | `x → z → x̂`，学习 MSE。不是生成式的。 |
| VAE | AE with a sampler（带采样器的自编码器） | 编码器输出一个分布，KL 惩罚塑造代码空间。 |
| ELBO | Evidence lower bound（证据下界） | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`；当 `q = p(z\|x)` 时紧密。 |
| Reparameterization（重参数化） | `z = μ + σ·ε` | 将随机节点重写为确定性 + 纯噪声。使通过采样的反向传播成为可能。 |
| Prior（先验） | `p(z)` | 潜空间的目标分布，通常为 `N(0, I)`。 |
| Posterior collapse（后验崩溃） | "KL term wins"（KL 项获胜） | 编码器忽略 `x`，输出先验；解码器必须幻觉。 |
| β-VAE | Tunable KL weight（可调 KL 权重） | `loss = recon + β·KL`。更高的 β = 更解耦但更模糊。 |
| VQ-VAE | Discrete latent（离散潜空间） | 用最近码本向量替换连续 `z`；支持 Transformer 建模。 |

## Production note: the VAE is the hottest path in a diffusion server（生产笔记：VAE 是扩散服务器中最热的路径）

在 Stable Diffusion / Flux / SD3 管道中，VAE 在每个请求中被调用两次 —— 一次编码（如果执行 img2img / inpainting）和一次解码。在 1024² 时，解码器传递通常是整个管道中最大的激活内存峰值，因为它将 `128×128×16` 的潜空间上采样回 `1024×1024×3`。两个实际后果：

- **Slice or tile the decode（切片或分块解码）。** `diffusers` 暴露了 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。分块用一个小的接缝伪影交换 `O(tile²)` 内存而不是 `O(H·W)`。对于消费者 GPU 上的 1024²+ 是必不可少的。
- **bf16 decoder, fp32 numerics for the final resize（bf16 解码器，最终调整大小的 fp32 数值）。** SD 1.x VAE 是以 fp32 发布的，并且在 1024²+ 时被转换为 fp16 会*静默地产生 NaN*。SDXL 提供 `madebyollin/sdxl-vae-fp16-fix` —— 始终优先使用 fp16-fix 变体或使用 bf16。

## Further Reading（延伸阅读）

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) —— VAE 论文。
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) —— 解耦 β-VAE。
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) —— VQ-VAE。
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) —— 最先进的图像 VAE。
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) —— Stable Diffusion；VAE 作为编码器。
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) —— Encodec，音频 VAE 标准。
