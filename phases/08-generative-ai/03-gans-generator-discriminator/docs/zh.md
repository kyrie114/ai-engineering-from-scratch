# GANs — Generator vs Discriminator（GAN —— 生成器与判别器）

> Goodfellow 2014 年的技巧是完全跳过密度。两个网络。一个制造赝品。一个抓住它们。它们战斗直到赝品与真实品无法区分。这不应该有效。它经常无效。当它有效时，样本仍然是狭窄领域文献中最锐利的。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop)（Phase 3 第 02 课 反向传播）, Phase 3 · 08 (Optimizers)（Phase 3 第 08 课 优化器）, Phase 8 · 02 (VAE)（Phase 8 第 02 课 VAE）
**Time:** ~75 minutes（约 75 分钟）

## The Problem（问题）

VAE 产生模糊的样本，因为它们的 MSE 解码器损失对于*均值*图像是贝叶斯最优的 —— 而许多合理数字的均值是一个模糊的数字。你想要一个奖励*合理性*的损失，而不是与任何一个目标的逐像素接近。合理性没有闭式解。你必须学会它。

Goodfellow 的想法：训练一个分类器 `D(x)` 来区分真实图像和赝品。训练一个生成器 `G(z)` 来欺骗 `D`。`G` 的损失信号是 `D` 当前认为使某物看起来真实的任何东西。随着 `G` 的改进，这个信号会更新，追逐一个移动的目标。如果两个网络都收敛，`G` 已经学习了数据分布，而从未写下 `log p(x)`。

这就是对抗训练。数学是一个极小极大博弈：

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

在 2026 年，GAN 不再是 SOTA 生成器（扩散和流匹配夺得了这个王冠）。但 StyleGAN 2/3 仍然是发布过的最锐利的人脸模型，GAN 判别器在扩散训练中被用作*感知损失*，对抗训练为快速 1 步蒸馏（SDXL-Turbo、SD3-Turbo、LCM）提供动力，让你可以发布实时扩散。

## The Concept（概念）

![GAN training: generator and discriminator in minimax（GAN 训练：生成器和判别器在极小极大博弈中）](../assets/gan.svg)

**Generator `G(z)`（生成器 `G(z)`）。** 将一个噪声向量 `z ~ N(0, I)` 映射到一个样本 `x̂`。一个解码器形状的网络（密集或转置卷积）。

**Discriminator `D(x)`（判别器 `D(x)`）。** 将一个样本映射到一个标量概率（或分数）。真实 → 1，赝品 → 0。

**Loss（损失）。** 两个交替更新：

- **Train `D`（训练 `D`）：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。真实=1，赝品=0 的二元交叉熵。
- **Train `G`（训练 `G`）：** `loss_G = -log D(G(z))`。这是 Goodfellow 使用的*非饱和*形式（原始的 `log(1 - D(G(z)))` 在 `D` 自信时会饱和并杀死梯度）。

**Training loop（训练循环）。** 一步 `D`，一步 `G`。重复。

**Why it works（为什么它有效）。** 如果 `G` 完美匹配 `p_data`，那么 `D` 不能做得比 chance 更好，并且到处输出 0.5；`G` 不再获得梯度。平衡。

**Why it breaks（为什么它崩溃）。** 模式崩溃（`G` 找到一个 `D` 无法分类的模式并永远铸造它），梯度消失（`D` 学习太快并且 `log D` 饱和），训练不稳定（学习率、批量大小、任何东西）。

## Variants that made GANs work（使 GAN 有效的变体）

| Year（年份） | Innovation（创新） | Fix（修复） |
|------|------------|-----|
| 2015 | DCGAN | Conv/deconv, batch norm, LeakyReLU —— 第一个稳定的架构。 |
| 2017 | WGAN, WGAN-GP | 用 Wasserstein 距离 + 梯度惩罚替换 BCE。修复梯度消失。 |
| 2017 | Spectral normalization | 对判别器进行 Lipschitz 边界约束。在 2026 年的判别器中仍然使用。 |
| 2018 | Progressive GAN | 先训练低分辨率，再添加层。第一个 megapixel 结果。 |
| 2019 | StyleGAN / StyleGAN2 | 映射网络 + 自适应实例归一化。固定领域照片真实感的 SOTA。 |
| 2021 | StyleGAN3 | 无别名、平移等变 —— 2026 年仍然是面部黄金标准。 |
| 2022 | StyleGAN-XL | 有条件的、类感知的、更大规模。 |
| 2024 | R3GAN | 以更强的正则化重新品牌；在 1024² 上无需技巧即可工作。 |

```figure
gan-minimax
```

## Build It（动手实现）

`code/main.py` 在 1-D 数据上训练一个微型 GAN：两个高斯分布的混合。生成器和判别器是单隐藏层 MLP。我们手工实现前向、反向和极小极大循环。目标是看到两个关键失败模式（模式崩溃 + 梯度消失）是如何发生的。

### Step 1: non-saturating loss（非饱和损失）

普通的 Goodfellow 损失 `log(1 - D(G(z)))` 在 D 以高置信度将 G 的赝品分类为赝品时趋于 0。在那个点上，G 的梯度基本上为零 —— G 无法改进。非饱和形式 `-log D(G(z))` 有相反的渐近线：当 D 自信时它爆炸，给 G 一个强信号。

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### Step 2: one discriminator step per generator step（每生成器一步一个判别器步）

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

为 G 提供新鲜赝品，否则梯度是陈旧的。

### Step 3: watch for mode collapse（注意模式崩溃）

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

典型症状：两个真实模式之一停止被生成。判别器停止纠正它，因为它从未被当作赝品看到。

## Pitfalls（陷阱）

- **Discriminator too strong（判别器太强）。** 将 D 的学习率降低 2-5 倍，或添加实例/层噪声。如果 D 达到 >95% 准确率，G 就死了。
- **Generator memorizes a mode（生成器记住一个模式）。** 向 D 输入添加噪声，使用一个 minibatch-discriminator 层，或切换到 WGAN-GP。
- **Batch norm leaking statistics（Batch norm 泄露统计信息）。** 真实批次 + 赝品批次通过同一个 BN 层流动会混合它们的统计信息。改用实例归一化或谱归一化。
- **Inception-score gaming（Inception 分数游戏）。** FID 和 IS 在低样本数量下是嘈杂的。在评估时使用 ≥10k 样本。
- **One-shot sampling is a lie for conditional tasks（单次采样对有条件任务来说是谎言）。** 你仍然需要 CFG 尺度、截断技巧和重新采样才能获得可用的输出。

## Use It（实际应用）

2026 年的 GAN 栈：

| Situation（场景） | Pick（选择） |
|-----------|------|
| Photoreal human faces, fixed pose（照片级真实感人脸，固定姿态） | StyleGAN3（最锐利，最小） |
| Anime / stylized faces（动漫/风格化人脸） | StyleGAN-XL or Stable Diffusion LoRA |
| Image-to-image translation（图像到图像转换） | Pix2Pix / CycleGAN (Phase 8 · 04) or ControlNet (Phase 8 · 08) |
| Fast 1-step text-to-image（快速 1 步文本到图像） | Adversarial distillation of diffusion (SDXL-Turbo, SD3-Turbo) |
| Perceptual loss inside a diffusion trainer（扩散训练器内的感知损失） | Small GAN discriminator on image crops |
| Anything multi-modal, open-ended（任何多模态、开放式的） | Don't —— use diffusion or flow matching |

GAN 是锐利的但是狭窄的。一旦你的领域开放 —— 照片、任意文本提示、视频 —— 切换到扩散。对抗技巧作为组件（感知损失、蒸馏）继续存在，而不是作为一个独立的生成器。

## Ship It（交付）

保存为 `outputs/skill-gan-debugger.md`。该技能接收一个失败的 GAN 运行（损失曲线、样本网格、数据集大小）并输出一个可能原因的排名列表、单行修复和一个重新运行协议。

## Exercises（练习）

1. **Easy（简单）。** 使用库存设置运行 `code/main.py`。然后设置 `D_LR = 5 * G_LR` 并重新运行。G 的损失崩溃到一个常数的速度有多快？
2. **Medium（中等）。** 用 WGAN 损失替换 Goodfellow BCE 损失：`loss_D = E[D(fake)] - E[D(real)]`，`loss_G = -E[D(fake)]`，并将 D 的权重裁剪到 `[-0.01, 0.01]`。训练更稳定吗？比较挂钟收敛。
3. **Hard（困难）。** 将 1-D 示例扩展到 2-D 数据（环上的 8 个高斯分布的混合）。在步骤 1k、5k、10k 时跟踪生成器捕获了多少个 8 个模式。实现 minibatch discrimination 并重新测量。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Generator（生成器） | "G" | 噪声到样本的网络，`G: z → x̂`。 |
| Discriminator（判别器） | "D" | 分类器 `D: x → [0, 1]`，真实 vs 赝品。 |
| Minimax（极小极大） | "The game"（游戏） | 联合目标的 `min_G max_D`。 |
| Non-saturating loss（非饱和损失） | "The fix"（修复） | 对 G 使用 `-log D(G(z))` 而不是 `log(1 - D(G(z)))`。 |
| Mode collapse（模式崩溃） | "G memorized one thing"（G 记住了一件事） | 尽管数据多样，生成器产生很少 distinct 的输出。 |
| WGAN | "Wasserstein" | 用 Earth-Mover 距离 + 梯度惩罚替换 BCE；更平滑的梯度。 |
| Spectral norm（谱范数） | "Lipschitz trick"（Lipschitz 技巧） | 约束 D 的权重范数以边界其斜率；稳定训练。 |
| StyleGAN | "The one that works"（有效的那个） | 映射网络 + AdaIN；人脸领域的最佳，2026 年仍然如此。 |

## Production note: one-shot inference is GAN's lasting advantage（生产笔记：单次推理是 GAN 持久的优势）

GAN 不再在开放域生成的样本质量上获胜，但它们在推理成本上仍然获胜。在生产推理文献词汇中，GAN 有：

- **No prefill, no decode stages（无 prefill，无解码阶段）。** 单个 `G(z)` 前向传递。TTFT ≈ 总延迟。
- **No KV-cache pressure（无 KV-cache 压力）。** 唯一的状态是权重。批量大小受激活内存限制，而不是缓存。
- **Trivial continuous batching（简单的连续批处理）。** 由于每个请求占用相同的固定 FLOPs，服务器目标占用的静态批次通常是最优的。不需要飞行中调度器。

这就是为什么 GAN 蒸馏（SDXL-Turbo、SD3-Turbo、ADD、LCM）是 2026 年快速文本到图像的主导技术：它将一个 20-50 步扩散管道压缩成 1-4 个 GAN 风格的前向传递，同时保持扩散基的分布。对抗损失作为训练时旋钮继续存在，用于将慢速生成器转变为快速生成器。

## Further Reading（延伸阅读）

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) —— 原始 GAN 论文。
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) —— 第一个稳定的架构。
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) —— WGAN。
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) —— SN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) —— StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) —— StyleGAN3。
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) —— SDXL-Turbo。
