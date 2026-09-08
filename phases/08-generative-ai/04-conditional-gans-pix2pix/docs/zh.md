# Conditional GANs & Pix2Pix（条件 GAN 与 Pix2Pix）

> 2014-2017 年的第一个重大突破是控制 GAN 生成什么。附加一个标签，或一张图像，或一个句子。Pix2Pix 完成了图像版本，并且在狭窄的图像到图像任务上，它仍然击败每一个通用文本到图像模型。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs)（Phase 8 第 03 课 GAN）, Phase 4 · 06 (U-Net)（Phase 4 第 06 课 U-Net）, Phase 3 · 07 (CNNs)（Phase 3 第 07 课 卷积神经网络）
**Time:** ~75 minutes（约 75 分钟）

## The Problem（问题）

一个无条件 GAN 采样任意人脸。对演示有用，对生产无用。你想要：*将草图映射到照片*，*将地图映射到航拍照片*，*将白天场景映射到夜晚*，*给灰度图像着色*。在所有这些中，你给定一个输入图像 `x` 并且必须用一些语义对应关系输出 `y`。每个 `x` 有许多合理的 `y`。均方误差将它们压平成糊状。对抗损失不会，因为“看起来真实”是锐利的。

条件 GAN（Mirza & Osindero，2014）添加一个条件 `c` 作为 `G` 和 `D` 的输入。Pix2Pix（Isola et al.，2017）将其专业化：条件是完整的输入图像，生成器是 U-Net，判别器是一个*基于 patch 的*分类器（PatchGAN），损失是对抗 + L1。即使在 2026 年，这个配方在狭窄的图像到图像领域上也优于从头开始的文本到图像模型，因为它是在*配对数据*上训练的 —— 你恰好拥有你需要的信号。

## The Concept（概念）

![Pix2Pix: U-Net generator, PatchGAN discriminator（Pix2Pix：U-Net 生成器，PatchGAN 判别器）](../assets/pix2pix.svg)

**Conditional G（条件 G）。** `G(x, z) → y`。在 Pix2Pix 中，`z` 是 G 内部的 dropout（没有输入噪声 —— Isola 发现显式噪声被忽略了）。

**Conditional D（条件 D）。** `D(x, y) → [0, 1]`。输入是*对*（条件，输出）。这是关键区别：D 必须判断 `y` 是否与 `x` 一致，而不仅仅是 `y` 是否看起来真实。

**U-Net generator（U-Net 生成器）。** 带有跨瓶颈 skip 连接的编码器-解码器。对于输入和输出共享低频结构的任务（边缘、轮廓）至关重要。没有 skips，高频细节会消失。

**PatchGAN discriminator（PatchGAN 判别器）。** 不是输出一个单一的真实/赝品分数，D 输出一个 `N×N` 网格，其中每个单元判断一个约 70×70 像素的感受野。平均。这是一个马尔可夫随机场假设：真实感是局部的。训练更快，参数更少，输出更锐利。

**Loss（损失）。**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 项稳定训练并将 G 推向已知目标。L1 比 L2 给出更锐利的边缘（中位数，不是均值）。`λ = 100` 是 Pix2Pix 的默认值。

## CycleGAN —— 当没有配对时

Pix2Pix 需要配对的 `(x, y)` 数据。CycleGAN（Zhu et al.，2017）以额外的损失为代价放弃了这个要求：*循环一致性*损失。两个生成器 `G: X → Y` 和 `F: Y → X`。训练它们使 `F(G(x)) ≈ x` 和 `G(F(y)) ≈ y`。这让你可以在没有配对示例的情况下将马翻译成斑马，夏天翻译成冬天。

在 2026 年，无配对图像到图像主要通过扩散（ControlNet、IP-Adapter）而不是 CycleGAN 完成，但循环一致性思想几乎存在于每一个无配对域自适应论文中。

```figure
gx-patchgan
```

## Build It（动手实现）

`code/main.py` 在 1-D 数据上实现了一个微型条件 GAN。条件 `c` 是一个类标签（0 或 1）。任务：为给定类生成来自条件分布的样本。

### Step 1: append condition to both G and D inputs（将条件附加到 G 和 D 输入）

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

独热编码是最简单的方式。更大的模型使用学习的嵌入、FiLM 调制或交叉注意力。

### Step 2: train conditional（训练条件模型）

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器必须匹配*给定条件*的真实分布，而不是边缘分布。

### Step 3: verify per-class output（验证每个类的输出）

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## Pitfalls（陷阱）

- **Condition ignored（条件被忽略）。** G 学习边缘化，D 从不惩罚，因为条件信号太弱。修复方法：更激进地对 D 进行条件化（在早期层，而不仅仅是晚期），使用投影判别器（Miyato & Koyama 2018）。
- **L1 weight too low（L1 权重太低）。** G 漂移到任意看起来真实的输出，而不是忠实的输出。对于 Pix2Pix 风格的任务，从 λ≈100 开始。
- **L1 weight too high（L1 权重太高）。** G 产生模糊的输出，因为 L1 仍然是一个 L_p 范数。一旦训练稳定，就斜坡下降。
- **Ground-truth leakage in D（D 中的真实标签泄露）。** 将 `(x, y)` 连接为 D 输入，而不仅仅是 `y`。没有这个 D 无法检查一致性。
- **Mode collapse per class（每个类的模式崩溃）。** 每个类可以独立崩溃。运行类条件多样性检查。

## Use It（实际应用）

2026 年图像到图像任务的状态：

| Task（任务） | Best approach（最佳方法） |
|------|---------------|
| Sketch → photo, same domain, paired data（草图 → 照片，相同领域，配对数据） | Pix2Pix / Pix2PixHD（仍然快速，仍然锐利） |
| Sketch → photo, unpaired（草图 → 照片，无配对） | ControlNet with a Scribble conditioning model |
| Semantic seg → photo（语义分割 → 照片） | SPADE / GauGAN2 or SD + ControlNet-Seg |
| Style transfer（风格迁移） | Diffusion with IP-Adapter or LoRA; GAN methods are legacy |
| Depth → photo（深度 → 照片） | ControlNet-Depth over Stable Diffusion |
| Super-resolution（超分辨率） | Real-ESRGAN (GAN), ESRGAN-Plus, or SD-Upscale (diffusion) |
| Colorization（着色） | ColTran, diffusion-based colorizers, or Pix2Pix-color |
| Daytime → nighttime, seasons, weather（白天 → 夜晚，季节，天气） | CycleGAN or ControlNet-based |

当你 (a) 有数千个配对示例，(b) 任务狭窄且可重复，以及 (c) 你需要快速推理时，Pix2Pix 仍然是正确的工具。在通用开放域任务上，扩散获胜。

## Ship It（交付）

保存为 `outputs/skill-img2img-chooser.md`。该技能接收一个任务描述、数据可用性（配对 vs 无配对，N 个样本）和延迟/质量预算，然后输出：方法（Pix2Pix、CycleGAN、ControlNet 变体、SDXL + IP-Adapter）、训练数据要求、推理成本和评估协议（LPIPS、FID、特定任务）。

## Exercises（练习）

1. **Easy（简单）。** 修改 `code/main.py` 以添加第三个类。确认 G 仍然将每个类的噪声映射到正确的模式。
2. **Medium（中等）。** 在 1-D 设置中用感知式损失替换 L1（例如，一个小型冻结 D 作为特征提取器）。它改变了条件分布的锐度吗？
3. **Hard（困难）。** 在 1-D 设置中草图一个 CycleGAN：两个分布，两个生成器，循环损失。展示它在没有配对数据的情况下学会在它们之间映射。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Conditional GAN（条件 GAN） | "GAN with labels"（带标签的 GAN） | `G(z, c)`，`D(x, c)`。两个网络都看到条件。 |
| Pix2Pix | "Image-to-image GAN"（图像到图像 GAN） | 带有 U-Net G 和 PatchGAN D + L1 损失的配对 cGAN。 |
| U-Net | "Encoder-decoder with skips"（带 skips 的编码器-解码器） | 对称 conv 网络；skips 保留高频。 |
| PatchGAN | "Local-realism classifier"（局部真实感分类器） | D 输出每个 patch 的分数而不是全局分数。 |
| CycleGAN | "Unpaired image translation"（无配对图像翻译） | 两个 G + 循环一致性损失；无配对数据。 |
| SPADE | "GauGAN" | 用语义图规范化中间激活；分割到图像。 |
| FiLM | "Feature-wise linear modulation"（逐特征线性调制） | 来自条件的逐特征仿射变换；廉价条件化。 |

## Production note: Pix2Pix as a latency-bound baseline（生产笔记：Pix2Pix 作为延迟边界基线）

当你有配对数据和一个狭窄任务（草图 → 渲染，语义图 → 照片，白天 → 夜晚）时，Pix2Pix 的单次推理在延迟上比扩散快一个数量级。生产比较通常是：

| Path（路径） | Steps（步数） | Typical latency at 512² on a single L4（单个 L4 上 512² 的典型延迟） |
|------|-------|----------------------------------------|
| Pix2Pix (U-Net forward)（Pix2Pix (U-Net 前向)） | 1 | ~30 ms |
| SD-Inpaint or SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix 在静态批处理中的吞吐量获胜（每个请求是相同的 FLOPs）。扩散在质量和通用性上获胜。现代做法通常是发布一个用于狭窄任务的 Pix2Pix 风格蒸馏模型和一个用于尾部输入的扩散回退。

## Further Reading（延伸阅读）

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) —— cGAN 论文。
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) —— Pix2Pix。
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) —— CycleGAN。
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) —— Pix2PixHD。
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) —— SPADE / GauGAN。
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) —— 投影 D。
