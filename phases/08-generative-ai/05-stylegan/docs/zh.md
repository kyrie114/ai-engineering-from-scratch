# StyleGAN（StyleGAN）

> 大多数生成器同时将 `z` 搅入每一层。StyleGAN 将其分开：首先将 `z` 映射到一个中间 `w`，然后通过 AdaIN 在每一个分辨率级别*注入* `w`。那一个改变解耦了潜空间，并使照片级真实感人脸成为连续七年的 solved problem。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs)（Phase 8 第 03 课 GAN）, Phase 4 · 08 (Normalization)（Phase 4 第 08 课 归一化）, Phase 3 · 07 (CNNs)（Phase 3 第 07 课 卷积神经网络）
**Time:** ~45 minutes（约 45 分钟）

## The Problem（问题）

一个 DCGAN 通过一堆转置卷积将 `z` 映射到图像。问题：`z` 控制一切 —— 姿态、光照、身份、背景 —— 纠缠在一起。沿着 `z` 的一个轴移动，所有四个都改变。你不能要求模型“同一个人，不同姿态”，因为表示不是以那种方式因式分解的。

Karras et al.（2019，NVIDIA）提议：停止将 `z` 直接送入 conv 层。将一个常数 `4×4×512` 张量作为网络输入。学习一个 8 层 MLP，将 `z ∈ Z → w ∈ W` 映射。通过*自适应实例归一化*（AdaIN）在每一分辨率注入 `w`：规范化每个 conv 特征图，然后通过 `w` 的仿射投影进行缩放和移位。添加每层噪声以控制随机细节（皮肤毛孔、发丝）。

结果：`W` 对于“高级风格”（姿态、身份）与“精细风格”（光照、颜色）有大致正交的轴。你可以通过在低分辨率级别使用图像 A 的 `w` 而在高分辨率级别使用图像 B 的 `w` 来交换两幅图像之间的风格。这解锁了编辑、跨领域风格化和整个“StyleGAN-inversion”研究线。

## The Concept（概念）

![StyleGAN: mapping network + AdaIN + per-layer noise（StyleGAN：映射网络 + AdaIN + 每层噪声）](../assets/stylegan.svg)

**Mapping network（映射网络）。** `f: Z → W`，一个 8 层 MLP。`Z = N(0, I)^512`。`W` 不被强制为高斯分布 —— 它学习一个数据自适应的形状。

**Synthesis network（合成网络）。** 从一个学习的常数 `4×4×512` 开始。每个分辨率块：`upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`。分辨率翻倍：4、8、16、32、64、128、256、512、1024。

**AdaIN。**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

其中 `y_scale` 和 `y_bias` 来自 `w` 的仿射投影。按特征图规范化，然后重新设计。这里的“风格”是特征图的一阶和二阶统计量。

**Per-layer noise（每层噪声）。** 添加到每个特征图的单通道高斯噪声，按每个通道的学习因子缩放。控制随机细节而不影响全局结构。

**Truncation trick（截断技巧）。** 在推理时，采样 `z`，计算 `w = mapping(z)`，然后 `w' = ŵ + ψ·(w - ŵ)`，其中 `ŵ` 是许多样本上 `w` 的均值。`ψ < 1` 以质量换取多样性。几乎每个 StyleGAN 演示都使用 `ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3

| Version（版本） | Year（年份） | Innovation（创新） |
|---------|------|------------|
| StyleGAN | 2019 | Mapping network + AdaIN + noise + progressive growing（映射网络 + AdaIN + 噪声 + 渐进式增长）。 |
| StyleGAN2 | 2020 | Weight demodulation 替换 AdaIN（修复液滴伪影）；skip/residual 架构；路径长度正则化。 |
| StyleGAN3 | 2021 | 无别名卷积 + 等变核；消除纹理粘在像素网格上。 |
| StyleGAN-XL | 2022 | 类条件，1024²，ImageNet。 |
| R3GAN | 2024 | 以更强的正则化重新品牌；在 FFHQ-1024 上以 20 倍更少的参数缩小与扩散的差距。 |

在 2026 年，StyleGAN3 仍然是 (a) 高 FPS 的狭窄领域照片真实感，(b) 少样本域自适应（在一个新数据集上以 100 张图像训练，冻结映射网络），(c) 基于反转的编辑（找到重建真实照片的 `w`，然后编辑该 `w`）的默认选择。对于开放域文本到图像，它不是工具 —— 扩散是。

```figure
gx-stylegan-mapping
```

## Build It（动手实现）

`code/main.py` 在 1-D 中实现了一个玩具“style-GAN lite”：一个映射 MLP，一个合成函数，它接受一个学习的常数向量并用 `w` 派生的缩放/偏置对其进行调制，以及每层噪声。它展示了通过仿射调制注入 `w` 匹配或优于将 `z` 连接到生成器的输入。

### Step 1: mapping network（映射网络）

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### Step 2: adaptive instance normalization（自适应实例归一化）

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

每个特征图的缩放和偏置来自通过线性投影的 `w`。

### Step 3: per-layer noise（每层噪声）

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

每个通道的 sigma 是可学习的。

## Pitfalls（陷阱）

- **Droplet artifacts（液滴伪影）。** StyleGAN 1 在特征图中产生了一个块状液滴，因为 AdaIN 将均值归零。StyleGAN 2 的权重解调通过缩放卷积权重而不是激活来修复它。
- **Texture sticking（纹理粘滞）。** StyleGAN 1 和 2 纹理跟随像素坐标，而不是对象坐标（在插值时可见）。StyleGAN 3 的无别名卷积通过窗口 sinc 滤波器修复了这个问题。
- **Mode coverage（模式覆盖）。** 截断 `ψ < 0.7` 看起来很干净，但从一个狭窄的锥体采样；如果你需要多样性，使用 `ψ = 1.0`。
- **Inversion is lossy（反转是有损的）。** 将真实照片反转到 `W` 通常通过优化或编码器（e4e、ReStyle、HyperStyle）完成。结果在多次迭代中漂移。

## Use It（实际应用）

| Use case（用例） | Approach（方法） |
|----------|----------|
| Photoreal human faces (anime, product, narrow)（照片级真实感人脸（动漫、产品、狭窄）） | StyleGAN3 FFHQ / custom fine-tune |
| Face editing from a photo（从照片编辑人脸） | e4e inversion + StyleSpace / InterFaceGAN directions |
| Face swap / reenactment（人脸交换/重演） | StyleGAN + encoder + blending |
| Avatar pipelines（虚拟形象管道） | StyleGAN3 w/ ADA for low-data fine-tune |
| Domain adaptation from a few images（从几张图像进行域自适应） | Freeze mapping network, fine-tune synthesis |
| Multi-modal or text-conditioned generation（多模态或文本条件生成） | Don't —— use diffusion |

对于答案是“一个人的照片”的产品级演示，StyleGAN 在推理成本（单次前向传递，在 4090 上 <10ms）和相同质量条的锐度上击败扩散。

## Ship It（交付）

保存为 `outputs/skill-stylegan-inversion.md`。该技能接收一张真实照片并输出：反转方法（e4e / ReStyle / HyperStyle）、预期的潜空间损失、编辑预算（你可以在 `W` 中移动多远而不产生伪影），以及已知良好的编辑方向列表（年龄、表情、姿态）。

## Exercises（练习）

1. **Easy（简单）。** 使用 `adain_on=True` 和 `adain_on=False` 运行 `code/main.py`。比较固定潜空间与扰动潜空间的输出分布。
2. **Medium（中等）。** 实现混合正则化：对于一个训练批次，计算 `w_a`，`w_b`，并在合成的第一半应用 `w_a`，在第二半应用 `w_b`。解码器是否学习了解耦的风格？
3. **Hard（困难）。** 获取一个预训练的 StyleGAN3 FFHQ 模型（ffhq-1024.pkl）。通过在标记样本上训练一个 SVM 找到控制“微笑”的 `w` 方向；报告在身份漂移之前你可以推动多远。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Mapping network（映射网络） | "The MLP" | `f: Z → W`，8 层，将潜空间几何与数据统计解耦。 |
| W space（W 空间） | "The style space"（风格空间） | 映射网络的输出；大致解耦。 |
| AdaIN | "Adaptive instance norm"（自适应实例归一化） | 规范化特征图，然后通过 `w` 投影进行缩放 + 偏置。 |
| Truncation trick（截断技巧） | "Psi" | `w = mean + ψ·(w - mean)`，ψ<1 以质量换取多样性。 |
| Path-length regularization（路径长度正则化） | "PL reg" | 惩罚 `w` 每单位变化引起的图像大变化；使 `W` 更平滑。 |
| Weight demodulation（权重解调） | "The StyleGAN2 fix"（StyleGAN2 修复） | 规范化卷积权重而不是激活；杀死液滴伪影。 |
| Alias-free（无别名） | "StyleGAN3's trick"（StyleGAN3 的技巧） | 窗口 sinc 滤波器；消除纹理粘在像素网格上。 |
| Inversion（反转） | "Find w for a real image"（为真实图像找到 w） | 优化或编码 `x → w` 使 `G(w) ≈ x`。 |

## Production note: why StyleGAN still ships in 2026（生产笔记：为什么 StyleGAN 在 2026 年仍然发布）

StyleGAN3 在 4090 上生成一个 1024² FFHQ 人脸不到 10 ms —— `num_steps = 1`，没有 VAE 解码，没有交叉注意力传递。在生产术语中，这是任何图像生成器的下限延迟。一个 50 步 SDXL + VAE 解码管道在相同分辨率下约 3 秒。这是一个**300 倍差距**，对于狭窄领域产品（虚拟形象服务、身份证件管道、库存人脸生成），它在 TCO 上获胜。

两个操作后果：

- **No scheduler, no batcher（无调度器，无批处理器）。** 目标占用的静态批次是最优的。连续批处理（对 LLM 和扩散至关重要）提供零收益，因为每个请求占用相同的 FLOPs。
- **Truncation `ψ` is the safety knob（截断 `ψ` 是安全旋钮）。** `ψ < 0.7` 从映射网络范围的狭窄锥体采样。这是服务层对样本方差的唯一控制杆。在峰值负载时降低 `ψ`，为高级用户提高它。

## Further Reading（延伸阅读）

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) —— StyleGAN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) —— StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) —— StyleGAN3。
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) —— e4e inversion。
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) —— StyleGAN-XL。
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) —— 现代极简 GAN 配方。
