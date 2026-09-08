# Visual Autoregressive Modeling (VAR): Next-Scale Prediction（视觉自回归建模 (VAR)：下一尺度预测）

> 扩散模型在时间上迭代采样（去噪步骤）。VAR 在尺度上迭代采样 —— 它预测一个 1x1 令牌，然后 2x2，然后 4x4，直到最终分辨率，每个尺度以之前的为条件。2024 年的论文表明 VAR 匹配图像生成的 GPT 风格缩放定律，并在相同计算预算下击败 DiT。这节课构建核心机制。

**Type:** Build
**Languages:** Python (with PyTorch)（Python（带 PyTorch））
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention)（Phase 7 第 03 课 多头注意力）, Phase 8 Lesson 06 (DDPM)（Phase 8 第 06 课 DDPM）
**Time:** ~90 minutes（约 90 分钟）

## The Problem（问题）

自回归生成在语言建模中占主导地位，因为它可预测地扩展：更多计算，更多参数，更低的困惑度，更好的输出。图像生成在 2024 年之前有两个主要的 AR 尝试：PixelRNN/PixelCNN（逐像素）和 DALL-E 1 / Parti / MuseGAN（在 VQ-VAE 代码上逐令牌）。

两者都 suffer from a generation-order problem（都 suffer from 生成顺序问题）。像素和令牌排列在 2D 网格中，但 AR 模型必须以 1D 光栅顺序访问它们。一个早期的角像素不知道图像最终会变成什么。生成质量比 GPT-on-text 扩展得更差，在匹配计算下从未达到扩散模型质量。

VAR 通过改变生成的内容来修复生成顺序问题。VAR 不是在一个空间中逐个预测图像令牌，而是以增加的分辨率预测整个图像。步骤 1：预测一个 1x1 令牌（整体图像“摘要”）。步骤 2：预测一个 2x2 令牌网格（更粗糙的特征）。步骤 3：预测一个 4x4 网格。步骤 K：预测最终的 `(H/8)x(W/8)` 网格。

每个尺度关注所有之前的尺度（在“尺度顺序”上因果），并在其自己的尺度内并行。顺序问题消失了：尺度 k 的整个图像在一次 Transformer 传递中产生。

## The Concept（概念）

### VQ-VAE Multi-Scale Tokenizer（VQ-VAE 多尺度分词器）

VAR 需要一个**多尺度离散分词器**。对于一个图像 x，它产生一个渐进更高分辨率令牌网格的序列：

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

每个 `z_k` 使用相同的码本（典型大小 4096-16384）。每个尺度的分词不是独立的 —— 它被训练使得在每个尺度上对残差求和重建 f：

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是一个**残差 VQ** 变体。尺度 k 捕获尺度 1..k-1 遗漏的内容。解码器取所有尺度嵌入的和并产生图像。

多尺度 VQ 分词器训练一次（像 VQGAN）然后冻结。所有生成工作由顶部的自回归模型完成。

### Next-Scale Prediction（下一尺度预测）

生成模型是一个 Transformer，它看到所有之前尺度的令牌并预测下一个尺度的令牌。

输入序列结构：
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

位置嵌入编码尺度索引和尺度内的空间位置。注意力在尺度顺序上是因果的：尺度 k 的令牌，位置 `(i, j)` 可以关注尺度 1..k 的所有令牌，并关注尺度 k 本身的令牌，这些令牌在任何使用的 intra-scale 顺序中更早出现（VAR 使用固定位置注意力，没有 intra-scale 因果 —— 一个尺度内的所有位置并行预测）。

训练损失：在每个尺度 k，给定所有先前尺度令牌预测令牌 `z_k`。离散 VQ 代码上的交叉熵损失。与 GPT 相同的结构，除了“序列”现在是尺度结构的。

### Generation（生成）

在推理时：
```
generate z_1 = sample from p(z_1)                    # 1 token（1 个令牌）
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel（4 个令牌并行）
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel（16 个令牌并行）
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

对于 K = 10 个尺度，生成是 10 个 Transformer 前向传递。每次传递在其整个尺度内并行产生 —— 没有尺度内的逐令牌自回归。对于一个 256x256 图像，这大约是 10 次传递 vs DiT 的 28-50 次。

### Why Next-Scale Wins Over Next-Token（为什么下一尺度胜过下一令牌）

三个结构胜利：
1. **Coarse-to-fine aligns with natural image statistics（由粗到细与自然图像统计对齐）。** 人类视觉感知和图像数据集都表现出尺度相关的规律性：低频结构是稳定和可预测的；高频细节依赖于低频内容。下一尺度预测利用这一点。
2. **Parallel generation within scale（尺度内并行生成）。** 不像 GPT 风格令牌 AR，VAR 在一个步骤中产生一个尺度的所有令牌。有效生成长度是对数的而不是线性的。
3. **No generation order bias（无生成顺序偏差）。** 尺度 k 的令牌看到尺度 k-1 的全部；没有“左边”或“上面”偏差迫使早期令牌在晚期上下文可用之前提交。

### Scaling Law（缩放定律）

Tian et al. 证明 VAR 对 ImageNet 上的 FID 遵循幂律缩放曲线 —— 就像 GPT 对困惑度一样。加倍参数或计算可靠地减半错误。这是第一个图像生成模型以与语言模型一样干净的方式展示这种缩放行为。结果是 VAR 缩放预测变得从计算可预测，而不是每个架构的经验猜测。

### Relationship to Diffusion（与扩散的关系）

VAR 和扩散共享相同的数据压缩故事：两者都将生成问题分解为一系列更容易的子问题。

- Diffusion（扩散）：逐渐添加噪声，学习撤销一步。
- VAR：逐渐添加分辨率，学习预测下一个尺度。

它们是问题上的不同轴。两者都产生可行的条件分布。经验上 VAR 在推理时更快（更少的传递，尺度内全部并行），并在相同计算下匹配或击败 DiT 的类条件 ImageNet。文本条件 VAR（VARclip、HART）是一个活跃的研究方向。

```figure
gx-var-next-scale
```

## Build It（动手实现）

在 `code/main.py` 中，你将：
1. 在合成“图像”数据（2D Gaussian rings）上构建一个微型**多尺度 VQ 分词器**。
2. 训练一个**VAR 风格 Transformer** 来下一尺度预测令牌。
3. 通过调用 Transformer 4 次（4 个尺度）和解码来采样。
4. 验证尺度有序训练使生成在尺度内并行。

这是一个玩具实现。要点是看到尺度结构注意力掩码和尺度内并行生成实际工作。

## Ship It（交付）

这节课产生 `outputs/skill-var-tokenizer-designer.md` —— 一个用于设计多尺度分词器的技能：尺度数量、尺度比率、码本大小、残差共享、解码器架构。

## Exercises（练习）

1. **Scale count ablation（尺度数量消融）。** 用 4、6、8、10 个尺度训练 VAR。测量重建质量 vs 自回归传递数量。更多尺度 = 更细的残差 = 更好的质量但更多传递。
2. **Codebook size（码本大小）。** 用码本大小 512、4096、16384 训练分词器。更大的码本给出更好的重建但更难预测。找到膝盖。
3. **Parallel-within-scale check（尺度内并行检查）。** 对于一个训练好的 VAR，显式测量注意力模式。在尺度 k 内，模型是否关注跨尺度位置但不关注 intra-scale？验证掩码实现。
4. **VAR vs DiT scaling（VAR vs DiT 缩放）。** 对于相同的 ImageNet 类条件任务，在匹配参数预算（例如，33M、130M、458M）下训练 VAR 和 DiT。绘制 FID vs 计算。VAR 应该在每个尺寸上领先 DiT —— 在小规模上重现论文结果。
5. **Text conditioning（文本条件化）。** 将 VAR 扩展为通过 adaLN 接受文本嵌入（CLIP pooled）作为额外条件输入。这是 HART 配方。在文本对齐采样上 FID 提高了多少？

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| VAR | "Visual AutoRegressive"（视觉自回归） | 通过 VQ 令牌金字塔的下一尺度预测进行图像生成 |
| Next-scale prediction（下一尺度预测） | "Predict coarser, then finer"（预测更粗糙，然后更精细） | 模型预测令牌在增加分辨率尺度，以所有先前尺度为条件 |
| Multi-scale VQ tokenizer（多尺度 VQ 分词器） | "Residual VQ"（残差 VQ） | 产生 K 个增加分辨率令牌网格的 VQ-VAE，解码器求和所有尺度 |
| Scale k（尺度 k） | "Pyramid level k"（金字塔级别 k） | K 个分辨率级别之一，从 k=1 的 1x1 到 k=K 的 `(H/p)x(W/p)` |
| Parallel-within-scale（尺度内并行） | "One forward per scale"（每个尺度一次前向） | 尺度 k 的所有令牌在一次 Transformer 传递中预测，不是自回归地 |
| Causal-across-scales（跨尺度因果） | "Scale-ordered attention"（尺度有序注意力） | 尺度 k 的令牌可以关注尺度 1..k 的所有令牌，但不能关注尺度 k+1..K |
| Residual VQ（残差 VQ） | "Additive tokenization"（加性分词） | 每个尺度的令牌编码较低尺度留下的残差；解码器求和所有尺度嵌入 |
| VAR scaling law（VAR 缩放定律） | "Image GPT scaling"（图像 GPT 缩放） | FID 在计算中遵循可预测的幂律，像语言模型的困惑度 |
| HART | "Hybrid VAR + text"（混合 VAR + 文本） | 结合 MaskGIT 风格迭代解码与 VAR 尺度结构的文本条件 VAR 变体 |
| Scale position embedding（尺度位置嵌入） | "(scale, row, col) triple"（（尺度、行、列）三元组） | 位置编码携带尺度索引和尺度内的空间坐标 |

## Further Reading（延伸阅读）

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) —— VAR 论文，规范参考
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) —— DiT，扩散比较基线
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) —— VQGAN，VAR 多尺度分词器扩展的令牌化器家族
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) —— VQ-VAE，离散图像分词的基础
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) —— 文本条件 VAR
