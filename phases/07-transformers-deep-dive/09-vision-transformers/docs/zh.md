# Vision Transformers (ViT)（视觉 Transformer）

> 一张图像是补丁的网格。一个句子是 token 的网格。同一个 transformer 吃掉两者。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer)（完整 Transformer）, Phase 4 · 03 (CNNs)（CNN）, Phase 4 · 14 (Vision Transformers intro)（视觉 Transformer 入门）
**Time:** ~45 minutes（约 45 分钟）

## The Problem（问题）

2020 年之前，计算机视觉意味着卷积。ImageNet、COCO 和检测基准上的每个 SOTA 都使用 CNN backbone。Transformer 是给语言用的。

Dosovitskiy et al. (2020) —— "An Image is Worth 16x16 Words" —— 表明你可以完全去掉卷积。把图像切成固定大小的补丁，线性投影每个补丁到嵌入，把序列喂给 vanilla transformer 编码器。在足够规模（ImageNet-21k 预训练或更大）下，ViT 匹配或击败基于 ResNet 的模型。

ViT 是 2026 年更广泛模式的开始：一个架构，多种模态。Whisper 把音频 token 化。ViT 把图像 token 化。机器人的 action tokens。视频的 pixel tokens。Transformer 不在乎——喂它一个序列，它就会学习。

到 2026 年，ViT 及其后代（DeiT、Swin、DINOv2、ViT-22B、SAM 3）拥有大部分视觉。CNNs 仍然在边缘设备和延迟敏感任务上获胜。其他一切都有 ViT 在栈中的某个地方。

## The Concept（概念）

![Image → patches → tokens → transformer（图像 → 补丁 → token → transformer）](../assets/vit.svg)

### Step 1 — patchify（补丁化）

把一个 `H × W × C` 图像分割成一个 `N × (P·P·C)` 的扁平补丁序列。典型设置：`224 × 224` 图像，`16 × 16` 补丁 → 196 个补丁，每个 768 个值。

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

补丁大小是杠杆。更小的补丁 = 更多 token，更好分辨率，二次注意力成本。更大的补丁 = 更粗，更便宜。

### Step 2 — 线性嵌入

一个单一的学到的矩阵把每个扁平补丁投影到 `d_model`。等价于一个 kernel size 为 `P`、stride 为 `P` 的卷积。在 PyTorch 中这字面上就是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)` —— 一个 2 行实现。

### Step 3 — 前置 `[CLS]` token，加位置嵌入

- 前置一个可学习的 `[CLS]` token。它的最终隐藏状态是用于分类的图像表示。
- 添加可学习的位置嵌入（ViT-original）或正弦 2D（后来的变体）。
- 在 2024+ RoPE 扩展到 2D 用于位置，有时没有显式嵌入。

### Step 4 — 标准 transformer 编码器

堆叠 L 个 `LayerNorm → Self-Attention → + → LayerNorm → MLP → +` 块。和 BERT 相同。没有视觉特定的层。这是论文的教学高潮。

### Step 5 — 头

对于分类：取 `[CLS]` 隐藏状态 → linear → softmax。对于 DINOv2 或 SAM，丢弃 `[CLS]`，直接使用补丁嵌入。

### 重要的变体

| Model（模型） | Year（年份） | Change（变化） |
|-------|------|--------|
| ViT | 2020 | 原始的。固定补丁大小，完整全局注意力。 |
| DeiT | 2021 | 蒸馏；只在 ImageNet-1k 上可训练。 |
| Swin | 2021 | 带移位窗口的分层。固定次二次成本。 |
| DINOv2 | 2023 | 自监督（无标签）。最好的通用视觉特征。 |
| ViT-22B | 2023 | 22B 参数；scaling laws 适用。 |
| SigLIP | 2023 | ViT + 语言对，sigmoid 对比损失。 |
| SAM 3 | 2025 | Segment anything；ViT-Large + promptable mask decoder。 |

### 为什么花了一段时间

ViT 需要*大量*数据才能匹配 CNNs，因为它没有任何 CNN 归纳偏置（translation invariance、locality）。没有 >100M 标签图像或强自监督预训练，CNNs 在匹配计算下仍然获胜。DeiT 在 2021 年用蒸馏技巧修复了这个问题；DINOv2 在 2023 年用自监督永久修复了它。

```figure
n5-patch-stream
```

## Build It（动手实现）

见 `code/main.py`。纯 stdlib patchify + linear embedding + 健全性检查。没有训练——任何 realistic 规模的 ViT 都需要 PyTorch 和数小时的 GPU 时间。

### Step 1: fake image（假图像）

一个 24 × 24 RGB 图像，作为 `(R, G, B)` 元组的行列表。我们使用 6×6 补丁 → 16 个补丁，每个 108-d 嵌入向量。

### Step 2: patchify（补丁化）

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

光栅顺序：网格上的 row-major。每个 ViT 都使用这个顺序。

### Step 3: linear embed（线性嵌入）

将每个扁平补丁乘以一个随机 `(patch_flat_size, d_model)` 矩阵。验证在 prepending `[CLS]` 后输出形状是 `(N_patches + 1, d_model)`。

### Step 4: 计算 realistic ViT 的参数数量

打印 ViT-Base 的参数计数：12 层，12 个头，d=768，patch=16。与 ResNet-50（~25M）比较。ViT-Base 约 86M。ViT-Large ~307M。ViT-Huge ~632M。

## Use It（实际应用）

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation（图像表示）
```

**DINOv2 嵌入是 2026 年图像特征的默认。** 冻结 backbone，训练一个 tiny 头。适用于分类、检索、检测、captioning。Meta 的 DINOv2 checkpoint 在每个非文本视觉任务上都优于 CLIP。

**补丁大小选择。** 小模型使用 16×16（ViT-B/16）。密集预测（segmentation）使用 8×8 或 14×14（SAM、DINOv2）。非常大的模型使用 14×14。

## Ship It（交付）

见 `outputs/skill-vit-configurator.md`。这个 skill 在给定数据集大小、分辨率和计算预算的情况下，为一个新的视觉任务选择 ViT 变体和补丁大小。

## Exercises（练习）

1. **Easy（简单）。** 运行 `code/main.py`。验证补丁数量等于 `(H/P) * (W/P)`，扁平补丁维度等于 `P*P*C`。
2. **Medium（中等）。** 实现 2D 正弦位置嵌入——为每个补丁的 `row` 和 `col` 两个独立正弦码，连接起来。把它们喂入一个 tiny PyTorch ViT，并在 CIFAR-10 上与可学习位置嵌入比较准确率。
3. **Hard（困难）。** 构建一个 3 层 ViT（PyTorch），在 1,000 张 MNIST 图像上用 4×4 补丁训练。测量测试准确率。现在在相同 1,000 张图像上添加 DINOv2 预训练（简化：只是训练编码器从掩码补丁预测补丁嵌入）。准确率提高了吗？

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Patch（补丁） | "The vision-transformer token"（视觉 transformer token） | 图像 `P × P × C` 区域的像素值的扁平向量。 |
| Patchify（补丁化） | "Chop + flatten"（切 + 展平） | 把图像切成不重叠的补丁，把每个展平为向量。 |
| `[CLS]` token | "The image summary"（图像摘要） | 前置的可学习 token；它的最终嵌入是图像表示。 |
| Inductive bias（归纳偏置） | "What the model assumes"（模型假设什么） | ViT 比 CNNs 有更少的先验；需要更多数据来弥补差距。 |
| DINOv2 | "Self-supervised ViT"（自监督 ViT） | 用图像增强 + momentum teacher 无标签训练。2026 年最好的通用图像特征。 |
| SigLIP | "CLIP's successor"（CLIP 的继任者） | ViT + 用 sigmoid 对比损失训练的文本编码器；在匹配计算上优于 CLIP。 |
| Swin | "Windowed ViT"（窗口化 ViT） | 带局部注意力 + 移位窗口的分层 ViT；次二次。 |
| Register tokens（注册 token） | "2023 trick"（2023 技巧） | 几个额外的可学习 token，吸收注意力 sink；改进 DINOv2 特征。 |

## Further Reading（延伸阅读）

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 论文。
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT。
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin。
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2。
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2 的 register-token 修复。
