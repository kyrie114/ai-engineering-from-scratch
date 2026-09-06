# 视觉 Transformer（ViT）（Vision Transformers (ViT)）

> 把图像切成一个个 patch（图像块），把每个 patch 当作一个词，跑一个标准 transformer。回不去了。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 Lesson 02 (Self-Attention), Phase 4 Lesson 04 (Image Classification)
**Time:** ~45 minutes

## 学习目标（Learning Objectives）

- 从零实现 patch 嵌入、可学习的位置嵌入、class token 和 transformer 编码器块，搭出一个最小的 ViT
- 解释为什么人们曾认为 ViT 需要海量预训练数据，直到 DeiT 和 MAE 证明并非如此
- 比较 ViT、Swin 和 ConvNeXt 的架构先验（无先验、局部窗口注意力、卷积骨干）
- 使用 `timm` 和标准的 linear-probe / 微调配方，在小型数据集上微调预训练的 ViT

## 问题所在（The Problem）

十年来，卷积就是计算机视觉的同义词。CNN 拥有强大的归纳偏置（inductive bias）——局部性、平移等变性——没人认为这些能被替代。后来 Dosovitskiy et al. (2020) 证明：把一个普通 transformer 直接用在展平的图像 patch 上，完全不带卷积机制，在大规模数据上也能追平甚至击败最好的 CNN。

代价是“大规模”。ViT 在 ImageNet-1k 上输给 ResNet；先在 ImageNet-21k 或 JFT-300M 上预训练、再在 ImageNet-1k 上微调的 ViT 就赢了。当时的结论是：transformer 缺少有用的先验，但可以从足够多的数据里学出来。后续工作（DeiT、MAE、DINO）表明，只要训练配方对——强数据增强、自监督预训练、蒸馏——ViT 在小数据上也能训得很好。

到 2026 年，纯 CNN 在边缘设备上仍有竞争力（ConvNeXt 是其中最强的），但其余赛道都是 transformer 的天下：分割（Mask2Former、SegFormer）、检测（DETR、RT-DETR）、多模态（CLIP、SigLIP）、视频（VideoMAE、VJEPA）。你该掌握的就是 ViT 的块结构。

## 核心概念（The Concept）

### 流水线（The pipeline）

```mermaid
flowchart LR
    IMG["图像<br/>(3, 224, 224)"] --> PATCH["Patch 嵌入<br/>conv 16x16 s=16<br/>-> (768, 14, 14)"]
    PATCH --> FLAT["展平成<br/>(196, 768) 个词元"]
    FLAT --> CAT["前置<br/>[CLS] token"]
    CAT --> POS["加上可学习的<br/>位置嵌入"]
    POS --> ENC["N 个 transformer<br/>编码器块"]
    ENC --> CLS["取 [CLS]<br/>token 的输出"]
    CLS --> HEAD["MLP 分类器"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a
```

七步：patch -> 词元 -> 注意力 -> 分类器。每个变体（DeiT、Swin、ConvNeXt、MAE 预训练）只改动这七步中的一两步，其余原封不动。

### Patch 嵌入（Patch embedding）

第一个卷积是秘诀。卷积核大小 16、步幅 16，于是 224x224 的图像变成 14x14 的 16x16 patch 网格，每个 patch 被投影成 768 维嵌入。这一个卷积同时完成了切 patch 和线性投影。

```
Input:  (3, 224, 224)
Conv (3 -> 768, k=16, s=16, no padding):
Output: (768, 14, 14)
Flatten spatial: (196, 768)
```

196 个 patch = 196 个词元（token）。每个词元的特征维度是 768（ViT-B）、1024（ViT-L）或 1280（ViT-H）。

### 分类词元（Class token）

一个单独的可学习向量，拼接到序列最前面：

```
tokens = [CLS; patch_1; patch_2; ...; patch_196]   shape (197, 768)
```

经过 N 个 transformer 块之后，`[CLS]` 的输出就是整张图像的全局表示。分类头只读这一个向量。

### 位置嵌入（Positional embedding）

transformer 对空间位置没有内置概念。给每个词元加一个可学习向量：

```
tokens = tokens + learned_pos_embedding   (also shape (197, 768))
```

这个嵌入是模型的一个参数；基于梯度的训练会让它适应 2D 图像结构。正弦 2D 替代方案存在，但实践中很少用。

### Transformer 编码器块（Transformer encoder block）

标准配置：多头自注意力（multi-head self-attention）、MLP、残差连接、pre-LayerNorm。

```
x = x + MSA(LN(x))
x = x + MLP(LN(x))

MLP is two-layer with GELU: Linear(d -> 4d) -> GELU -> Linear(4d -> d)
```

ViT-B/16 堆叠 12 个这样的块，每块 12 个注意力头，总计 86M 参数。

### 为什么用 pre-LN（Why pre-LN）

早期 transformer 用 post-LN（`x = LN(x + sublayer(x))`），没有 warmup 就很难训过 6-8 层。pre-LN（`x = x + sublayer(LN(x))`）不需要 warmup 也能稳定训练更深的网络。每个 ViT、每个现代 LLM 用的都是 pre-LN。

### Patch 大小的取舍（Patch size trade-off）

- 16x16 patch -> 196 个词元，标准配置。
- 32x32 patch -> 49 个词元，更快但分辨率更低。
- 8x8 patch -> 784 个词元，更精细但 O(n^2) 的注意力成本增长得很快。

patch 越大 = 词元越少 = 越快但空间细节越少。SwinV2 在分层窗口里使用 4x4 patch。

### DeiT 在 ImageNet-1k 上训练 ViT 的配方（DeiT's recipe for training ViT on ImageNet-1k）

原版 ViT 需要 JFT-300M 才能击败 CNN。DeiT（Touvron et al., 2020）仅凭 ImageNet-1k 就把 ViT-B 训到了 81.8% top-1，靠四处改动：

1. 重度数据增强：RandAugment、Mixup、CutMix、Random Erasing。
2. 随机深度（stochastic depth）：训练时随机丢掉整块。
3. 重复增强：同一张图像每个 batch 采样 3 次。
4. 从 CNN 教师蒸馏（可选，能进一步提升精度）。

现代每一个 ViT 训练配方都是 DeiT 的后代。

### Swin 与 ConvNeXt（Swin vs ConvNeXt）

- **Swin**（Liu et al., 2021）—— 基于窗口的注意力。每个块只在局部窗口内注意；交替的块会移动窗口，让信息跨窗口混合。在保留注意力算子的同时，把 CNN 式的局部性先验带了回来。
- **ConvNeXt**（Liu et al., 2022）—— 重新设计的 CNN，采用与 Swin 相同的架构选择（深度卷积、LayerNorm、GELU、倒置瓶颈）。它证明差距不在“注意力 vs 卷积”，而在“现代训练配方 + 架构”。

2026 年，ConvNeXt-V2 和 Swin-V2 都是生产级选择；正确选项取决于你的推理栈（ConvNeXt 对边缘设备编译得更友好）和预训练语料。

### MAE 预训练（MAE pretraining）

掩码自编码器（Masked Autoencoder，He et al., 2022）：随机遮住 75% 的 patch，训练编码器只处理可见的 25%，再训练一个小解码器从编码器的输出重建被遮住的 patch。预训练结束后，丢掉解码器，微调编码器。

MAE 让 ViT 仅凭 ImageNet-1k 就能训好、还能打到 SOTA，是当前默认的自监督配方。

```figure
batchnorm-inference
```

## 动手构建（Build It）

### 步骤 1：Patch 嵌入（Step 1: Patch embedding）

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=192, image_size=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
```

一个卷积、一次展平、一次转置。这就是从图像到词元的全部步骤。

### 步骤 2：Transformer 块（Step 2: Transformer block）

Pre-LN、多头自注意力、带 GELU 的 MLP、残差连接。

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
```

`nn.MultiheadAttention` 负责按头拆分、缩放点积和输出投影。`batch_first=True` 让形状为 `(N, seq, dim)`。

### 步骤 3：ViT 本体（Step 3: The ViT）

```python
class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 num_classes=10, dim=192, depth=6, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch = PatchEmbedding(in_channels, patch_size, dim, image_size)
        num_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x[:, 0])
        return self.head(x)

vit = ViT(image_size=64, patch_size=16, num_classes=10, dim=192, depth=6, num_heads=3)
x = torch.randn(2, 3, 64, 64)
print(f"output: {vit(x).shape}")
print(f"params: {sum(p.numel() for p in vit.parameters()):,}")
```

约 2.8M 参数——一个 CPU 就能跑动的小 ViT。真正的 ViT-B 是 86M；同一个类定义，把参数换成 `dim=768, depth=12, num_heads=12`。

### 步骤 4：健全性检查——单张图像推理（Step 4: Sanity check — single image inference）

```python
logits = vit(torch.randn(1, 3, 64, 64))
print(f"logits: {logits}")
print(f"probs:  {logits.softmax(-1)}")
```

应该能无错跑完。概率之和为 1。

## 生产实践（Use It）

`timm` 自带每一个 ViT 变体和 ImageNet 预训练权重。一行代码：

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

2026 年，`timm` 是视觉 transformer 的生产默认。它用同一套 API 支持 ViT、DeiT、Swin、Swin-V2、ConvNeXt、ConvNeXt-V2、MaxViT、MViT、EfficientFormer 以及其他几十个模型。

多模态工作（图像 + 文本）用 `transformers`，里面有 CLIP、SigLIP、BLIP-2、LLaVA。这些模型的图像编码器全都是 ViT 变体。

## 交付产出（Ship It）

本课产出：

- `outputs/prompt-vit-vs-cnn-picker.md` —— 一个提示词，根据数据集规模、算力和推理栈，在 ViT、ConvNeXt 或 Swin 之间做选择。
- `outputs/skill-vit-patch-and-pos-embed-inspector.md` —— 一个技能，校验 ViT 的 patch 嵌入和位置嵌入形状是否与模型期望的序列长度一致，能抓住最常见的移植 bug。

## 练习（Exercises）

1. **（简单）** 打印上面这个小型 ViT 前向传播中每个中间张量的形状。确认：输入 `(N, 3, 64, 64)` -> patch `(N, 16, 192)` -> 加 CLS 后 `(N, 17, 192)` -> 分类器输入 `(N, 192)` -> 输出 `(N, num_classes)`。
2. **（中等）** 在第 4 课的 synthetic-CIFAR 数据集上微调预训练的 `timm` ViT-S/16。与在同样数据上微调 ResNet-18 做比较。报告训练时间和最终精度。
3. **（困难）** 为这个小型 ViT 实现 MAE 预训练：遮住 75% 的 patch，训练编码器 + 一个小解码器来重建被遮住的 patch。在预训练前后分别评估线性探测（linear-probe）精度。

## 关键术语（Key Terms）

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|----------------------|
| Patch 嵌入 | “第一个卷积” | 一个 kernel size = stride = patch size 的卷积；把图像变成词元嵌入网格 |
| Class token | “[CLS]” | 一个拼到词元序列前面的可学习向量；它的最终输出是整张图像的全局表示 |
| 位置嵌入 | “可学习位置” | 加到每个词元上的可学习向量，让 transformer 知道每个 patch 来自哪里 |
| Pre-LN | “子层之前做 LayerNorm” | 稳定的 transformer 变体：`x + sublayer(LN(x))` 而不是 `LN(x + sublayer(x))` |
| 多头注意力 | “并行注意力” | 标准 transformer 注意力拆成 num_heads 个独立子空间，之后再拼接 |
| ViT-B/16 | “Base，patch 16” | 标准尺寸：dim=768、depth=12、heads=12、patch_size=16、image=224；约 86M 参数 |
| DeiT | “数据高效 ViT” | 仅用 ImageNet-1k 配强增强训练的 ViT；证明大规模预训练数据集并非严格必需 |
| MAE | “掩码自编码器” | 自监督预训练：遮住 75% 的 patch 再重建；当前主流的 ViT 预训练配方 |

## 延伸阅读（Further Reading）

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) —— ViT 论文
- [DeiT: Data-efficient Image Transformers (Touvron et al., 2020)](https://arxiv.org/abs/2012.12877) —— 如何仅用 ImageNet-1k 训练 ViT
- [Masked Autoencoders are Scalable Vision Learners (He et al., 2022)](https://arxiv.org/abs/2111.06377) —— MAE 预训练
- [timm 文档](https://huggingface.co/docs/timm) —— 生产中会用到的每个视觉 transformer 的参考
