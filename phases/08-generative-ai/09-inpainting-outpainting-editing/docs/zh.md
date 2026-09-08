# Inpainting, Outpainting & Image Editing（图像修复、外绘与编辑）

> 文本到图像制造新东西。图像修复修复旧东西。在生产中，70% 的可计费图像工作是编辑 —— 交换背景、移除标志、扩展画布、重新生成一只手。图像修复是扩散证明其价值的地方。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion)（Phase 8 第 07 课 潜空间扩散）, Phase 8 · 08 (ControlNet & LoRA)（Phase 8 第 08 课 ControlNet 与 LoRA）
**Time:** ~75 minutes（约 75 分钟）

## The Problem（问题）

一个客户发来一张完美的产品照片，背景中有一个分散注意力的标志。你想擦除标志并让其他所有内容保持像素级相同。你不能从零开始运行文本到图像 —— 结果将有不同的颜色、不同的光照、不同的产品角度。你想*仅*重新生成掩码区域内的内容，并且你希望重新生成尊重周围的上下文。

那就是图像修复。变体：

- **Inpainting（图像修复）。** 在掩码内重新生成，保留外部像素。
- **Outpainting（外绘）。** 在掩码外重新生成（或超出画布），保留内部。
- **Image editing（图像编辑）。** 重新生成整个图像但保持与原始图像的语义或结构保真度（SDEdit、InstructPix2Pix）。

2026 年每一个扩散管道都发布了一个图像修复模式。Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它们都基于相同的原理工作。

## The Concept（概念）

![Inpainting: mask-aware denoising with context-preserving reinjection（图像修复：具有上下文保持重注入的掩码感知去噪）](../assets/inpainting.svg)

### The naive approach (and why it's wrong)（朴素方法（以及为什么它是错的））

用掩码运行标准文本到图像。在每个采样步骤中，用前向扩散干净图像替换噪声潜空间的未掩码区域。它工作……效果不好。边界伪影渗透进来，因为模型没有关于掩码区域内有什么的信息。

### The proper inpainting model（正确的图像修复模型）

训练一个修改后的 U-Net，它接受 9 个输入通道而不是 4 个：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外的通道是 VAE 编码源图像的副本加上一个单通道掩码。在训练时，你随机掩码图像的区域并训练模型仅对掩码区域去噪，同时未掩码区域作为干净条件信号给出。在推理时，模型可以“看到”掩码区域周围的内容并产生连贯的补全。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 都使用这个 9 通道（或类似）输入。Diffusers `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit (Meng et al., 2022) —— 自由编辑

将噪声添加到源图像直到某个中间 `t`，然后用新提示从 `t` 向下到 0 运行反向链。无需重新训练。起始 `t` 的选择权衡保真度与创作自由度：

- `t/T = 0.3` → 几乎与源相同，小的风格变化
- `t/T = 0.6` → 适度编辑，保留粗结构
- `t/T = 0.9` → 从近噪声生成，最小的源保留

### InstructPix2Pix (Brooks et al., 2023)

在 `(input_image, instruction, output_image)` 三元组上微调扩散模型。在推理时，以输入图像和文本指令（“让它日落”，“添加一条龙”）为条件。两个 CFG 尺度：图像尺度和文本尺度。

### RePaint (Lugmayr et al., 2022)

保留一个标准无条件扩散模型。在每次反向步骤中，重新采样 —— 偶尔跳回到更噪声的状态并重新去噪。避免边界伪影。当你没有训练好的图像修复模型时使用。

```figure
inpaint-mask-reinject
```

## Build It（动手实现）

`code/main.py` 在 5 维数据上实现了一个玩具 1-D 图像修复方案。我们在 5 维混合数据上训练 DDPM，其中每个样本是来自两个聚类之一的 5 个浮点数。在推理时，我们“掩码”5 个维度中的 2 个，在每一步注入未掩码 3 个的噪声前向版本，并重新生成仅掩码的维度。

### Step 1: 5-D DDPM data（5-D DDPM 数据）

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### Step 2: train denoiser over all 5 dims（在所有 5 个维度上训练去噪器）

标准 DDPM。网络为 5 维噪声输入输出 5 维噪声预测。

### Step 3: at inference, mask-aware reverse（在推理时，掩码感知反向）

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source（用干净源的噪声新版本替换未掩码维度）
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

这是朴素方法，它在玩具 1-D 数据上工作。真实图像修复使用 9 通道输入，因为纹理一致性比边界更重要。

### Step 4: outpainting（外绘）

外绘是掩码反转的图像修复：掩码新的（以前不存在的）画布，用原始填充其余部分。相同的训练目标。

## Pitfalls（陷阱）

- **Seams（接缝）。** 朴素方法留下可见边界，因为梯度信息不会跨掩码流动。修复方法：将掩码扩展 8-16 像素，或使用适当的图像修复模型。
- **Mask leakage（掩码泄露）。** 如果条件图像的未掩码区域质量低或有噪声，它会污染掩码内的生成。去噪或轻微模糊。
- **CFG interacts with mask size（CFG 与掩码大小交互）。** 在小掩码上的高 CFG = 饱和补丁。对小编辑降低 CFG。
- **SDEdit fidelity cliff（SDEdit 保真度悬崖）。** 从 `t/T = 0.5` 到 `t/T = 0.6` 可能失去主体的身份。扫描和检查点。
- **Prompt mismatch（提示不匹配）。** 提示应该描述*整个*图像，而不仅仅是新内容。“A cat sitting on a chair”而不是“a cat”。

## Use It（实际应用）

| Task（任务） | Pipeline（管道） |
|------|----------|
| Remove object, small mask（移除物体，小掩码） | SD-Inpaint or Flux-Fill, standard prompt |
| Replace sky（替换天空） | SD-Inpaint + "blue sky at sunset" |
| Extend canvas（扩展画布） | SDXL outpaint mode (8px feather) or Flux-Fill with outpaint mask |
| Regenerate hand / face（重新生成手/脸） | SD-Inpaint with prompt re-describing the subject + ControlNet-Openpose |
| Change style of one region（改变一个区域的风格） | SDEdit at `t/T=0.5` on masked region |
| "Make it sunset"（“让它日落”） | InstructPix2Pix or Flux-Kontext |
| Background replacement（背景替换） | SAM mask → SD-Inpaint |
| Ultra-high-fidelity（超高保真度） | Flux-Fill or GPT-Image (hosted) for hardest cases |

SAM（Meta 的 Segment Anything，2023）+ 扩散 inpaint 是 2026 年的背景移除管道。SAM 2（2024）适用于视频。

## Ship It（交付）

保存为 `outputs/skill-editing-pipeline.md`。该技能接收一个原始图像 + 编辑描述 + 可选掩码（或 SAM 提示），并输出：掩码生成方法、基础模型、CFG 尺度（图像 + 文本）、SDEdit-t 或图像修复模式，以及 QA 检查表。

## Exercises（练习）

1. **Easy（简单）。** 在 `code/main.py` 中，改变掩码维度比例从 0.2 到 0.8。在什么比例时图像修复质量（掩码维度中的残差）等于无条件生成？
2. **Medium（中等）。** 实现 RePaint：在每 10 个反向步骤中，跳回 5 步（添加噪声）并重新去噪。测量它是否减少了掩码边缘的边界残差。
3. **Hard（困难）。** 使用 Hugging Face diffusers 比较：SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-Fill 在 20 张人脸再生任务上。分别评分姿态遵循和身份保留。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Inpainting（图像修复） | "Fill the hole"（填补空洞） | 在掩码内重新生成；保留外部像素。 |
| Outpainting（外绘） | "Extend the canvas"（扩展画布） | 在画布外重新生成；保留内部。 |
| 9-channel U-Net | "Proper inpainting model"（正确的图像修复模型） | 具有 `noisy \| encoded-source \| mask` 作为输入的 U-Net。 |
| SDEdit | "Img2img with noise level"（具有噪声级别的 img2img） | 将噪声添加到时间 `t`，用新提示去噪。 |
| InstructPix2Pix | "Text-only edits"（仅文本编辑） | 在 `(image, instruction, output)` 三元组上微调的扩散。 |
| RePaint | "No retraining"（无重新训练） | 在反向期间定期重新添加噪声以减少接缝。 |
| SAM | "Segment Anything" | 通过点击或框的掩码生成器；与 inpaint 配对。 |
| Flux-Kontext | "Edit with context"（用上下文编辑） | 接受参考图像 + 指令进行编辑的 Flux 变体。 |

## Production note: edit pipelines are latency-sensitive（生产笔记：编辑管道对延迟敏感）

用户编辑图像期望亚 5 秒的往返。在 1024² 上 30 步 SDXL-Inpaint 在 L4 上是 3-4 秒，加上 SAM 掩码生成（约 200 ms）和 VAE 编码/解码（约 500 ms 合并）。在生产框架中，这是 TTFT 约束而不是吞吐量约束 —— 批次 1，低并发，最小化每个阶段：

- **SAM-H is the slow one（SAM-H 是慢的那个）。** 在 1024² 上 SAM-H 约 200 ms；SAM-ViT-B 约 40 ms，有轻微质量损失。SAM 2（视频）增加时间开销；不要对单图像编辑使用它。
- **Skip the encode when possible（尽可能跳过编码）。** `pipe.image_processor.preprocess(img)` 编码到潜空间。如果你有先前生成的潜空间（在迭代编辑 UI 中典型），通过 `latents=...` 直接传递它们以跳过一次 VAE 编码。
- **Mask dilation matters for throughput too（掩码扩展对吞吐量也很重要）。** 小掩码意味着 U-Net 前向传递的大部分被浪费（未掩码像素无论如何都被钳位）。`diffusers` 的 `StableDiffusionInpaintPipeline` 无论如何都运行完整 U-Net；只有 9 通道正确 inpaint 变体利用掩码计算。
- **Flux-Kontext is the 2025 answer（Flux-Kontext 是 2025 年的答案）。** 在 `(source_image, instruction)` 上的单个前向传递 —— 没有单独的掩码，没有 SDEdit 噪声扫描。在 H100 上它在约 1.5 秒内发布一个编辑。建筑教训：折叠阶段。

## Further Reading（延伸阅读）

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) —— 无训练图像修复。
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) —— SDEdit。
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) —— 文本指令编辑。
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) —— SAM，掩码源。
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) —— 视频 SAM。
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) —— 注意力级编辑。
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) —— 2024 工具。
