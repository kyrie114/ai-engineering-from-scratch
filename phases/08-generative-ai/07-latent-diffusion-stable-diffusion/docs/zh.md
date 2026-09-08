# Latent Diffusion & Stable Diffusion（潜空间扩散与 Stable Diffusion）

> 在 512×512 图像上进行像素空间扩散是一场计算战争罪。Rombach et al.（2022）注意到你不需要全部 786k 维度来生成图像 —— 你只需要足够捕捉语义结构，以及一个单独的解码器来处理其余部分。在 VAE 的潜空间内运行扩散。那一个想法就是 Stable Diffusion。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 02 (VAE)（Phase 8 第 02 课 VAE）, Phase 8 · 06 (DDPM)（Phase 8 第 06 课 DDPM）, Phase 7 · 09 (ViT)（Phase 7 第 09 课 ViT）
**Time:** ~75 minutes（约 75 分钟）

## The Problem（问题）

在 512² 上进行像素空间扩散意味着 U-Net 在形状为 `[B, 3, 512, 512]` 的张量上运行。对于一个 500M 参数的 U-Net，每个采样步骤约 100 GFLOPS。五十步是每张图像 5 TFLOPS。在十亿张图像上训练，计算账单是荒谬的。

这些 FLOPs 中的大部分都用于将感知上不重要的细节通过网络推动 —— 一个有损 VAE 可以压缩掉的高频纹理。Rombach 的想法：训练一个 VAE 一次（*第一阶段*），冻结它，并在 4 通道 64×64 潜空间（*第二阶段*）中完全运行扩散。相同的 U-Net。1/16 的像素。对于可比较的质量约 64x 更少的 FLOPs。

这就是 Stable Diffusion 配方。SD 1.x / 2.x 在 `64×64×4` 潜空间上使用了一个 860M 的 U-Net，SDXL 在 `128×128×4` 上使用了一个 2.6B 的 U-Net，SD3 用流匹配将 U-Net 替换为扩散 Transformer（DiT）。Flux.1-dev（Black Forest Labs，2024）发布了一个 12B 参数的 DiT-MMDiT。所有都在相同的两阶段基质上运行。

## The Concept（概念）

![Latent diffusion: VAE compression + diffusion in latent space（潜空间扩散：VAE 压缩 + 潜空间中的扩散）](../assets/latent-diffusion.svg)

**Two stages, separately trained（两个阶段，分别训练）。**

1. **Stage 1 — VAE（第一阶段 —— VAE）。** 编码器 `E(x) → z`，解码器 `D(z) → x`。目标压缩：每个空间轴 8 倍下采样 + 调整通道使总潜空间大小约为像素计数的 1/16。损失 = 重建（L1 + LPIPS 感知）+ KL（小权重使 `z` 不被强制太高斯，因为我们不需要从 `z` 进行精确采样）。通常用对抗损失训练，使解码图像锐利。
2. **Stage 2 — diffusion on `z`（第二阶段 —— `z` 上的扩散）。** 将 `z = E(x_real)` 视为数据。训练一个 U-Net（或 DiT）去噪 `z_t`。在推理时：通过扩散采样 `z_0`，然后 `x = D(z_0)`。

**Text conditioning（文本条件化）。** 两个额外的组件。一个冻结的文本编码器（SD 1.x 用 CLIP-L，SD 2/XL 用 CLIP-L+OpenCLIP-G，SD3 和 Flux 用 T5-XXL）。一个交叉注意力注入：每个 U-Net 块接收 `[Q = image features, K = V = text tokens]` 并将它们混合在一起。令牌是文本影响图像的唯 way。

**The loss function is identical to Lesson 06（损失函数与第 06 课相同）。** 对噪声的相同 DDPM / 流匹配 MSE。你只是交换了数据域。

## Architecture variants（架构变体）

| Model（模型） | Year（年份） | Backbone（主干） | Latent shape（潜空间形状） | Text encoder（文本编码器） | Params（参数） |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 tokens)（CLIP-L（77 个令牌）） | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner（U-Net + 细化器） | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | Distilled（蒸馏） | 128×128×4 | same（相同） | 1-4 step sampling（1-4 步采样） |
| SD3 | 2024 | MMDiT (multimodal DiT) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT distilled | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 step |

趋势：用 DiT（潜空间 patch 上的 Transformer）替换 U-Net，扩大文本编码器（T5 比 CLIP 更适合提示遵循），增加潜空间通道（4 → 16 给出更多细节空间）。

```figure
noise-schedule
```

## Build It（动手实现）

`code/main.py` 在第 06 课的 DDPM 之上堆叠了一个玩具 1-D“VAE”（恒等编码器 + 解码器，用于演示；真实的 VAE 将是一个 conv net），并添加了具有无分类器引导的类条件化。它展示了相同的扩散损失无论是在原始 1-D 值上还是在编码值上运行都有效 —— 这是关键见解。

### Step 1: encoder/decoder（编码器/解码器）

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale（玩具“压缩”到更小尺度）
def decode(z):    return z * 2.0
```

一个真实的 VAE 有训练的权重。为了教学，这个线性映射足以展示扩散在 `z` 上运行而不关心原始数据空间。

### Step 2: diffusion in `z`-space（`z` 空间中的扩散）

与第 06 课相同的 DDPM。网络看到的数据是 `z = E(x)`。在采样 `z_0` 之后，用 `D(z_0)` 解码。

### Step 3: classifier-free guidance（无分类器引导）

在训练期间，10% 的时间丢弃类标签（用空令牌替换）。在推理时，计算 `ε_cond` 和 `ε_uncond`，然后：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 无引导（完全多样性），`w = 3` = 默认，`w = 7+` = 饱和/过锐。

### Step 4: text conditioning (concept, not code)（文本条件化（概念，非代码））

用冻结的文本编码器输出替换类标签。通过交叉注意力将文本嵌入馈送到 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这是类条件扩散模型与 Stable Diffusion 之间唯一的实质性区别。

## Pitfalls（陷阱）

- **VAE-scale mismatch（VAE 尺度不匹配）。** SD 1.x VAE 有一个缩放常数（`scaling_factor ≈ 0.18215`），在编码后应用。忘记这个会使 U-Net 在方差 wildly wrong 的潜空间上训练。每个检查点都带有一个。
- **Text encoder silently wrong（文本编码器静默错误）。** SD3 需要 T5-XXL 且 >=128 个令牌，回退到仅 CLIP 是有损的。始终检查 `use_t5=True`，否则提示保真度会崩溃。
- **Mixing latent spaces（混合潜空间）。** SDXL、SD3、Flux 都使用不同的 VAE。在 SDXL 潜空间上训练的 LoRA 在 SD3 上不工作。Hugging Face diffusers 0.30+ 拒绝加载不匹配的检查点。
- **CFG too high（CFG 太高）。** `w > 10` 产生饱和的、油腻的图像，并以多样性为代价过度拟合提示。最佳点是 `w = 3-7`。
- **Negative prompts leaking（负面提示泄露）。** 空负面提示变为空令牌；填写的负面提示变为 `ε_uncond`。它们不相同；一些管道静默默认为空。

## Use It（实际应用）

2026 年生产栈：

| Target（目标） | Recommended backbone（推荐主干） |
|--------|----------------------|
| Narrow domain, paired data, training a model from scratch（狭窄领域，配对数据，从头训练模型） | SDXL fine-tune (LoRA / full) —— 最快发布 |
| Open-domain text-to-image, open weights（开放域文本到图像，开放权重） | Flux.1-dev (12B, Apache / non-commercial) or SD3.5-Large |
| Fastest inference, open weights（最快推理，开放权重） | Flux.1-schnell (1-4 step, Apache) or SDXL-Lightning |
| Best prompt adherence, hosted（最佳提示遵循，托管） | GPT-Image / DALL-E 3 (still), Midjourney v7, Imagen 4 |
| Edit workflows（编辑工作流） | Flux.1-Kontext (Dec 2024) —— 原生接受图像 + 文本 |
| Research, baseline（研究，基线） | SD 1.5 —— 古老但研究充分 |

## Ship It（交付）

保存为 `outputs/skill-sd-prompter.md`。该技能接收一个文本提示 + 目标风格，并输出：模型 + 检查点、CFG 尺度、采样器、负面提示、分辨率、可选的 ControlNet/IP-Adapter 组合，以及每步 QA 检查表。

## Exercises（练习）

1. **Easy（简单）。** 使用引导 `w ∈ {0, 1, 3, 7, 15}` 运行 `code/main.py`。按类记录平均样本。在什么 `w` 时类均值偏离真实数据均值？
2. **Medium（中等）。** 用 tanh-MLP 编码器/解码器对替换玩具线性编码器，并带有重建损失。在新潜空间上重新训练扩散。样本质量改变了吗？
3. **Hard（困难）。** 使用 diffusers 设置一个真实的 Stable Diffusion 推理：加载 `sdxl-base`，用 CFG=7 运行 30 Euler 步，计时。现在切换到 `sdxl-turbo`，4 步，CFG=0。相同的主题，不同的质量 —— 描述发生了什么以及为什么。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| First stage（第一阶段） | "The VAE" | 训练的编码器/解码器对；将 512² 压缩到 64²。 |
| Second stage（第二阶段） | "The U-Net" | 潜空间上的扩散模型。 |
| CFG | "Guidance scale"（引导尺度） | `(1+w)·ε_cond - w·ε_uncond`；调整条件强度。 |
| Null token（空令牌） | "Empty prompt embed"（空提示嵌入） | 用于 `ε_uncond` 的无条件嵌入。 |
| Cross-attention（交叉注意力） | "How text gets in"（文本如何进入） | 每个 U-Net 块将文本令牌作为 K 和 V 进行注意力。 |
| DiT | "Diffusion Transformer"（扩散 Transformer） | 用潜空间 patch 上的 Transformer 替换 U-Net；更好地扩展。 |
| MMDiT | "Multi-modal DiT"（多模态 DiT） | SD3 的架构：文本和图像流与联合注意力。 |
| VAE scaling factor（VAE 缩放因子） | "Magic number"（魔法数字） | 将潜空间除以约 5.4，使扩散在单位方差空间中运行。 |

## Production note: running Flux-12B on an 8GB consumer GPU（生产笔记：在 8GB 消费级 GPU 上运行 Flux-12B）

参考 Flux 集成是规范的“我有一个消费级 GPU，我能发布这个吗？”配方。技巧是生产推理文献中列出的相同三旋钮配方应用于扩散 DiT：

1. **Staggered loading（交错加载）。** Flux 有三个网络永远不需要在 VRAM 中共存：T5-XXL 文本编码器（fp32 约 10 GB）、CLIP-L（小）、12B MMDiT 和 VAE。首先编码提示，*删除* 编码器，加载 DiT，去噪，*删除* DiT，加载 VAE，解码。消费级 8GB GPU 一次只适合一个阶段。
2. **4-bit quantization via bitsandbytes（通过 bitsandbytes 的 4 位量化）。** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)` 用于 T5 编码器和 DiT。将内存降低 8 倍，质量下降对于文本到图像是不可察觉的（Aritra 的基准测试中链接在笔记本中）。
3. **CPU offload（CPU 卸载）。** `pipe.enable_model_cpu_offload()` 在每个前向传递推进时自动在 CPU 和 GPU 之间交换模块。增加 10-20% 延迟但使管道能够运行。

内存计算是：`10 GB T5 / 8 = 1.25 GB` 量化，`12B params × 0.5 bytes = ~6 GB` 量化 DiT，加上激活。在 stas00 的术语中，这是 TP=1 推理的极端 —— 没有模型并行，最大量化。对于生产，你会在 H100 上运行 TP=2 或 TP=4；对于一个开发笔记本电脑，这是配方。

## Further Reading（延伸阅读）

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) —— Stable Diffusion。
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) —— SDXL。
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) —— DiT。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) —— SD3，MMDiT。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) —— CFG。
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) —— Flux.1 系列。
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) —— 上面每个检查点的参考实现。
