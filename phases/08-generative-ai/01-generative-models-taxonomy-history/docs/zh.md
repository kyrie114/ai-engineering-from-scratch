# Generative Models — Taxonomy & History（生成模型 —— 分类与历史）

> 每个图像模型、文本模型、视频模型和 3D 模型都适合五个桶之一。选错桶，你会和数学斗争数周。选对桶，该领域过去十二年的进展会干净地堆在你的脑子里。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 2 (ML Fundamentals)（Phase 2 机器学习基础）, Phase 3 (Deep Learning Core)（Phase 3 深度学习核心）, Phase 7 · 14 (Transformers)（Phase 7 第 14 课 Transformer）
**Time:** ~45 minutes（约 45 分钟）

## The Problem（问题）

生成模型做一份工作：给定从某个未知分布 `p_data(x)` 中抽取的训练样本，输出看起来像来自同一分布的新样本。Faces、sentences、MIDI files、protein structures —— 如果你眯起眼睛看，都是同一个问题。

棘手的是 `p_data` 生活在一个有数百万维度的空间里（一个 512x512 RGB 图像是 ~786k 维度），样本坐在那个空间里的一个薄流形上，你只有大约 10M 示例。Brute-force 密度是 hopeless 的。每个生成模型都是一个 compromise，用一种 hard problem 交换一个 slightly less hard one。

五个家族在过去十二年中幸存下来。知道每个家族做的 compromise 告诉你为什么它在一些任务上获胜并在其他任务上崩溃。

## The Concept（概念）

![Five families of generative models — taxonomy by what they model（五类生成模型 —— 按它们建模的内容分类）](../assets/taxonomy.svg)

**1. 显式密度，可处理（Explicit density, tractable）。** 写 `log p(x)` 为你实际可以评估的和。Autoregressive 模型（PixelCNN、WaveNet、GPT）factorize `p(x) = ∏ p(x_i | x_<i)`。Normalizing flows（RealNVP、Glow）把 `p(x)` 构建为一个简单基的 invertible transform。优点：exact likelihood，clean training loss。缺点：autoregressive inference 是 sequential（对长序列慢），flows 需要 invertible architecture（architecturally restrictive）。

**2. 显式密度，近似（Explicit density, approximate）。** 从下方 bound `log p(x)`（ELBO）并优化 bound。VAE（Kingma 2013）使用一个 encoder-decoder 带一个 variational posterior。Diffusion 模型（DDPM、Ho 2020）训练一个 denoiser，隐式优化一个 weighted ELBO。Diffusion 是 2026 年主导的图像、视频和 3D backbone。

**3. 隐式密度（Implicit density）。** 完全跳过 density；学习一个 generator `G(z)`，产生样本和一个 discriminator `D(x)`，告诉 real from fake。GAN（Goodfellow 2014）。推理快（一个前向传播）但在训练期间 notoriously unstable。StyleGAN 1/2/3 仍然是 fixed-domain photorealism（faces、bedrooms）的 state of the art，即使在 2026 年。

**4. Score-based / continuous-time。** 直接学习 log-density `∇_x log p(x)`（score）的梯度。Song & Ermon (2019) 表明 score matching 把 diffusion 推广到 SDE。Flow matching（Lipman 2023）是 2024-2026 hotness：simulate-free training，straighter paths，比 DDPM 快 4-10x 采样。Stable Diffusion 3、Flux、AudioCraft 2 都使用 flow matching。

**5. 基于 token 的离散代码 autoregressive。** 用 VQ-VAE 或 residual quantizer 把高维数据压缩成一个短的离散 token 序列，然后用 Transformer 对 token 序列建模。Parti、MuseNet、AudioLM、VALL-E、Sora's patch tokenizer 都使用这个。这是 bucket 1 加一个 learned tokenizer。

## A brief history（简史）

| Year（年份） | Model（模型） | Why it mattered（为什么重要） |
|------|-------|-----------------|
| 2013 | VAE (Kingma) | 第一个有可用训练损失的深度生成模型。 |
| 2014 | GAN (Goodfellow) | 隐式密度，无 likelihood —— 令人震惊的 sharp 样本。 |
| 2015 | DRAW, PixelCNN | Sequential image generation（序列图像生成）。 |
| 2017 | Glow, RealNVP | Invertible flows；带深度的 exact likelihood。 |
| 2017 | Progressive GAN | 第一个 megapixel faces。 |
| 2019 | StyleGAN / StyleGAN2 | Photorealistic faces 仍然难以在那个领域击败。 |
| 2020 | DDPM (Ho) | Diffusion 变得实用。 |
| 2021 | CLIP, DALL-E 1, VQGAN | Text-to-image 进入主流。 |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | Latent diffusion + text conditioning = commodity。 |
| 2022 | ControlNet, LoRA | 对预训练 diffusion 的 fine control。 |
| 2023 | SDXL, Midjourney v5, Flow matching | Scale + better training dynamics。 |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | Video diffusion；flow matching 获胜。 |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | Production-grade video。 |
| 2026 | Consistency + Rectified Flow | 从 diffusion backbone 一步采样。 |

## The five-question triage（五问题分诊）

当一个新生成模型论文发布时，在阅读 method section 之前回答这五个问题。

1. **What is being modeled?（建模的是什么？）** Pixels、latents、discrete tokens、3D Gaussians、meshes、waveforms？
2. **Is the density explicit or implicit?（密度是显式还是隐式？）** 他们写下了 `log p(x)` 吗？
3. **Sampling: one-shot or iterative?（采样：one-shot 还是 iterative？）** Iterative 意味着更慢的推理；one-shot 通常意味着 adversarial 或 distilled。
4. **Conditioning: unconditional, class, text, image, pose?（条件：无条件、class、text、image、pose？）** 这决定了 loss 和 architecture scaffolding。
5. **Evaluation: FID, CLIP score, IS, human preference, task accuracy?（评估：FID、CLIP score、IS、human preference、task accuracy？）** 每个都有 known failure mode（见 Lesson 14）。

你会为本阶段的每节课重新回答这五个。到最后，它们会变成 reflex。

```figure
autoencoder-bottleneck
```

## Build It（动手实现）

这节课的代码是一个 lightweight visualization：用三个 toy approach（kernel density、discrete histogram 和一个 nearest-sample "GAN-ish" generator）拟合一个 1-D mixture-of-Gaussians，这样你可以在一个你可以打印在屏幕上的问题上看到 explicit vs implicit density 之间的区别。

运行 `code/main.py`。它从 two-mode Gaussian mixture 中抽取 2000 个样本，然后打印：

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

注意：前两个让你问“这个点有多可能？”第三个不能。这就是 *explicit vs implicit* 区别，它将在未来的每节课中重要。

## Use It（实际应用）

2026 年哪个家族，哪个任务？

| Task（任务） | Best family（最佳家族） | Why（为什么） |
|------|-------------|-----|
| Photoreal faces, narrow domain（photoreal faces，窄领域） | StyleGAN 2/3 | 仍然最 sharp，推理最快。 |
| General text-to-image（通用 text-to-image） | Latent diffusion + flow matching | SD3、Flux.1、DALL-E 3。 |
| Fast text-to-image（快速 text-to-image） | Rectified flow + distillation | SDXL-Turbo、SD3-Turbo、LCM。 |
| Text-to-video | Diffusion Transformer + flow matching | Sora、Veo 2、Kling。 |
| Speech + music | Token-based AR (AudioLM, VALL-E, MusicGen) or flow matching (AudioCraft 2) | Discrete tokens 便宜扩展。 |
| 3D scenes | Gaussian Splatting fit, diffusion prior | 3D-GS 用于重建，diffusion 用于 novel-view。 |
| Density estimation (no sampling)（密度估计（无采样）） | Flows | 唯一有 exact `log p(x)` 的家族。 |
| Simulation / physics（模拟 / 物理） | Flow matching, score SDE | Straight-line paths, smooth vector field。 |

## Ship It（交付）

保存为 `outputs/skill-model-chooser.md`。

这个 skill 接收一个任务描述并输出：(1) 使用哪个家族，(2) 三个开源和三个 hosted 选项的 ranked list，(3) 你应该 watch 的 likely failure mode，(4) 一个 compute/time budget。

## Exercises（练习）

1. **Easy（简单）。** 对于以下五个产品，识别家族和 backbone：ChatGPT image、Midjourney v7、Sora、Runway Gen-3、ElevenLabs。证据应该来自 public technical report。
2. **Medium（中等）。** 你明天要读的论文声称比 diffusion 快 100x 采样。写下三个问题来检查这个加速在 conditioning 和高分辨率下是否幸存。
3. **Hard（困难）。** 拿一个你关心的领域（例如 protein structure、CAD、molecules、trajectories）。为该领域当前 SOTA 模型回答五问题分诊，并 sketch 一个更好的模型会改变什么。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Generative model（生成模型） | "It makes new stuff"（它制造新东西） | 学习 `p_data(x)` 的 sampler，可选择暴露 `log p(x)`。 |
| Explicit density（显式密度） | "You can evaluate it"（你可以评估它） | 模型提供一个 closed-form 或 tractable 的 `log p(x)`。 |
| Implicit density（隐式密度） | "GAN-style" | 只有一个 sampler —— 没有办法评估给定点的 `p(x)`。 |
| ELBO | "Evidence lower bound"（证据下界） | `log p(x)` 的一个 tractable lower bound；VAE 和 diffusion 优化它。 |
| Score（分数） | "Gradient of log-density"（log-density 的梯度） | `∇_x log p(x)`；diffusion 和 SDE 模型学习这个场。 |
| Manifold hypothesis（流形假设） | "Data lives on a surface"（数据生活在一个表面上） | 高维数据集中在一个低维流形上；为什么 dimensionality reduction 有效。 |
| Autoregressive（自回归） | "Predict the next piece"（预测下一块） | 把 joint factorize 为 conditionals 的 product。 |
| Latent（潜空间） | "Compressed code"（压缩代码） | 低维表示，解码器可以从它重建输入。 |

## Production note: five families, five inference shapes（生产笔记：五家族，五推理形状）

每个家族映射到一个不同的 inference-server cost curve。production-inference 文献把 LLM 推理框定为 prefill + decode；同样的分解也适用于这里：

- **Autoregressive (bucket 1 and 5)。** Sequential decode 主导延迟；KV-cache、continuous batching 和 speculative decoding 都直接适用。
- **VAE / diffusion / flow-matching (buckets 2 and 4)。** 没有 LLM 意义上的 decode。Cost = `num_steps × step_cost`，`step_cost` 是在完整 latent resolution 下的 transformer 或 U-Net forward。生产旋钮是 step count（DDIM / DPM-Solver / distillation）、batch size 和 precision（bf16 / fp8 / int4）。
- **GAN (bucket 3)。** 一个前向传播。没有 schedule，没有 KV-cache。TTFT ≈ total latency。这就是为什么 StyleGAN 仍然在窄领域 UX 上获胜。

当你在论文摘要中看到“比 diffusion 更快”时，把它翻译成“更少的步数 × 相同的步数成本”或“相同的步数 × 更便宜的步数成本”。其他一切都是 marketing。

## Further Reading（延伸阅读）

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) —— GAN 论文。
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) —— VAE 论文。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) —— DDPM 论文。
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) —— diffusion 作为 SDE。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) —— flow matching 论文。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) —— Stable Diffusion 3。
