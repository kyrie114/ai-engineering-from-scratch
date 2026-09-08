# Video Generation（视频生成）

> 一个图像是一个 2-D 张量。一个视频是一个 3-D 的。理论是相同的；计算难 10-100 倍。OpenAI 的 Sora（2024 年 2 月）证明了它是可能的。到 2026 年，Veo 2、Kling 1.5、Runway Gen-3、Pika 2.0 和 WAN 2.2 在 1080p 从文本发布生产视频 —— 开放权重栈（CogVideoX、HunyuanVideo、Mochi-1、WAN 2.2）落后 12 个月。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion)（Phase 8 第 07 课 潜空间扩散）, Phase 7 · 09 (ViT)（Phase 7 第 09 课 ViT）, Phase 8 · 06 (DDPM)（Phase 8 第 06 课 DDPM）
**Time:** ~45 minutes（约 45 分钟）

## The Problem（问题）

一个 10 秒 1080p 视频以 24fps 是 240 帧 1920×1080×3 像素。每个片段约 1.5 GB 的原始数据。像素空间扩散是不可行的。你需要：

1. **Spatiotemporal compression（时空压缩）。** 一个编码视频（而不是帧）成时空 patch 序列的 VAE。
2. **Temporal coherence（时间一致性）。** 帧需要在几秒内共享内容、光照和对象身份。网络必须建模运动。
3. **Compute budget（计算预算）。** 视频训练对于相同模型大小比图像贵 10-100 倍。
4. **Conditioning（条件化）。** 文本、图像（第一帧）、音频或另一个视频。大多数生产模型接受所有四个。

解决这个问题的架构是应用于时空 patch 的扩散 Transformer（DiT），在巨大的（提示、字幕、视频）数据集上训练。与第 06 课相同的扩散损失。

## The Concept（概念）

![Video diffusion: patchify, DiT, decode（视频扩散：patchify、DiT、解码）](../assets/video-generation.svg)

### Patchify

用一个 3D VAE（学习的时空压缩）编码视频。潜空间的形状是 `[T_latent, H_latent, W_latent, C_latent]`。拆分为大小为 `[t_p, h_p, w_p]` 的 patch。对于 Sora 风格的模型，`t_p = 1`（每帧 patch）或 `t_p = 2`（每两帧）。一个 10 秒 1080p 视频压缩到约 20,000-100,000 个 patch。

### Spatiotemporal DiT（时空 DiT）

一个 Transformer 处理 flat patch 序列。每个 patch 有一个 3D 位置嵌入（时间 + y + x）。注意力通常是因式分解的：

- **Spatial attention（空间注意力）** 在每个帧的 patch 内。
- **Temporal attention（时间注意力）** 在相同空间位置的帧之间。
- **Full 3D attention（完整 3D 注意力）** 贵 16-100 倍；仅在低分辨率或研究中使用的。

### Text conditioning（文本条件化）

与大型文本编码器的交叉注意力（Sora 用 T5-XXL，CogVideoX-5B 用 T5-XXL）。长提示很重要 —— Sora 的训练集有 GPT 生成的密集重字幕，平均每个片段 200 个令牌。

### Training（训练）

时空潜空间上的标准扩散损失（ε 或 v 预测）。数据：网络视频 + 约 100M 精选片段 + 合成文本字幕。计算：即使是小型研究运行也需要 10,000+ GPU 小时；Sora 规模是 100,000+。

## The 2026 production landscape（2026 年生产格局）

| Model（模型） | Date（日期） | Max duration（最大时长） | Max res（最大分辨率） | Open weights?（开放权重？） | Notable（值得注意的） |
|-------|------|--------------|---------|---------------|---------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | No（否） | First model to show world simulator properties at scale（第一个展示大规模世界模拟器属性的模型） |
| Sora Turbo | 2024-12 | 20s | 1080p | No（否） | Production Sora at 5x faster inference（生产 Sora 快 5 倍推理） |
| Veo 2 (Google) | 2024-12 | 8s | 4K | No（否） | Highest quality + physics in 2025（2025 年最高质量 + 物理） |
| Veo 3 | 2025 Q3 | 15s | 4K | No（否） | Native audio and stronger camera control（原生音频和更强的相机控制） |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10s | 1080p | No（否） | Best human motion in 2025 Q1（2025 Q1 最佳人体运动） |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | No（否） | Professional video tools on top（顶部的专业视频工具） |
| Pika 2.0 | 2024-10 | 5s | 1080p | No（否） | Strongest character consistency（最强角色一致性） |
| CogVideoX (THUDM) | 2024 | 10s | 720p | Yes (2B, 5B) | First open 5B-scale video（第一个开放 5B 规模视频） |
| HunyuanVideo (Tencent) | 2024-12 | 5s | 720p | Yes (13B) | Open SOTA late 2024（2024 年末开放 SOTA） |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | Yes (10B) | Most permissively licensed（最宽松许可） |
| WAN 2.2 (Alibaba) | 2025-07 | 5s | 720p | Yes | Strongest open model mid-2025（2025 年中最佳开放模型） |

开放权重正在以比图像空间更快的速度缩小差距：HunyuanVideo + WAN 2.2 LoRA 在 2026 年中期已经为大多数开源工作流提供动力。

```figure
video-diffusion-denoise
```

## Build It（动手实现）

`code/main.py` 模拟了核心时空 DiT 想法：patchify 一个小型合成视频，添加每个 patch 的位置嵌入，并用一个 Transformer 风格的注意力 over patch 去噪整个序列。没有 numpy；纯 Python。我们展示即使是在 1-D 中，当相邻帧 patch 共享一个去噪器和位置嵌入时，时间一致性也会出现。

### Step 1: patchify a synthetic 1-D "video"（patchify 一个合成 1-D“视频”）

```python
def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory（“视频”是遵循平滑轨迹的 1-D 值序列）
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### Step 2: position embedding per frame（每帧位置嵌入）

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### Step 3: denoiser sees the whole sequence（去噪器看到整个序列）

与其独立去噪每一帧，我们的小型网络连接所有帧值 + 它们的位置嵌入，并预测所有帧联合的噪声。

### Step 4: temporal coherence test（时间一致性测试）

训练后，采样一个视频。测量帧到帧的 delta。如果模型已经学习了时间结构，delta 保持比独立采样每一帧更小。

## Pitfalls（陷阱）

- **Independent per-frame sampling = flicker（独立每帧采样 = 闪烁）。** 如果你对每一帧单独运行图像扩散，输出会闪烁，因为每一帧的噪声是独立的。视频扩散通过注意力或共享噪声将帧耦合来修复这个问题。
- **Naive 3D attention = OOM（朴素 3D 注意力 = OOM）。** 在 10 秒 1080p 潜空间上的完整 3D 注意力是数百亿次操作。因式分解为空间 + 时间。
- **Data captioning matters more than size（数据字幕比大小更重要）。** Sora 相对于先前工作的主要升级是训练了约 10 倍更详细的字幕（GPT-4 重新标记片段）。OpenAI 的技术报告对此是明确的。
- **First-frame conditioning（第一帧条件化）。** 大多数生产模型也接受一个图像作为第一帧。这是“图像到视频”模式；训练包括这个变体。
- **Physics drift（物理漂移）。** 长片段（>10s）积累微妙的inconsistencies。滑动窗口生成 + 关键帧锚定有帮助。

## Use It（实际应用）

| Use case（用例） | 2026 pick（2026 选择） |
|----------|-----------|
| Highest-quality text-to-video, hosted（最高质量文本到视频，托管） | Veo 3 or Sora |
| Camera-controlled cinematic（相机控制的电影感） | Runway Gen-3 with motion brushes |
| Character consistency across clips（跨片段角色一致性） | Pika 2.0 or Kling 2.1 |
| Open weights, fast fine-tune（开放权重，快速微调） | WAN 2.2 + LoRA |
| Image-to-video（图像到视频） | WAN 2.2-I2V, Kling 2.1 I2V, or Runway |
| Audio-to-video lip sync（音频到视频口型同步） | Veo 3 (native audio) or a dedicated lip-sync model |
| Video editing（视频编辑） | Runway Act-Two, Kling Motion Brush, Flux-Kontext (still-frame) |

2024 年到 2026 年相同质量下每秒视频的成本下降了 20 倍。

## Ship It（交付）

保存为 `outputs/skill-video-brief.md`。该技能接收一个视频简报（时长、宽高比、风格、相机计划、主题一致性、音频），并输出：模型 + 托管、提示脚手架（相机语言、主题描述、运动描述符）、种子 + 可重复性协议，以及帧级 QA 检查表。

## Exercises（练习）

1. **Easy（简单）。** 在 `code/main.py` 中，比较 (a) 独立每帧采样和 (b) 联合序列采样的帧到帧 delta。报告 delta 的均值和方差。
2. **Medium（中等）。** 添加一个第一帧条件：将帧 0 固定到一个给定值并采样其余部分。测量固定值如何传播。
3. **Hard（困难）。** 使用 HuggingFace diffusers 在本地 GPU 上运行 CogVideoX-2B。计时 720p 6 秒片段的 20 推理步。分析时空注意力以识别瓶颈。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Video VAE（视频 VAE） | "3-D VAE" | 编码器，将 `(T, H, W, C)` 压缩为时空潜空间。 |
| Patches（补丁） | "The tokens"（令牌） | 潜空间的固定大小 3-D 块；DiT 的输入。 |
| Factorized attention（因式分解注意力） | "Spatial + temporal"（空间 + 时间） | 先运行空间上的注意力，然后运行时间上的注意力；跳过完整 3-D 注意力。 |
| Image-to-video (I2V)（图像到视频） | "Animate this photo"（动画化这张照片） | 模型接受图像 + 文本，输出从它开始的视频。 |
| Keyframe conditioning（关键帧条件化） | "Anchor frames"（锚帧） | 固定特定帧以控制视频的弧线。 |
| Motion brush（运动画笔） | "Directional hint"（方向提示） | UI 输入，用户将运动向量画在图像上。 |
| Re-captioning（重字幕） | "Dense captions"（密集字幕） | 使用 LLM 用详细提示重新标记训练片段。 |
| Flicker（闪烁） | "Temporal artifact"（时间伪影） | 帧到帧不一致；用耦合去噪修复。 |

## Production note: video latents are a memory-bandwidth problem（生产笔记：视频潜空间是一个内存带宽问题）

一个 10 秒 1080p 片段在 24fps 是 240 帧 × 1920 × 1080 × 3 ≈ 1.5 GB 的原始像素。在 4× 视频 VAE 压缩（`2 × spatial × 2 × temporal`）之后，潜空间每个请求约 100 MB。在批次 1 上通过时空 DiT 运行 30 步，你正在通过 HBM 移动约 3 GB/步 —— 内存带宽，而不是 FLOPs，是瓶颈。

三个生产旋钮，都直接来自生产推理文献的推理章节：

- **TP across the DiT（跨 DiT 的 TP）。** 文本到视频模型通常 ≥10B 参数。在 4 个 H100 上的 TP=4 是标准的；405B 级模型的 PP=2 × TP=2。每步延迟大致随 TP 线性下降，直到全归约墙。
- **Frame batching = continuous batching（帧批处理 = 连续批处理）。** 在生成时，视频概念上是通过注意力链接的帧批次。连续批处理（飞行中调度）适用：当帧 `t-1` 被返回时开始渲染帧 `t+1`，如果模型架构允许滑动窗口生成。
- **Clip-level prefill cache（片段级预填充缓存）。** 对于图像到视频，第一帧条件化类似于 LLM 的提示预填充：计算一次，跨时间解码器传递重用。这有效地是视频的 KV-cache。

## Further Reading（延伸阅读）

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) —— Sora 技术报告。
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) —— CogVideoX。
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) —— HunyuanVideo。
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) —— Mochi-1。
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) —— 2025 年中期开放 SOTA。
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) —— 开创性视频扩散论文。
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) —— Stable Video Diffusion 的祖先。
