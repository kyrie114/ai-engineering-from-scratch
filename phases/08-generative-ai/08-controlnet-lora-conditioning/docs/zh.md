# ControlNet, LoRA & Conditioning（ControlNet、LoRA 与条件化）

> 文本单独来说是一个笨拙的控制信号。ControlNet 让你克隆一个预训练的扩散模型，并用深度图、姿态骨架、涂鸦或边缘图像来引导它。LoRA 让你通过训练 1000 万个参数来微调一个 20 亿参数的模型。它们一起将 Stable Diffusion 从一个玩具变成了 2026 年每个代理商都发布的图像管道。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion)（Phase 8 第 07 课 潜空间扩散）, Phase 10 (LLMs from Scratch —— for LoRA foundation)（Phase 10 从零实现 LLM —— LoRA 基础）
**Time:** ~75 minutes（约 75 分钟）

## The Problem（问题）

一个像“一个女人穿着红色连衣裙在繁忙的街道上遛狗”的提示给模型没有关于*狗在哪里*、*女人是什么姿态*或*街道的透视*的信息。文本固定了你需要指定图像的大约 10%。其余的是视觉的，不能用文字有效地描述。

为每个信号（姿态、深度、canny、分割）从头开始训练一个新的条件模型是禁止的。你想要保持 2.6B 参数的 SDXL 主干冻结，附加一个读取条件的小型侧网络，并让它推动主干的中间特征。那就是 ControlNet。

你还想教模型新概念（你的脸、你的产品、你的风格），而不重新训练完整模型。你想要一个 100 倍更小的增量。那就是 LoRA —— 低秩适配器，插入到现有注意力权重中。

ControlNet + LoRA + 文本 = 2026 从业者的工具包。大多数生产图像管道在 SDXL / SD3 / Flux 基础上层叠 2-5 个 LoRA、1-3 个 ControlNet 和一个 IP-Adapter。

## The Concept（概念）

![ControlNet clones the encoder; LoRA adds low-rank deltas（ControlNet 克隆编码器；LoRA 添加低秩增量）](../assets/controlnet-lora.svg)

### ControlNet (Zhang et al., 2023)

拿一个预训练的 SD。*克隆* U-Net 的编码器一半。冻结原始的。训练克隆以接受额外的条件输入（边缘、深度、姿态）。通过*零卷积* skip 连接（初始化为零的 1×1 conv —— 从无操作开始，学习增量）将克隆连接回原始的 U-Net 解码器一半。

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

零卷积初始化意味着 ControlNet 从恒等开始 —— 即使在训练之前也没有害处。在 1M（提示、条件、图像）三元组上用标准扩散损失训练。

每模态 ControlNet 作为小型侧模型发布（SDXL 约 360M，SD 1.5 约 70M）。你可以在推理时组合它们：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu et al., 2021)

对于模型中的任何线性层 `W ∈ R^{d×d}`，冻结 `W` 并添加一个低秩增量：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。注意力的秩 4-16 是标准的，重度微调的秩 64-128。新参数数量：`2 · d · r` 而不是 `d²`。对于 `d=640` 的 SDXL 注意力，`r=16`：每个适配器 20k 参数而不是 410k —— 一个 20 倍的减少。在整个模型中：一个 LoRA 通常是 20-200MB 而基础是 5GB。

在推理时，你可以缩放 LoRA：`W' = W + α · B @ A`。`α = 0.5-1.5` 是正常的。多个 LoRA 堆叠是相加的（附带它们以非线性方式相互作用的通常警告）。

### IP-Adapter (Ye et al., 2023)

一个接受*图像*作为条件化（与文本一起）的小型适配器。使用 CLIP 图像编码器产生图像令牌，将它们与文本令牌一起注入交叉注意力。每个基础模型约 20MB。让你可以“用这个参考图像的风格生成一个图像”，而不需要 LoRA。

## Composability matrix（可组合性矩阵）

| Tool（工具） | What it controls（它控制什么） | Size（大小） | When to use（何时使用） |
|------|------------------|------|-------------|
| ControlNet | Spatial structure (pose, depth, edges)（空间结构（姿态、深度、边缘）） | 70-360MB | Exact layout, composition（精确布局，构图） |
| LoRA | Style, subject, concept（风格、主题、概念） | 20-200MB | Personalization, style（个性化，风格） |
| IP-Adapter | Style or subject from reference image（来自参考图像的风格或主题） | 20MB | No text can describe the look（没有文本可以描述外观） |
| Textual Inversion | Single concept as a new token（作为新令牌的单一概念） | 10KB | Legacy, mostly replaced by LoRA（遗留，大部分被 LoRA 取代） |
| DreamBooth | Full fine-tune on a subject（对主题的完整微调） | 2-5GB | Strong identity, high compute（强身份，高计算） |
| T2I-Adapter | Lighter ControlNet alternative（更轻的 ControlNet 替代品） | 70MB | Edge devices, inference budget（边缘设备，推理预算） |

ControlNet ≈ 空间。LoRA ≈ 语义。两者都用。

```figure
v4-controlnet-zero
```

## Build It（动手实现）

`code/main.py` 在 1-D 上模拟了两个机制：

1. **LoRA。** 一个预训练的线性层 `W`。冻结它。训练一个低秩 `B @ A`，使 `W + BA` 匹配一个目标线性层。展示 `r = 1` 足以完美学习一个秩 1 修正。
2. **ControlNet-lite。** 一个“冻结基础”预测器和一个读取额外信号的“侧网络”。侧网络的输出由一个初始化为零的可学习标量门控（我们的零卷积版本）。训练并观察门斜坡上升。

### Step 1: LoRA math（LoRA 数学）

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### Step 2: zero-init side network（零初始化侧网络）

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0（门初始化为 0）
h = base(x) + gated
```

在步骤 0 时，输出与基础完全相同。早期训练缓慢更新 `gate` —— 没有灾难性漂移。

## Pitfalls（陷阱）

- **Over-scaling LoRAs（过度缩放 LoRA）。** `α = 2` 或 `α = 3` 是一个常见的“让它更强”技巧，产生过度风格化/破碎的输出。保持 `α ≤ 1.5`。
- **ControlNet weight conflict（ControlNet 权重冲突）。** 在权重 1.0 使用姿态 ControlNet 和在权重 1.0 使用深度 ControlNet 通常会过冲。权重和 ≈ 1.0 是一个安全默认值。
- **LoRA on the wrong base（在错误基础上的 LoRA）。** SDXL LoRA 在 SD 1.5 上静默无操作，因为注意力维度不匹配。Diffusers 在 0.30+ 中会警告。
- **Textual Inversion drift（文本反转漂移）。** 在一个检查点上训练的令牌在另一个检查点上漂移严重。LoRA 更便携。
- **LoRA weight-merging and storage（LoRA 权重合并和存储）。** 你可以将 LoRA 烘焙到基础模型权重中以获得更快的推理（没有运行时添加），但你失去了在运行时缩放 `α` 的能力。保留两个版本。

## Use It（实际应用）

| Goal（目标） | 2026 pipeline（2026 管道） |
|------|---------------|
| Reproduce a brand's art style（复制品牌的藝術風格） | LoRA trained on ~30 curated images at rank 32 |
| Put my face in a generated image（将我的脸放在生成的图像中） | DreamBooth or LoRA + IP-Adapter-FaceID |
| Specific pose + prompt（特定姿态 + 提示） | ControlNet-Openpose + SDXL + text |
| Depth-aware composition（深度感知构图） | ControlNet-Depth + SD3 |
| Reference + prompt（参考 + 提示） | IP-Adapter + text |
| Exact layout（精确布局） | ControlNet-Scribble or ControlNet-Canny |
| Background replace（背景替换） | ControlNet-Seg + Inpainting (Lesson 09) |
| Fast 1-step style（快速 1 步风格） | LCM-LoRA on SDXL-Turbo |

## Ship It（交付）

保存为 `outputs/skill-sd-toolkit-composer.md`。该技能接收一个任务（输入资产：提示，可选的参考图像，可选姿态，可选深度，可选涂鸦）并输出：工具栈、权重和可重复的种子协议。

## Exercises（练习）

1. **Easy（简单）。** 在 `code/main.py` 中，改变 LoRA 秩 `r` 从 1 到 4。在什么秩时 LoRA 完全匹配一个秩 2 目标增量？
2. **Medium（中等）。** 在两个目标变换上训练两个独立的 LoRA。一起加载它们并展示它们的相加交互。交互何时破坏线性？
3. **Hard（困难）。** 使用 diffusers 堆叠：SDXL-base + Canny-ControlNet (weight 0.8) + a style LoRA (α 0.8) + IP-Adapter (weight 0.6)。随着栈权重的变化，测量 FID-vs-prompt-adherence 权衡。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| ControlNet | "Spatial control"（空间控制） | 克隆编码器 + 零卷积 skips；读取一个条件图像。 |
| Zero convolution（零卷积） | "Starts as identity"（从恒等开始） | 初始化为零的 1×1 conv；ControlNet 从无操作开始。 |
| LoRA | "Low-rank adapter"（低秩适配器） | `W + B @ A`，`r << d`；比完整微调少 100 倍参数。 |
| rank r（秩 r） | "The knob"（旋钮） | LoRA 压缩；4-16 典型，64+ 用于重度个性化。 |
| α | "LoRA strength"（LoRA 强度） | LoRA 增量的运行时缩放。 |
| IP-Adapter | "Reference image"（参考图像） | 通过 CLIP 图像令牌的小型图像条件适配器。 |
| DreamBooth | "Full subject fine-tune"（完整主题微调） | 在约 30 张主题图像上训练完整模型。 |
| Textual Inversion | "New token"（新令牌） | 仅学习一个新词嵌入；遗留，大部分被取代。 |

## Production note: LoRA swaps, ControlNet lanes, multi-tenant serving（生产笔记：LoRA 交换、ControlNet 车道、多租户服务）

一个真实的文本到图像 SaaS 在相同的基础检查点上服务数百个 LoRA 和十几个 ControlNet。服务问题看起来很像 LLM 多租户（生产文献在连续批处理和 LoRAX / S-LoRA 下涵盖 LLM 案例）：

- **Hot-swap LoRAs, do not merge（热插拔 LoRA，不要合并）。** 将 `W' = W + α·B·A` 合并到基础中给出约 3-5% 更快的每步推理，但冻结了 `α` 和基础。将 LoRA 作为秩 r 增量热保持在 VRAM 中；diffusers 暴露 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` 用于每请求激活。交换成本是 `2 · d · r · num_layers` 权重 —— MB 级，亚秒级。
- **ControlNet as a second attention lane（ControlNet 作为第二注意力车道）。** 克隆的编码器与基础并行运行。两个权重 1.0 的 ControlNet = 每步两个额外前向传递，不是一个合并传递。批量大小空间二次下降。为每个活动 ControlNet 预算约 1.5 倍步数成本。
- **Quantized LoRAs too（量化 LoRA 也是）。** 如果你量化了基础（见第 07 课，Flux 在 8GB 上），LoRA 增量也干净地量化到 8 位或 4 位。QLoRA 风格的加载让你可以在 4 位 Flux 基础上堆叠 5-10 个 LoRA 而不 blow memory。

Flux 特定：Niels 的 Flux-on-8GB 笔记本将基础量化为 4 位；在该量化基础上堆叠一个风格 LoRA（`pipe.load_lora_weights("user/style-lora")`）在 `weight_name="pytorch_lora_weights.safetensors"` 仍然有效。这是大多数 SaaS 代理商在 2026 年发布的配方。

## Further Reading（延伸阅读）

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) —— ControlNet。
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) —— LoRA（最初为 LLM；移植到扩散）。
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) —— IP-Adapter。
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) —— ControlNet 的更轻替代品。
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) —— DreamBooth。
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) —— 参考管道。
