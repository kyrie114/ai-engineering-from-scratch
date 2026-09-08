# 3D Generation（3D 生成）

> 3D 是 2D-to-3D 杠杆最强的模态。2023 年的突破是 3D Gaussian Splatting。2024-2026 年的生成推动在顶部层叠多视图扩散 + 3D 重建，从单个提示或照片产生对象和场景。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 4 (Vision)（Phase 4 视觉）, Phase 8 · 07 (Latent Diffusion)（Phase 8 第 07 课 潜空间扩散）
**Time:** ~45 minutes（约 45 分钟）

## The Problem（问题）

3D 内容是痛苦的：

- **Representation（表示）。** 网格、点云、体素网格、有符号距离场（SDF）、神经辐射场（NeRF）、3D Gaussians。每一个都有权衡。
- **Data scarcity（数据稀缺）。** ImageNet 有 14M 图像。最大的干净 3D 数据集（Objaverse-XL，2023）有约 10M 对象，大多数质量低。
- **Memory（内存）。** 一个 512³ 体素网格是 128M 个体素；一个有用的场景 NeRF 每射线需要 1M 个样本。生成比重建更难。
- **Supervision（监督）。** 对于 2D 图像，你有像素。对于 3D，你通常有一小部分 2D 视图，必须提升到 3D。

2026 年栈将两个问题分开。首先，用扩散模型生成*2D 多视图图像*。其次，将*3D 表示*（通常是 Gaussian splatting）拟合到那些图像。

## The Concept（概念）

![3D generation: multi-view diffusion + 3D reconstruction（3D 生成：多视图扩散 + 3D 重建）](../assets/3d-generation.svg)

### Representation: 3D Gaussian Splatting (Kerbl et al., 2023)（表示：3D Gaussian Splatting（Kerbl et al.，2023））

将一个场景表示为约 1M 个 3D Gaussians 的云。每个有 59 个参数：位置（3）、协方差（6，或四元数 4 + 尺度 3）、不透明度（1）、球谐颜色（3 度 0 级 3，3 度 48）。

渲染 = 投影 + alpha 混合。快（在 4090 上 1080p 约 100 fps）。可微。通过梯度下降拟合到真实照片。一个场景在消费级 GPU 上 5-30 分钟内拟合。

顶部的两个 2023-2024 创新：
- **Generative Gaussian splats（生成式 Gaussian splats）。** 像 LGM、LRM、InstantMesh 这样的模型从一张或几张图像直接预测 Gaussian 云。
- **4D Gaussian Splatting（4D Gaussian Splatting）。** 具有每帧偏移的 Gaussians 用于动态场景。

### Multi-view diffusion（多视图扩散）

微调一个预训练的图像扩散模型，从文本提示或单张图像生成同一对象的多个一致视图。Zero123（Liu et al.，2023）、MVDream（Shi et al.，2023）、SV3D（Stability，2024）、CAT3D（Google，2024）。通常输出对象周围的 4-16 个视图，通过 Gaussian splatting 或 NeRF 提升到 3D。

### Text-to-3D pipelines（文本到 3D 管道）

| Model（模型） | Input（输入） | Output（输出） | Time（时间） |
|-------|-------|--------|------|
| DreamFusion (2022) | text | NeRF via SDS | ~1 hour per asset |
| Magic3D | text | mesh + texture | ~40 min |
| Shap-E (OpenAI, 2023) | text | implicit 3D | ~1 min |
| SJC / ProlificDreamer | text | NeRF / mesh | ~30 min |
| LRM (Meta, 2023) | image | triplane | ~5 s |
| InstantMesh (2024) | image | mesh | ~10 s |
| SV3D (Stability, 2024) | image | novel views | ~2 min |
| CAT3D (Google, 2024) | 1-64 images | 3D NeRF | ~1 min |
| TripoSR (2024) | image | mesh | ~1 s |
| Meshy 4 (2025) | text + image | PBR mesh | ~30 s |
| Rodin Gen-1.5 (2025) | text + image | PBR mesh | ~60 s |
| Tencent Hunyuan3D 2.0 (2025) | image | mesh | ~30 s |

2025-2026 方向：具有适合游戏引擎的 PBR 材质的直接文本到网格模型。多视图扩散中间步骤仍然是通用对象的最佳性能配方。

### NeRF (for context)（NeRF（用于上下文））

神经辐射场（Mildenhall et al.，2020）。一个微型 MLP 接收 `(x, y, z, view direction)` 并输出 `(color, density)`。通过沿射线积分渲染。在质量上击败基于网格的新视图合成，但渲染慢 100-1000 倍。对于大多数实时使用被 Gaussian splatting 取代，但在研究中仍然占主导地位。

```figure
v4-3d-multiview
```

## Build It（动手实现）

`code/main.py` 实现了一个玩具 2D“Gaussian splatting”拟合：将合成目标图像（平滑渐变）表示为一组 2D Gaussian splats 的总和。通过梯度下降优化位置、颜色和协方差以匹配目标。你看到两个核心操作：前向渲染（splat + alpha 混合）和通过梯度下降拟合。

### Step 1: 2D Gaussian splat（2D Gaussian splat）

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### Step 2: render by summing splats（通过求和 splats 渲染）

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真实的 3D Gaussian splatting 按深度对 Gaussians 排序并按顺序 alpha 混合。我们的 2D 玩具只是求和。

### Step 3: fit by gradient descent（通过梯度下降拟合）

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## Pitfalls（陷阱）

- **View inconsistency（视图不一致）。** 如果你独立生成 4 个视图并且它们不同意对象结构，3D 拟合是模糊的。修复方法：具有共享注意力的多视图扩散。
- **Back-side hallucination（背面幻觉）。** 单张图像 → 3D 必须发明看不见的一侧。质量差异很大。
- **Gaussian splat explosion（Gaussian splat 爆炸）。** 无约束训练增长到 10M 个 splats 并过拟合。 Densification + 修剪启发式（来自 3D-GS 原始论文）是必不可少的。
- **Topology issues（拓扑问题）。** 来自隐式场（SDF）的网格通常有孔或自相交。在发布前运行 remesher（例如 blender 的体素 remesh）。
- **License of training data（训练数据许可）。** Objaverse 有混合许可；商业使用因模型而异。

## Use It（实际应用）

| Task（任务） | 2026 pick（2026 选择） |
|------|-----------|
| Scene reconstruction from photos（从照片重建场景） | Gaussian splatting (3DGS, Gsplat, Scaniverse) |
| Text-to-3D object for games（游戏文本到 3D 对象） | Meshy 4 or Rodin Gen-1.5 (PBR output) |
| Image-to-3D | Hunyuan3D 2.0, TripoSR, InstantMesh |
| Novel-view synthesis from few images（从少图像新视图合成） | CAT3D, SV3D |
| Dynamic scene reconstruction（动态场景重建） | 4D Gaussian Splatting |
| Avatar / clothed human（虚拟形象/穿衣人） | Gaussian Avatar, HUGS |
| Research / SOTA（研究/SOTA） | Whatever dropped last week（上周掉落的任何东西） |

对于在游戏或电子商务管道中发布生产 3D：Meshy 4 或 Rodin Gen-1.5 输出直接进入 Unity / Unreal 的 PBR 网格。

## Ship It（交付）

保存为 `outputs/skill-3d-pipeline.md`。该技能接收一个 3D 简报（输入：文本 / 一张图像 / 少图像；输出：网格 / splat / NeRF；使用：渲染 / 游戏 / VR），并输出：管道（多视图扩散 + 拟合，或直接网格模型）、基础模型、迭代预算、拓扑后处理、所需的材质通道。

## Exercises（练习）

1. **Easy（简单）。** 用 4、16、64 Gaussians 运行 `code/main.py`。报告最终 MSE 与目标。
2. **Medium（中等）。** 扩展到颜色 Gaussians（RGB）。确认重建匹配目标颜色模式。
3. **Hard（困难）。** 使用 gsplat 或 Nerfstudio，从 50 张照片捕获中重建一个真实对象。报告拟合时间和保留视图上的最终 SSIM。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| 3D Gaussian Splatting（3D Gaussian Splatting） | "3DGS" | 场景作为 3D Gaussians 的云；可微 alpha 混合渲染。 |
| NeRF | "Neural radiance field"（神经辐射场） | 在 3D 点输出颜色 + 密度的 MLP；通过射线积分渲染。 |
| Triplane（三平面） | "Three 2-D planes"（三个 2-D 平面） | 将 3D 分解为三个 2-D 轴对齐特征网格；比体积更便宜。 |
| SDS | "Score distillation sampling"（分数蒸馏采样） | 通过使用 2D 扩散分数作为伪梯度来训练 3D 模型。 |
| Multi-view diffusion（多视图扩散） | "Many views at once"（一次许多视图） | 输出一批一致相机视图的扩散模型。 |
| PBR | "Physically-based rendering"（基于物理的渲染） | 具有 albedo、粗糙度、金属、法线通道的材质。 |
| Densification（致密化） | "Grow splats"（增长 splats） | 3DGS 训练启发式：在高梯度区域拆分/克隆 splats。 |

## Production note: 3D has no shared substrate yet（生产笔记：3D 还没有共享基质）

与图像（潜空间扩散 + DiT）和视频（时空 DiT）不同，3D 在 2026 年没有单一的主导运行时。生产决策树在表示上分叉：

- **NeRF / triplane（NeRF / 三平面）。** 推理是光线步进 + 每个样本的一个 MLP 前向。一个 512² 渲染需要数百万次 MLP 前向。激进地批处理射线样本；SDPA/xformers 适用。
- **Multi-view diffusion + LRM reconstruction（多视图扩散 + LRM 重建）。** 两阶段管道。第 1 阶段（多视图 DiT）是像第 07 课一样的扩散服务器。第 2 阶段（LRM Transformer）是对视图的一次前向传递。整体延迟配置文件是“扩散 + 一次性” —— 相应地选择每阶段服务原语。
- **SDS / DreamFusion（SDS / DreamFusion）。** 每资产优化，不是推理。构建作业，不是请求处理器。

对于大多数 2026 产品，正确答案是“在请求上运行多视图扩散模型，异步重建到 3DGS，服务 3DGS 用于实时查看”。这干净地将工作负载在 GPU 推理服务器（快速）和离线优化器（慢）之间分开。

## Further Reading（延伸阅读）

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) —— NeRF。
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) —— 3DGS。
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) —— SDS。
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) —— Zero123。
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) —— 多视图扩散。
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) —— LRM。
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) —— CAT3D。
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) —— SV3D。
