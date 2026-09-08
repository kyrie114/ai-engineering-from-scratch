# 从零构建 3D 高斯泼溅（3D Gaussian Splatting from Scratch）

> 场景是数百万个 3D 高斯的云。每个高斯都有一个位置、方向、缩放、不透明度，以及一个取决于视角方向的颜色。光栅化它们，通过光栅化反向传播，完成。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 13 (3D Vision & NeRF), Phase 1 Lesson 12 (Tensor Operations), Phase 4 Lesson 10 (Diffusion basics optional)
**Time:** ~90 minutes

## 学习目标（Learning Objectives）

- 解释为什么 3D 高斯泼溅（3D Gaussian Splatting）在 2026 年取代 NeRF 成为照片级真实感 3D 重建的生产默认方案
- 说明每个高斯的六个参数（位置、旋转四元数、缩放、不透明度、球谐函数颜色、可选特征）以及每个参数贡献多少浮点数
- 从零使用 alpha 合成（alpha compositing）实现 2D 高斯泼溅光栅化器，然后展示 3D 情况如何投影到相同的循环
- 使用 `nerfstudio`、`gsplat` 或 `SuperSplat` 从 20-50 张照片重建场景，并导出到 `KHR_gaussian_splatting` glTF 扩展或 OpenUSD 26.03 `UsdVolParticleField3DGaussianSplat` 架构

## 问题（The Problem）

NeRF 将场景存储为 MLP 的权重。每个渲染像素是沿光线数百次 MLP 查询。训练需要数小时，渲染需要数秒，且权重无法编辑——如果你想在场景中移动一把椅子，就必须重新训练。

3D 高斯泼溅（Kerbl, Kopanas, Leimkühler, Drettakis, SIGGRAPH 2023）取代了这一切。场景是一组显式的 3D 高斯。渲染是 GPU 光栅化，速度 100+ fps。训练需要几分钟。编辑是直接的：平移一部分高斯，你就移动了椅子。到 2026 年，Khronos Group 已批准高斯泼溅的 glTF 扩展，OpenUSD 26.03 附带高斯泼溅架构，Zillow 和 Apartments.com 用它们渲染房地产，大多数 3D 重建研究论文都是核心 3DGS 思想的变体。

心智模型很简单，但数学有足够的运动部件，大多数介绍从光栅化开始，跳过投影和球谐函数。本课构建完整流程——先 2D 版本，然后是 3D 扩展。

## 概念（The Concept）

### 每个高斯携带什么（What a Gaussian carries）

一个 3D 高斯是空间中的参数化 blob，具有以下属性：

```
position         mu         (3,)    world 坐标中的中心
rotation         q          (4,)    编码方向的单位四元数
scale            s          (3,)    每个轴的对数缩放（渲染时取指数）
opacity          alpha      (1,)    后 sigmoid 不透明度 [0, 1]
SH coefficients  c_lm       (3 * (L+1)^2,)   与视角相关的颜色
```

旋转 + 缩放构建 3x3 协方差：`Sigma = R S S^T R^T`。这是 3D 中高斯的形状。球谐函数让颜色随视角变化——高光、微弱光泽、视角相关的发光——而无需存储每视角纹理。SH 阶数 3 时，每个颜色通道有 16 个系数，每个高斯仅颜色就有 48 个浮点数。

场景通常有 100-500 万个高斯。每个存储大约 60 个浮点数（3 + 4 + 3 + 1 + 48 + 其他）。一个五百万高斯场景是 240 MB——远小于等效的带逐点纹理的点云，比高分辨率下 NeRF MLP 权重重映小一个数量级。

### 光栅化，而非光线步进（Rasterisation, not ray marching）

```mermaid
flowchart LR
    SCENE["数百万个 3D 高斯<br/>（位置、旋转、缩放、<br/>不透明度、SH 颜色）"] --> PROJ["投影到 2D<br/>（相机外参 + 内参）"]
    PROJ --> TILES["分配到瓦片<br/>（16×16 屏幕空间）"]
    TILES --> SORT["每瓦片<br/>深度排序"]
    SORT --> ALPHA["从前到后<br/>Alpha 合成"]
    ALPHA --> PIX["像素颜色"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

五步，全部 GPU 友好。无每像素 MLP 查询。单块 RTX 3080 Ti 以 147 fps 渲染 600 万个泼溅。

### 投影步骤（The projection step）

在世界位置 mu 处带 3D 协方差 Sigma 的 3D 高斯，投影到屏幕位置 mu' 处带 2D 协方差 Sigma' 的 2D 高斯：

```
mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = viewing transform (rotation + translation of camera)
J = Jacobian of the perspective projection at mu'
```

2D 高斯的足迹是一个椭圆，其轴是 Sigma' 的特征向量。椭圆内的每个像素接收高斯的贡献，权重为 exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))。

### Alpha 合成规则（The alpha-compositing rule）

对于一个像素，覆盖它的高斯从后到前排序（或等价地从前到后使用反转公式）。颜色使用自 1980 年代以来每个半透明光栅化器都使用的相同方程合成：

```
C_pixel = sum_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       到 i 的透射率
alpha_i = opacity_i * exp(-0.5 * d^T Sigma'^-1 d)   局部贡献
c_i = eval_SH(SH_i, view_direction)    视角相关颜色
```

这与 NeRF 体渲染的方程完全相同，只是在显式稀疏高斯集上而非沿光线的密集采样。这个等式就是渲染质量匹配 NeRF 的原因——两者都在积分相同的辐射场方程。

### 为何可微分（Why this is differentiable）

每一步——投影、瓦片分配、Alpha 合成、SH 评估——相对于高斯参数都是可微的。给定真实图像，计算渲染像素损失，通过光栅化器反向传播，用梯度下降更新所有 (mu, q, s, alpha, c_lm)。约 30,000 次迭代后，高斯找到正确的位置、缩放和颜色。

### 致密化与剪枝（Densification and pruning）

固定集合的高斯无法覆盖复杂场景。训练包括两种自适应机制：

- **克隆（Clone）** 一个高斯，当其梯度幅度高但缩放小时，在当前位置克隆——重建需要更多细节。
- **分裂（Split）** 一个大缩放高斯为两个更小的高斯，当其梯度高时——一个高斯太光滑，无法拟合该区域。
- **剪枝（Prune）** 不透明度低于阈值的高斯——它们没有贡献。

致密化每 N 次迭代运行一次。场景通常从约 10 万个初始高斯（从 SfM 点播种）增长到训练结束时的 100-500 万。

### 球谐函数简述（Spherical harmonics in one paragraph）

视角相关颜色是单位球面上一个函数 c(direction)。球谐函数是球体的傅里叶基。截断到阶数 L，每个通道得到 (L+1)^2 个基函数。对新视角评估颜色，是学习到的 SH 系数与在视角方向评估的基之间的点积。阶数 0 = 1 个系数 = 常数颜色。阶数 3 = 16 个系数 = 足以捕捉朗伯着色、高光和轻微反射。SD 高斯泼溅论文默认使用阶数 3。

### 2026 年生产栈（The 2026 production stack）

```
1. 采集         智能手机 / DJI 无人机 / 手持扫描仪
2. SfM / MVS     COLMAP 或 GLOMAP 推导相机姿态 + 稀疏点
3. 训练 3DGS     nerfstudio / gsplat / inria official / PostShot（RTX 4090 上约 10-30 分钟）
4. 编辑          SuperSplat / SplatForge（清理浮动物、分割）
5. 导出          .ply -> glTF KHR_gaussian_splatting 或 .usd（OpenUSD 26.03）
6. 查看          Cesium / Unreal / Babylon.js / Three.js / Vision Pro
```

### 4D 与生成式变体（4D and generative variants）

- **4D 高斯泼溅** —— 高斯是时间的函数；用于体积视频（Superman 2026，A$AP Rocky 的 "Helicopter"）。
- **生成式泼溅** —— 文本生成泼溅模型（World Labs 的 Marble），幻觉整个场景。
- **3D 高斯无迹变换** —— NVIDIA NuRec 的变体，用于自动驾驶仿真。

```figure
cv3-gaussian-splat
```

## 构建它（Build It）

### 步骤 1：一个 2D 高斯（A 2D Gaussian）

我们先构建一个 2D 光栅化器。3D 情况在投影后简化为它。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_2d_gaussian(means, covs, points):
    """
    means:  (G, 2)      中心
    covs:   (G, 2, 2)   协方差矩阵
    points: (H, W, 2)   像素坐标
    returns: (G, H, W)  每个像素每个高斯的密度
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)
```

`einsum` 为每个（高斯，像素）对执行二次型 `diff^T Sigma^-1 diff`。

### 步骤 2：2D 泼溅光栅化器（2D splatting rasteriser）

从前到后 alpha 合成。2D 中的深度无意义，因此使用学习的每个高斯标量进行排序。

```python
def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     在 [0, 1] 中
    depths:    (G,)     用于排序的每个高斯标量
    image_size: (H, W)
    returns:   (H, W, 3) 渲染图像
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

不快——真正实现使用基于瓦片的 CUDA 内核——但数学完全正确且完全可微。

### 步骤 3：可训练的 2D 泼溅场景（A trainable 2D splat scene）

```python
class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # 2D 中的单角度
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)
```

`log_scale`、`opacity_logit` 和 `colour_logits` 都是无约束参数，在渲染时通过正确的激活映射。这是每个 3DGS 实现的标准模式。

### 步骤 4：将 2D 高斯拟合到目标图像（Fit 2D Gaussians to a target image）

```python
import math
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Red circle
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # Blue square
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"step {step:3d}  mse {loss.item():.4f}")
```

200 步后，64 个高斯稳定到两个形状。这就是全部思想——在显式几何基元上进行梯度下降。

### 步骤 5：从 2D 到 3D（From 2D to 3D）

3D 扩展保持相同的循环。新增内容：

1. 每个高斯的旋转是四元数而非单角度。
2. 协方差为 `R S S^T R^T`，其中 R 由四元数构建，S = diag(exp(log_scale))。
3. 投影 (mu, Sigma) -> (mu', Sigma') 使用相机外参和 mu 处透视投影的 Jacobian。
4. 颜色变为球谐函数展开；在视角方向评估。
5. 深度排序来自实际相机空间 z 而非学习的标量。

每个生产实现（`gsplat`、`inria/gaussian-splatting`、`nerfstudio`）都在 GPU 上使用基于瓦片的 CUDA 内核精确执行这些步骤。

### 步骤 6：球谐函数评估（Spherical harmonics evaluation）

最高阶数 3 的 SH 基每个通道有 16 项。评估：

```python
def eval_sh_degree_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)   最后一维是 RGB 通道
    dirs:      (..., 3)       单位向量
    returns:   (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # 阶数 3 项在此省略以保持简洁；完整 16 系数版本在代码文件中
    return result
```

学习的 `sh_coeffs` 存储该高斯"每个方向上的颜色"。渲染时，你根据当前视角方向评估并获得一个 3 向量 RGB。

## 使用它（Use It）

对于真正的 3DGS 工作，使用 `gsplat`（Meta）或 `nerfstudio`：

```bash
pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data
```

`splatfacto` 是 nerfstudio 的 3DGS 训练器。在 RTX 4090 上，典型场景运行需要 10-30 分钟。

2026 年重要的导出选项：

- `.ply` —— 原始高斯云（可移植，文件最大）。
- `.splat` —— PlayCanvas / SuperSplat 量化格式。
- glTF `KHR_gaussian_splatting` —— Khronos 标准，跨查看器可移植（2026 年 2 月 RC）。
- OpenUSD `UsdVolParticleField3DGaussianSplat` —— USD 原生，用于 NVIDIA Omniverse 和 Vision Pro 流水线。

对于 4D / 动态场景，`4DGS` 和 `Deformable-3DGS` 使用随时间变化的均值和不透明度扩展了相同的机制。

## 交付成果（Ship It）

本课产出：

- `outputs/prompt-3dgs-capture-planner.md` —— 针对给定场景类型规划采集会话（照片数量、相机路径、光照）的提示。
- `outputs/skill-3dgs-export-router.md` —— 根据下游查看器或引擎选择正确导出格式（`.ply` / `.splat` / glTF / USD）的 skill。

## 练习（Exercises）

1. **(Easy)** 在另一个合成图像上运行 2D 泼溅训练器。在 [16, 64, 256] 中变化 `num_splats`，绘制每个值的 MSE 随步数变化曲线。找出收益递减点。
2. **(Medium)** 扩展 2D 光栅化器以支持通过阶数 2 球谐函数依赖标量"视角角度"的每高斯 RGB 颜色。在一对目标图像上训练，并验证模型重建了两者。
3. **(Hard)** 克隆 `nerfstudio` 并在任何你有的场景（书桌、植物、人脸、房间）的 20 张照片上训练 `splatfacto`。导出到 glTF `KHR_gaussian_splatting`，在查看器（Three.js `GaussianSplats3D`、SuperSplat、Babylon.js V9）中打开。报告训练时间、高斯数量和渲染 fps。

## 关键术语（Key Terms）

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| 3DGS | "Gaussian splats" | Explicit scene representation as millions of 3D Gaussians with per-Gaussian position, rotation, scale, opacity, SH colour |
| Covariance | "Shape of the Gaussian" | `Sigma = R S S^T R^T`; orientation and anisotropic scale of one Gaussian |
| Alpha compositing | "Back-to-front blend" | Same equation as NeRF's volumetric render, now over an explicit sparse set |
| Densification | "Clone and split" | Adaptive addition of new Gaussians where reconstruction is under-fit |
| Pruning | "Delete low-opacity" | Remove Gaussians that have collapsed to near-zero opacity during training |
| Spherical harmonics | "View-dependent colour" | Fourier basis on the sphere; stores colour as a function of viewing direction |
| Splatfacto | "nerfstudio's 3DGS" | The easiest path to training 3DGS in 2026 |
| `KHR_gaussian_splatting` | "glTF standard" | Khronos 2026 extension that makes 3DGS portable across viewers and engines |

## 拓展阅读（Further Reading）

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) —— 原始论文
- [gsplat (Meta/nerfstudio)](https://github.com/nerfstudio-project/gsplat) —— 生产级 CUDA 光栅化器
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) —— 参考训练配方
- [Khronos KHR_gaussian_splatting extension](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md) —— 2026 年可移植格式
- [OpenUSD 26.03 release notes](https://openusd.org/release/) —— `UsdVolParticleField3DGaussianSplat` 架构
- [THE FUTURE 3D State of Gaussian Splatting 2026](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026) —— 行业概览
