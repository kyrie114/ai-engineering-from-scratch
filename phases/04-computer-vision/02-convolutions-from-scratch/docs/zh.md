# 从零实现卷积（Convolutions from Scratch）

> 卷积就是一个在图像上滑动的小型全连接层，在每一个位置共享同一组权重。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 4 Lesson 01 (Image Fundamentals)
**Time:** ~75 minutes

## 学习目标（Learning Objectives）

- 只用 NumPy 从零实现二维卷积，包括嵌套循环版本和向量化的 `im2col` 版本
- 对任意输入尺寸、卷积核尺寸、填充和步幅组合计算输出空间尺寸，并解释 `(H - K + 2P) / S + 1` 公式的由来
- 手工设计卷积核（边缘、模糊、锐化、Sobel），并解释每一个为什么会产生它那样的激活模式
- 把卷积堆叠成特征提取器，并把堆叠深度与感受野大小联系起来

## 问题（The Problem）

在 224x224 的 RGB 图像上，一个全连接层的每个神经元需要 224 * 224 * 3 = 150,528 个输入权重。一个只有 1,000 个单元的隐藏层就已经有 1.5 亿个参数——这还是在你学到任何有用东西之前。更糟的是，这一层并不知道左上角的狗和右下角的狗是同一个模式。它把每个像素位置当作独立的，这对图像来说恰恰是错的：把一只猫平移三个像素，不应该逼着网络重新学习这个概念。

图像模型需要的两个性质是**平移等变性（translation equivariance）**（输入平移，输出跟着平移）和**参数共享（parameter sharing）**（同一个特征检测器在每处运行）。全连接层一样都给不了你。卷积把两者免费奉上。

卷积不是为深度学习发明的。驱动 JPEG 压缩、Photoshop 里的高斯模糊、工业视觉中的边缘检测以及有史以来每一种音频滤波器的，都是同一个操作。CNN 能在 2012 到 2020 年间统治 ImageNet，原因是：对于"相邻取值相关、同一模式可能出现在任何地方"的数据，卷积是正确的先验。

## 核心概念（The Concept）

### 一个卷积核，滑动（One kernel, sliding）

二维卷积取一个小权重矩阵，称为卷积核（kernel，或称 filter），让它在输入上滑动，并在每个位置计算逐元素乘积的和。这个和就是一个输出像素。

```mermaid
flowchart LR
    subgraph IN["输入 (H x W)"]
        direction LR
        I1["5 x 5 图像"]
    end
    subgraph K["卷积核 (3 x 3)"]
        K1["学习到的<br/>权重"]
    end
    subgraph OUT["输出 (H-2 x W-2)"]
        O1["3 x 3 特征图"]
    end
    I1 --> |"滑动卷积核<br/>在每个位置<br/>计算点积"| O1
    K1 --> O1

    style IN fill:#dbeafe,stroke:#2563eb
    style K fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

在 5x5 输入上的一个具体 3x3 例子（无填充，步幅 1）：

```
Input X (5 x 5):                Kernel W (3 x 3):

  1  2  0  1  2                   1  0 -1
  0  1  3  1  0                   2  0 -2
  2  1  0  2  1                   1  0 -1
  1  0  2  1  3
  2  1  1  0  1

The kernel slides across every valid 3 x 3 window. Output Y is 3 x 3:

 Y[0,0] = sum( W * X[0:3, 0:3] )
 Y[0,1] = sum( W * X[0:3, 1:4] )
 Y[0,2] = sum( W * X[0:3, 2:5] )
 Y[1,0] = sum( W * X[1:4, 0:3] )
 ... and so on
```

这一个公式——**共享权重、局部性、滑动窗口**——就是全部思想。其余都是簿记工作。

### 输出尺寸公式（Output size formula）

给定输入空间尺寸 `H`、卷积核尺寸 `K`、填充 `P`、步幅 `S`：

```
H_out = floor( (H - K + 2P) / S ) + 1
```

把它背下来。每个架构你都要算它几十次。

| 场景 | H | K | P | S | H_out |
|------|---|---|---|---|-------|
| 有效卷积，无填充 | 32 | 3 | 0 | 1 | 30 |
| same 卷积（保持尺寸） | 32 | 3 | 1 | 1 | 32 |
| 下采样 2 倍 | 32 | 3 | 1 | 2 | 16 |
| 2x2 池化 | 32 | 2 | 0 | 2 | 16 |
| 大感受野 | 32 | 7 | 3 | 2 | 16 |

"same padding" 指的是当 S == 1 时选一个 P 使 H_out == H。对奇数 K，就是 P = (K - 1) / 2。这就是 3x3 卷积核占主导的原因——它是仍然拥有中心的最小奇数卷积核。

### 填充（Padding）

没有填充时，每次卷积都会缩小特征图。堆 20 层，你的 224x224 图像就变成 184x184，既在边界上浪费算力，又让需要形状匹配的残差连接变复杂。

```
Zero padding (P = 1) on a 5 x 5 input:

  0  0  0  0  0  0  0
  0  1  2  0  1  2  0
  0  0  1  3  1  0  0
  0  2  1  0  2  1  0       Now the kernel can centre on pixel
  0  1  0  2  1  3  0       (0, 0) and still have three rows and
  0  2  1  1  0  1  0       three columns of values to multiply.
  0  0  0  0  0  0  0
```

实践中会遇到的模式：`zero`（最常见）、`reflect`（镜像边缘，避免生成式模型出现生硬边框）、`replicate`（复制边缘）、`circular`（环绕，用于环形问题）。

### 步幅（Stride）

步幅是滑动的步长。`stride=1` 是默认值。`stride=2` 把空间维度减半，是在 CNN 内部不借助独立池化层进行下采样的经典方式——每种现代架构（ResNet、ConvNeXt、MobileNet）都在某处用带步幅的卷积取代了 max-pool。

```
Stride 1 on a 5 x 5 input, 3 x 3 kernel:

  starts: (0,0) (0,1) (0,2)        -> output row 0
          (1,0) (1,1) (1,2)        -> output row 1
          (2,0) (2,1) (2,2)        -> output row 2

  Output: 3 x 3

Stride 2 on the same input:

  starts: (0,0) (0,2)              -> output row 0
          (2,0) (2,2)              -> output row 1

  Output: 2 x 2
```

### 多输入通道（Multiple input channels）

真实图像有三个通道。对 RGB 输入做 3x3 卷积，实际上是在一个 3x3x3 的体块上进行：每个输入通道一个 3x3 切片。在每个空间位置，你对全部三个切片做乘法求和，再加一个偏置。

```
Input:   (C_in,  H,  W)        3 x 5 x 5
Kernel:  (C_in,  K,  K)        3 x 3 x 3 (one kernel)
Output:  (1,     H', W')       2D map

For a layer that produces C_out output channels, you stack C_out kernels:

Weight:  (C_out, C_in, K, K)   e.g. 64 x 3 x 3 x 3
Output:  (C_out, H', W')       64 x 3 x 3

Parameter count: C_out * C_in * K * K + C_out   (the + C_out is biases)
```

最后一行就是你规划模型时要算的东西。对 3 通道输入做 64 通道的 3x3 卷积，参数量是 `64 * 3 * 3 * 3 + 64 = 1,792`。很便宜。

### im2col 技巧（The im2col trick）

嵌套循环易读但慢。GPU 想要的是大矩阵乘法。技巧是：把输入的每个感受野窗口摊平成大矩阵的一列，把卷积核摊平成一行，整个卷积就变成了一次 matmul。

```mermaid
flowchart LR
    X["输入<br/>(C_in, H, W)"] --> IM2COL["im2col<br/>(提取窗口)"]
    IM2COL --> COLS["列矩阵<br/>(C_in * K * K, H_out * W_out)"]
    W["权重<br/>(C_out, C_in, K, K)"] --> FLAT["展平<br/>(C_out, C_in * K * K)"]
    FLAT --> MM["matmul"]
    COLS --> MM
    MM --> OUT["输出<br/>(C_out, H_out * W_out)<br/>reshape 为 (C_out, H_out, W_out)"]

    style X fill:#dbeafe,stroke:#2563eb
    style W fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

每个生产级卷积实现都是它的某种变体，再加上缓存分块技巧（直接卷积、Winograd、大核时的 FFT 卷积）。理解了 im2col，你就理解了核心。

### 感受野（Receptive field）

单个 3x3 卷积看 9 个输入像素。堆两个 3x3 卷积，第二层的一个神经元看到 5x5 的输入像素。三个 3x3 卷积给出 7x7。一般地：

```
RF after L stacked K x K convs (stride 1) = 1 + L * (K - 1)

With strides:   RF grows multiplicatively with stride along each layer.
```

"一路 3x3 到底"（VGG、ResNet、ConvNeXt）能够成立的全部原因在于：两个 3x3 卷积看到的输入区域与一个 5x5 卷积相同，但参数更少，中间还多了一个非线性。

```figure
convolution-kernel
```

## 动手实现（Build It）

### 第 1 步：填充数组（Step 1: Pad an array）

从最小的原语开始：一个围绕 H x W 数组补零的函数。

```python
import numpy as np

def pad2d(x, p):
    if p == 0:
        return x
    h, w = x.shape[-2:]
    out = np.zeros(x.shape[:-2] + (h + 2 * p, w + 2 * p), dtype=x.dtype)
    out[..., p:p + h, p:p + w] = x
    return out

x = np.arange(9).reshape(3, 3)
print(x)
print()
print(pad2d(x, 1))
```

末尾轴技巧 `x.shape[:-2]` 意味着同一个函数无需修改就能作用于 `(H, W)`、`(C, H, W)` 或 `(N, C, H, W)`。

### 第 2 步：用嵌套循环实现二维卷积（Step 2: 2D convolution with nested loops）

参考实现——慢，但毫无歧义。这原则上就是 `torch.nn.functional.conv2d` 在做的事。

```python
def conv2d_naive(x, w, b=None, stride=1, padding=0):
    c_in, h, w_in = x.shape
    c_out, c_in_w, kh, kw = w.shape
    assert c_in == c_in_w

    x_pad = pad2d(x, padding)
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w_in + 2 * padding - kw) // stride + 1

    out = np.zeros((c_out, h_out, w_out), dtype=np.float32)
    for oc in range(c_out):
        for i in range(h_out):
            for j in range(w_out):
                hs = i * stride
                ws = j * stride
                patch = x_pad[:, hs:hs + kh, ws:ws + kw]
                out[oc, i, j] = np.sum(patch * w[oc])
        if b is not None:
            out[oc] += b[oc]
    return out
```

四层嵌套循环（输出通道、行、列，外加对 C_in、kh、kw 的隐式求和）。这是你用来核对每一个更快实现的基准真相。

### 第 3 步：用手工设计的卷积核验证（Step 3: Verify with a hand-designed kernel）

构造一个垂直 Sobel 核，应用到合成的阶跃图像上，看垂直边缘亮起来。

```python
def synthetic_step_image():
    img = np.zeros((1, 16, 16), dtype=np.float32)
    img[:, :, 8:] = 1.0
    return img

sobel_x = np.array([
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]]
], dtype=np.float32)[None]

x = synthetic_step_image()
y = conv2d_naive(x, sobel_x, padding=1)
print(y[0].round(1))
```

预期在第 7 列（亮度从左到右增加）出现大的正值，其余地方全为零。这一条打印输出就是你确认数学正确的健全性检查。

### 第 4 步：im2col（Step 4: im2col）

把输入中每个卷积核大小的窗口变成矩阵的一列。对 `C_in=3, K=3`，每列是 27 个数。

```python
def im2col(x, kh, kw, stride=1, padding=0):
    c_in, h, w = x.shape
    x_pad = pad2d(x, padding)
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w + 2 * padding - kw) // stride + 1

    cols = np.zeros((c_in * kh * kw, h_out * w_out), dtype=x.dtype)
    col = 0
    for i in range(h_out):
        for j in range(w_out):
            hs = i * stride
            ws = j * stride
            patch = x_pad[:, hs:hs + kh, ws:ws + kw]
            cols[:, col] = patch.reshape(-1)
            col += 1
    return cols, h_out, w_out
```

它仍然是 Python 循环，但繁重的计算即将变成一次向量化的 matmul。

### 第 5 步：用 im2col + matmul 加速卷积（Step 5: Fast conv via im2col + matmul）

用一次矩阵乘法替换四重循环。

```python
def conv2d_im2col(x, w, b=None, stride=1, padding=0):
    c_out, c_in, kh, kw = w.shape
    cols, h_out, w_out = im2col(x, kh, kw, stride, padding)
    w_flat = w.reshape(c_out, -1)
    out = w_flat @ cols
    if b is not None:
        out += b[:, None]
    return out.reshape(c_out, h_out, w_out)
```

正确性检查：运行两个实现并比较。

```python
rng = np.random.default_rng(0)
x = rng.normal(0, 1, (3, 16, 16)).astype(np.float32)
w = rng.normal(0, 1, (8, 3, 3, 3)).astype(np.float32)
b = rng.normal(0, 1, (8,)).astype(np.float32)

y_naive = conv2d_naive(x, w, b, padding=1)
y_im2col = conv2d_im2col(x, w, b, padding=1)

print(f"max abs diff: {np.max(np.abs(y_naive - y_im2col)):.2e}")
```

`max abs diff` 应该在 `1e-5` 左右——差异来自浮点累加顺序，不是 bug。

### 第 6 步：一组手工设计的卷积核（Step 6: A bank of hand-designed kernels）

五个滤波器，展示单个卷积层在任何训练开始之前能表达什么。

```python
KERNELS = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
    "blur_3x3": np.ones((3, 3), dtype=np.float32) / 9.0,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
}

def apply_kernel(img2d, kernel):
    x = img2d[None].astype(np.float32)
    w = kernel[None, None]
    return conv2d_im2col(x, w, padding=1)[0]
```

应用到任何灰度图像上：blur 让图像变柔，sharpen 让边缘更锐利，Sobel-x 点亮垂直边缘，Sobel-y 点亮水平边缘。这些恰恰是 AlexNet 和 VGG 中*第一个*训练出来的卷积层最终学到的东西——因为无论后续任务是什么，好的图像模型都需要边缘和斑块检测器。

## 直接使用（Use It）

PyTorch 的 `nn.Conv2d` 把同一个操作包上了 autograd、CUDA 内核和 cuDNN 优化。形状语义完全一致。

```python
import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
print(conv)
print(f"weight shape: {tuple(conv.weight.shape)}   # (C_out, C_in, K, K)")
print(f"bias shape:   {tuple(conv.bias.shape)}")
print(f"param count:  {sum(p.numel() for p in conv.parameters())}")

x = torch.randn(8, 3, 224, 224)
y = conv(x)
print(f"\ninput  shape: {tuple(x.shape)}")
print(f"output shape: {tuple(y.shape)}")
```

把 `padding=1` 换成 `padding=0`，输出降到 222x222。把 `stride=1` 换成 `stride=2`，输出降到 112x112。就是你上面背过的那个公式。

## 交付成果（Ship It）

本课产出：

- `outputs/prompt-cnn-architect.md`——一个提示词，给定输入尺寸、参数预算和目标感受野，设计出每一步都带正确 K/S/P 的 `Conv2d` 层堆叠。
- `outputs/skill-conv-shape-calculator.md`——一个技能，逐层走查网络规格，返回每个块的输出形状、感受野和参数量。

## 练习（Exercises）

1. **（简单）** 给定 128x128 灰度输入和 `[Conv3x3(s=1,p=1), Conv3x3(s=2,p=1), Conv3x3(s=1,p=1), Conv3x3(s=2,p=1)]` 的堆叠，手工计算每一层的输出空间尺寸和感受野。用 PyTorch 的 `nn.Sequential` 搭一串哑卷积来验证。
2. **（中等）** 扩展 `conv2d_naive` 和 `conv2d_im2col`，让它们接受一个 `groups` 参数。证明 `groups=C_in=C_out` 复现了深度可分卷积（depthwise convolution），并且其参数量是 `C * K * K` 而不是 `C * C * K * K`。
3. **（困难）** 手写 `conv2d_im2col` 的反向传播：给定输出的梯度，计算 `x` 和 `w` 的梯度。在同样的输入和权重上与 `torch.autograd.grad` 对比验证。诀窍在于：im2col 的梯度是 `col2im`，而且它必须对重叠窗口做累加。

## 关键术语（Key Terms）

| 术语 | 人们常说什么 | 实际含义 |
|------|-------------|---------|
| 卷积（Convolution） | "滑动滤波器" | 在每个空间位置以共享权重应用的可学习点积；数学上是互相关（cross-correlation），但所有人都叫它卷积 |
| 卷积核 / 滤波器（Kernel / filter） | "特征检测器" | 形状为 (C_in, K, K) 的小权重张量，它与一个输入窗口的点积产生一个输出像素 |
| 步幅（Stride） | "每次跳多远" | 相邻两次卷积核放置之间的步长；步幅 2 把每个空间维度减半 |
| 填充（Padding） | "边缘补零" | 在输入周围补上的额外数值，让卷积核能对准边界像素；`same` 填充保持输出尺寸等于输入尺寸 |
| 感受野（Receptive field） | "神经元能看到多少" | 某个输出激活所依赖的原始输入区域，随深度和步幅增长 |
| im2col | "GEMM 技巧" | 把每个感受野窗口重排成列，让卷积变成一次大矩阵乘法——所有快速卷积内核的核心 |
| 深度可分卷积（Depthwise conv） | "每个通道一个卷积核" | `groups == C_in` 的卷积，每个输出通道只由对应的输入通道计算而来；MobileNet 和 ConvNeXt 的骨干 |
| 平移等变性（Translation equivariance） | "输入平移，输出平移" | 输入平移 k 像素则输出平移 k 像素的性质；共享权重免费带来它 |

## 延伸阅读（Further Reading）

- [A guide to convolution arithmetic for deep learning (Dumoulin & Visin, 2016)](https://arxiv.org/abs/1603.07285)——那些每门课都悄悄照抄的填充/步幅/空洞卷积权威图示
- [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/)——经典讲义，包括 im2col 的原始解释
- [The Annotated ConvNet (fast.ai)](https://nbviewer.org/github/fastai/fastbook/blob/master/13_convolutions.ipynb)——从手工卷积一路走到训练好的数字分类器的笔记本
- [Receptive Field Arithmetic for CNNs (Dang Ha The Hien)](https://distill.pub/2019/computing-receptive-fields/)——论文级质量的感受野计算交互式讲解
