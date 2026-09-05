# 线性代数直觉（Linear Algebra Intuition）

> 每个 AI 模型都只是戴着花哨帽子的矩阵数学。

**Type:** Learn
**Languages:** Python, Julia
**Prerequisites:** Phase 0
**Time:** ~60 minutes

## 学习目标（Learning Objectives）

- 用 Python 从零实现向量与矩阵运算（加法、点积、矩阵乘法）
- 从几何角度解释点积、投影和 Gram-Schmidt 正交化在做什么
- 使用行化简判定一组向量的线性无关性、秩与基
- 将线性代数概念与它们在 AI 中的应用联系起来：嵌入、注意力分数和 LoRA

## 问题所在（The Problem）

随手翻开任何一篇 ML 论文，第一页里就会出现向量、矩阵、点积和变换。没有线性代数直觉，这些只是符号；有了它，你就能看清神经网络到底在做什么——在空间中移动点。

你不需要成为数学家。你需要看清这些运算在几何上意味着什么，然后亲手把它们实现出来。

## 核心概念（The Concept）

### 向量既是点，也是方向（Vectors Are Points (and Directions)）

向量就是一列数字。但这些数字是有含义的——它们是空间中的坐标。

**二维向量 [3, 2]：**

| x | y | 含义 |
|---|---|-------|
| 3 | 2 | 该向量从平面上的原点 (0,0) 指向 (3, 2) |

该向量的模长为 sqrt(3^2 + 2^2) = sqrt(13)，指向右上方。

在 AI 中，向量可以表示一切：
- 一个词 → 由 768 个数字组成的向量，也就是它在嵌入（embedding）空间中的"含义"
- 一张图像 → 由数百万个像素值组成的向量
- 一个用户 → 一个表示偏好的向量

### 矩阵是变换（Matrices Are Transformations）

矩阵把一个向量变成另一个向量。它可以旋转、缩放、拉伸或投影。

```mermaid
graph LR
    subgraph Before
        A["点 A"]
        B["点 B"]
    end
    subgraph Matrix["矩阵乘法"]
        M["M（变换）"]
    end
    subgraph After
        A2["点 A'"]
        B2["点 B'"]
    end
    A --> M
    B --> M
    M --> A2
    M --> B2
```

在 AI 中，矩阵就是模型本身：
- 神经网络权重 → 把输入变换为输出的矩阵
- 注意力分数 → 决定关注什么内容的矩阵
- 嵌入矩阵 → 把词映射为向量的矩阵

### 点积衡量相似度（The Dot Product Measures Similarity）

两个向量的点积（dot product）告诉你它们有多相似。

```
a · b = a₁×b₁ + a₂×b₂ + ... + aₙ×bₙ

Same direction:      a · b > 0  (similar)
Perpendicular:       a · b = 0  (unrelated)
Opposite direction:  a · b < 0  (dissimilar)
```

搜索引擎、推荐系统和 RAG 的工作原理恰恰如此——找出点积很大的向量。

### 线性无关（Linear Independence）

如果一组向量中没有任何一个向量能写成其余向量的组合，就称它们线性无关（linearly independent）。如果 v1、v2、v3 相互独立，它们张成一个三维空间；如果其中一个是其余向量的组合，它们就只能张成一个平面。

这对 AI 为什么重要：你的特征矩阵的各列应当线性无关。如果两个特征完全相关（线性相关），模型就无法区分它们各自的影响。这会在回归中引发多重共线性（multicollinearity）——权重矩阵变得不稳定，输入的微小变化会导致输出的剧烈波动。

**具体示例：**

```
v1 = [1, 0, 0]
v2 = [0, 1, 0]
v3 = [2, 1, 0]   # v3 = 2*v1 + v2
```

v1 和 v2 相互独立——两者既不是对方的倍数，也不是对方的组合。但 v3 = 2*v1 + v2，所以 {v1, v2, v3} 是一组线性相关的向量。这三个向量都落在 xy 平面内。无论怎样组合，你都无法得到 [0, 0, 1]。你有三个向量，却只有两个自由维度。

在数据集中：如果 feature_3 = 2*feature_1 + feature_2，那么加入 feature_3 不会给模型带来任何新信息。更糟的是，它会让正规方程（normal equations）变得奇异（singular）——权重没有唯一解。

### 基与秩（Basis and Rank）

基（basis）是张成整个空间的一组数目最少的线性无关向量。基向量的个数就是空间的维数。

三维空间的标准基是 {[1,0,0], [0,1,0], [0,0,1]}。但三维空间中任意三个无关向量都能构成一个有效的基。选择基就是在选择坐标系。

矩阵的秩（rank）= 线性无关列的个数 = 线性无关行的个数。如果 rank < min(rows, cols)，矩阵就是亏秩的（rank-deficient）。这意味着：
- 方程组有无穷多解（或无解）
- 信息在变换中丢失了
- 矩阵不可逆

| 情形 | 秩 | 对 ML 的意义 |
|-----------|------|---------------------|
| 满秩（rank = min(m, n)） | 可能的最大值 | 存在唯一的最小二乘解。模型状态良好。 |
| 亏秩（rank < min(m, n)） | 低于最大值 | 特征冗余。权重有无穷多解。需要正则化。 |
| 秩为 1 | 1 | 每一列都是同一个向量的缩放副本。所有数据都落在一条直线上。 |
| 接近亏秩（奇异值很小） | 数值上偏低 | 矩阵病态（ill-conditioned）。输入的微小噪声会导致输出的巨大变化。使用 SVD 截断或岭回归。 |

### 投影（Projection）

把向量 **a** 投影到向量 **b** 上，得到的是 **a** 在 **b** 方向上的分量：

```
proj_b(a) = (a dot b / b dot b) * b
```

残差 (a - proj_b(a)) 与 b 垂直。这种正交分解是最小二乘拟合的基础。

投影在 ML 中无处不在：
- 线性回归最小化观测点到列空间的距离——其解本身就是一次投影
- PCA 把数据投影到方差最大的方向上
- transformer 中的注意力机制计算查询（query）在键（key）上的投影

```mermaid
graph LR
    subgraph Projection["a 在 b 上的投影"]
        direction TB
        O["原点"] --> |"b（方向）"| B["b"]
        O --> |"a（原始）"| A["a"]
        O --> |"proj_b(a)"| P["投影"]
        A -.-> |"残差（垂直）"| P
    end
```

**示例：** a = [3, 4], b = [1, 0]

proj_b(a) = (3*1 + 4*0) / (1*1 + 0*0) * [1, 0] = 3 * [1, 0] = [3, 0]

投影把 y 分量丢掉了。这就是最简单的降维——把你不需要的方向扔掉。

### Gram-Schmidt 正交化（Gram-Schmidt Process）

把任意一组无关向量转换成标准正交（orthonormal）基。标准正交意味着每个向量长度为 1，且两两垂直。

算法如下：
1. 取第一个向量，将其归一化
2. 取第二个向量，减去它在第一个向量上的投影，再归一化
3. 取第三个向量，减去它在之前所有向量上的投影，再归一化
4. 对其余向量重复这一过程

```
Input:  v1, v2, v3, ... (linearly independent)

u1 = v1 / |v1|

w2 = v2 - (v2 dot u1) * u1
u2 = w2 / |w2|

w3 = v3 - (v3 dot u1) * u1 - (v3 dot u2) * u2
u3 = w3 / |w3|

Output: u1, u2, u3, ... (orthonormal basis)
```

这就是 QR 分解的内部原理。Q 是标准正交基，R 保存投影系数。QR 分解用于：
- 求解线性方程组（比高斯消元更稳定）
- 计算特征值（QR 算法）
- 最小二乘回归（标准数值方法）

```figure
eigen-directions
```

## 动手构建（Build It）

### 第 1 步：从零实现向量（Python）（Step 1: Vectors from scratch (Python)）

```python
class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def __repr__(self):
        return f"Vector({self.components})"


a = Vector([1, 2, 3])
b = Vector([4, 5, 6])

print(f"a + b = {a + b}")
print(f"a · b = {a.dot(b)}")
print(f"|a| = {a.magnitude():.4f}")
print(f"cosine similarity = {a.cosine_similarity(b):.4f}")
```

### 第 2 步：从零实现矩阵（Python）（Step 2: Matrices from scratch (Python)）

```python
class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def __repr__(self):
        return f"Matrix({self.rows})"


rotation_90 = Matrix([[0, -1], [1, 0]])
point = Vector([3, 1])

rotated = rotation_90 @ point
print(f"Original: {point}")
print(f"Rotated 90°: {rotated}")
```

### 第 3 步：为什么这对 AI 很重要（Step 3: Why this matters for AI）

```python
import random

random.seed(42)
weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
input_vector = Vector([1.0, 0.5, -0.3])

output = weights @ input_vector
print(f"Input (3D): {input_vector}")
print(f"Output (2D): {output}")
print("This is what a neural network layer does -- matrix multiplication.")
```

### 第 4 步：Julia 版本（Step 4: Julia version）

```julia
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

println("a + b = ", a + b)
println("a · b = ", a ⋅ b)       # Julia supports unicode operators
println("|a| = ", √(a ⋅ a))
println("cosine = ", (a ⋅ b) / (√(a ⋅ a) * √(b ⋅ b)))

# Matrix-vector multiplication
W = [0.1 -0.2 0.3; 0.4 0.5 -0.1]
x = [1.0, 0.5, -0.3]
println("Wx = ", W * x)
println("This is a neural network layer.")
```

### 第 5 步：从零实现线性无关与投影（Python）（Step 5: Linear independence and projection from scratch (Python)）

```python
def is_linearly_independent(vectors):
    n = len(vectors)
    dim = len(vectors[0].components)
    mat = Matrix([v.components[:] for v in vectors])
    rows = [row[:] for row in mat.rows]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n


def project(a, b):
    scalar = a.dot(b) / b.dot(b)
    return Vector([scalar * x for x in b.components])


def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = project(w, u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


v1 = Vector([1, 0, 0])
v2 = Vector([1, 1, 0])
v3 = Vector([1, 1, 1])
basis = gram_schmidt([v1, v2, v3])
for i, u in enumerate(basis):
    print(f"u{i+1} = {u}")
    print(f"  |u{i+1}| = {u.magnitude():.6f}")

print(f"u1 · u2 = {basis[0].dot(basis[1]):.6f}")
print(f"u1 · u3 = {basis[0].dot(basis[2]):.6f}")
print(f"u2 · u3 = {basis[1].dot(basis[2]):.6f}")
```

## 用库实现（Use It）

现在用 NumPy 实现同样的事情——这才是你实际会用到的：

```python
import numpy as np

a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6], dtype=float)

print(f"a + b = {a + b}")
print(f"a · b = {np.dot(a, b)}")
print(f"|a| = {np.linalg.norm(a):.4f}")
print(f"cosine = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")

W = np.random.randn(2, 3) * 0.1
x = np.array([1.0, 0.5, -0.3])
print(f"Wx = {W @ x}")
```

### 用 NumPy 计算秩、投影与 QR（Rank, Projection, and QR with NumPy）

```python
import numpy as np

A = np.array([[1, 2], [2, 4]])
print(f"Rank: {np.linalg.matrix_rank(A)}")

a = np.array([3, 4])
b = np.array([1, 0])
proj = (np.dot(a, b) / np.dot(b, b)) * b
print(f"Projection of {a} onto {b}: {proj}")

Q, R = np.linalg.qr(np.random.randn(3, 3))
print(f"Q is orthogonal: {np.allclose(Q @ Q.T, np.eye(3))}")
print(f"R is upper triangular: {np.allclose(R, np.triu(R))}")
```

### PyTorch——张量就是带自动微分的向量（PyTorch -- Tensors Are Vectors with Autodiff）

```python
import torch

x = torch.randn(3, requires_grad=True)
y = torch.tensor([1.0, 0.0, 0.0])

similarity = torch.dot(x, y)
similarity.backward()

print(f"x = {x.data}")
print(f"y = {y.data}")
print(f"dot product = {similarity.item():.4f}")
print(f"d(dot)/dx = {x.grad}")
```

点积对 x 的梯度就是 y 本身。PyTorch 自动算出了这个结果。神经网络中的每个运算都是由这样的运算构建的——矩阵乘法、点积、投影——而自动微分（autodiff）会沿着它们追踪梯度。

你刚刚从零实现了 NumPy 一行代码就能做到的事情。现在你明白了幕后到底发生了什么。

## 交付成果（Ship It）

本课产出：
- `outputs/prompt-linear-algebra-tutor.md`——一个让 AI 助手通过几何直觉教授线性代数的提示词

## 关联（Connections）

本课的每个概念都对应着现代 AI 的具体部分：

| 概念 | 出现的地方 |
|---------|------------------|
| 点积 | transformer 中的注意力分数、RAG 中的余弦相似度 |
| 矩阵乘法 | 每个神经网络层、每个线性变换 |
| 线性无关 | 特征选择、避免多重共线性 |
| 秩 | 判断方程组是否可解、LoRA（低秩适配） |
| 投影 | 线性回归（投影到列空间）、PCA |
| Gram-Schmidt / QR | 数值求解器、特征值计算 |
| 标准正交基 | 稳定的数值计算、白化变换 |

LoRA 值得单独一提。它通过把权重更新分解为低秩矩阵来微调大语言模型。LoRA 不更新 4096x4096 的权重矩阵（16M 参数），而是更新两个大小分别为 4096x16 和 16x4096 的矩阵（131K 参数）。秩为 16 的约束意味着 LoRA 假设权重更新位于完整 4096 维空间中的一个 16 维子空间里。这就是线性代数在实实在在地发挥作用。

## 练习（Exercises）

1. 实现 `Vector.angle_between(other)`，返回两个向量之间以度为单位的角度
2. 创建一个将 x 坐标扩大 2 倍、y 坐标扩大 3 倍的二维缩放矩阵，然后把它作用到向量 [1, 1] 上
3. 给定 5 个随机的类词向量（维度 50），用余弦相似度找出最相似的两个
4. 验证 Gram-Schmidt 的输出确实是标准正交的：检查每一对向量的点积为 0，且每个向量的模长为 1
5. 创建一个秩为 2 的 3x3 矩阵，并用 `rank()` 方法验证。然后解释这些列张成了什么几何对象。
6. 把向量 [1, 2, 3] 投影到 [1, 1, 1] 上。结果在几何上代表什么？

## 关键术语（Key Terms）

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 向量 | "一支箭头" | 表示 n 维空间中一个点或方向的一列数字 |
| 矩阵 | "一张数字表格" | 把向量从一个空间映射到另一个空间的变换 |
| 点积 | "相乘再求和" | 衡量两个向量对齐程度的量——相似度搜索的核心 |
| 嵌入 | "某种 AI 魔法" | 表示某个事物（词、图像、用户）含义的向量 |
| 线性无关 | "它们不重叠" | 集合中没有任何向量可以写成其余向量的组合 |
| 秩 | "有多少个维度" | 矩阵中线性无关列（或行）的个数 |
| 投影 | "影子" | 一个向量在另一个向量方向上的分量 |
| 基 | "坐标轴" | 张成该空间的一组数目最少的无关向量 |
| 标准正交 | "相互垂直的单位向量" | 两两垂直且长度都为 1 的向量 |
