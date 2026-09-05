# 向量、矩阵与运算（Vectors, Matrices & Operations）

> 每个神经网络都只是多绕了几步的矩阵乘法。

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 minutes

## 学习目标（Learning Objectives）

- 构建一个支持逐元素运算、矩阵乘法、转置、行列式和求逆的 Matrix 类
- 区分逐元素乘法与矩阵乘法，并说明各自适用的场景
- 仅用从零实现的 Matrix 类搭建单个全连接神经网络层（`relu(W @ x + b)`）
- 解释广播（broadcasting）规则，以及偏置相加在神经网络框架中的工作方式

## 问题所在（The Problem）

你想搭建一个神经网络。你读代码时看到了这一行：

```
output = activation(weights @ input + bias)
```

那个 `@` 是矩阵乘法。`weights` 是一个矩阵，`input` 是一个向量。如果你不知道这些运算在做什么，这行代码就是魔法；如果你知道，它就是一个层完整的前向传播，只用了三次运算。

模型处理的每张图像都是像素值组成的矩阵。每个词嵌入（embedding）都是一个向量。每个神经网络的每一层都是一次矩阵变换。不熟练掌握矩阵运算就无法构建 AI 系统，就像不理解变量就无法写代码一样。

本课将从零开始建立这种熟练度。

## 核心概念（The Concept）

### 向量：有序的数字列表（Vectors: ordered lists of numbers）

向量是一列有方向、有模长的数字。在 AI 中，向量表示数据点、特征或参数。

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

二维向量 `[3, 4]` 指向平面上的坐标 (3, 4)。它的长度（模长）是 5（3-4-5 三角形）。

### 矩阵：数字网格（Matrices: grids of numbers）

矩阵是一个二维网格，由行和列组成。一个 m x n 矩阵有 m 行 n 列。

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

在神经网络中，权重矩阵把输入向量变换成输出向量。一个有 784 个输入、128 个输出的层使用 128x784 的权重矩阵。

### 为什么形状很重要（Why shapes matter）

矩阵乘法有一条严格的规则：`(m x n) @ (n x p) = (m x p)`。内层维度必须匹配。

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

如果你在 PyTorch 里遇到过形状不匹配的报错，原因就在这里。

### 运算一览（The operations map）

| 运算 | 作用 | 在神经网络中的用途 |
|-----------|-------------|-------------------|
| 加法 | 逐元素合并 | 给输出加偏置 |
| 标量乘法 | 缩放每个元素 | 学习率 * 梯度 |
| 矩阵乘法 | 变换向量 | 层的前向传播 |
| 转置 | 行列互换 | 反向传播 |
| 行列式 | 汇总成一个数 | 检查可逆性 |
| 逆 | 撤销一次变换 | 求解线性方程组 |
| 单位矩阵 | 什么都不做的矩阵 | 初始化、残差连接 |

### 逐元素乘法与矩阵乘法（Element-wise vs matrix multiplication）

这个区别让初学者反复栽跟头。

逐元素乘法：对应位置相乘。两个矩阵必须形状相同。

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

矩阵乘法：行与列做点积。内层维度必须匹配。

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

不同的运算、不同的结果、不同的规则。

### 广播（Broadcasting）

当你把一个偏置向量加到输出矩阵上时，形状并不匹配。广播会把较小的数组拉伸到合适的形状。

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

每个现代框架都会自动做这件事。理解它能让你避免困惑：形状看起来不对，代码却跑通了。

```figure
vector-projection
```

## 动手构建（Build It）

### 第 1 步：Vector 类（Step 1: Vector class）

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### 第 2 步：带核心运算的 Matrix 类（Step 2: Matrix class with core operations）

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### 第 3 步：看看它如何运行（Step 3: See it work）

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### 第 4 步：连接到神经网络（Step 4: Connect to neural networks）

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

这就是一个单独的全连接层：`output = relu(W @ x + b)`。所有神经网络中的每个全连接层做的都是这件事。

## 用库实现（Use It）

NumPy 用更少的行数完成上面的所有事情，而且快上几个数量级。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python 中的 `@` 运算符会调用 `__matmul__`。NumPy 用 C 和 Fortran 编写的优化 BLAS 例程来实现它。同样的数学，快 100 倍。

NumPy 中的广播：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy 自动把一维偏置广播到两行上。这就是所有神经网络框架中偏置相加的工作方式。

## 交付成果（Ship It）

本课产出一个通过几何直觉教授矩阵运算的提示词，见 `outputs/prompt-matrix-operations.md`。

这里构建的 Matrix 类，是我们在第 3 阶段第 10 课要构建的迷你神经网络框架的基础。

## 练习（Exercises）

1. **验证逆矩阵。** 计算 `A @ A.inverse_2x2()`，确认得到单位矩阵。用三个不同的 2x2 矩阵试一试。行列式为零时会发生什么？

2. **实现 3x3 逆矩阵。** 扩展 Matrix 类，用伴随矩阵法（adjugate method）计算 3x3 矩阵的逆。用 NumPy 的 `np.linalg.inv` 做对照测试。

3. **搭建两层网络。** 只用你的 Matrix 类（不用 NumPy）创建一个两层神经网络：输入 (3) -> 隐藏 (4) -> 输出 (2)。初始化随机权重，运行一次前向传播，并验证所有形状都正确。

## 关键术语（Key Terms）

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 向量 | "一支箭头" | 一列有序的数字。在 AI 中：高维空间里的一个点。 |
| 矩阵 | "一张数字表格" | 一个线性变换。它把向量从一个空间映射到另一个空间。 |
| 矩阵乘法 | "把数字乘起来就行" | 第一个矩阵的每一行与第二个矩阵的每一列之间的点积。顺序很重要。 |
| 转置 | "翻个面" | 行列互换。把 m x n 矩阵变成 n x m。在反向传播中至关重要。 |
| 行列式 | "矩阵算出来的某个数" | 衡量矩阵把面积（二维）或体积（三维）放大了多少倍。为零意味着变换把某个维度压扁了。 |
| 逆 | "把矩阵撤销掉" | 能逆转该变换的矩阵。只在行列式不为零时存在。 |
| 单位矩阵 | "那个无聊的矩阵" | 相当于乘以 1 的矩阵。用于残差连接（ResNet）。 |
| 广播 | "魔法般的形状修复" | 沿缺失的维度复制，把较小的数组拉伸成与较大数组匹配的形状。 |
| 逐元素 | "普通的乘法" | 对应位置相乘。两个数组必须形状相同（或可以广播）。 |

## 延伸阅读（Further Reading）

- [3Blue1Brown：线性代数的本质](https://www.3blue1brown.com/topics/linear-algebra) - 本课涵盖的每个运算的可视化直觉
- [NumPy 广播文档](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy 遵循的确切规则
- [斯坦福 CS229 线性代数复习](http://cs229.stanford.edu/section/cs229-linalg.pdf) - 面向 ML 的线性代数简明参考
