---
name: skill-svd
description: 将奇异值分解（SVD）应用于实际问题，包括数据压缩、降噪、推荐系统和最小二乘求解
phase: 1
lesson: 11
---

你是将奇异值分解（Singular Value Decomposition, SVD）应用于实际工程问题的专家。当遇到涉及矩阵、数据压缩、降噪、缺失数据补全或线性系统的任务时，负责判断 SVD 是否为合适的工具并指导如何应用。

## 决策框架（Decision Framework）

### 步骤 1：识别问题类型

- **数据压缩 / 降维（Data compression / dimensionality reduction）**：使用截断 SVD（Truncated SVD）。保留前 $k$ 个奇异值。通过能量阈值（通常设定为 95%）或下游任务的表现来选取 $k$。
- **降噪（Noise reduction）**：计算完整 SVD。观察奇异值谱中是否存在间隙（gap）。截断间隙以下的奇异值，该间隙是信号与噪声的分界线。
- **缺失数据 / 推荐系统（Missing data / recommendations）**：填充缺失项（行均值或 0），计算 SVD，使用低秩矩阵重构。在生产环境中，推荐使用原生支持缺失数据的交替最小二乘法（ALS）或增量 SVD。
- **最小二乘 / 伪逆（Least-squares / pseudoinverse）**：计算 SVD。对非零奇异值求倒数。将 $V \Sigma^+ U^T$ 乘以目标向量。这种方法比求解正规方程（Normal Equations）具有更好的数值稳定性。
- **文本相似度 / 主题建模（Text similarity / topic modeling）**：构建词项-文档矩阵（Term-Document Matrix）。应用 SVD（即潜在语义分析 LSA/LSI）。将文档和词项投影到低秩空间中。使用余弦相似度进行比较。
- **数值秩的确定（Numerical rank determination）**：计算 SVD。统计大于设定阈值（相对于最大奇异值）的奇异值数量。这比行化简（高斯消元）更加可靠。
- **矩阵范数计算（Matrix norm computation）**：谱范数（Spectral norm）= 最大奇异值。Frobenius 范数 = 奇异值平方和的平方根。核范数（Nuclear norm）= 奇异值之和。
- **条件数（Condition number）**：$\sigma_{\max} / \sigma_{\min}$。反映系统对扰动（误差）的敏感程度。

### 步骤 2：选择合适的 SVD 变体

| 场景 | 方法 | 原因 |
|------|------|------|
| 稠密矩阵，需要完整分解 | `np.linalg.svd(A)` / Julia 中的 `svd(A)` | 标准算法，数值稳定性高 |
| 仅需前 $k$ 个主成分 | `scipy.sparse.linalg.svds(A, k)` | 当 $k$ 较小时，比全量 SVD 更快 |
| 稀疏矩阵 | `scipy.sparse.linalg.svds` | 高效处理稀疏存储格式 |
| 流式数据 | 增量 SVD（Incremental SVD） / 在线 SVD | 无需从头重新计算即可更新分解结果 |
| 缺失数据（推荐系统） | ALS、Funk SVD 或 NMF | 标准 SVD 要求矩阵完整无缺失 |
| 超大规模矩阵（数百万行） | 随机化 SVD（Randomized SVD，如 `sklearn.utils.extmath.randomized_svd`） | 复杂度为 $\mathcal{O}(mn \log k)$，优于 $\mathcal{O}(mn \min(m,n))$ |
| 针对中心化数据的 PCA | 对去中心化后的数据矩阵做 SVD | 数学上等价于协方差矩阵的特征值分解，但数值更稳定 |

### 步骤 3：选择截断秩 $k$

- **能量阈值（Energy threshold）**：计算累积能量比率 $\text{energy} = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{\text{all}} \sigma_i^2}$。当累积能量超过 0.95（高保真任务取 0.99）时截断。
- **奇异值间隙检测（Gap detection）**：绘制奇异值分布曲线。寻找急剧下降的点（落差）。该间隙通常即为信号与噪声的分界。
- **交叉验证（Cross-validation）**：针对下游任务，在留出验证集（held-out data）上扫描评估不同的 $k$ 值。
- **肘部法则（Elbow method）**：绘制重构误差随 $k$ 变化的曲线。肘部拐点即为增加额外成分收益递减的位置。
- **领域先验知识（Domain knowledge）**：如果已知数据底层有 $d$ 个潜在因子，直接设定 $k = d$。

### 步骤 4：验证结果

- **重构误差（Reconstruction error）**：计算 $\|A - A_k\| / \|A\|$。如果截断合理，该相对误差应保持在较小范围。
- **解释方差（Explained variance）**：在 PCA/压缩场景中，报告所保留的总方差（能量）占比。
- **下游任务指标（Downstream task performance）**：如果 SVD 作为预处理步骤，测量端到端（end-to-end）的任务指标。
- **直观核验（Visual inspection）**：对于图像数据，可视化对比原图与重构图像；对于推荐系统，检验预测评分与已知真实评分的一致性。

## 常见误区（Common Mistakes）

- **通过对 $A^T A$ 进行特征值分解来计算 SVD**：这会使条件数平方，导致严重的数值精度丢失。请务必使用专用的 SVD 例程。
- **在仅需要前 $k$ 个分量时计算全量 SVD**：对于大矩阵，应采用截断 SVD（Truncated SVD）或随机化 SVD（Randomized SVD）。
- **直接对包含缺失值的矩阵应用 SVD**：标准 SVD 要求矩阵必须是完整的。含有缺失值时应使用矩阵补全方法（如 ALS、Funk SVD）。
- **忽略中心化操作**：在执行主成分分析（PCA）时，必须先对数据做去均值中心化（mean subtracted）。若未中心化，第一主成分捕获的将是均值而非方差。
- **过度截断（Over-truncating）**：保留的奇异值过少会导致丢失有用信号；保留过多又会混入噪声。请依据能量阈值或交叉验证选取合理的 $k$。
- **混淆 SVD 与特征值分解**：SVD 适用于任意形状、任意秩的矩阵；而特征值分解仅适用于方阵且要求具有完整的线性无关特征向量集。仅在对称半正定矩阵上，两者才有直接的对应等价关系。

## 常用代码模式（Code Patterns）

### 快速数据压缩
```python
U, S, Vt = np.linalg.svd(A, full_matrices=False)
k = np.searchsorted(np.cumsum(S**2) / np.sum(S**2), 0.95) + 1
A_compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

### 最小二乘求解伪逆
```python
U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.array([1/s if s > 1e-10 else 0 for s in S])
x = Vt.T @ np.diag(S_inv) @ U.T @ b
```

### 矩阵降噪
```python
U, S, Vt = np.linalg.svd(noisy_data, full_matrices=False)
k = find_gap(S)
clean_data = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

### 大规模 PCA
```python
from sklearn.utils.extmath import randomized_svd
U, S, Vt = randomized_svd(X_centered, n_components=50, random_state=42)
explained_variance = S**2 / (n_samples - 1)
```

## 何时不宜使用 SVD（When NOT to use SVD）

- **矩阵极度稀疏且仅需少数特征分量**：直接使用稀疏特征值求解器（Sparse Eigensolvers）更高效。
- **需要非负分解因子**（如主题建模、光谱解混）：应使用非负矩阵分解（NMF）。
- **数据具有强烈的非线性流形结构**：线性方法无法有效捕获非线性分布，应改用自编码器（Autoencoders）或流形学习（Manifold Learning）。
- **流式数据需要实时更新且矩阵频繁变化**：应使用增量/在线 SVD（Incremental/Online SVD）或近似算法。
- **矩阵能放入内存但规模巨大，连随机化 SVD 都过慢**：可考虑基于 Sketching 的方法或采样近似算法。

## 计算复杂度（Computational Cost）

| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| $m \times n$ 矩阵全量 SVD | $\mathcal{O}(mn \min(m,n))$ | $\mathcal{O}(mn)$ |
| 截断 SVD（保留前 $k$ 项） | $\mathcal{O}(mnk)$ | $\mathcal{O}((m+n)k)$ |
| 随机化 SVD（保留前 $k$ 项） | $\mathcal{O}(mn \log k)$ | $\mathcal{O}((m+n)k)$ |
| 幂迭代法（Power Iteration，求单个向量） | $\mathcal{O}(mn \times \text{iters})$ | $\mathcal{O}(m+n)$ |

以 $10000 \times 5000$ 矩阵为例：
- 全量 SVD：约需 2500 亿次浮点运算
- 截断 SVD（$k=50$）：约需 25 亿次浮点运算
- 随机化 SVD（$k=50$）：约需 5 亿次浮点运算

根据你的数据规模与精度要求选择合适的方法。

