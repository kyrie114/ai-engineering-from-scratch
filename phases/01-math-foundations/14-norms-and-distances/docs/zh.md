# 范数与距离（Norms and Distances）

> 你的距离函数定义了什么算"相似"。选错了，下游的一切都会跟着出错。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**Time:** ~90 minutes

## 学习目标（Learning Objectives）

- 从零实现 L1、L2、余弦、马氏距离（Mahalanobis）、Jaccard 和编辑距离函数
- 为给定的机器学习任务选择合适的距离度量，并解释其他选项为何不行
- 把 L1 和 L2 范数与 LASSO、Ridge 正则化及其几何约束区域联系起来
- 演示同一数据集在不同度量下会得到不同的最近邻

## 问题（The Problem）

你有两个向量。它们可能是词嵌入（word embedding），可能是用户画像，也可能是像素数组。你需要知道：它们有多接近？

答案完全取决于你选了哪种距离函数。两个数据点在一种度量下是最近邻，在另一种度量下可能相距甚远。你的 KNN 分类器、推荐引擎、向量数据库、聚类算法、损失函数——全都依赖这个选择。选错了，模型就会优化错误的目标。

没有普适的最优距离。L2 适合空间数据，余弦相似度统治 NLP，Jaccard 处理集合，编辑距离处理字符串，马氏距离考虑相关性，Wasserstein 搬运概率质量。每一种都编码了关于"相似"含义的不同假设。

本课从零构建每一个主要距离函数，告诉你各自在什么时候是合适的工具，并演示同一份数据在不同度量下会得到完全不同的最近邻。

## 概念（The Concept）

### 范数：度量向量的大小（Norms: measuring vector magnitude）

范数（norm）度量向量的"大小"。两个向量之间的任何距离函数都可以写成它们之差的范数：d(a, b) = ||a - b||。所以理解范数就是理解距离。

### L1 范数（曼哈顿距离）（L1 Norm (Manhattan distance)）

L1 范数对所有分量的绝对值求和。

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

它被叫作曼哈顿距离，是因为它衡量的是你在一个只能沿轴行走的城市街区网格里要走多远。不能走对角线。

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

什么时候用 L1：
- 高维稀疏数据（文本特征、one-hot 编码）
- 需要对离群点鲁棒时（单个巨大的差异不会主导结果）
- 特征选择问题（L1 正则化会促进稀疏性）

与 L1 正则化（Lasso）的联系：在损失函数中加入 ||w||_1 会惩罚权重绝对值之和，把小的权重压到恰好为零，从而实现自动特征选择。L1 惩罚在权重空间中形成菱形约束区域，而菱形的角正好落在轴上——那里的一些权重为零。

与损失函数的联系：平均绝对误差（MAE）就是预测值与目标值之间 L1 距离的平均。它对所有误差线性惩罚，因此与 MSE 相比对离群点更鲁棒。

### L2 范数（欧几里得距离）（L2 Norm (Euclidean distance)）

L2 范数是直线距离，即各分量平方和的平方根。

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

这就是几何课上学过的距离，n 维空间里的勾股定理。

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

什么时候用 L2：
- 低到中等维度的连续数据
- 各特征尺度相近时
- 物理距离（空间数据、传感器读数）
- 像素级别的图像相似度

与 L2 正则化（Ridge）的联系：在损失函数中加入 ||w||_2^2 会惩罚大的权重。与 L1 不同，它不会把权重压到零，而是按比例把所有权重向零收缩。L2 惩罚形成圆形约束区域，轴上没有角点，因此权重会变小，但很少恰好为零。

与损失函数的联系：均方误差（MSE）是 L2 距离平方的平均。平方惩罚让大误差比小误差受到更重的惩罚。

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp 范数：一般化家族（Lp Norms: the general family）

L1 和 L2 都是 Lp 范数的特例：

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

不同的 p 值产生不同形状的"单位球"（到原点距离为 1 的所有点的集合）：

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-infinity 范数（切比雪夫距离）（L-infinity Norm (Chebyshev distance)）

当 p 趋于无穷大时，Lp 范数收敛到分量绝对值的最大值。

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

两点间的距离由差异最大的那个维度决定，其余维度全部被忽略。

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

什么时候用 L-infinity：
- 任一维度的最坏偏差都很重要时
- 棋盘游戏（国际象棋里的国王走的是 L-infinity 距离：向任意方向走一步都算 1）
- 制造公差（每个维度都必须在规格范围内）

### 余弦相似度与余弦距离（Cosine Similarity and Cosine Distance）

余弦相似度（cosine similarity）度量两个向量之间的夹角，忽略它们的大小。

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

取值范围从 -1（方向相反）到 +1（方向相同）。垂直向量的余弦相似度为 0。

余弦距离把它转换成距离：cosine_distance = 1 - cosine_similarity。取值范围从 0（方向相同）到 2（方向相反）。

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

为什么余弦相似度在 NLP 和嵌入中占主导：对文本而言，文档长度不应影响相似度。一篇关于猫的文档即使是另一篇同类文档的两倍长，两者也应当是"相似"的。余弦相似度忽略大小（长度），只关心方向。词分布相同但长度不同的两篇文档指向同一方向，余弦相似度为 1.0。

什么时候用余弦相似度：
- 文本相似度（TF-IDF 向量、词嵌入、句嵌入）
- 任何"大小是噪声、方向是信号"的领域
- 推荐系统（用户偏好向量）
- 嵌入搜索（向量数据库几乎都用余弦或点积）

### 点积相似度对比余弦相似度（Dot Product Similarity vs Cosine Similarity）

两个向量的点积是：

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

余弦相似度是点积除以两个大小后的归一化结果。当两个向量都已单位化（大小为 1）时，点积与余弦相似度完全相同。

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

两者不同之处：点积包含大小信息，大小更大的向量会得到更高的点积得分。这在某些检索系统里很重要——你想让"热门"条目排得更靠前。大小在这里充当了隐式的质量或重要性信号。

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

实践建议：
- 想要纯方向相似度时，用余弦相似度
- 大小带有有意义的信息时，用点积
- 许多向量数据库（Pinecone、Weaviate、Qdrant）允许你在两者之间选择
- 如果你的嵌入已经做了 L2 归一化，选哪个都一样

### 马氏距离（Mahalanobis Distance）

欧几里得距离对所有维度一视同仁。但如果特征之间有相关性或尺度不同，L2 会给出误导性结果。

马氏距离（Mahalanobis distance）考虑了数据的协方差结构。

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

其中 S 是数据的协方差矩阵。

直观理解：马氏距离先对数据做去相关和归一化（白化，whitening），再在变换后的空间里计算 L2 距离。如果 S 是单位矩阵（特征不相关且方差为 1），马氏距离就退化为欧几里得距离。

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

什么时候用马氏距离：
- 离群点检测（离均值马氏距离很大的点就是离群点）
- 特征尺度不一且有相关性时的分类
- 有足够数据估计出可靠协方差矩阵时
- 制造业质量控制（多变量过程监控）

### Jaccard 相似度（用于集合）（Jaccard Similarity (for sets)）

Jaccard 相似度度量两个集合的重叠程度。

```
J(A, B) = |A intersect B| / |A union B|
```

取值范围从 0（无重叠）到 1（集合相同）。Jaccard 距离 = 1 - Jaccard 相似度。

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

什么时候用 Jaccard：
- 比较标签、类别或特征的集合
- 基于词是否出现（而非频率）的文档相似度
- 近重复检测（用 MinHash 近似 Jaccard）
- 比较二值特征向量（出现/缺失数据）
- 评估分割模型（IoU（Intersection over Union）就是 Jaccard）

### 编辑距离（Levenshtein 距离）（Edit Distance (Levenshtein Distance)）

编辑距离（edit distance）统计把一个字符串变成另一个所需的最少单字符操作数。操作包括：插入、删除或替换。

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

用动态规划计算。填一张矩阵，其中条目 (i, j) 是字符串 A 的前 i 个字符与字符串 B 的前 j 个字符之间的编辑距离。

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

什么时候用编辑距离：
- 拼写检查与纠正
- DNA 序列比对（使用加权操作）
- 模糊字符串匹配
- 脏文本数据去重

### KL 散度（不是距离，却常被当作距离用）（KL Divergence (not a distance, but used like one)）

KL 散度（KL divergence）度量一个概率分布与另一个的差异。第 09 课讲过它，但它也属于这里的讨论，因为尽管它不是距离，人们却常把它当距离用。

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

关键性质：KL 散度不是对称的。

```
D_KL(P || Q) != D_KL(Q || P)
```

这意味着它不满足距离度量的基本要求，也不满足三角不等式。它是散度（divergence），不是距离。

前向 KL（D_KL(P || Q)）是"求均值"（mean-seeking）的：Q 试图覆盖 P 的所有模态。
反向 KL（D_KL(Q || P)）是"求模态"（mode-seeking）的：Q 聚焦于 P 的单个模态。

你会在这些地方见到 KL 散度：
- VAE（ELBO 中的 KL 项把潜变量分布拉向先验）
- 知识蒸馏（学生模型试图匹配教师模型的分布）
- RLHF（KL 惩罚让微调后的模型贴近基础模型）
- 策略梯度方法（约束策略更新）

### Wasserstein 距离（推土机距离）（Wasserstein Distance (Earth Mover's Distance)）

Wasserstein 距离度量把一个概率分布变成另一个所需的最小"工作量"。可以这么想：如果一个分布是一堆土，另一个是一个坑，你要搬多少土、搬多远？

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

对一维分布，它简化为累积分布函数绝对差的积分：

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

Wasserstein 为什么重要：
- 它是真正的度量（对称，满足三角不等式）
- 即使分布不重叠也能提供梯度（KL 散度会趋于无穷）
- 这一性质使它成为 Wasserstein GAN（WGAN）的核心，WGAN 解决了原始 GAN 的训练不稳定问题

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

什么时候用 Wasserstein：
- GAN 训练（WGAN、WGAN-GP）
- 比较可能不重叠的分布
- 最优传输问题
- 图像检索（比较颜色直方图）

### 为什么不同任务需要不同的距离（Why Different Tasks Need Different Distances）

| 任务 | 最合适的距离 | 原因 |
|------|--------------|-----|
| 文本相似度 | 余弦 | 大小是噪声，方向才是语义 |
| 图像像素比较 | L2 | 空间关系重要，特征尺度相近 |
| 稀疏高维特征 | L1 | 鲁棒，不会放大罕见的大差异 |
| 集合重叠（标签、类别） | Jaccard | 数据天然是集合而非向量 |
| 字符串匹配 | 编辑距离 | 操作对应人工编辑的直觉 |
| 离群点检测 | 马氏距离 | 考虑特征相关性和尺度 |
| 分布比较 | KL 散度 | 度量用 Q 代替 P 损失的信息 |
| GAN 训练 | Wasserstein | 分布不重叠也能提供梯度 |
| 嵌入（向量数据库） | 余弦或点积 | 嵌入的训练目标就是把语义编进方向 |
| 推荐 | 点积 | 大小可以编码热度或置信度 |
| DNA 序列 | 加权编辑距离 | 替换代价随碱基对不同而变化 |
| 制造业质检 | L-infinity | 任一维度的最坏偏差都很重要 |

### 与损失函数的联系（Connection to Loss Functions）

损失函数就是作用在预测值与目标值之上的距离函数。

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### 与正则化的联系（Connection to Regularization）

正则化在损失函数中加入权重的范数惩罚。

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

为什么 L1 产生稀疏性而 L2 不会：想象 2D 权重空间中的约束区域。L1 是菱形，L2 是圆形。损失函数的等高线（椭圆）最有可能在菱形的角上与它相切，那里的一个权重为零；而与圆相切在光滑点上，那里两个权重都非零。

### 最近邻搜索（Nearest Neighbor Search）

每种距离函数都对应一个最近邻搜索问题：给定查询点，找出数据集中离它最近的点。

在 n 个点、d 维的数据集中，精确最近邻搜索每次查询的复杂度是 O(n * d)。数据集很大时这就太慢了。

近似最近邻（ANN）算法用少量精度换取巨大的速度提升：

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW（Hierarchical Navigable Small World，分层可导航小世界图）是现代向量数据库中的主流算法。它构建一个多层图，每个节点连接到它的近似最近邻。搜索从顶层开始（稀疏、长跳），逐层下降到底层（稠密、短跳）。

```figure
norm-unit-balls
```

## 动手构建（Build It）

### 第 1 步：全部范数与距离函数（Step 1: All norm and distance functions）

完整实现见 `code/distances.py`。每个函数都只用基础 Python 数学从零构建。

### 第 2 步：同一数据，不同距离，不同邻居（Step 2: Same data, different distances, different neighbors）

`distances.py` 中的演示创建一个数据集、选一个查询点，展示最近邻如何随距离度量而变化。L1 下"最近"的点，在 L2 或余弦下未必最近。

### 第 3 步：嵌入相似度搜索（Step 3: Embedding similarity search）

代码里包含一个模拟的嵌入相似度搜索，分别用余弦相似度和 L2 距离找出与查询最相似的"文档"，展示两种排序可能不同。

## 使用它（Use It）

最常见的实际用途：在向量数据库中查找相似条目。

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

当你调用 `model.encode(text)` 然后去向量数据库里搜索时，幕后发生的就是这些。嵌入模型把文本映射成向量，向量数据库计算你的查询向量与每条存储向量的余弦相似度（或点积），并用 ANN 算法避免逐一检查所有向量。

## 练习（Exercises）

1. 计算 (1, 2, 3) 与 (4, 0, 6) 之间的 L1、L2 和 L-infinity 距离。验证对任意点对 L-inf <= L2 <= L1 恒成立，并证明这个顺序为什么有保证。

2. 构造两个余弦相似度高（> 0.9）但 L2 距离大（> 10）的向量，从几何上解释发生了什么。再构造两个余弦相似度低（< 0.3）但 L2 距离小（< 0.5）的向量。

3. 实现一个函数，输入数据集和查询点，分别返回 L1、L2、余弦和马氏距离下的最近邻。找一个数据集，让这四种度量对哪个点最近各执一词。

4. 用 CDF 方法手算 [0.5, 0.5, 0, 0] 与 [0, 0, 0.5, 0.5] 之间的 Wasserstein 距离。再算 [0.25, 0.25, 0.25, 0.25] 与 [0, 0, 0.5, 0.5] 之间的。哪个更大？为什么？

5. 实现 MinHash 来近似 Jaccard 相似度。生成 100 个随机集合，计算所有集合对的精确 Jaccard，再分别用 50、100 和 200 个哈希函数比较 MinHash 近似结果。画出近似误差曲线。

## 关键术语（Key Terms）

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 范数（norm） | "向量的大小" | 把向量映射为非负标量的函数，满足三角不等式、绝对齐次性，且仅对零向量取零 |
| L1 范数 | "曼哈顿距离" | 分量绝对值之和。在优化中产生稀疏性，对离群点鲁棒 |
| L2 范数 | "欧几里得距离" | 分量平方和的平方根。欧几里得空间中的直线距离 |
| Lp 范数 | "广义范数" | 分量绝对值的 p 次幂之和的 p 次方根。L1 和 L2 是特例 |
| L-infinity 范数 | "最大范数"或"切比雪夫距离" | 分量绝对值的最大值。Lp 在 p 趋于无穷时的极限 |
| 余弦相似度 | "向量间的夹角" | 点积除以两个大小后的归一化。取值 -1 到 +1，忽略向量长度 |
| 余弦距离 | "1 减余弦相似度" | 把余弦相似度转换为距离。取值 0 到 2 |
| 点积 | "未归一化的余弦" | 逐分量乘积之和。等于余弦相似度乘以两个大小 |
| 马氏距离 | "考虑相关性的距离" | 用数据协方差矩阵白化（去相关并归一化）后的空间中的 L2 距离 |
| Jaccard 相似度 | "集合重叠度" | 交集大小除以并集大小。用于集合，不是向量 |
| 编辑距离 | "Levenshtein 距离" | 把一个字符串变成另一个所需的最少插入、删除和替换次数 |
| KL 散度 | "分布之间的距离" | 并非真正的距离（不对称）。度量用 Q 编码 P 额外多出的比特数 |
| Wasserstein 距离 | "推土机距离" | 把质量从一个分布搬运到另一个所需的最小工作量。是真正的度量 |
| 近似最近邻 | "ANN 搜索" | 以远快于精确搜索的速度找出近似最近点的算法（HNSW、LSH、IVF） |
| HNSW | "向量数据库的招牌算法" | 分层可导航小世界图。用于快速近似最近邻搜索的多层图 |
| L1 正则化 | "Lasso" | 在损失中加入权重的 L1 范数。把权重压向零（稀疏性） |
| L2 正则化 | "Ridge"或"权重衰减" | 在损失中加入权重的 L2 范数平方。把权重向零收缩但不产生稀疏性 |
| Elastic Net | "L1 + L2" | 结合 L1 与 L2 正则化。处理相关特征组时比单独用任何一个都好 |

## 延伸阅读（Further Reading）

- [FAISS：高效相似度搜索库](https://github.com/facebookresearch/faiss)——Meta 的十亿级 ANN 搜索库
- [Wasserstein GAN（Arjovsky 等，2017）](https://arxiv.org/abs/1701.07875)——把推土机距离引入 GAN 的论文
- [局部敏感哈希（Indyk & Motwani，1998）](https://dl.acm.org/doi/10.1145/276698.276876)——奠基性的 ANN 算法
- [词表示的高效估计（Mikolov 等，2013）](https://arxiv.org/abs/1301.3781)——Word2Vec，余弦相似度自此成为嵌入的默认度量
- [sklearn.neighbors 文档](https://scikit-learn.org/stable/modules/neighbors.html)——scikit-learn 中距离度量与近邻算法的实用指南
