# 无监督学习（Unsupervised Learning）

> 没有标签，没有老师。算法自己发现结构。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1 (Norms & Distances, Probability & Distributions), Phase 2 Lessons 1-6
**Time:** ~90 minutes

## 学习目标（Learning Objectives）

- 从零实现 K-Means、DBSCAN 和高斯混合模型（Gaussian Mixture Model），并比较它们的聚类行为
- 使用轮廓系数（silhouette score）和肘部法则（elbow method）评估聚类质量，选出最优的 K
- 解释 DBSCAN 何时优于 K-Means，并指出哪种算法能处理非球形簇和离群点
- 构建一个基于聚类方法的异常检测流水线，标记偏离正常模式的点

## 问题（The Problem）

到目前为止，每一课机器学习内容都假设数据有标签："这是输入，这是正确输出。"现实世界中标签很昂贵。医院有数百万份病历，却没有人逐条标注疾病类别。电商网站有数百万个用户会话，却没有人手工标注客户群。安全团队有网络日志，但没有人标记每一条异常。

无监督学习（unsupervised learning）在没有人告诉它要找什么的情况下发现模式。它把相似的数据点聚在一起，发现隐藏的结构，并让异常浮出水面。如果说监督学习是拿着答案解析学习教材，那么无监督学习就是盯着原始数据，直到模式自己显现。

难点在于：没有标签，你就无法直接衡量"对"还是"错"。你需要不同的工具来判断算法发现的结构是否有意义。

## 核心概念（The Concept）

### 聚类：把相似的东西归为一组（Clustering: Grouping Similar Things Together）

聚类（clustering）把每个数据点分到一个组（簇，cluster）中，使同一组内的点彼此之间比与其他组的点更相似。永恒的问题是："相似"到底指什么？

```mermaid
flowchart LR
    A[Raw Data] --> B{Choose Method}
    B --> C[K-Means]
    B --> D[DBSCAN]
    B --> E[Hierarchical]
    B --> F[GMM]
    C --> G[Flat, spherical clusters]
    D --> H[Arbitrary shapes, noise detection]
    E --> I[Tree of nested clusters]
    F --> J[Soft assignments, elliptical clusters]
```

### K-Means：主力算法（K-Means: The Workhorse）

K-Means 把数据划分为恰好 K 个簇。每个簇有一个质心（centroid，即它的重心），每个点都属于离它最近的质心。

Lloyd 算法：

1. 随机选 K 个点作为初始质心
2. 把每个数据点分配给最近的质心
3. 把每个质心重新计算为其所辖点的均值
4. 重复第 2-3 步，直到分配不再变化

目标函数（惯性，inertia）度量每个点到其所属质心的平方距离总和。K-Means 最小化它，但只能找到局部最小值。不同的初始化可能得到不同的结果。

### 选择 K（Choosing K）

两种标准方法：

**肘部法则（elbow method）：** 对 K = 1, 2, 3, ..., n 依次运行 K-Means，画出惯性随 K 变化的曲线，寻找"肘部"——再增加簇也无法显著降低惯性的位置。

**轮廓系数（silhouette score）：** 对每个点，度量它与所属簇内其他点的相似程度（a），以及它到最近其他簇的相似程度（b）。轮廓系数为 (b - a) / max(a, b)，取值范围从 -1（分错簇）到 +1（聚类良好）。对所有点取平均得到全局分数。

### DBSCAN：基于密度的聚类（DBSCAN: Density-Based Clustering）

K-Means 假设簇是球形的，并且要求你预先指定 K。DBSCAN 这两个假设都不需要。它把簇看作由稀疏区域分隔开的密集区域。

两个参数：
- **eps**：邻域的半径
- **min_samples**：形成密集区域所需的最少点数

三种类型的点：
- **核心点（core point）**：eps 距离内至少有 min_samples 个点
- **边界点（border point）**：在某个核心点的 eps 范围内，但它本身不是核心点
- **噪声点（noise point）**：既非核心点也非边界点。它们就是离群点。

DBSCAN 把彼此在 eps 范围内的核心点连成同一个簇。边界点加入邻近核心点的簇。噪声点不属于任何簇。

优点：能发现任意形状的簇、自动确定簇的数量、识别离群点。缺点：对密度差异大的簇处理不佳。

### 层次聚类（Hierarchical Clustering）

构建一棵由嵌套簇组成的树（树状图，dendrogram）。

凝聚式（自底向上）：
1. 开始时每个点自成一簇
2. 合并距离最近的两个簇
3. 重复直到只剩一个簇
4. 在期望的层级切割树状图，得到 K 个簇

簇之间的"接近程度"可以这样度量：
- **单连接（single linkage）**：两簇间任意两点距离的最小值
- **全连接（complete linkage）**：任意两点距离的最大值
- **平均连接（average linkage）**：所有点对的平均距离
- **Ward 法**：使总簇内方差增量最小的合并方式

### 高斯混合模型（Gaussian Mixture Models (GMM)）

K-Means 给出硬分配：每个点恰好属于一个簇。GMM 给出软分配：每个点以一定概率属于每个簇。

GMM 假设数据由 K 个高斯分布的混合生成，每个高斯分布有自己的均值和协方差。期望最大化（Expectation-Maximization, EM）算法在以下两步之间交替：

- **E 步**：计算每个点属于每个高斯分布的概率
- **M 步**：更新每个高斯分布的均值、协方差和混合权重，以最大化数据的似然

GMM 能建模椭圆形的簇（不像 K-Means 只限于球形），并且天然能处理相互重叠的簇。

### 何时用哪个（When to Use Which）

| 方法 | 最适合 | 避免用于 |
|--------|----------|------------|
| K-Means | 大型数据集、球形簇、已知 K | 形状不规则、存在离群点 |
| DBSCAN | K 未知、任意形状、离群点检测 | 密度不一、维度非常高 |
| 层次聚类 | 小数据集、需要树状图、K 未知 | 大型数据集（内存 O(n^2)） |
| GMM | 重叠簇、需要软分配 | 特大型数据集、维度过多 |

### 用聚类做异常检测（Anomaly Detection with Clustering）

聚类天然支持异常检测：
- **K-Means**：远离所有质心的点是异常
- **DBSCAN**：噪声点按定义就是异常
- **GMM**：在所有高斯分布下概率都很低的点是异常

```figure
kmeans-step
```

## 动手实现（Build It）

### 第 1 步：从零实现 K-Means（Step 1: K-Means from scratch）

```python
import math
import random


def euclidean_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def kmeans(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n_features = len(data[0])

    centroids = random.sample(data, k)

    for iteration in range(max_iterations):
        clusters = [[] for _ in range(k)]
        assignments = []

        for point in data:
            distances = [euclidean_distance(point, c) for c in centroids]
            nearest = distances.index(min(distances))
            clusters[nearest].append(point)
            assignments.append(nearest)

        new_centroids = []
        for cluster in clusters:
            if len(cluster) == 0:
                new_centroids.append(random.choice(data))
                continue
            centroid = [
                sum(point[j] for point in cluster) / len(cluster)
                for j in range(n_features)
            ]
            new_centroids.append(centroid)

        if all(
            euclidean_distance(old, new) < 1e-6
            for old, new in zip(centroids, new_centroids)
        ):
            print(f"  Converged at iteration {iteration + 1}")
            break

        centroids = new_centroids

    return assignments, centroids
```

### 第 2 步：肘部法则与轮廓系数（Step 2: Elbow method and silhouette score）

```python
def compute_inertia(data, assignments, centroids):
    total = 0.0
    for point, cluster_id in zip(data, assignments):
        total += euclidean_distance(point, centroids[cluster_id]) ** 2
    return total


def silhouette_score(data, assignments):
    n = len(data)
    if n < 2:
        return 0.0

    clusters = {}
    for i, c in enumerate(assignments):
        clusters.setdefault(c, []).append(i)

    if len(clusters) < 2:
        return 0.0

    scores = []
    for i in range(n):
        own_cluster = assignments[i]
        own_members = [j for j in clusters[own_cluster] if j != i]

        if len(own_members) == 0:
            scores.append(0.0)
            continue

        a = sum(euclidean_distance(data[i], data[j]) for j in own_members) / len(own_members)

        b = float("inf")
        for cluster_id, members in clusters.items():
            if cluster_id == own_cluster:
                continue
            avg_dist = sum(euclidean_distance(data[i], data[j]) for j in members) / len(members)
            b = min(b, avg_dist)

        if max(a, b) == 0:
            scores.append(0.0)
        else:
            scores.append((b - a) / max(a, b))

    return sum(scores) / len(scores)


def find_best_k(data, max_k=10):
    print("Elbow method:")
    inertias = []
    for k in range(1, max_k + 1):
        assignments, centroids = kmeans(data, k)
        inertia = compute_inertia(data, assignments, centroids)
        inertias.append(inertia)
        print(f"  K={k}: inertia={inertia:.2f}")

    print("\nSilhouette scores:")
    for k in range(2, max_k + 1):
        assignments, centroids = kmeans(data, k)
        score = silhouette_score(data, assignments)
        print(f"  K={k}: silhouette={score:.4f}")

    return inertias
```

### 第 3 步：从零实现 DBSCAN（Step 3: DBSCAN from scratch）

```python
def dbscan(data, eps, min_samples):
    n = len(data)
    labels = [-1] * n
    cluster_id = 0

    def region_query(point_idx):
        neighbors = []
        for i in range(n):
            if euclidean_distance(data[point_idx], data[i]) <= eps:
                neighbors.append(i)
        return neighbors

    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = region_query(i)

        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seed_set = list(neighbors)
        seed_set.remove(i)

        j = 0
        while j < len(seed_set):
            q = seed_set[j]

            if not visited[q]:
                visited[q] = True
                q_neighbors = region_query(q)
                if len(q_neighbors) >= min_samples:
                    for nb in q_neighbors:
                        if nb not in seed_set:
                            seed_set.append(nb)

            if labels[q] == -1:
                labels[q] = cluster_id

            j += 1

        cluster_id += 1

    return labels
```

### 第 4 步：高斯混合模型（EM 算法）（Step 4: Gaussian Mixture Model (EM algorithm)）

```python
def gmm(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n = len(data)
    d = len(data[0])

    indices = random.sample(range(n), k)
    means = [list(data[i]) for i in indices]
    variances = [1.0] * k
    weights = [1.0 / k] * k

    def gaussian_pdf(x, mean, variance):
        d = len(x)
        coeff = 1.0 / ((2 * math.pi * variance) ** (d / 2))
        exponent = -sum((xi - mi) ** 2 for xi, mi in zip(x, mean)) / (2 * variance)
        return coeff * math.exp(max(exponent, -500))

    for iteration in range(max_iterations):
        responsibilities = []
        for i in range(n):
            probs = []
            for j in range(k):
                probs.append(weights[j] * gaussian_pdf(data[i], means[j], variances[j]))
            total = sum(probs)
            if total == 0:
                total = 1e-300
            responsibilities.append([p / total for p in probs])

        old_means = [list(m) for m in means]

        for j in range(k):
            r_sum = sum(responsibilities[i][j] for i in range(n))
            if r_sum < 1e-10:
                continue

            weights[j] = r_sum / n

            for dim in range(d):
                means[j][dim] = sum(
                    responsibilities[i][j] * data[i][dim] for i in range(n)
                ) / r_sum

            variances[j] = sum(
                responsibilities[i][j]
                * sum((data[i][dim] - means[j][dim]) ** 2 for dim in range(d))
                for i in range(n)
            ) / (r_sum * d)
            variances[j] = max(variances[j], 1e-6)

        shift = sum(
            euclidean_distance(old_means[j], means[j]) for j in range(k)
        )
        if shift < 1e-6:
            print(f"  GMM converged at iteration {iteration + 1}")
            break

    assignments = []
    for i in range(n):
        assignments.append(responsibilities[i].index(max(responsibilities[i])))

    return assignments, means, weights, responsibilities
```

### 第 5 步：生成测试数据并运行全部演示（Step 5: Generate test data and run everything）

```python
def make_blobs(centers, n_per_cluster=50, spread=0.5, seed=42):
    random.seed(seed)
    data = []
    true_labels = []
    for label, (cx, cy) in enumerate(centers):
        for _ in range(n_per_cluster):
            x = cx + random.gauss(0, spread)
            y = cy + random.gauss(0, spread)
            data.append([x, y])
            true_labels.append(label)
    return data, true_labels


def make_moons(n_samples=200, noise=0.1, seed=42):
    random.seed(seed)
    data = []
    labels = []
    n_half = n_samples // 2
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = math.cos(angle) + random.gauss(0, noise)
        y = math.sin(angle) + random.gauss(0, noise)
        data.append([x, y])
        labels.append(0)
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = 1 - math.cos(angle) + random.gauss(0, noise)
        y = 1 - math.sin(angle) - 0.5 + random.gauss(0, noise)
        data.append([x, y])
        labels.append(1)
    return data, labels


if __name__ == "__main__":
    centers = [[2, 2], [8, 3], [5, 8]]
    data, true_labels = make_blobs(centers, n_per_cluster=50, spread=0.8)

    print("=== K-Means on 3 blobs ===")
    assignments, centroids = kmeans(data, k=3)
    print(f"  Centroids: {[[round(c, 2) for c in cent] for cent in centroids]}")
    sil = silhouette_score(data, assignments)
    print(f"  Silhouette score: {sil:.4f}")

    print("\n=== Elbow Method ===")
    find_best_k(data, max_k=6)

    print("\n=== DBSCAN on 3 blobs ===")
    db_labels = dbscan(data, eps=1.5, min_samples=5)
    n_clusters = len(set(db_labels) - {-1})
    n_noise = db_labels.count(-1)
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")

    print("\n=== GMM on 3 blobs ===")
    gmm_assignments, gmm_means, gmm_weights, _ = gmm(data, k=3)
    print(f"  Means: {[[round(m, 2) for m in mean] for mean in gmm_means]}")
    print(f"  Weights: {[round(w, 3) for w in gmm_weights]}")
    gmm_sil = silhouette_score(data, gmm_assignments)
    print(f"  Silhouette score: {gmm_sil:.4f}")

    print("\n=== DBSCAN on moons (non-spherical clusters) ===")
    moon_data, moon_labels = make_moons(n_samples=200, noise=0.1)
    moon_db = dbscan(moon_data, eps=0.3, min_samples=5)
    n_moon_clusters = len(set(moon_db) - {-1})
    n_moon_noise = moon_db.count(-1)
    print(f"  Found {n_moon_clusters} clusters, {n_moon_noise} noise points")

    print("\n=== K-Means on moons (will fail to separate) ===")
    moon_km, moon_centroids = kmeans(moon_data, k=2)
    moon_sil = silhouette_score(moon_data, moon_km)
    print(f"  Silhouette score: {moon_sil:.4f}")
    print("  K-Means splits moons poorly because they are not spherical")

    print("\n=== Anomaly detection with DBSCAN ===")
    anomaly_data = list(data)
    anomaly_data.append([20.0, 20.0])
    anomaly_data.append([-5.0, -5.0])
    anomaly_data.append([15.0, 0.0])
    anomaly_labels = dbscan(anomaly_data, eps=1.5, min_samples=5)
    anomalies = [
        anomaly_data[i]
        for i in range(len(anomaly_labels))
        if anomaly_labels[i] == -1
    ]
    print(f"  Detected {len(anomalies)} anomalies")
    for a in anomalies[-3:]:
        print(f"    Point {[round(v, 2) for v in a]}")
```

## 直接使用（Use It）

使用 scikit-learn，同样的算法只需一行代码：

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score as sklearn_silhouette

km = KMeans(n_clusters=3, random_state=42).fit(data)
db = DBSCAN(eps=1.5, min_samples=5).fit(data)
agg = AgglomerativeClustering(n_clusters=3).fit(data)
gmm_model = GaussianMixture(n_components=3, random_state=42).fit(data)
```

从零实现的版本让你确切看到这些库在计算什么。K-Means 在分配与重算之间迭代；DBSCAN 从密集的种子生长出簇；GMM 在期望与最大化之间交替。库版本增加了数值稳定性、更聪明的初始化（K-Means++）和 GPU 加速，但核心逻辑是一样的。

## 成果交付（Ship It）

本课产出了可运行的 K-Means、DBSCAN 和 GMM 从零实现。这些聚类代码可以作为更高级无监督方法的基础来复用。

## 练习（Exercises）

1. 实现 K-Means++ 初始化：不再随机挑选质心，而是随机选第一个质心，之后每个质心按与其到最近已有质心距离的平方成正比的概率挑选。比较它与随机初始化的收敛速度。
2. 在代码中加入层次凝聚聚类。实现 Ward 连接，并生成树状图（用嵌套的合并列表表示）。在不同层级切割它，并与 K-Means 的结果比较。
3. 构建一个简单的异常检测流水线：对同一份数据运行 DBSCAN 和 GMM，标记两种方法一致认为是离群点的点（DBSCAN 中的噪声点、GMM 中的低概率点）。度量二者的重合程度，并讨论它们在什么情况下会不一致。

## 关键术语（Key Terms）

| 术语 | 人们的说法 | 实际含义 |
|------|----------------|----------------------|
| 聚类 | "把相似的东西归在一起" | 把数据划分为若干子集，使组内相似度（按某种特定距离度量）超过组间相似度 |
| 质心 | "簇的中心" | 分配到某个簇的所有点的均值；K-Means 用它代表一个簇 |
| 惯性 | "簇有多紧" | 每个点到其所属质心的平方距离之和；越小越紧 |
| 轮廓系数 | "簇之间分得有多开" | 对每个点计算 (b - a) / max(a, b)，其中 a 是簇内平均距离，b 是到最近其他簇的平均距离 |
| 核心点 | "密集区域中的点" | 在 DBSCAN 中，eps 距离内至少有 min_samples 个邻居的点 |
| EM 算法 | "软版 K-Means" | 期望最大化：迭代计算隶属概率（E 步）并更新分布参数（M 步） |
| 树状图 | "簇组成的树" | 展示层次聚类中簇的合并顺序与合并距离的树形图 |
| 异常 | "离群点" | 不符合预期模式的数据点；在 DBSCAN 中表现为噪声，在 GMM 中表现为低概率 |

## 延伸阅读（Further Reading）

- [Stanford CS229 - Unsupervised Learning](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) - Andrew Ng 关于聚类与 EM 的讲义
- [scikit-learn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html) - 所有聚类算法的实用对比，附可视化示例
- [DBSCAN original paper (Ester et al., 1996)](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) - 提出基于密度聚类的原始论文
