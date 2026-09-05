# 面向机器学习的图论（Graph Theory for Machine Learning）

> 图是刻画关系的数据结构。如果你的数据里存在连接，你就需要图论。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-03 (linear algebra, matrices)
**Time:** ~90 minutes

## 学习目标（Learning Objectives）

- 构建支持邻接矩阵/邻接表表示的图类，并实现 BFS 与 DFS 遍历
- 计算图拉普拉斯矩阵，并用其特征值检测连通分量、对节点聚类
- 把一轮 GNN 风格的消息传递实现为归一化邻接矩阵乘法
- 应用谱聚类，利用 Fiedler 向量对图进行划分

## 问题所在（The Problem）

社交网络、分子、知识库、引文网络、路线图——它们都是图。传统机器学习把数据当作扁平的表格。每行相互独立，每个特征是一列。但当连接的结构很重要时，表格就失效了。

想想一个社交网络。你想预测一个用户会买什么商品。他自己的购买历史固然重要，但他朋友们的购买历史更重要。连接本身就携带信号。

再想一个分子。你想预测它是否与某个蛋白质结合。原子固然重要，但真正要紧的是原子之间如何成键。结构本身就是数据。

图神经网络（GNN）是深度学习中增长最快的方向。它们支撑着药物发现、社交推荐、欺诈检测和知识图谱推理。每个 GNN 都建立在同一个基础之上：基础图论。

你需要四样东西：
1. 把图表示成矩阵的方法（这样你才能对它们做乘法）
2. 用于探索图结构的遍历算法
3. 拉普拉斯矩阵（Laplacian）——谱图理论中最重要的矩阵
4. 消息传递（message passing）——让 GNN 得以工作的运算

## 核心概念（The Concept）

### 图：节点与边（Graphs: Nodes and Edges）

图 G = (V, E) 由顶点（节点）V 和边 E 组成。每条边连接两个节点。

**有向与无向。** 在无向图中，边 (u, v) 意味着 u 连接到 v，同时 v 也连接到 u。在有向图（digraph）中，边 (u, v) 意味着 u 指向 v，但不一定反向成立。

**带权与无权。** 在无权图中，边要么存在要么不存在。在带权图中，每条边有一个数值权重——距离、成本或强度。

| 图类型 | 示例 |
|-----------|---------|
| 无向、无权 | Facebook 好友网络 |
| 有向、无权 | Twitter 关注网络 |
| 无向、带权 | 路网图（距离） |
| 有向、带权 | 网页链接（PageRank 分数） |

### 邻接矩阵（The Adjacency Matrix）

邻接矩阵 A 是核心表示。对一个有 n 个节点的图：

```
A[i][j] = 1    if there is an edge from node i to node j
A[i][j] = 0    otherwise
```

对无向图，A 是对称的：A[i][j] = A[j][i]。对带权图，A[i][j] = 边 (i, j) 的权重。

**示例——一个三角形：**

```
Nodes: 0, 1, 2
Edges: (0,1), (1,2), (0,2)

A = [[0, 1, 1],
     [1, 0, 1],
     [1, 1, 0]]
```

邻接矩阵是每个 GNN 的输入。对 A 做的矩阵运算对应着图上的运算。

### 度（Degree）

节点的度是与它相连的边数。对有向图，有入度（进来的边）和出度（出去的边）。

度矩阵 D 是对角矩阵：

```
D[i][i] = degree of node i
D[i][j] = 0    for i != j
```

以上面的三角形为例：D = diag(2, 2, 2)，因为每个节点都与其他两个节点相连。

度告诉你节点的重要性。高度数 = 中心节点。网络的度分布揭示其结构。社交网络服从幂律（少数中心节点，大量叶子节点）。随机图的度服从泊松分布。

### BFS 与 DFS（BFS and DFS）

两种最基本的图遍历算法。你两个都需要。

**广度优先搜索（BFS）：** 先探索所有邻居，再探索邻居的邻居。使用队列（FIFO）。

```
BFS from node 0:
  Visit 0
  Queue: [1, 2]        (neighbors of 0)
  Visit 1
  Queue: [2, 3]        (add neighbors of 1)
  Visit 2
  Queue: [3]           (neighbors of 2 already visited)
  Visit 3
  Queue: []            (done)
```

BFS 在无权图中寻找最短路径。起点到任意节点的距离等于该节点第一次被发现时的 BFS 层级。这就是社交网络中的跳数距离用 BFS 来算的原因。

**深度优先搜索（DFS）：** 尽可能深地走，再回溯。使用栈（LIFO）或递归。

```
DFS from node 0:
  Visit 0
  Stack: [1, 2]        (neighbors of 0)
  Visit 2               (pop from stack)
  Stack: [1, 3]         (add neighbors of 2)
  Visit 3               (pop from stack)
  Stack: [1]
  Visit 1               (pop from stack)
  Stack: []             (done)
```

DFS 的用处：
- 寻找连通分量（从未访问的节点出发跑 DFS）
- 检测环（DFS 树中的回边）
- 拓扑排序（DFS 完成顺序的逆序）

| 算法 | 数据结构 | 能找到 | 使用场景 |
|-----------|---------------|-------|----------|
| BFS | 队列 | 最短路径 | 社交网络距离、知识图谱遍历 |
| DFS | 栈 | 连通分量、环 | 连通性、拓扑排序 |

### 图拉普拉斯矩阵（The Graph Laplacian）

L = D - A。谱图理论中最重要的矩阵。

对三角形而言：

```
D = [[2, 0, 0],    A = [[0, 1, 1],    L = [[2, -1, -1],
     [0, 2, 0],         [1, 0, 1],         [-1, 2, -1],
     [0, 0, 2]]         [1, 1, 0]]         [-1, -1,  2]]
```

拉普拉斯矩阵有一些非凡的性质：

1. **L 是半正定的。** 所有特征值 >= 0。

2. **零特征值的个数等于连通分量的个数。** 连通图恰好有一个零特征值。有 3 个不连通分量的图有三个零特征值。

3. **最小的非零特征值（Fiedler 值）度量连通性。** Fiedler 值大说明图连接良好。Fiedler 值小说明图有一个薄弱点——瓶颈。

4. **Fiedler 值对应的特征向量（Fiedler 向量）揭示最佳切分。** 取值为正的节点归为一组，取值为负的归为另一组。这就是谱聚类。

```mermaid
graph TD
    subgraph "从图到矩阵"
        G["图 G"] --> A["邻接矩阵 A"]
        G --> D["度矩阵 D"]
        A --> L["拉普拉斯矩阵 L = D - A"]
        D --> L
    end
    subgraph "谱分析"
        L --> E["L 的特征值"]
        L --> V["L 的特征向量"]
        E --> C["连通分量（零特征值）"]
        E --> F["连通性（Fiedler 值）"]
        V --> S["谱聚类"]
    end
```

### 谱性质（Spectral Properties）

邻接矩阵和拉普拉斯矩阵的特征值无需任何遍历就能揭示结构性质。

**谱聚类**的工作方式如下：
1. 计算拉普拉斯矩阵 L
2. 求 L 的 k 个最小特征向量（跳过第一个，它在连通图中是全 1 向量）
3. 把这些特征向量当作每个节点的新坐标
4. 在这些坐标上运行 k-means

为什么可行？L 的特征向量编码了图上"最平滑"的函数。连接良好的节点得到相近的特征向量取值，被瓶颈隔开的节点取值不同。特征向量天然地把簇分开。

**与随机游走的联系。** 归一化拉普拉斯矩阵与图上的随机游走相关。随机游走的平稳分布与节点度成正比。混合时间（游走收敛得多快）取决于谱隙。

### 消息传递（Message Passing）

图神经网络的核心运算。每个节点从邻居收集消息，聚合它们，并更新自己的状态。

```
h_v^(k+1) = UPDATE(h_v^(k), AGGREGATE({h_u^(k) : u in neighbors(v)}))
```

最简单的形式里，AGGREGATE = 求均值，UPDATE = 线性变换 + 激活：

```
h_v^(k+1) = sigma(W * mean({h_u^(k) : u in neighbors(v)}))
```

这是伪装起来的矩阵乘法。如果 H 是所有节点特征组成的矩阵，A 是邻接矩阵：

```
H^(k+1) = sigma(A_norm * H^(k) * W)
```

其中 A_norm 是归一化邻接矩阵（每行之和为 1）。

一轮消息传递让每个节点"看到"它的直接邻居。两轮让它看到邻居的邻居。K 轮让每个节点获得其 K 跳邻域的信息。

```mermaid
graph LR
    subgraph "第 0 轮"
        A0["节点 A: [1,0]"]
        B0["节点 B: [0,1]"]
        C0["节点 C: [1,1]"]
    end
    subgraph "第 1 轮（聚合邻居）"
        A1["节点 A: avg(B,C) = [0.5, 1.0]"]
        B1["节点 B: avg(A,C) = [1.0, 0.5]"]
        C1["节点 C: avg(A,B) = [0.5, 0.5]"]
    end
    A0 --> A1
    B0 --> A1
    C0 --> A1
    A0 --> B1
    C0 --> B1
    A0 --> C1
    B0 --> C1
```

### 概念与机器学习应用（Concepts and ML Applications）

| 概念 | 机器学习应用 |
|---------|---------------|
| 邻接矩阵 | GNN 的输入表示 |
| 图拉普拉斯矩阵 | 谱聚类、社区发现 |
| BFS/DFS | 知识图谱遍历、路径查找 |
| 度分布 | 节点重要性、特征工程 |
| 消息传递 | GNN 层（GCN、GAT、GraphSAGE） |
| L 的特征值 | 社区发现、图划分 |
| 谱聚类 | 无监督节点分组 |
| PageRank | 节点重要性、网页搜索 |

```figure
graph-degree-distribution
```

## 动手实现（Build It）

### 步骤 1：从零实现 Graph 类（Step 1: Graph class from scratch）

```python
class Graph:
    def __init__(self, n_nodes, directed=False):
        self.n = n_nodes
        self.directed = directed
        self.adj = {i: {} for i in range(n_nodes)}

    def add_edge(self, u, v, weight=1.0):
        self.adj[u][v] = weight
        if not self.directed:
            self.adj[v][u] = weight

    def neighbors(self, node):
        return list(self.adj[node].keys())

    def degree(self, node):
        return len(self.adj[node])

    def adjacency_matrix(self):
        import numpy as np
        A = np.zeros((self.n, self.n))
        for u in range(self.n):
            for v, w in self.adj[u].items():
                A[u][v] = w
        return A

    def degree_matrix(self):
        import numpy as np
        D = np.zeros((self.n, self.n))
        for i in range(self.n):
            D[i][i] = self.degree(i)
        return D

    def laplacian(self):
        return self.degree_matrix() - self.adjacency_matrix()
```

邻接表（`self.adj`）高效地存储邻居。邻接矩阵转换使用 numpy，因为所有谱运算都需要它。

### 步骤 2：BFS 与 DFS（Step 2: BFS and DFS）

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    order = []
    distances = {}
    queue = deque([(start, 0)])
    visited.add(start)
    while queue:
        node, dist = queue.popleft()
        order.append(node)
        distances[node] = dist
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return order, distances


def dfs(graph, start):
    visited = set()
    order = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in reversed(graph.neighbors(node)):
            if neighbor not in visited:
                stack.append(neighbor)
    return order
```

BFS 用 deque（双端队列）实现 O(1) 的 popleft。DFS 把列表当栈用。两者都恰好访问每个节点一次——O(V + E) 时间。

### 步骤 3：连通分量与拉普拉斯特征值（Step 3: Connected components and Laplacian eigenvalues）

```python
def connected_components(graph):
    visited = set()
    components = []
    for node in range(graph.n):
        if node not in visited:
            order, _ = bfs(graph, node)
            visited.update(order)
            components.append(order)
    return components


def laplacian_eigenvalues(graph):
    import numpy as np
    L = graph.laplacian()
    eigenvalues = np.linalg.eigvalsh(L)
    return eigenvalues
```

`eigvalsh` 用于对称矩阵——无向图的拉普拉斯矩阵总是对称的。它按升序返回特征值。数一数零的个数，就能得到连通分量的数量。

### 步骤 4：谱聚类（Step 4: Spectral clustering）

```python
def spectral_clustering(graph, k=2):
    import numpy as np
    L = graph.laplacian()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    features = eigenvectors[:, 1:k+1]

    labels = np.zeros(graph.n, dtype=int)
    for i in range(graph.n):
        if features[i, 0] >= 0:
            labels[i] = 0
        else:
            labels[i] = 1
    return labels
```

k=2 时，Fiedler 向量的符号把图分成两个簇。k>2 时，你需要对前 k 个特征向量运行 k-means（排除平凡的全 1 特征向量）。

### 步骤 5：消息传递（Step 5: Message passing）

```python
def message_passing(graph, features, weight_matrix):
    import numpy as np
    A = graph.adjacency_matrix()
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A_norm = A / row_sums
    aggregated = A_norm @ features
    output = aggregated @ weight_matrix
    return output
```

这就是一轮 GNN 消息传递。每个节点的新特征是其邻居特征的加权平均，再经过权重矩阵变换。叠加多轮就能把信息传播得更远。

## 直接使用（Use It）

有了 networkx 和 numpy，同样的操作一行就能搞定：

```python
import networkx as nx
import numpy as np

G = nx.karate_club_graph()

A = nx.adjacency_matrix(G).toarray()
L = nx.laplacian_matrix(G).toarray()

eigenvalues = np.linalg.eigvalsh(L.astype(float))
print(f"Smallest eigenvalues: {eigenvalues[:5]}")
print(f"Connected components: {nx.number_connected_components(G)}")

communities = nx.community.greedy_modularity_communities(G)
print(f"Communities found: {len(communities)}")

pr = nx.pagerank(G)
top_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"Top 5 PageRank nodes: {top_nodes}")
```

networkx 借助优化的 C 后端处理任意规模的图。生产环境请用它。用你从零实现的版本来理解它做了什么。

### 用 numpy 做谱分析（numpy spectral analysis）

```python
import numpy as np

A = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0]
])

D = np.diag(A.sum(axis=1))
L = D - A

eigenvalues, eigenvectors = np.linalg.eigh(L)
print(f"Eigenvalues: {np.round(eigenvalues, 4)}")
print(f"Fiedler value: {eigenvalues[1]:.4f}")
print(f"Fiedler vector: {np.round(eigenvectors[:, 1], 4)}")

fiedler = eigenvectors[:, 1]
group_a = np.where(fiedler >= 0)[0]
group_b = np.where(fiedler < 0)[0]
print(f"Cluster A: {group_a}")
print(f"Cluster B: {group_b}")
```

Fiedler 向量承担了全部重活。取值为正的在一个簇，为负的在另一个簇。无需任何迭代优化——只需要一次特征分解。

## 交付成果（Ship It）

本课产出：
- `outputs/skill-graph-analysis.md` —— 分析图结构数据的技能参考

## 知识关联（Connections）

| 概念 | 出现之处 |
|---------|------------------|
| 邻接矩阵 | GCN、GAT、GraphSAGE 的输入 |
| 拉普拉斯矩阵 | 谱聚类、ChebNet 滤波器 |
| BFS | 知识图谱遍历、最短路径查询 |
| 消息传递 | 每个 GNN 层、神经消息传递 |
| 谱隙 | 图连通性、随机游走的混合时间 |
| 度分布 | 幂律网络、节点特征工程 |
| 连通分量 | 预处理、处理非连通图 |
| PageRank | 节点重要性排序、注意力初始化 |

GNN 值得特别一提。GCN（Kipf & Welling, 2017）中的图卷积运算使用加了自环的邻接矩阵 A_hat = A + I：

```text
H^(l+1) = sigma(D_hat^(-1/2) * A_hat * D_hat^(-1/2) * H^(l) * W^(l))
```

其中 A_hat = A + I（邻接矩阵加自环），D_hat 是 A_hat 的度矩阵。自环确保每个节点在聚合时包含自己的特征。这正是带对称归一化的消息传递。D_hat^(-1/2) * A_hat * D_hat^(-1/2) 就是归一化邻接矩阵。拉普拉斯矩阵之所以出现，是因为这种归一化与 L_sym = I - D^(-1/2) * A * D^(-1/2) 相关。理解了拉普拉斯矩阵，就理解了 GCN 为什么有效。

## 练习（Exercises）

1. **从零实现 PageRank。** 从均匀分数开始。每一步：score(v) = (1-d)/n + d * sum(score(u)/out_degree(u))，对所有指向 v 的 u 求和。取 d=0.85。运行直到收敛（变化 < 1e-6）。在一个小的网页图上测试。

2. **用谱聚类发现社区。** 构造一个有两个明显分离簇的图（例如两个团，只用一条边相连）。运行谱聚类，验证它找到了正确的切分。随着跨簇的边增多会发生什么？

3. **实现 Dijkstra 算法**，求带权图中的最短路径。在同一张图上使用均匀权重，把结果与 BFS 比较。

4. **搭建一个 2 层消息传递网络。** 用不同的权重矩阵做两次消息传递。证明 2 轮之后，每个节点拥有其 2 跳邻域的信息。

5. **分析一个真实世界的图。** 使用空手道俱乐部图（Karate Club，34 个节点、78 条边）。计算度分布、拉普拉斯特征值和谱聚类。把谱聚类结果与已知的真实划分进行比较。

## 关键术语（Key Terms）

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|----------------------|
| 图 | "节点和边" | 编码成对关系的数学结构 G=(V,E) |
| 邻接矩阵 | "连接表" | 一个 n x n 矩阵，若节点 i 与 j 相连则 A[i][j] = 1 |
| 度 | "节点有多连通" | 与节点相连的边数 |
| 拉普拉斯矩阵 | "D 减 A" | L = D - A，其特征值揭示图结构的矩阵 |
| Fiedler 值 | "代数连通度" | L 的最小非零特征值，衡量图的连通程度 |
| BFS | "逐层搜索" | 先访问所有邻居再深入的遍历方式，可找到最短路径 |
| DFS | "先走到底" | 沿一条路径走到底再回溯的遍历方式 |
| 消息传递 | "节点与邻居对话" | 每个节点从邻居聚合信息，是 GNN 的核心 |
| 谱聚类 | "按特征向量聚类" | 利用拉普拉斯矩阵的特征向量对图进行划分 |
| 连通分量 | "独立的一块" | 任意两个节点都互相可达的极大子图 |

## 延伸阅读（Further Reading）

- **Kipf & Welling (2017)** -- "Semi-Supervised Classification with Graph Convolutional Networks." 开创现代 GNN 的论文。证明了谱图卷积可以简化为消息传递。
- **Spielman (2012)** -- "Spectral Graph Theory" 讲义。关于拉普拉斯矩阵、谱隙与图划分的权威入门。
- **Hamilton (2020)** -- "Graph Representation Learning." 一本从基础讲到应用的 GNN 专著。
- **Bronstein et al. (2021)** -- "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges." 统一框架论文。
- **Veličković et al. (2018)** -- "Graph Attention Networks." 用注意力机制扩展消息传递。
