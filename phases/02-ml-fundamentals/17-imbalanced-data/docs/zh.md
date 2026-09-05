# 处理不平衡数据（Handling Imbalanced Data）

> 当 99% 的数据都是"正常"时，准确率就是个谎言。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 2, Lessons 01-09 (especially evaluation metrics)
**Time:** ~90 minutes

## 学习目标（Learning Objectives）

- 从零实现 SMOTE，并解释合成过采样（synthetic oversampling）与随机复制（random duplication）的区别
- 用 F1、AUPRC 和马修斯相关系数（Matthews Correlation Coefficient）而不是准确率来评估不平衡分类器
- 比较类别权重（class weighting）、阈值调整（threshold tuning）和重采样（resampling）策略，针对给定的不平衡比例选择正确的方法
- 构建一个结合 SMOTE、类别权重和阈值优化的完整不平衡数据处理流水线

## 问题（The Problem）

你构建了一个欺诈检测模型，准确率 99.9%，你开始庆祝。然后你发现：它把每一笔交易都预测为"非欺诈"。

这不是 bug。当只有 0.1% 的交易是欺诈时，这是理性选择。模型学到的是：永远猜多数类能把总误差降到最低。它在技术上是对的，也完全没用。

凡是真实分类问题都逃不开这一点。疾病诊断：阳性率 1%。网络入侵：攻击占 0.01%。制造缺陷：次品率 0.5%。垃圾邮件过滤：20% 是垃圾邮件。流失预测：5% 的用户流失。少数类越关键，它往往越稀少。

准确率失效是因为它把所有正确预测一视同仁。正确标记一笔正常交易和正确抓住一次欺诈，在准确率里都只是加一分。但抓住欺诈才是这个模型存在的全部理由。我们需要指标、技术和训练策略，逼着模型关注那个稀有却重要的类。

## 概念（The Concept）

### 准确率为何失效（Why Accuracy Fails）

考虑一个 1000 个样本的数据集：990 个负类，10 个正类。一个永远预测负类的模型：

|  | 预测为正 | 预测为负 |
|--|---|---|
| 实际为正 | 0 (TP) | 10 (FN) |
| 实际为负 | 0 (FP) | 990 (TN) |

准确率 = (0 + 990) / 1000 = 99.0%

模型抓住的欺诈是零，疾病是零，缺陷是零，可准确率说 99%。这就是准确率在不平衡问题上的危险之处。

### 更好的指标（Better Metrics）

**精确率（precision）** = TP / (TP + FP)。所有被标为正的样本里，有多少真的是正？精确率高意味着误报少。

**召回率（recall）** = TP / (TP + FN)。所有真正的正样本里，我们抓住了多少？召回率高意味着漏掉的正样本少。

**F1 分数** = 2 * precision * recall / (precision + recall)。调和平均数。比算术平均更能惩罚精确率与召回率之间的极端失衡。

**F-beta 分数** = (1 + beta^2) * precision * recall / (beta^2 * precision + recall)。beta > 1 时召回率更重要，beta < 1 时精确率更重要。F2 在欺诈检测中常见（漏掉欺诈比误报更糟）。

**AUPRC**（精确率-召回率曲线下面积，Area Under Precision-Recall Curve）。类似 AUC-ROC，但对不平衡数据更有信息量。随机分类器的 AUPRC 等于正类占比（不像 ROC 那样是 0.5），这让改进更容易看出来。

**马修斯相关系数（Matthews Correlation Coefficient）** = (TP * TN - FP * FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))。取值范围 -1 到 +1。只有模型在两个类别上都表现好时才会给高分。即使两类规模悬殊也保持均衡。

对上面那个"永远预测负类"的模型：precision = 0/0（无定义，通常置为 0）、recall = 0/10 = 0、F1 = 0、MCC = 0。这些指标正确地判定该模型一文不值。

### 不平衡数据处理流水线（The Imbalanced Data Pipeline）

```mermaid
flowchart TD
    A[Imbalanced Dataset] --> B{Imbalance Ratio?}
    B -->|Mild: 80/20| C[Class Weights]
    B -->|Moderate: 95/5| D[SMOTE + Threshold Tuning]
    B -->|Severe: 99/1| E[SMOTE + Class Weights + Threshold]
    C --> F[Train Model]
    D --> F
    E --> F
    F --> G[Evaluate with F1 / AUPRC / MCC]
    G --> H{Good Enough?}
    H -->|No| I[Try Different Strategy]
    H -->|Yes| J[Deploy with Monitoring]
    I --> B
```

### SMOTE：合成少数类过采样技术（SMOTE: Synthetic Minority Oversampling Technique）

随机过采样直接复制已有的少数类样本。能奏效，但有过拟合风险，因为模型会反复看到一模一样的点。

SMOTE 生成合理但不是复制品的合成少数类样本。算法如下：

1. 对每个少数类样本 x，在其他少数类样本中找它的 k 个最近邻
2. 随机选一个邻居
3. 在 x 与该邻居之间的线段上生成一个新样本

公式：`new_sample = x + random(0, 1) * (neighbor - x)`

它在真实少数类点之间做插值，在同一特征空间区域里生成样本，而不是简单复制已有数据。

```mermaid
flowchart LR
    subgraph Original["原始少数类样本"]
        P1["x1 (1.0, 2.0)"]
        P2["x2 (1.5, 2.5)"]
        P3["x3 (2.0, 1.5)"]
    end
    subgraph SMOTE["SMOTE 生成"]
        direction TB
        S1["选中 x1，邻居 x2"]
        S2["随机 t = 0.4"]
        S3["new = x1 + 0.4*(x2-x1)"]
        S4["new = (1.2, 2.2)"]
        S1 --> S2 --> S3 --> S4
    end
    Original --> SMOTE
    subgraph Result["扩充后的样本集"]
        R1["x1 (1.0, 2.0)"]
        R2["x2 (1.5, 2.5)"]
        R3["x3 (2.0, 1.5)"]
        R4["synthetic (1.2, 2.2)"]
    end
    SMOTE --> Result
```

### 重采样策略对比（Sampling Strategies Compared）

**随机过采样（Random Oversampling）**：复制少数类样本，使其数量与多数类一致。
- 优点：简单，不损失信息
- 缺点：完全重复的样本导致过拟合，训练时间增加

**随机欠采样（Random Undersampling）**：删除多数类样本，使其数量与少数类一致。
- 优点：训练快，简单
- 缺点：丢掉可能有用的多数类数据，方差更高

**SMOTE**：通过插值生成合成少数类样本。
- 优点：生成新数据点，比随机过采样更不易过拟合
- 缺点：可能在决策边界附近产生噪声样本，不考虑多数类分布

| 策略 | 数据变化 | 风险 | 何时使用 |
|----------|-------------|------|-------------|
| 过采样 | 复制少数类 | 过拟合 | 小数据集、中度不平衡 |
| 欠采样 | 删除多数类 | 信息损失 | 大数据集、想训练快 |
| SMOTE | 加入合成少数类 | 边界噪声 | 中度不平衡、少数类样本足够做 k-NN |

### 类别权重（Class Weights）

不改数据，改模型对待错误的方式。给少数类的错分分配更高权重。

对一个 950 个负样本、50 个正样本的二分类问题：
- 负类权重 = n_samples / (2 * n_negative) = 1000 / (2 * 950) = 0.526
- 正类权重 = n_samples / (2 * n_positive) = 1000 / (2 * 50) = 10.0

正类的权重是 19 倍。错分一个正样本的代价等于错分 19 个负样本。模型被迫关注少数类。

在逻辑回归中，这会修改损失函数：

```
weighted_loss = -sum(w_i * [y_i * log(p_i) + (1-y_i) * log(1-p_i)])
```

其中 w_i 取决于样本 i 的类别。

类别权重在期望意义上与过采样数学等价，但不用生成新数据点。因此更快，也避免了复制样本带来的过拟合风险。

### 阈值调整（Threshold Tuning）

大多数分类器输出概率。默认阈值是 0.5：P(positive) >= 0.5 就判正。但 0.5 是随意定的。类别不平衡时，最优阈值通常低得多。

流程：
1. 训练一个模型
2. 在验证集上拿到预测概率
3. 把阈值从 0.0 扫到 1.0
4. 在每个阈值下计算 F1（或你选定的指标）
5. 选使指标最大的阈值

```mermaid
flowchart LR
    A[Model] --> B[Predict Probabilities]
    B --> C[Sweep Thresholds 0.0 to 1.0]
    C --> D[Compute F1 at Each]
    D --> E[Pick Best Threshold]
    E --> F[Use in Production]
```

对一笔欺诈交易，模型可能输出 P(fraud) = 0.15。阈值 0.5 下它被判为非欺诈；阈值 0.10 下它被正确抓住。概率校准好不好不如排序重要 -- 只要欺诈得到的概率高于非欺诈，就存在能把它们分开的阈值。

### 代价敏感学习（Cost-Sensitive Learning）

类别权重的推广。不用统一代价，而是指定具体的错分代价：

|  | 预测为正 | 预测为负 |
|--|---|---|
| 实际为正 | 0（正确） | C_FN = 100 |
| 实际为负 | C_FP = 1 | 0（正确） |

漏掉一笔欺诈交易（FN）的代价比一次误报（FP）高 100 倍。模型优化的是总代价，不是总错误数。

当你能估计现实世界的代价时，这是最讲原则的方法。漏诊癌症的代价与导致一次多余活检的误报完全不同。把代价显式写出来，才能做出正确的权衡。

### 决策流程图（Decision Flowchart）

```mermaid
flowchart TD
    A[Start: Imbalanced Dataset] --> B{How imbalanced?}
    B -->|"< 70/30"| C["轻度不平衡：先试类别权重"]
    B -->|"70/30 到 95/5"| D["中度不平衡：SMOTE + 类别权重"]
    B -->|"> 95/5"| E["重度不平衡：组合多种策略"]
    C --> F{Enough data?}
    D --> F
    E --> F
    F -->|"< 1000 samples"| G["过采样或 SMOTE，避免欠采样"]
    F -->|"1000-10000"| H["SMOTE + 阈值调整"]
    F -->|"> 10000"| I["可以欠采样，或用类别权重"]
    G --> J[Train + Evaluate with F1/AUPRC]
    H --> J
    I --> J
    J --> K{Recall high enough?}
    K -->|No| L[Lower threshold]
    K -->|Yes| M{Precision acceptable?}
    M -->|No| N[Raise threshold or add features]
    M -->|Yes| O[Ship it]
```

```figure
class-imbalance
```

## 动手构建（Build It）

### 第 1 步：生成不平衡数据集（Step 1: Generate an imbalanced dataset）

```python
import numpy as np


def make_imbalanced_data(n_majority=950, n_minority=50, seed=42):
    rng = np.random.RandomState(seed)

    X_maj = rng.randn(n_majority, 2) * 1.0 + np.array([0.0, 0.0])
    X_min = rng.randn(n_minority, 2) * 0.8 + np.array([2.5, 2.5])

    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)])

    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]
```

### 第 2 步：从零实现 SMOTE（Step 2: SMOTE from scratch）

```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def find_k_neighbors(X, idx, k):
    distances = []
    for i in range(len(X)):
        if i == idx:
            continue
        d = euclidean_distance(X[idx], X[i])
        distances.append((i, d))
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]


def smote(X_minority, k=5, n_synthetic=100, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(X_minority)
    k = min(k, n_samples - 1)
    synthetic = []

    for _ in range(n_synthetic):
        idx = rng.randint(0, n_samples)
        neighbors = find_k_neighbors(X_minority, idx, k)
        neighbor_idx = neighbors[rng.randint(0, len(neighbors))]
        t = rng.random()
        new_point = X_minority[idx] + t * (X_minority[neighbor_idx] - X_minority[idx])
        synthetic.append(new_point)

    return np.array(synthetic)
```

### 第 3 步：随机过采样与欠采样（Step 3: Random oversampling and undersampling）

```python
def random_oversample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_resampled = list(X)
    y_resampled = list(y)

    for cls, count in zip(classes, counts):
        if count < max_count:
            cls_indices = np.where(y == cls)[0]
            n_needed = max_count - count
            chosen = rng.choice(cls_indices, size=n_needed, replace=True)
            X_resampled.extend(X[chosen])
            y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


def random_undersample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    X_resampled = []
    y_resampled = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        chosen = rng.choice(cls_indices, size=min_count, replace=False)
        X_resampled.extend(X[chosen])
        y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]
```

### 第 4 步：带类别权重的逻辑回归（Step 4: Logistic regression with class weights）

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_regression_weighted(X, y, weights, lr=0.01, epochs=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = sigmoid(z)
        error = pred - y
        weighted_error = error * weights

        gradient_w = (X.T @ weighted_error) / n_samples
        gradient_b = np.mean(weighted_error)

        w -= lr * gradient_w
        b -= lr * gradient_b

    return w, b


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weight_map = {}
    for cls, count in zip(classes, counts):
        weight_map[cls] = n_samples / (n_classes * count)
    return np.array([weight_map[yi] for yi in y])
```

### 第 5 步：阈值调整（Step 5: Threshold tuning）

```python
def find_optimal_threshold(y_true, y_probs, metric="f1"):
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif metric == "recall":
            score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == "precision":
            score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
```

### 第 6 步：评估函数（Step 6: Evaluation functions）

```python
def confusion_matrix_values(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }
```

### 第 7 步：比较所有方法（Step 7: Compare all approaches）

```python
X, y = make_imbalanced_data(950, 50, seed=42)
split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Baseline: no treatment
w_base, b_base = logistic_regression_weighted(
    X_train, y_train, np.ones(len(y_train)), lr=0.1, epochs=300
)
probs_base = sigmoid(X_test @ w_base + b_base)
preds_base = (probs_base >= 0.5).astype(int)

# Oversampled
X_over, y_over = random_oversample(X_train, y_train)
w_over, b_over = logistic_regression_weighted(
    X_over, y_over, np.ones(len(y_over)), lr=0.1, epochs=300
)
preds_over = (sigmoid(X_test @ w_over + b_over) >= 0.5).astype(int)

# SMOTE
minority_mask = y_train == 1
X_minority = X_train[minority_mask]
synthetic = smote(X_minority, k=5, n_synthetic=len(y_train) - 2 * int(minority_mask.sum()))
X_smote = np.vstack([X_train, synthetic])
y_smote = np.concatenate([y_train, np.ones(len(synthetic))])
w_sm, b_sm = logistic_regression_weighted(
    X_smote, y_smote, np.ones(len(y_smote)), lr=0.1, epochs=300
)
preds_smote = (sigmoid(X_test @ w_sm + b_sm) >= 0.5).astype(int)

# Class weights
sample_weights = compute_class_weights(y_train)
w_cw, b_cw = logistic_regression_weighted(
    X_train, y_train, sample_weights, lr=0.1, epochs=300
)
probs_cw = sigmoid(X_test @ w_cw + b_cw)
preds_cw = (probs_cw >= 0.5).astype(int)

# Threshold tuning (tune on held-out validation set, not test set)
probs_val = sigmoid(X_val @ w_cw + b_cw)
best_thresh, best_f1 = find_optimal_threshold(y_val, probs_val, metric="f1")
preds_thresh = (probs_cw >= best_thresh).astype(int)
```

代码文件把以上全部放在一个脚本里运行并打印结果。

## 直接使用（Use It）

用 scikit-learn 和 imbalanced-learn，这些技术都是一行代码：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

model_weighted = LogisticRegression(class_weight="balanced")
model_weighted.fit(X_train, y_train)
print(classification_report(y_test, model_weighted.predict(X_test)))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression()
model_smote.fit(X_resampled, y_resampled)
print(classification_report(y_test, model_smote.predict(X_test)))

pipeline = Pipeline([
    ("smote", SMOTE()),
    ("model", LogisticRegression(class_weight="balanced")),
])
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))
```

从零实现能让你看清每种技术到底做了什么：SMOTE 不过是少数类上的 k-NN 插值，类别权重是把损失乘上系数，阈值调整就是遍历截断点的 for 循环。没有任何魔法。

## 发布成果（Ship It）

本课产出：
- `outputs/skill-imbalanced-data.md` -- 一份处理不平衡分类问题的决策清单

## 练习（Exercises）

1. **Borderline-SMOTE**：修改 SMOTE 实现，只为靠近决策边界的少数类点生成合成样本（即 k 近邻中包含多数类样本的那些点）。在类别有重叠的数据集上与标准 SMOTE 比较结果。

2. **代价矩阵优化**：实现代价敏感学习，把代价矩阵作为参数。写一个函数，输入代价矩阵，返回使期望代价最小的最优预测。用不同代价比（1:10、1:100、1:1000）测试，画出精确率-召回率权衡的变化。

3. **阈值校准**：实现 Platt 缩放（在模型原始输出上拟合逻辑回归，得到校准后的概率）。比较校准前后的精确率-召回率曲线。证明校准不改变排序（AUC 不变），但让概率更有意义。

4. **平衡 bagging 集成**：训练多个模型，每个模型用一份平衡的自助采样（全部少数类 + 随机多数类子集）。对它们的预测取平均。把这种方法与用 SMOTE 的单个模型比较，同时度量性能和跨次运行的方差。

5. **不平衡比例实验**：取一个平衡数据集，逐步提高不平衡比例（50/50、70/30、90/10、95/5、99/1）。每个比例下分别用和不用 SMOTE 训练，画出两种方法的 F1 对不平衡比例的曲线。比例达到多少时 SMOTE 才开始产生有意义的差别？

## 关键术语（Key Terms）

| 术语 | 人们的说法 | 实际含义 |
|------|----------------|----------------------|
| 类别不平衡（Class imbalance） | "某一类的样本多得多" | 数据集中类别分布显著偏斜，导致模型偏向多数类 |
| SMOTE | "合成过采样" | 通过在已有少数类样本与其 k 近邻少数类之间插值来生成新的少数类样本 |
| 类别权重（Class weights） | "让稀有类的错误更贵" | 给损失函数乘上按类别区分的权重，让模型对少数类错分惩罚更重 |
| 阈值调整（Threshold tuning） | "移动决策边界" | 把分类的概率截断点从默认 0.5 改成能优化目标指标的值 |
| 精确率-召回率权衡（Precision-recall tradeoff） | "鱼与熊掌不可兼得" | 降低阈值能抓住更多正类（召回率更高），但也会标记更多假阳性（精确率更低），反之亦然 |
| AUPRC | "PR 曲线下面积" | 把精确率-召回率曲线汇总成一个数字；类别严重不平衡时比 AUC-ROC 更有信息量 |
| 马修斯相关系数（Matthews Correlation Coefficient） | "均衡的指标" | 预测标签与真实标签之间的相关性，只有模型在两个类别上都表现好时才会得高分 |
| 代价敏感学习（Cost-sensitive learning） | "不同的错误代价不同" | 把现实世界的错分代价纳入训练目标，让模型优化总代价而非错误个数 |
| 随机过采样（Random oversampling） | "复制少数类" | 重复少数类样本以平衡类别数量；简单，但有过拟合到重复样本的风险 |

## 延伸阅读（Further Reading）

- [SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al., 2002)](https://arxiv.org/abs/1106.1813) -- SMOTE 原始论文，至今仍是不平衡学习领域被引最多的工作
- [Learning from Imbalanced Data (He & Garcia, 2009)](https://ieeexplore.ieee.org/document/5128907) -- 覆盖采样、代价敏感与算法层面方法的全面综述
- [imbalanced-learn 文档](https://imbalanced-learn.org/stable/) -- 提供 SMOTE 变体、欠采样策略和流水线集成的 Python 库
- [The Precision-Recall Plot Is More Informative than the ROC Plot (Saito & Rehmsmeier, 2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) -- 不平衡问题上何时以及为什么更该用 PR 曲线而非 ROC 曲线
