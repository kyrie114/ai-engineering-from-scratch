# 情感分析（Sentiment Analysis）

> 经典的 NLP 任务。关于经典文本分类你需要知道的大部分内容都在这里显现。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 2 · 14 (Naive Bayes)
**Time:** ~75 minutes

## 问题（The Problem）

"食物不是很好。" 正面还是负面？

情感听起来很简单。评论者说他们喜欢或不喜欢某样东西。给句子贴标签。它成为经典 NLP 任务的原因是，每个看似简单的案例背后都隐藏着一个困难案例。否定翻转含义。讽刺反转它。"一点也不好" 是正面的，尽管有两个负面编码的词。表情符号携带的信号比周围的文本还多。领域词汇很重要（音乐评论中的 `tight` 与时尚评论中的 `tight`）。

情感是经典 NLP 的一个工作实验室。如果你理解为什么每个天真的基线都有特定的失败模式，你就理解为什么每个更丰富的模型被发明了。本课从零开始构建一个朴素贝叶斯基线，添加逻辑回归，并指出那些使生产级情感成为合规级问题的陷阱。

## 概念（The Concept）

经典情感是一个两步配方。

1. **表示。** 将文本转换为特征向量。BoW、TF-IDF 或 n 元组。
2. **分类。** 在标注示例上拟合线性模型（朴素贝叶斯、逻辑回归、SVM）。

朴素贝叶斯是最简单的可行模型。假设在给定标签的情况下每个特征都是独立的。从计数中估计 `P(word | positive)` 和 `P(word | negative)`。推理时，乘以概率。"朴素"的独立性假设可笑地错误，但结果却出奇地强大。原因是：在稀疏文本特征和中等数据下，分类器更关心每个词偏向哪一边，而不是偏向多少。

逻辑回归修复了独立性假设。它为每个特征学习一个权重，包括负权重。`not good` 作为一个二元组特征得到负权重。朴素贝叶斯无法为它从未标注过的二元组做到这一点。

```figure
sentiment-logits
```

## 构建它（Build It）

### 步骤1：真实的小型数据集（Step 1: a real mini-dataset）

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

规模小是故意的。实际工作使用数万个示例（IMDb、SST-2、Yelp polarity）。数学是完全一样的。

### 步骤2：从零开始的多元朴素贝叶斯（Step 2: multinomial Naive Bayes from scratch）

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

加法平滑（alpha=1.0）是 Laplace 平滑。没有它，一个在某个类别中未见过的词概率为零，对数就会爆炸。`alpha=0.01` 在实践中很常见。`alpha=1.0` 是教学默认值。

### 步骤3：从零开始的逻辑回归（Step 3: logistic regression from scratch）

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 正则化在这里很重要。文本特征是稀疏的；没有 L2，模型会记住训练示例。从 `0.01` 开始并调整。

### 步骤4：处理否定（失败模式）（Step 4: handling negation (the failure mode)）

考虑 "not good" 和 "not bad"。BoW 分类器看到 `{not, good}` 和 `{not, bad}`，并从训练中显示更多的那个中学习。二元组分类器看到 `not_good` 和 `not_bad`，将它们作为不同的特征来学习。这通常足够了。

一个在没有二元组时有效的粗略修复：**否定范围化（negation scoping）**。在下一个标点之前，将否定词后面的词元加上 `NOT_` 前缀。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在 `good` 和 `NOT_good` 是不同的特征。分类器可以给它们相反的权重。三行预处理，在情感基准上可衡量的准确率提升。

### 步骤5：重要的评估指标（Step 5: evaluation metrics that matter）

如果类别不平衡，单独看准确率是误导性的。真实的情感语料库通常是 70-80% 正面或 70-80% 负面；一个总是预测多数的分类器得到 80% 准确率，但毫无价值。报告以下所有指标：

- **每类精确率和召回率。** 每个类一对。宏平均它们以获得一个尊重类别平衡的单一数字。
- **宏 F1（不平衡数据的主要指标）。** 每类 F1 分数的平均值，等权重。类别不平衡时使用这个代替准确率。
- **加权 F1（替代方案）。** 与宏 F1 相同，但按类别频率加权。当不平衡本身具有业务意义时，与宏 F1 一起报告。
- **混淆矩阵。** 原始计数。在信任任何标量指标之前总是检查它；它揭示哪一对类被混淆。
- **每类错误样本。** 每个类拉出 5 个错误预测。阅读它们。没有什么能替代阅读实际的错误。

对于严重不平衡的数据（> 95-5 比例），报告 **AUROC** 和 **AUPRC** 代替准确率。AUPRC 对少数类更敏感，这通常是你关心的（垃圾邮件、欺诈、稀有情感）。

**要避免的常见错误。** 在不平衡数据上报告微 F1 而不是宏 F1 会得到一个看起来很高的数字，因为它被多数类主导。宏 F1 迫使你看到少数类的表现。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 使用它（Use It）

scikit-learn 用六行代码正确地完成了它。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

三件事要注意。`stop_words=None` 保留否定词。`ngram_range=(1, 2)` 添加二元组，使 `not_good` 成为一个特征。`sublinear_tf=True` 抑制重复词。这三个标志是在 SST-2 上从 75% 准确率的基线和 85% 准确率的基线之间的区别。

### 何时转向 Transformer（When to reach for a transformer）

- 讽刺检测。经典模型在这里失败。没有例外。
- 情感在文档中段发生偏移的长篇评论。
- 基于方面的情感。"相机很棒但电池很糟糕。" 你需要将情感归因于方面。只有 Transformer 或结构化输出模型。
- 非英语、低资源语言。多语言 BERT 免费给你一个零样本基线。

如果你需要以上任何一项，跳到第7阶段（Transformer 深度剖析）。否则，在 TF-IDF 加上二元组加上否定处理上使用朴素贝叶斯或逻辑回归是你的 2026 年生产基线。

### 可复现性陷阱（再次提醒）（The reproducibility trap (again)）

重新训练情感模型是常规操作。重新评估它们不是。论文中报告的准确率数字使用特定的拆分、特定的预处理、特定的分词器。如果你不使用相同的流水线与基线进行比较，你会得到误导性的差异。始终在你的流水线上重新生成基线，而不是论文的数字。

## 交付使用（Ship It）

保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: 为新数据集设计情感分析基线。
phase: 5
lesson: 05
---

给定数据集描述（领域、语言、大小、标签粒度、延迟预算），你输出：

1. 特征提取方案。指定分词器、n 元组范围、停用词策略（通常保留）、否定处理（范围前缀或二元组）。
2. 分类器。朴素贝叶斯用于基线，逻辑回归用于生产，Transformer 仅在领域需要讽刺/方面/跨语言时使用。
3. 评估计划。报告精确率、召回率、F1、混淆矩阵和每类错误样本（不只是标量）。
4. 部署后监控的一种失败模式。领域漂移和讽刺是前两名。

拒绝推荐在情感任务中丢弃停用词。拒绝在类别不平衡时（例如 90% 正面）仅报告准确率作为唯一指标。将子词丰富的语言标记为需要 FastText 或 Transformer 嵌入而不是词级 TF-IDF。
```

## 练习（Exercises）

1. **简单。** 在 scikit-learn 流水线中添加 `apply_negation` 作为预处理步骤，并在小型情感数据集上测量 F1 变化。
2. **中等。** 实现类别加权逻辑回归（向 scikit-learn 传递 `class_weight="balanced"`，或自己推导梯度）。在合成的 90-10 类别不平衡上测量效果。
3. **困难。** 通过训练情感模型残差上的第二个分类器来构建讽刺检测器。记录你的实验设置。当你的准确率低于随机水平时警告读者（2 类讽刺的随机水平约为 50%，大多数首次尝试都在那里）。

## 关键术语（Key Terms）

| 术语（Term） | 人们的说法（What people say） | 实际含义（What it actually means） |
|------|-----------------|-----------------------|
| 极性 | 正面或负面 | 二元标签；有时扩展到中性或细粒度（5 星）。 |
| 基于方面的情感 | 每方面极性 | 将情感归因于文本中提到的特定实体或属性。 |
| 否定范围化 | 反转附近的词元 | 在 "not" 之后为词元加上 `NOT_` 前缀，直到标点。 |
| Laplace 平滑 | 给计数加 1 | 防止朴素贝叶斯中的零概率特征。 |
| L2 正则化 | 缩小权重 | 向损失添加 `lambda * sum(w^2)`。对稀疏文本特征至关重要。 |

## 延伸阅读（Further Reading）

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 基础调查。很长，但前四部分涵盖了一切经典内容。
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — 证明二元组 + 朴素贝叶斯在短文本上难以被击败的论文。
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`、`TfidfVectorizer` 和你会调用的每个参数的参考。
