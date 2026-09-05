# 朴素贝叶斯（Naive Bayes）

> "朴素"这个假设是错的，但它照样好用，这正是它的美妙之处。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 2, Lessons 01-07 (classification, Bayes' theorem)
**Time:** ~75 minutes

## 学习目标（Learning Objectives）

- 从零实现带拉普拉斯平滑（Laplace smoothing）的多项式朴素贝叶斯（Multinomial Naive Bayes），用于文本分类
- 解释为什么朴素独立性假设在数学上是错的，却在实践中给出正确的类别排序
- 比较多项式（Multinomial）、伯努利（Bernoulli）和高斯（Gaussian）三种朴素贝叶斯变体，并针对给定的特征类型选出合适的一种
- 在高维稀疏数据上把朴素贝叶斯与逻辑回归（logistic regression）对比，并解释其中起作用的偏差-方差权衡（bias-variance tradeoff）

## 问题（The Problem）

你需要对文本分类。邮件分为垃圾邮件或正常邮件，客户评论分为正面或负面，工单分入不同类别。你有成千上万个特征（每个词一个），而训练数据有限。

大多数分类器在这里都会卡住。逻辑回归需要足够的样本才能可靠地估计数千个权重。决策树一次只按一个词切分，极易过拟合。在一万维空间里 KNN 毫无意义，因为每个点与其他点的距离都差不多。

朴素贝叶斯能应付这个局面。它做了一个在数学上错误的假设（在给定类别的条件下，每个特征彼此独立），却在文本分类上胜过那些"更聪明"的模型，尤其是训练集很小的时候。它一遍扫描即可完成训练，能扩展到数百万个特征，还能给出概率估计（尽管由于独立性假设，校准往往很差）。

理解"错误的假设为何能带来好的预测"，会让你领悟机器学习中一个根本道理：最好的模型不是最"正确"的那个，而是在你的数据上偏差-方差权衡最好的那个。

## 概念（The Concept）

### 贝叶斯定理快速回顾（Bayes' Theorem (Quick Review)）

贝叶斯定理把条件概率翻转过来：

```
P(class | features) = P(features | class) * P(class) / P(features)
```

我们想要的是 `P(class | features)` -- 给定文档中的词，该文档属于某个类别的概率。它可以由以下几项算出：
- `P(features | class)` -- 在该类别的文档中看到这些词的似然（likelihood）
- `P(class)` -- 类别的先验概率（prior）（垃圾邮件总体上有多常见？）
- `P(features)` -- 证据（evidence），对所有类别都相同，比较时可以忽略

`P(class | features)` 最高的类别胜出。

### 朴素的独立性假设（The Naive Independence Assumption）

要精确计算 `P(features | class)`，需要估计所有特征的联合概率。词表若有一万个词，你就要在 2^10,000 种可能的组合上估计分布。不可能。

朴素假设是：在给定类别的条件下，每个特征相互条件独立。

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

你不再估计那个不可能搞定的联合分布，而是估计 n 个简单的逐特征分布，每个只需要一个计数。

这个假设显然是错的。在任何文档里，"machine" 和 "learning" 都不是独立的。但分类器需要的不是正确的概率估计，而是正确的排序 -- 哪个类别的概率最高。独立性假设会引入系统性偏差，但这些偏差对所有类别的影响是相似的，因此排序依然正确。

### 它为什么依然有效（Why It Still Works）

三个原因：

1. **要排序，不要校准。** 分类只需要排名第一的类别是对的。即使 P(spam) = 0.99999 而真实概率是 0.7，分类器仍然会正确地判定为垃圾邮件。我们不需要正确的概率，只需要正确的赢家。

2. **高偏差、低方差。** 独立性假设是一个强先验。它对模型施加了很强的约束，从而防止过拟合。在训练数据有限时，一个略有偏差但稳定的模型，胜过一个理论正确但极不稳定的模型。这正是偏差-方差权衡在起作用。

3. **特征冗余相互抵消。** 相关特征提供的是冗余证据。分类器会重复计算这份证据，但它同样会为正确的类别重复计算。如果 "machine" 和 "learning" 总是一起出现，两者都为"科技"类提供证据。NB 把它们数了两遍，但数两遍的都是正确的那个类。

第四个原因是工程层面的：朴素贝叶斯极快。训练只需对数据做一遍词频统计，预测就是一次矩阵乘法。你可以几秒钟内在上百万篇文档上完成训练。这种速度意味着你可以更快迭代、尝试更多特征组合、跑更多实验，这是慢模型给不了的。

### 一步步算一遍（The Math Step by Step）

我们用一个具体例子走一遍。假设有两个类别：垃圾邮件（spam）和非垃圾邮件（not-spam）。词表里有三个词："free"、"money"、"meeting"。

训练数据：
- 垃圾邮件中 "free" 出现 80 次，"money" 60 次，"meeting" 10 次（共 150 个词）
- 非垃圾邮件中 "free" 出现 5 次，"money" 10 次，"meeting" 100 次（共 115 个词）
- 40% 的邮件是垃圾邮件，60% 不是

使用拉普拉斯平滑（alpha=1）：

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

新邮件包含："free"（2 次）、"money"（1 次）、"meeting"（0 次）。

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

垃圾邮件以巨大优势胜出。"free" 出现两次是支持垃圾邮件的有力证据。注意，"meeting" 没有出现对两边对数和的贡献都是零（0 * log(P)) -- 在多项式朴素贝叶斯中，缺席的词不产生任何影响。显式建模"词缺席"的是伯努利朴素贝叶斯。

### 三种变体（Three Variants）

朴素贝叶斯有三种变体，各自用不同方式建模 `P(feature | class)`。

#### 多项式朴素贝叶斯（Multinomial Naive Bayes）

把每个特征建模为计数。最适合特征是词频或 TF-IDF 值的文本数据。

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

`alpha` 就是拉普拉斯平滑（下文解释）。这个变体是文本分类的主力。

#### 高斯朴素贝叶斯（Gaussian Naive Bayes）

把每个特征建模为正态分布。最适合连续特征。

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

每个类别对每个特征都有自己的均值和方差。当特征在每个类别内部确实近似钟形分布时，效果很好。

#### 伯努利朴素贝叶斯（Bernoulli Naive Bayes）

把每个特征建模为二值（出现或不出现）。最适合短文本或二值特征向量。

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

与多项式版本不同，伯努利版本会显式惩罚词的缺席。如果 "free" 通常出现在垃圾邮件里，而这封邮件里没有，伯努利版本会把这一点当作反对"垃圾邮件"的证据。

### 何时用哪种变体（When to Use Each Variant）

| 变体 | 特征类型 | 最适合 | 例子 |
|---------|-------------|----------|---------|
| 多项式（Multinomial） | 计数或词频 | 文本分类、词袋（bag-of-words） | 垃圾邮件过滤、主题分类 |
| 高斯（Gaussian） | 连续值 | 特征近似正态的表格数据 | 鸢尾花分类、传感器数据 |
| 伯努利（Bernoulli） | 二值（0/1） | 短文本、二值特征向量 | 短信垃圾过滤、出现/缺席特征 |

### 拉普拉斯平滑（Laplace Smoothing）

如果一个词出现在测试数据里，却从未在某个类别的训练数据里出现过，会怎样？

不做平滑时：`P(word | class) = 0/N = 0`。乘积中只要有一个零，`P(class | features)` 就是 0，其他证据再多也没用。一个没见过的词就能摧毁整个预测，无论其他证据多么支持它。

拉普拉斯平滑给每个特征的计数加上一个小常数 `alpha`（通常为 1）：

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

在 alpha=1 时，每个词至少有一个极小的概率。测试邮件里出现 "discombobulate" 这样的词不再会杀死垃圾邮件概率。这种平滑有贝叶斯解释：它等价于给词分布放了一个均匀的狄利克雷先验（Dirichlet prior）。

alpha 越大，平滑越强（分布越均匀）；alpha 越小，模型越信任数据。alpha 是一个需要你调的超参数（hyperparameter）。

alpha 的影响：

| Alpha | 效果 | 何时使用 |
|-------|--------|-------------|
| 0.001 | 几乎不平滑，信任数据 | 训练集非常大、预计不会出现未见特征 |
| 0.1 | 轻度平滑 | 较大的训练集 |
| 1.0 | 标准拉普拉斯平滑 | 默认起点 |
| 10.0 | 重度平滑，把分布压平 | 训练集非常小、预计会有很多未见特征 |

### 对数空间计算（Log-Space Computation）

把几百个概率（每个都小于 1）连乘会导致浮点下溢（floating-point underflow）。真实值虽是一个极小的正数，但在浮点数里乘积会变成零。

解决办法：在对数空间里工作。不连乘概率，而是把它们的对数相加：

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

这让预测变成了一个点积：

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

就是矩阵乘法。这正是朴素贝叶斯预测如此之快的原因 -- 它和单层线性模型做的是同一种运算。

### 朴素贝叶斯对逻辑回归（Naive Bayes vs Logistic Regression）

两者都是文本上的线性分类器，区别在于建模的对象。

| 维度 | 朴素贝叶斯 | 逻辑回归 |
|--------|------------|-------------------|
| 类型 | 生成式（建模 P(X\|Y)) | 判别式（建模 P(Y\|X)) |
| 训练 | 统计词频 | 优化损失函数 |
| 小数据 | 更好（强先验有帮助） | 更差（样本不足以估计权重） |
| 大数据 | 更差（错误假设开始拖后腿） | 更好（边界更灵活） |
| 特征 | 假设特征独立 | 能处理相关性 |
| 速度 | 一遍扫描，非常快 | 迭代优化 |
| 校准 | 概率不准 | 概率更好 |

经验法则：先用朴素贝叶斯。如果数据足够多而 NB 已经到顶，就换逻辑回归。

### 分类流水线（Classification Pipeline）

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

实践中我们在对数空间工作以避免浮点下溢。不把许多小概率连乘，而是把它们的对数相加：

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

```figure
naive-bayes
```

## 动手构建（Build It）

`code/naive_bayes.py` 中的代码从零实现了 MultinomialNB 和 GaussianNB。

### MultinomialNB 的实现（MultinomialNB）

从零开始的实现：

1. **fit(X, y)**：对每个类别，统计每个特征的频率，加上拉普拉斯平滑，计算对数概率，并保存类别先验（类别频率的对数）。

2. **predict_log_proba(X)**：对每个样本，为所有类别计算 log P(class) 加上各特征 log P(feature_i | class) 之和。这是一次矩阵乘法：X @ log_probs.T + log_priors。

3. **predict(X)**：返回对数概率最高的类别。

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

关键洞察：拟合完成后，预测只是一次矩阵乘法加一个偏置。这就是朴素贝叶斯快的原因。

### GaussianNB 的实现（GaussianNB）

对连续特征，我们为每个类别的每个特征估计均值和方差：

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

预测时对每个特征使用高斯概率密度函数（PDF），并在特征间相乘（在对数空间里相加）。

### 演示：文本分类（Demo: Text Classification）

代码生成合成的词袋（bag-of-words）数据，模拟两个类别（科技文章对体育文章）。每个类别有不同的词频分布，MultinomialNB 用词频计数对它们分类。

合成数据的构造方式是：创建 200 个"词"（特征列）。词 0-39 在科技文章中高频、在体育文章中低频；词 80-119 在体育文章中高频、在科技文章中低频；词 40-79 在两者中都是中频。这样就构造出一个贴近现实的场景：有些词是强烈的类别指示器，有些词则是噪声。

### 演示：连续特征（Demo: Continuous Features）

代码生成类似鸢尾花（Iris）的数据（3 个类别、4 个特征、高斯簇）。GaussianNB 用每个类别的均值和方差进行分类。每个类别有不同的中心（均值向量）和不同的散布程度（方差），模拟现实中不同类别之间度量系统性不同的数据。

代码还演示了：
- **平滑对比：** 用不同的 alpha 值训练 MultinomialNB，展示平滑强度对准确率的影响。
- **训练规模实验：** 展示随着训练数据从 20 条增长到 1600 条，NB 的准确率如何提升。NB 在样本很少时就能达到不错的准确率 -- 这是它的主要优势。
- **混淆矩阵（confusion matrix）：** 按类别给出精确率（precision）、召回率（recall）和 F1 分数，展示 NB 在哪里犯错。

### 预测速度（Prediction Speed）

朴素贝叶斯的预测就是一次矩阵乘法。对 n 个样本、d 个特征、k 个类别：
- MultinomialNB：一次矩阵乘法 (n x d) @ (d x k) = O(n * d * k)
- GaussianNB：n * k 次高斯 PDF 计算，每次遍历 d 个特征 = O(n * d * k)

两者在每个维度上都是线性的。对比 KNN（需要对所有训练点计算距离）或带 RBF 核的 SVM（需要对所有支持向量做核计算）：在预测阶段，NB 快好几个数量级。

## 直接使用（Use It）

在 sklearn 里，两个变体都是一行代码：

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

用 sklearn 做文本分类：

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

`naive_bayes.py` 中的代码会在同一份数据上把从零实现的版本与 sklearn 对比，验证正确性。

### TF-IDF 搭配朴素贝叶斯（TF-IDF with Naive Bayes）

原始词频对每个词的每次出现一视同仁。但 "the"、"is" 这类常见词在每个类别里都频繁出现 -- 它们不携带任何信息。TF-IDF（词频-逆文档频率，Term Frequency - Inverse Document Frequency）会降低常见词的权重，提高稀有而有区分力的词的权重。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF 值非负，因此可以和 MultinomialNB 配合使用。TF-IDF + MultinomialNB 的组合是文本分类最强的基线之一。在训练样本少于 10,000 的数据集上，它经常击败更复杂的模型。

### 用 BernoulliNB 处理短文本（BernoulliNB for Short Text）

对短文本（推文、短信、聊天消息），BernoulliNB 可能优于 MultinomialNB。短文本词数很少，MultinomialNB 依赖的频率信息噪声很大；BernoulliNB 只关心词出现与否，这对短文本更可靠。

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

CountVectorizer 的 `binary=True` 开关把所有计数转成 0/1。不开这个开关，BernoulliNB 也能跑，但它看到的计数并不是它设计时针对的输入。

### 校准 NB 概率（Calibrating NB Probabilities）

NB 的概率校准很差。当 NB 给出 P(spam) = 0.95 时，真实概率可能只有 0.7。如果你需要可靠的概率估计（比如设定阈值，或与其他模型组合），请用 sklearn 的 CalibratedClassifierCV：

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

它通过交叉验证在 NB 的原始分数之上拟合一个逻辑回归。得到的概率会接近真实的类别频率。

### 常见陷阱（Common Gotchas）

1. **特征取值为负。** MultinomialNB 要求特征非负。如果特征有负值（比如某些设置下的 TF-IDF 或标准化后的特征），改用 GaussianNB，或者把特征平移成正值。

2. **零方差特征。** GaussianNB 要除以方差。如果某特征在某个类别内方差为零（所有取值相同），概率计算就会崩掉。代码给所有方差加了一个很小的平滑项（1e-9）来防止这种情况。

3. **类别不平衡。** 如果 99% 的邮件都不是垃圾邮件，先验 P(not-spam) = 0.99 会强到压过似然证据。你可以手动设定类别先验，或使用 sklearn 的 class_prior 参数。

4. **特征缩放。** MultinomialNB 不需要缩放（它处理的是计数）。GaussianNB 也不需要（它估计逐特征统计量）。这相对逻辑回归和 SVM 是个优势，后两者对特征尺度敏感。

## 发布成果（Ship It）

本课产出：
- `outputs/skill-naive-bayes-chooser.md` -- 一个帮你挑对 NB 变体的决策技能
- `code/naive_bayes.py` -- 从零实现的 MultinomialNB 和 GaussianNB，附 sklearn 对比

### 朴素贝叶斯何时失效（When Naive Bayes Fails）

当独立性假设导致排序错误（而不仅是概率不准）时，NB 就失效了。这发生在：

1. **特征之间有强交互。** 如果类别取决于两个特征的组合而非其中任何一个（类似 XOR 的模式），NB 会完全漏掉。单独每个特征都不提供证据，而 NB 又无法非线性地组合它们。

2. **高度相关的特征给出相反证据。** 如果特征 A 说"是垃圾邮件"、特征 B 说"不是垃圾邮件"，而 A 和 B 又完全相关（现实中它们总是一致），NB 就会在本无冲突的地方看到冲突证据。

3. **训练集非常大。** 数据足够多时，逻辑回归这类判别模型能学到真实的决策边界并超过 NB。小数据时帮了忙的独立性假设，此时反而拖累了模型。

实践中，这些失效模式在文本分类里很少出现。文本特征数量多、单个弱，独立性假设的误差往往会相互抵消。对于特征少且强相关的表格数据，优先考虑逻辑回归或树模型。

## 练习（Exercises）

1. **平滑实验。** 在文本数据上用 0.01、0.1、1.0、10.0、100.0 这几个 alpha 值训练 MultinomialNB，画出准确率对 alpha 的曲线。性能峰值在哪里？为什么 alpha 过大反而有害？

2. **特征独立性检验。** 拿一个真实的文本数据集，选两个明显相关的词（"machine" 和 "learning"）。计算 P(word1 | class) * P(word2 | class)，并与 P(word1 AND word2 | class) 比较。独立性假设错得有多离谱？它影响分类准确率吗？

3. **实现 Bernoulli。** 扩展代码，加一个 BernoulliNB 类。把词袋转成二值（出现/缺席），在文本数据上与 MultinomialNB 比较准确率。伯努利版本什么时候赢？

4. **NB 对逻辑回归。** 在文本数据上训练两者。从 100 个训练样本开始，逐步增加到 10,000。画出两者准确率随训练集规模变化的曲线。逻辑回归在什么规模上反超朴素贝叶斯？

5. **垃圾邮件过滤器。** 构建一个完整的垃圾邮件分类器：对原始邮件文本分词、建立词表、构造词袋特征、训练 MultinomialNB、用精确率和召回率评估（而不只是准确率 -- 为什么？）。

## 关键术语（Key Terms）

| 术语 | 人们的说法 | 实际含义 |
|------|----------------|----------------------|
| 朴素贝叶斯（Naive Bayes） | "简单的概率分类器" | 应用贝叶斯定理、并假设特征在给定类别条件下相互独立的分类器 |
| 条件独立（Conditional independence） | "特征之间互不影响" | P(A, B \| C) = P(A \| C) * P(B \| C) -- 在已知 C 之后，知道 B 不会让你对 A 有任何新的了解 |
| 拉普拉斯平滑（Laplace smoothing） | "加一平滑" | 给每个特征加一个小的计数，防止零概率主导预测 |
| 先验（Prior） | "看到数据之前的信念" | P(class) -- 在观察任何特征之前每个类别的概率 |
| 似然（Likelihood） | "数据和假设的贴合程度" | P(features \| class) -- 已知类别时观察到这些特征的概率 |
| 后验（Posterior） | "看到数据之后的信念" | P(class \| features) -- 观察特征之后更新过的类别概率 |
| 生成模型（Generative model） | "建模数据如何生成" | 先学 P(X \| Y) 和 P(Y)，再用贝叶斯定理得到 P(Y \| X) 的模型 |
| 判别模型（Discriminative model） | "建模决策边界" | 直接学 P(Y \| X)、不建模 X 如何生成的模型 |
| 对数概率（Log probability） | "避免下溢" | 用 log P 代替 P 进行计算，防止许多小数的乘积在浮点数里变成零 |

## 延伸阅读（Further Reading）

- [scikit-learn 朴素贝叶斯文档](https://scikit-learn.org/stable/modules/naive_bayes.html) -- 三种变体及其数学细节
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) -- 多项式对伯努利的经典文本分类对比
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) -- 对文本 NB 的改进
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) -- 证明小数据下 NB 比 LR 收敛更快
