# 词嵌入 — 从零开始的 Word2Vec（Word Embeddings — Word2Vec from Scratch）

> 一个词就是它陪伴的公司。在这个想法上训练一个浅层网络，几何就会浮现。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 minutes

## 问题（The Problem）

TF-IDF 知道 `dog` 和 `puppy` 是不同的词。它不知道它们的意思几乎相同。在 `dog` 上训练的分类器无法泛化到关于 `puppy` 的评论。你可以通过列出同义词来弥补，但这在稀有词、领域术语和你未预料到的每一种语言上都失败了。

你希望得到一种表示，其中 `dog` 和 `puppy` 在空间中落在相近的位置。`king - man + woman` 落在 `queen` 附近。在 `dog` 上训练的模型免费传递一些信号给 `puppy`。

Word2Vec 给了我们这个空间。两层神经网络，万亿词元训练运行，2013 年发布。架构几乎简单得尴尬。结果重塑了 NLP 长达十年。

## 概念（The Concept）

**分布假设（Distributional hypothesis）**（Firth, 1957）："你从它所陪伴的公司来认识一个词。" 如果两个词出现在相似的上下文中，它们的意思可能相似。

Word2Vec 有两种风格，都利用了这一点。

- **Skip-gram。** 给定中心词，预测周围的词。`cat -> (the, sat, on)`，窗口大小为 2。
- **CBOW（连续词袋）。** 给定周围的词，预测中心词。`(the, sat, on) -> cat`。

Skip-gram 训练较慢但处理稀有词更好。它成为了默认选择。

网络有一个没有非线性的隐藏层。输入是词汇表上的独热向量。输出是词汇表上的 softmax。训练后，丢弃输出层。隐藏层权重就是嵌入。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

技巧：对 10 万个词进行 softmax 计算成本高得令人望而却步。Word2Vec 使用 **负采样（negative sampling）** 将其转换为二分类任务。预测"这个上下文词是否出现在这个中心词附近，是或否"。为每个训练对采样一小部分负（非共现）词，而不是在整个词汇表上计算 softmax。

```figure
word-vector-arithmetic
```

## 构建它（Build It）

### 步骤1：从语料库生成训练对（Step 1: training pairs from a corpus）

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口中的每一对（中心词，上下文词）都是正训练样本。

### 步骤2：嵌入表（Step 2: embedding tables）

两个矩阵。`W` 是中心词嵌入表（你要保留的那个）。`W'` 是上下文词表（通常丢弃，有时与 `W` 平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小的随机初始化。词汇表大小 1 万，维度 100 是现实的；教学上，50 个词 × 16 维足以看到几何结构。

### 步骤3：负采样目标（Step 3: negative sampling objective）

对于每个正样本对 `(center, context)`，从词汇表中采样 `k` 个随机词作为负样本。训练模型使正样本的点积 `W[center] · W'[context]` 较高，负样本较低。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

神奇公式：正样本对上的逻辑损失（希望 sigmoid 接近 1）加上负样本对上的逻辑损失（希望 sigmoid 接近 0）。梯度流向两个表。完整推导见原始论文；如果你想彻底理解，用纸笔走一遍。

### 步骤4：在小语料库上训练（Step 4: train on a toy corpus）

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在大型语料库上训练足够多的轮次后，共享上下文的词具有相似的中心嵌入。在小语料库上，你会隐约看到效果。在数十亿词元上，你会戏剧性地看到它。

### 步骤5：类比技巧（Step 5: the analogy trick）

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的 300 维 Google News 向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。不是因为模型知道什么是皇室。因为向量 `(king - man)` 捕获了类似"皇家"的东西，将其添加到 `woman` 会落在皇家女性区域附近。

## 使用它（Use It）

从零编写 Word2Vec 是教学用途。生产 NLP 使用 `gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

对于实际工作，你几乎从不自己训练 Word2Vec。你下载预训练的向量。

- **GloVe** — 斯坦福的共现矩阵分解方法。50d、100d、200d、300d 检查点。良好的通用覆盖。第04课专门涵盖 GloVe。
- **fastText** — Facebook 的 Word2Vec 扩展，嵌入字符 n 元组。通过组合子词处理词表外词。第04课。
- **Google News 上的预训练 Word2Vec** — 300d，300 万词汇，2013 年发布。至今每天仍被下载。

### Word2Vec 在2026年仍然胜出的场景（When Word2Vec still wins in 2026）

- 轻量级领域特定检索。在一小时的笔记本电脑上训练医学摘要，获得通用模型无法捕获的专业化向量。
- 类比式特征工程。`gender_vector = mean(man - woman pairs)`。从其他词中减去它以获得性别中立轴。在公平性研究中仍在使用。
- 可解释性。100 维足够小，可以通过 PCA 或 t-SNE 绘图，并实际看到聚类形成。
- 推理必须在没有 GPU 的设备上运行的任何地方。Word2Vec 查找是单行获取。

### Word2Vec 在哪里失败（Where Word2Vec fails）

多义词墙（polysemy wall）。`bank` 只有一个向量。`river bank` 和 `financial bank` 共享它。`table`（电子表格 vs. 家具）共享它。下游分类器无法从向量中区分词义。

上下文嵌入（ELMo、BERT、此后的每个 Transformer）通过根据周围上下文为词的每次出现生成不同的向量来解决这个问题。这就是从 Word2Vec 到 BERT 的跳跃：从静态到上下文。第7阶段涵盖 Transformer 部分。

词表外问题是另一个失败。如果不在训练数据中，Word2Vec 从未见过 `Zoomer-approved`。没有回退方案。fastText 通过子词组合修复了这个问题（第04课）。

## 交付使用（Ship It）

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: 检查 word2vec 模型。运行类比、查找邻居、诊断质量。
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

你探测训练好的词嵌入以验证它们正常工作。给定 `gensim.models.KeyedVectors` 对象和词汇表，你运行：

1. 三个规范类比测试。`king : man :: queen : woman`。`paris : france :: tokyo : japan`。`walking : walked :: swimming : ?`。报告 top-1 结果及其余弦值。
2. 五个用户提供的领域特定词的最近邻测试。打印带余弦值的 top-5 邻居。
3. 一个对称性检查。`similarity(a, b) == similarity(b, a)` 在浮点精度范围内。
4. 一个退化检查。如果任何嵌入的范数低于 0.01 或高于 100，模型有训练 bug。标记它。

拒绝仅凭类比准确率就宣布模型良好。类比基准可以被操纵，且不会转移到下游任务。推荐内在评估 + 下游评估一起使用。
```

## 练习（Exercises）

1. **简单。** 在小型语料库（20 个关于猫和狗的句子）上运行训练循环。200 个轮次后，验证 `nearest(vocab, W, W[vocab["cat"]])` 在其 top 3 中返回 `dog`。如果不是，增加轮次或词汇表。
2. **中等。** 添加高频词二次采样。频率高于 `10^-5` 的词以与其频率成比例的概率从训练对中删除。测量对稀有词相似度的影响。
3. **困难。** 在 20 Newsgroups 语料库上训练模型。计算两个偏差轴：`he - she` 和 `doctor - nurse`。将职业词投影到两个轴上。报告哪些职业的偏差差距最大。这是公平性研究人员使用的探测类型。

## 关键术语（Key Terms）

| 术语（Term） | 人们的说法（What people say） | 实际含义（What it actually means） |
|------|-----------------|-----------------------|
| 词嵌入 | 词作为向量 | 从上下文中学习的密集、低维（通常 100-300）表示。 |
| Skip-gram | Word2Vec 技巧 | 从中心词预测上下文词。比 CBOW 慢，对稀有词更好。 |
| 负采样 | 训练捷径 | 用对 `k` 个随机词的二分类替换对整个词汇表的 softmax。 |
| 静态嵌入 | 每个词一个向量 | 无论上下文如何，向量都相同。在多义词上失败。 |
| 上下文嵌入 | 上下文敏感向量 | 基于周围词，词的每次出现有不同的向量。Transformer 产生的就是这种。 |
| OOV | 词表外 | 训练中未见过的词。Word2Vec 无法为这些词生成向量。 |

## 延伸阅读（Further Reading）

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — 负采样论文。简短且易读。
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — 最清晰的梯度推导，如果原始论文的数学让你感觉密集。
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — 实际有效的生产训练设置。
