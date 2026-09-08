# 词袋、TF-IDF 与文本表示（Bag of Words, TF-IDF, and Text Representation）

> 先计数，后思考。在定义明确的任务上，TF-IDF 在2026年仍优于嵌入。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 02 (Linear Regression from Scratch)
**Time:** ~75 minutes

## 问题（The Problem）

模型需要数字。你有的却是字符串。

每个 NLP 流水线都要回答同一个问题。如何将变长的词元流转换为分类器可以消费的固定大小向量。该领域给出的第一个答案是最简单的那个可行的方案：统计词频。生成一个向量。

这个向量承载的生产 NLP 比任何嵌入模型都多。垃圾邮件过滤器、主题分类器、日志异常检测、搜索排名（BM25 之前）、第一波情感分析、第一个十年的学术 NLP 基准。2026 年的从业者在狭窄的分类任务上仍然首选它。它快速、可解释，而且在词的存在是关键的场景下，通常与一个 4 亿参数的嵌入模型没有区别。

本课从零开始构建词袋（BoW），然后构建 TF-IDF。然后展示 scikit-learn 用三行代码完成相同工作。然后指出那个让你转向嵌入的失败模式。

## 概念（The Concept）

**词袋（Bag of Words, BoW）** 丢弃顺序。对于每个文档，统计词汇表中每个词出现的次数。向量长度等于词汇表大小。位置 `i` 是词 `i` 的计数。

**TF-IDF** 对 BoW 进行重新加权。在每个文档中都出现的词没有信息量，因此将其缩小。在整个语料库中罕见但在单个文档中频繁出现的词是信号，因此将其放大。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是文档中的词频，`df` 是文档频率（包含该词的文档数量），`N` 是文档总数。`log` 使普遍词的权重保持有界。

关键特性：两者都产生具有可解释轴的稀疏向量。你可以查看训练好的分类器的权重，读出哪些词将文档推向每个类别。你无法用一个 768 维的 BERT 嵌入做到这一点。

```figure
bow-tfidf
```

## 构建它（Build It）

### 步骤1：构建词汇表（Step 1: build the vocabulary）

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：分词后的文档列表（任何词级分词器都可以；本课的 `code/main.py` 使用简化的 lowercase 变体）。输出：`{word: index}` 字典。稳定的插入顺序意味着词索引 0 是第一个文档中看到的第一个词。惯例各不相同；scikit-learn 按字母顺序排序。

### 步骤2：词袋（Step 2: bag of words）

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行是文档。列是词汇表索引。条目 `[i][j]` 表示"词 `j` 在文档 `i` 中出现了多少次。" 文档1有 `cat` 两次，因为它确实如此。文档0有 `ran` 零次，因为它没有。

### 步骤3：词频与文档频率（Step 3: term frequency and document frequency）

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

两个值得命名的平滑技巧。`(n+1)/(d+1)` 避免了 `log(x/0)`。尾部的 `+1` 确保在每个文档中都出现的词仍有 IDF 1（不是 0），与 scikit-learn 的默认值一致。其他实现使用原始 `log(N/df)`。两者都有效；平滑版本更友好。

### 步骤4：TF-IDF（Step 4: TF-IDF）

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三个文档，五个词汇词（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 出现在全部三个中，因此其 IDF 较低。`dog` 只出现一次，因此其 IDF 较高。向量是稀疏的（大多数条目都很小），有区分度的词会凸显。

### 步骤5：L2 归一化行（Step 5: L2-normalize rows）

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

没有归一化，较长的文档会得到更大的向量并主导相似度分数。L2 归一化将每个文档放在单位超球面上。行之间的余弦相似度现在只是点积。

## 使用它（Use It）

scikit-learn 提供了生产版本。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 在一次调用中完成分词、词汇表和 BoW。`TfidfVectorizer` 增加了 IDF 加权和 L2 归一化。两者都返回稀疏矩阵。对于 10 万个文档，密集版本无法放入内存；在分类器需要密集输入之前保持稀疏。

改变一切的参数：

| 参数（Arg） | 效果（Effect） |
|-----|--------|
| `ngram_range=(1, 2)` | 包含二元语法。通常提升分类效果。 |
| `min_df=2` | 删除在少于 2 个文档中出现的词。在噪声数据上修剪词汇表。 |
| `max_df=0.95` | 删除在超过 95% 的文档中出现的词。在没有硬编码列表的情况下近似停用词移除。 |
| `stop_words="english"` | scikit-learn 内置的停用词列表。取决于任务——情感分析不应该丢弃否定词。 |
| `sublinear_tf=True` | 使用 `1 + log(tf)` 而不是原始 `tf`。当一个词在文档中重复出现多次时会有帮助。 |

### TF-IDF 何时仍然胜出（截至2026年）（When TF-IDF still wins (as of 2026)）

- 垃圾邮件检测、主题标注、日志异常标记。词的存在是关键的；语义细微差别不是。
- 低数据场景（数百个标注示例）。TF-IDF 加上逻辑回归没有预训练成本。
- 延迟很重要的任何地方。TF-IDF 加上线性模型在微秒内响应。通过 Transformer 嵌入一个文档需要 10-100 毫秒。
- 必须解释预测的系统。检查分类器的系数。顶部正向词就是原因。

### TF-IDF 何时失败（When TF-IDF fails）

语义 blindness 失败。考虑这两个文档：

- "The movie was not good at all."
- "The movie was excellent."

一个是负面评论。一个是正面评论。它们的 TF-IDF 重叠恰好是 `{the, movie, was}`。词袋分类器必须记住 `not` 靠近 `good` 会翻转标签。它可以在足够的数据上学到这一点，但永远不会像理解语法的模型那样优雅。

另一个失败：推理时的词表外词。在 IMDb 评论上训练的 BoW 模型对 `Zoomer-approved` 一无所知，如果这个词元从未出现在训练中。子词嵌入（第04课）处理这个问题。TF-IDF 无法处理。

### 混合方案：TF-IDF 加权嵌入（Hybrid: TF-IDF weighted embeddings）

2026 年中数据分类的实用默认方案：使用 TF-IDF 权重作为词嵌入上的注意力。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你从嵌入中获得语义容量，从 TF-IDF 中获得稀有词强调。分类器在池化向量上训练。在 5 万个标注示例以下的情感、主题和意图分类中，这比单独使用任何一种都表现更好。

## 交付使用（Ship It）

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: 给定一个文本分类任务，推荐 BoW、TF-IDF、嵌入或混合方案。
phase: 5
lesson: 02
---

你推荐文本向量化策略。给定任务描述，输出：

1. 表示方法（BoW、TF-IDF、transformer 嵌入或混合方案）。用一句话解释原因。
2. 具体的向量化器配置。命名库。引用参数（`ngram_range`、`min_df`、`max_df`、`sublinear_tf`、`stop_words`）。
3. 发布前应该测试的一种失败模式。

当用户有少于 500 个标注示例且没有 TF-IDF 基线表现出语义失败证据时，拒绝推荐嵌入。拒绝在情感分析中移除停用词（否定词携带信号）。将类别不平衡标记为需要不仅仅是向量化器的更改。

示例输入："对 3 万个客户支持工单进行 12 分类。大多数工单是 2-3 句话。仅英语。审计日志需要可解释性。"

示例输出：

- 表示方法：TF-IDF。3 万个示例不算小；可解释性要求排除了密集嵌入。
- 配置：`TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`。保留停用词，因为类别关键词有时就是停用词（"not working" vs "working"）。
- 需要测试的失败：验证 `min_df=3` 没有删除稀有类别关键词。运行按类过滤的 `get_feature_names_out` 并目视检查。
```

## 练习（Exercises）

1. **简单。** 在 L2 归一化的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档得分 1.0，词汇表不相交的文档得分 0.0。
2. **中等。** 向 `bag_of_words` 添加 `n-gram` 支持。参数 `n` 产生对 `n` 元组的计数。测试 `n=2` 在 `["the", "cat", "sat"]` 上是否产生 `["the cat", "cat sat"]` 的二元组计数。
3. **困难。** 使用 GloVe 100d 向量构建上面的 TF-IDF 加权嵌入混合方案（下载一次，缓存）。在 20 Newsgroups 数据集上比较分类准确率与纯 TF-IDF 和纯均值池化嵌入。报告各自在什么情况下胜出。

## 关键术语（Key Terms）

| 术语（Term） | 人们的说法（What people say） | 实际含义（What it actually means） |
|------|-----------------|-----------------------|
| BoW | 词频向量 | 一个文档中词汇表词的计数。丢弃顺序。 |
| TF | 词频 | 文档中一个词的计数，可选地按文档长度归一化。 |
| DF | 文档频率 | 至少包含该词的文档数量。 |
| IDF | 逆文档频率 | `log(N / df)` 平滑后。降低到处出现的词的权重。 |
| 稀疏向量 | 大部分为零 | 词汇表通常是 1 万-10 万个词；大多数词在任何给定文档中都不存在。 |
| 余弦相似度 | 向量夹角 | L2 归一化向量的点积。1 是相同，0 是正交。 |

## 延伸阅读（Further Reading）

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 规范 API 参考，以及对每个参数的说明。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — 使 TF-IDF 成为十年默认值的论文。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026 年关于旧方法何时胜出以及为什么的观点。
