# 主题模型——LDA 与 BERTopic（Topic Modeling — LDA and BERTopic）

> LDA：文档是主题的混合，主题是词上的分布。BERTopic：文档在嵌入空间中聚类，聚类就是主题。相同的目标，不同的分解。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word2Vec)
**Time:** ~45 minutes

## 问题（The Problem）

你有 10,000 条客户支持工单、50,000 篇新闻文章或 200,000 条推文。你需要在不阅读的情况下知道这个集合是关于什么的。你没有标记的类别。你甚至不知道有多少类别存在。

主题模型无需监督地回答这个问题。给它一个语料库，返回一个小的连贯主题集，并且对于每个文档，返回一个关于这些主题的分布。

两个算法家族占据主导地位。LDA（2003）将每个文档视为潜在主题的混合，每个主题视为词上的分布。推断是贝叶斯的。当你需要混合成员（mixed-membership）主题分配和可解释的词级概率分布时，它仍然在生产环境中交付。

BERTopic（2020）使用 BERT 编码文档，使用 UMAP 降低维度，使用 HDBSCAN 聚类，并通过基于类的 TF-IDF 提取主题词。它在短文本、社交媒体和任何语义相似性比词重叠更重要的事物上获胜。一个文档得到一个主题，这是长文内容的限制。

本课为两者建立直觉，并命名对于给定语料库应该选择哪一个。

## 概念（The Concept）

![LDA mixture model vs BERTopic clustering](../assets/topic-modeling.svg)

**LDA 生成故事。** 每个主题是一个词上的分布。每个文档是主题的混合。要在一个文档中生成一个词，从文档的混合中抽取一个主题，然后从该主题的分布中抽取一个词。推断反过来：给定观察到的词，推断每个文档的主题分布和每个主题的词分布。折叠吉布斯采样（Collapsed Gibbs sampling）或变分贝叶斯（variational Bayes）进行数学运算。

关键 LDA 输出：

- `doc_topic`：矩阵 `(n_docs, n_topics)`，每行和为 1（文档的主题混合）。
- `topic_word`：矩阵 `(n_topics, vocab_size)`，每行和为 1（主题的词分布）。

**BERTopic 管道。**

1. 使用句子 transformer（例如 `all-MiniLM-L6-v2`）编码每个文档。384 维向量。
2. 使用 UMAP 将维度降低到约 5 维。BERT 嵌入对聚类来说太高维了。
3. 使用 HDBSCAN 聚类。基于密度，产生可变大小的聚类和一个 "异常值（outlier）" 标签。
4. 对于每个聚类，通过基于类的 TF-IDF 在聚类的文档上计算以提取顶级词。

输出是每个文档一个主题（加上 -1 异常值标签）。可选地，通过 HDBSCAN 的概率向量进行软成员（soft membership）。

```figure
topic-drift
```

## 构建它（Build It）

### 步骤 1：通过 scikit-learn 使用 LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注意：去除停用词，min_df 和 max_df 过滤稀有和普遍的词，CountVectorizer（不是 TfidfVectorizer），因为 LDA 期望原始计数。

### 步骤 2：BERTopic（生产环境）

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

对 `Topic != -1` 的过滤会删除 BERTopic 的异常值桶（HDBSCAN 无法聚类的文档）。`min_topic_size` 控制 HDBSCAN 的最小聚类大小；BERTopic 的库默认是 10。本示例为了本课程的规模显式将其设置为 15。对于超过 10,000 个文档的语料库，增加到 50 或 100。

### 步骤 3：评估

两种方法都输出主题词。问题是这些词是否连贯。

- **主题连贯性（c_v）。** 结合滑动窗口上下文上顶级词对的 NPMI（归一化点态互信息），将分数聚合成主题向量，然后通过余弦相似度比较这些向量。越高越好。使用 `gensim.models.CoherenceModel` 配合 `coherence="c_v"`。
- **主题多样性（Topic diversity）。** 所有主题顶级词中唯一词的比例。越高越好（主题不重叠）。
- **定性检查（Qualitative inspection）。** 阅读每个主题的顶级词。它们命名了一个真实的事物吗？人类判断仍然是最后一道防线。

## 选择哪一个

| 情况（Situation） | 选择（Pick） |
|-----------|------|
| 短文本（推文、评论、标题） | BERTopic |
| 有主题混合的长文档 | LDA |
| 无 GPU / 有限计算 | LDA 或 NMF |
| 需要文档级多主题分布 | LDA |
| LLM 集成用于主题标签 | BERTopic（直接支持） |
| 资源受限的边缘部署 | LDA |
| 最大语义连贯性 | BERTopic |

最大的实际考虑是文档长度。BERT 嵌入会截断；LDA 计数适用于任何长度。对于超过嵌入模型上下文窗口的文档，要么分块 + 聚合，要么使用 LDA。

## 使用它（Use It）

2026 年技术栈：

- **BERTopic。** 短文本和任何语义重要的内容的默认选择。
- **`gensim.models.LdaModel`。** 经典 LDA 用于生产环境，成熟，久经考验。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 实验的简单 LDA。
- **NMF。** 非负矩阵分解。LDA 的快速替代品，短文本质量相当。
- **Top2Vec。** 与 BERTopic 类似的设计。社区较小，但在某些基准上表现良好。
- **FASTopic。** 更新，在非常大的语料库上比 BERTopic 更快。
- **基于 LLM 的标签。** 运行任何聚类，然后提示模型命名每个聚类。

## 发布它（Ship It）

保存为 `outputs/skill-topic-picker.md`：

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## 练习（Exercises）

1. **简单（Easy）。** 在 20 Newsgroups 数据集上用 5 个主题拟合 LDA。打印每个主题的前 10 个词。手动标记每个主题。算法找到了真实类别吗？
2. **中等（Medium）。** 在相同的 20 Newsgroups 子集上拟合 BERTopic。比较发现的主题数量、顶级词和与 LDA 的定性连贯性。哪一个更清晰地呈现真实类别？
3. **困难（Hard）。** 在你的语料库上为 LDA 和 BERTopic 计算 c_v 连贯性。分别以 5、10、20、50 个主题运行每种方法。绘制连贯性 vs 主题数量。报告哪种方法在主题数量上更稳定。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| 主题（Topic） | 语料库关于的事物 | 词上的概率分布（LDA）或相似文档的聚类（BERTopic）。 |
| 混合成员（Mixed membership） | 文档是多个主题 | LDA 为每个文档分配一个在所有主题上的分布。 |
| UMAP | 降维 | 保留局部结构的流形学习；在 BERTopic 中使用。 |
| HDBSCAN | 密度聚类 | 发现可变大小的聚类；为异常值产生 "噪音" 标签（-1）。 |
| c_v 连贯性 | 主题质量度量 | 滑动窗口内顶级主题词的平均互信息。 |

## 延伸阅读（Further Reading）

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA 论文。
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic 论文。
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — 引入 c_v 及其朋友的论文。
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — 生产参考。优秀的示例。
