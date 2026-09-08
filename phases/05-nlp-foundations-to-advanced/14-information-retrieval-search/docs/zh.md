# 信息检索与搜索（Information Retrieval and Search）

> BM25 精确但脆弱。密集召回范围广但漏掉关键词。混合是 2026 年默认。其余都是调优。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 minutes

## 问题（The Problem）

用户输入 "what happens if someone lies to get money" 并期望找到实际涵盖那项内容的法规："Section 420 IPC。"关键词搜索完全错过了（没有共享词汇）。语义搜索如果嵌入未在法律文本上训练也会错过。真实搜索必须处理两者。

信息检索（IR）是每个 RAG 系统、每个搜索栏、每个文档站点模糊查找下面的管道。2026 年在生产中有效的架构不是单一方法。它是一个互补方法的链条，每种方法捕捉前一种的失败。

本课构建每个组件并命名每种方法捕捉哪些失败。

## 概念（The Concept）

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

四层。挑选你需要的。

1. **稀疏检索（Sparse retrieval, BM25）。** 快速，精确匹配精确，对语义很差。在倒排索引上运行。在数百万文档上每查询少于 10 毫秒。正确获取法规引用、产品代码、错误消息、命名实体。
2. **密集检索（Dense retrieval）。** 将查询和文档编码为向量。最近邻搜索。捕捉释义和语义相似性。错过仅相差一个字符的关键字匹配。使用 FAISS 或向量数据库每查询 50-200 毫秒。
3. **融合（Fusion）。** 合并稀疏和密集的排序列表。倒数排名融合（Reciprocal Rank Fusion, RRF）是容易的默认选择，因为它忽略原始分数（处于不同尺度）只使用排序位置。当你知道一个信号在领域占主导地位时，加权融合是一个选项。
4. **交叉编码器重排序（Cross-encoder rerank）。** 取融合后的前 30 个。运行交叉编码器（查询 + 文档一起，对每对评分）。保留前 5 个。交叉编码器每对比双编码器慢，但准确得多。你通过在仅前 30 个上运行来摊还成本。

三路检索（BM25 + 密集 + 类似 SPLADE 的学习稀疏）在 2026 年基准测试中优于两路，但需要学习稀疏索引的基础设施。对于大多数团队，两路加交叉编码器重排序是最佳平衡点。

```figure
gx-hybrid-retrieval
```

## 构建它（Build It）

### 步骤 1：从零开始构建 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

两个值得知道的参数。`k1=1.5` 控制词频饱和；更高意味着对词重复给予更多权重。`b=0.75` 控制长度归一化；0 忽略文档长度，1 完全归一化。默认值是 Robertson 在原始论文中的推荐，很少需要调整。

### 步骤 2：使用双编码器进行密集检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

L2 归一化嵌入，使点积等于余弦（cosine）。`all-MiniLM-L6-v2` 是 384 维，快速，对于大多数英语检索来说足够强大。对于多语言工作，使用 `paraphrase-multilingual-MiniLM-L12-v2`。对于最高精度，`bge-large-en-v1.5` 或 `e5-large-v2`。

### 步骤 3：倒数排名融合

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 常数来自原始 RRF 论文。更高的 `k` 使排名差异的贡献扁平化；更低的 `k` 使顶部排名占主导。60 是公布的默认值，很少需要调整。

### 步骤 4：混合搜索 + 重排序

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三个阶段的组合。BM25 找到词汇匹配。密集找到语义匹配。RRF 在不进行分数校准的情况下合并两个排序列表。交叉编码器使用查询-文档对重新为前 30 个评分，捕捉双编码器遗漏的细粒度相关性。保留前 5 个。

### 步骤 5：评估

| 度量（Metric） | 含义（Meaning） |
|--------|---------|
| Recall@k | 在正确答案存在的查询中，有多少在 top-k 中？ |
| MRR（Mean Reciprocal Rank） | 第一个相关文档的 1/排名 的平均值。 |
| nDCG@k | 考虑相关性分级，不只是二元相关/不相关。 |

对于 RAG 来说，**Recall@k** 的检索器是最重要的数字。如果正确的段落不在检索到的集合中，你的阅读器无法回答。

调试提示：对于失败的查询，比较稀疏和密集排序。如果一个找到了正确的文档而另一个没有，你有一个词汇不匹配（修复：添加缺失的一半）或语义歧义（修复：更好的嵌入或重排序器）。

## 使用它（Use It）

2026 年技术栈：

| 规模（Scale） | 栈（Stack） |
|-------|-------|
| 1k-100k 文档 | 内存中 BM25 + `all-MiniLM-L6-v2` 嵌入 + RRF。没有单独的数据库。 |
| 100k-10M 文档 | FAISS 或 pgvector 用于密集 + Elasticsearch / OpenSearch 用于 BM25。并行运行。 |
| 10M+ 文档 | 支持混合的 Qdrant / Weaviate / Vespa / Milvus。交叉编码器重排序前 30 个。 |
| 最高质量前沿 | 三路（BM25 + 密集 + SPLADE）+ ColBERT 晚期交互重排序 |

无论你选择什么，都要为评估预留预算。在基准测试端到端 RAG 准确性之前，基准测试检索召回率。阅读器无法修复检索器遗漏的内容。

### 2026 年生产 RAG 的经验教训

- **80% 的 RAG 失败可追溯到摄取和分块，而不是模型。** 团队花费数周交换 LLM 和调优提示，而检索器安静地每三个查询返回错误的上下文。首先修复分块。
- **分块策略比分块大小更重要。** 固定大小的分割会破坏表格、代码和嵌套标题。句子感知是默认的；对于技术文档和产品手册，语义或基于 LLM 的分块有回报。
- **父文档模式（Parent-doc pattern）。** 检索小的 "子" 块以提高精度。当来自同一父节的多个子块出现时，交换父块以保留上下文。这始终在没有重新训练的情况下提升答案质量。
- **k_rerank=3 通常是最佳的。** 超过此值的每个额外块都会增加词元成本和生成延迟，而不会提升答案质量。如果 k=8 对你来说仍然比 k=3 更好，重排序器表现不佳。
- **HyDE / 查询扩展。** 从查询生成一个假设答案，嵌入它，检索。弥合短问题和长文档之间的措辞差距。免费精度提升，无需训练。
- **8K 词元以下的上下文预算。** 该限制下的一致命中意味着重排序器阈值太松。
- **版本化所有内容。** 提示、分块规则、嵌入模型、重排序器。任何漂移都会静默破坏答案质量。CI 门控（gates）忠实度、上下文精确率和无答案问题率，在用户看到之前阻止回归。
- **三路检索（BM25 + 密集 + 类似 SPLADE 的学习稀疏）优于两路** 在 2026 年基准测试中，特别是对于混合专有名词与语义的查询。当基础设施支持 SPLADE 索引时交付它。

适当设计的检索将幻觉减少了 70-90%，根据 2026 年的行业测量。大多数 RAG 性能提升来自更好的检索，而不是模型微调。

## 发布它（Ship It）

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 练习（Exercises）

1. **简单（Easy）。** 在 500 文档语料库上实现上述 `hybrid_search`。测试 20 个查询。比较 BM25 仅、密集仅和混合的 5 级召回率。
2. **中等（Medium）。** 添加 MRR 计算。对于每个已知正确文档的测试查询，找到正确文档在 BM25、密集和混合排序中的排名。报告每个的 MRR。
3. **困难（Hard）。** 使用 MultipleNegativesRankingLoss（Sentence Transformers）在你的领域微调一个密集编码器。从 500 个查询-文档对构建训练集。比较微调前后的召回率。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| BM25 | 关键词搜索 | Okapi BM25。通过词频、IDF 和长度对文档评分。 |
| 密集检索（Dense retrieval） | 向量搜索 | 将查询 + 文档编码为向量，找到最近邻。 |
| 双编码器（Bi-encoder） | 嵌入模型 | 独立编码查询和文档。查询时快速。 |
| 交叉编码器（Cross-encoder） | 重排序器模型 | 一起编码查询 + 文档。慢但准确。 |
| RRF | 排序融合 | 通过求和 `1/(k + rank)` 合并两个排序列表。 |
| Recall@k | 检索度量 | 相关文档在 top-k 中的查询比例。 |

## 延伸阅读（Further Reading）

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — 权威的 BM25 论述。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，问答的规范双编码器。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 缩小与密集差距的学习稀疏检索器。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 论文。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — 晚期交互检索。
