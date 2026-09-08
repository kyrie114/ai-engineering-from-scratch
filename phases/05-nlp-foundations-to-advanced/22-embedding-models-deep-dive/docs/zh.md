# 嵌入模型——2026 深入剖析（Embedding Models — The 2026 Deep Dive）

> Word2Vec 给你每个词一个向量。现代嵌入模型给你每个段落一个向量，跨语言，具有稀疏、密集和多向量视图，大小适合你的索引。选错，你的 RAG 就会检索到错误的东西。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec), Phase 5 · 14 (Information Retrieval)
**Time:** ~60 minutes

## 问题（The Problem）

你的 RAG 系统 40% 的时间检索到错误的段落。罪魁祸首很少是向量数据库或提示。它是嵌入模型。

2026 年选择嵌入意味着在五个轴上挑选：

1. **密集 vs 稀疏 vs 多向量。** 每个段落一个向量，或每个词元一个，或一个稀疏加权词袋。
2. **语言覆盖。** 单语言英语模型在仅英语任务上仍然获胜。多语言模型在语料库混合时获胜。
3. **上下文长度。** 512 个词元 vs 8,192 vs 32,768——真实有效容量通常是广告最大值的 60-70%。
4. **维度预算。** 3,072 个浮点数全精度 = 每个向量 12 KB。在 1 亿向量时，存储是每月 $1,300。Matryoshka 截断将其减少 4 倍。
5. **开放 vs 托管。** 开放权重意味着你控制栈和数据。托管意味着你为了始终最新而交换控制。

本课命名权衡，以便你可以基于证据选择，而不是基于上个季度的流行。

## 概念（The Concept）

![Dense, sparse, and multi-vector embeddings](../assets/embedding-modes.svg)

**密集嵌入（Dense embeddings）。** 每个段落一个向量（通常 384-3,072 维）。余弦相似度按语义接近度对段落排序。OpenAI `text-embedding-3-large`、BGE-M3 密集模式、Voyage-3。默认选择。

**稀疏嵌入（Sparse embeddings）。** SPLADE 风格。一个 transformer 为每个词汇表词元预测一个权重，然后将大多数归零。结果是一个大小为 |词汇表| 的稀疏向量。捕捉词汇匹配（像 BM25）但具有学习的词权重。在关键词繁重的查询上强劲。

**多向量（晚期交互，Multi-vector (late interaction)）。** ColBERTv2、Jina-ColBERT。每个词元一个向量。使用 MaxSim 评分：对于每个查询词元，找到最相似的文档词元，对分数求和。存储和评分更昂贵，但在长查询和领域特定语料库上获胜。

**BGE-M3：三合一。** 单个模型同时输出密集、稀疏和多向量表示。每个可以独立查询；分数通过加权和融合。当你想要一个检查点的灵活性时，2026 年的默认选择。

**Matryoshka 表示学习（Matryoshka Representation Learning）。** 训练使向量的前 N 维形成一个有用的独立嵌入。将 1,536 维向量截断为 256 维，精度损失约 1%，存储节省 6 倍。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+ 都支持。

### MTEB 排行榜讲述了一个部分的故事

大规模文本嵌入基准——在推出时（2022）跨 8 种任务类型的 56 个任务，在 MTEB v2 中扩展到 100+ 任务。2026 年初，Gemini Embedding 2 在检索中名列前茅（67.71 MTEB-R）。Cohere embed-v4 在通用方面领先（65.2 MTEB）。BGE-M3 在开放权重多语言中领先（63.0）。排行榜是必要的但不是充分的——始终在你的领域基准测试。

### 三层模式

| 使用案例 | 模式 |
|----------|---------|
| 快速初传 | 密集双编码器（BGE-M3、text-3-small） |
| 召回提升 | 稀疏（SPLADE、BGE-M3 sparse）+ RRF 融合 |
| 前 50 的精度 | 多向量（ColBERTv2）或交叉编码器重排序 |

大多数生产栈使用全部三种。

```figure
gx-matryoshka
```

## 构建它（Build It）

### 步骤 1：基线——使用 Sentence-BERT 的密集嵌入

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 使点积等于余弦相似度。始终设置它。

### 步骤 2：Matryoshka 截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后重新归一化。Nomic v1.5、OpenAI text-3 和 Voyage-4 训练使这对于前几个级别是无损的。非 Matryoshka 模型（原始 Sentence-BERT）在截断时急剧下降。

### 步骤 3：BGE-M3 多功能性

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

一个推理调用的三个索引。分数融合：

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

在你的领域上调优权重。

### 步骤 4：自定义任务上的 MTEB 评估

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在你的*代表性*子集上运行候选模型。不要单独信任排行榜排名——你的领域很重要。

### 步骤 5：从零开始的手工余弦

见 `code/main.py`。平均哈希技巧嵌入（stdlib-only）。无法与 transformer 嵌入竞争——但它展示了形状：分词 → 向量 → 归一化 → 点积。

## 陷阱

- **查询和文档使用相同模型。** 一些模型（Voyage、Jina-ColBERT）使用非对称编码——查询和文档通过不同的路径。始终检查模型卡。
- **缺少前缀。** `bge-*` 模型需要在查询前加上 `"Represent this sentence for searching relevant passages: "`。如果你忘记了，3-5 点的召回差距。
- **过度修剪 Matryoshka。** 1,536 → 256 通常安全。1,536 → 64 不安全。在你的评估集上验证。
- **上下文截断。** 大多数模型静默截断超过最大长度的输入。长文档需要分块（见第 23 课）。
- **忽略延迟尾部。** MTEB 分数隐藏 p99 延迟。一个 600M 模型可能比一个 335M 模型高 2 分，但每个查询贵 3 倍。

## 使用它（Use It）

2026 年技术栈：

| 情况 | 选择 |
|-----------|------|
| 仅英语，快速，API | `text-embedding-3-large` 或 `voyage-3-large` |
| 开放权重，英语 | `BAAI/bge-large-en-v1.5` |
| 开放权重，多语言 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| 长上下文（32k+） | Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B |
| 仅 CPU 部署 | Nomic Embed v2（137M 参数，MoE） |
| 存储受限 | Matryoshka 截断 + int8 量化 |
| 关键词繁重的查询 | 添加 SPLADE 稀疏，RRF 与密集融合 |

2026 模式：从 BGE-M3 或 text-3-large 开始，在你的领域上用 MTEB 评估，如果领域特定模型以超过 3 分的优势获胜，则交换。

## 发布它（Ship It）

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## 练习（Exercises）

1. **简单（Easy）。** 使用 `bge-small-en-v1.5` 以全维度（384）编码 100 个句子，然后以 Matryoshka 128 编码。在 10 个查询上测量 MRR 下降。
2. **中等（Medium）。** 在你的领域 500 个段落上比较 BGE-M3 密集、稀疏和 ColBERT。哪一个在 recall@10 上获胜？RRF 融合是否击败最佳单一模式？
3. **困难（Hard）。** 在你的前 2 个领域任务上跨三个候选模型运行 MTEB。报告 MTEB 分数、100 查询批次的 p99 延迟和 $/1M 查询。选择帕累托最优。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| 密集嵌入 | 向量 | 每个文本一个固定大小的向量。余弦相似度排序。 |
| 稀疏嵌入 | 学习的 BM25 | 每个词汇表词元一个权重；大多数为零；端到端训练。 |
| 多向量 | ColBERT 风格 | 每个词元一个向量；MaxSim 评分；更大的索引，更好的召回。 |
| Matryoshka | 俄罗斯套娃技巧 | 前 N 维本身就是有效的更小嵌入。 |
| MTEB | 基准 | 大规模文本嵌入基准——推出时 56 个任务，v2 中 100+ 个。 |
| BEIR | 检索基准 | 18 个零样本检索任务；经常引用用于跨领域鲁棒性。 |
| 非对称编码 | 查询 ≠ 文档路径 | 模型对查询和文档使用不同的投影。 |

## 延伸阅读（Further Reading）

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — 双编码器论文。
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — 排行榜论文。
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — 统一三模式模型。
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — 维度阶梯训练目标。
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — 生产中的晚期交互。
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — 实时排名。
