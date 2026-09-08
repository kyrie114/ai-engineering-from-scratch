# RAG 分块策略（Chunking Strategies for RAG）

> 分块配置对检索质量的影响与嵌入模型的选择一样大（Vectara NAACL 2025）。分块错了，任何重排序都救不了你。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60 minutes

## 问题（The Problem）

你把一份 50 页的合同放进 RAG 系统。用户问："终止条款是什么？"检索器返回封面页。为什么？因为模型是在 512 词元块上训练的，终止条款在 20 页之后，跨页分割，没有与查询关联的本地关键词。

修复不是 "买一个更好的嵌入模型。"修复是分块。多大？重叠？在哪里分割？有周围上下文？

2026 年 2 月的基准测试显示出令人惊讶的结果：

- Vectara 的 2026 研究：递归 512 词元分块在 69% → 54% 准确性上击败了语义分块。
- SPLADE + Mistral-8B 在 Natural Questions 上：重叠提供零可衡量的好处。
- 上下文悬崖（Context cliff）：响应质量在约 2,500 个词元的上下文周围急剧下降。

"明显"的答案（语义分块，20% 重叠，1000 个词元）通常是错的。本课为六种策略建立直觉，并告诉你何时使用哪一个。

## 概念（The Concept）

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**固定分块（Fixed chunking）。** 每 N 个字符或词元分割。最简单的基线。打断句子中间。良好的压缩，糟糕的连贯性。

**递归（Recursive）。** LangChain 的 `RecursiveCharacterTextSplitter`。首先尝试在 `\n\n` 上分割，然后 `\n`，然后 `.`，然后空格。干净地回退。2026 年默认。

**语义（Semantic）。** 嵌入每个句子。计算相邻句子之间的余弦相似度。在相似度低于阈值的地方分割。保留主题连贯性。较慢；有时产生伤害检索的 40 词元微小片段。

**句子（Sentence）。** 在句子边界上分割。每个块一个句子或 N 个句子的窗口。在约 5k 词元处以语义分块的一小部分成本匹配语义分块。

**父文档（Parent-document）。** 存储小的子块用于检索*和*用于上下文的较大父块。按子检索；返回父。优雅降级：糟糕的子块仍然返回合理的父块。

**晚期分块（Late chunking, 2024）。** 首先在词元级别嵌入整个文档，然后将词元嵌入池化为块嵌入。保留跨块上下文。与长上下文嵌入器（BGE-M3、Jina v3）一起工作。更高的计算成本。

**上下文检索（Contextual retrieval, Anthropic, 2024）。** 在块之前加上 LLM 生成的其在文档中位置的摘要（"This chunk is section 3.2 of the termination clauses..."）。在 Anthropic 自己的基准测试中提高 35-50% 检索。索引昂贵。

### 击败每个默认的规则

将块大小与查询类型匹配：

| 查询类型 | 块大小 |
|------------|-----------|
| 事实（"what is the CEO's name?"） | 256-512 词元 |
| 分析性 / 多跳 | 512-1024 词元 |
| 整节理解 | 1024-2048 词元 |

NVIDIA 2026 基准测试。块应该足够大以包含答案加上本地上下文，足够小以至于检索器的 top-K 返回专注于答案而不是上下文噪音。

```figure
n5-chunk-cuts
```

## 构建它（Build It）

### 步骤 1：固定和递归分块

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 步骤 2：语义分块

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

在你的领域上调优 `threshold`。太高 → 碎片。太低 → 一个巨大的块。

### 步骤 3：父文档

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键洞察：对父级去重。多个子级可以映射到同一个父级；返回所有都会浪费上下文。

### 步骤 4：上下文检索（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

索引上下文化块。在查询时，检索从额外的周围信号中受益。

### 步骤 5：评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

始终基准测试。"你的语料库的最佳"策略可能不匹配任何博客文章。

## 陷阱

- **仅在事实查询上评估分块。** 多跳查询揭示非常不同的赢家。使用按查询类型分层的评估集。
- **没有最小尺寸的语义分块。** 产生伤害检索的 40 词元片段。始终执行 `min_tokens`。
- **重叠作为货物崇拜。** 2026 年研究发现重叠通常提供零好处并使索引成本加倍。测量，不要假设。
- **没有最小/最大执行。** 5 个词元或 5000 个词元的块都破坏检索。钳制。
- **跨文档分块。** 绝不让一个块跨越两个文档。始终按文档分块，然后合并。

## 使用它（Use It）

2026 年技术栈：

| 情况 | 策略 |
|-----------|----------|
| 首次构建，未知语料库 | 递归，512 词元，无重叠 |
| 事实 QA | 递归，256-512 词元 |
| 分析性 / 多跳 | 递归，512-1024 词元 + 父文档 |
| 重度交叉引用（合同、论文） | 晚期分块或上下文检索 |
| 对话 / 对话语料库 | 轮次级块 + 说话者元数据 |
| 短话语（推文、评论） | 一个文档 = 一个块 |

从递归 512 开始。在 50 查询评估集上测量 recall@5。从那里调优。

## 发布它（Ship It）

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## 练习（Exercises）

1. **简单（Easy）。** 用固定（512,0）、递归（512,0）和递归（512,100）分块一个 20 页文档。比较块计数和边界质量。
2. **中等（Medium）。** 在 5 个文档上构建 30 查询评估集。测量 5 级召回率用于递归、语义和父文档。哪一个获胜？它匹配博客文章吗？
3. **困难（Hard）。** 实现上下文检索。测量基线递归的 MRR 改进。报告索引成本（LLM 调用）vs 准确性增益。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| 块（Chunk） | 文档的一块 | 被嵌入、索引和检索的子文档单元。 |
| 重叠（Overlap） | 安全边距 | 相邻块共享的 N 个词元；2026 年基准测试中通常无用。 |
| 语义分块 | 智能分块 | 在相邻句子嵌入相似度下降的地方分割。 |
| 父文档（Parent-document） | 两级检索 | 检索小子块，返回更大的父块。 |
| 晚期分块（Late chunking） | 嵌入后分块 | 在词元级别嵌入完整文档，池化为块向量。 |
| 上下文检索（Contextual retrieval） | Anthropic 的技巧 | 在索引之前附加到每个块的 LLM 生成的摘要。 |
| 上下文悬崖（Context cliff） | 2500 词元墙 | 2026 年 1 月在 RAG 中观察到的约 2.5k 上下文词元周围的响应质量下降。 |

## 延伸阅读（Further Reading）

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 生产中的默认。
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — 分块与嵌入选择一样重要。
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — 晚期分块论文。
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — 使用 LLM 生成的上下文前缀提高 35-50% 检索。
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 按查询类型的块大小。
