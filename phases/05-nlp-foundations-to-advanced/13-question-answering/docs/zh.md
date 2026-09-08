# 问答系统（Question Answering Systems）

> 三个系统塑造了现代问答。抽取式找到跨度（span）。检索增强（retrieval-augmented）将其锚定在文档中。生成式产生答案。每个现代 AI 助手都是三者的混合。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 11 (Machine Translation), Phase 5 · 10 (Attention Mechanism)
**Time:** ~75 minutes

## 问题（The Problem）

用户输入 "When did the first iPhone launch?" 并期望得到 "June 29, 2007。"而不是 "Apple's history is long and varied。" 而不是孤立无句的 "2007"。一个直接、有依据、正确的答案。

过去十年有三种问答架构占据主导地位。

- **抽取式问答（Extractive QA）。** 给定一个问题和一个已知包含答案的段落，在段落中找到答案跨度的起始和结束索引。SQuAD 是规范性基准。
- **开放域问答（Open-domain QA）。** 没有给定段落。先检索相关段落，然后抽取或生成答案。这是今天每个 RAG 管道的基础。
- **生成式/闭卷问答（Generative / Closed-book QA）。** 一个大语言模型从其参数记忆中回答。没有检索。推理最快，对事实最不可靠。

2026 年的趋势是混合的：检索最好的几个段落，然后提示生成式模型根据这些段落有依据地回答。这就是 RAG，第 14 课深入介绍了检索部分。本课构建问答部分。

## 概念（The Concept）

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**抽取式（Extractive）。** 使用 transformer（BERT 家族）一起编码问题和段落。训练两个预测答案在段落中的起始和结束词元索引的头。损失是对有效位置的交叉熵。输出是段落中的一个跨度。不会产生幻觉（按构造），不能处理段落无法回答的问题（按构造）。

**检索增强（Retrieval-augmented, RAG）。** 两个阶段。首先，检索器从语料库中找到前 `k` 个段落。其次，一个阅读器（抽取式或生成式）使用这些段落产生答案。检索器-阅读器的分离允许各自独立训练和评估。现代 RAG 通常在它们之间添加一个重排序器（reranker）。

**生成式（Generative）。** 一个仅解码器的 LLM（GPT, Claude, Llama）从学习到的权重中回答。没有检索步骤。对常识出色，对稀有或最近的事实灾难。幻觉率与预训练数据中事实频率成反比。

```figure
qa-span
```

## 构建它（Build It）

### 步骤 1：使用预训练模型进行抽取式问答

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 在 SQuAD 2.0 上训练，包括不可回答的问题。默认情况下，`question-answering` 管道返回得分最高的跨度，即使模型的空分数胜出——它不会自动返回空答案。要获得显式的 "无答案" 行为，请在管道调用中传递 `handle_impossible_answer=True`：管道仅在空分数超过每个跨度分数时才返回空答案。无论如何始终检查 `score` 字段。

### 步骤 2：检索增强管道（草图）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段管道。密集检索器（Sentence-BERT）通过语义相似性找到相关段落。抽取式阅读器（RoBERTa-SQuAD）从合并的前段落中提取答案跨度。适用于小型语料库。对于百万文档语料库，使用 FAISS 或向量数据库。

### 步骤 3：使用 RAG 进行生成式问答

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

提示模式很重要。显式告诉模型锚定在上下文中，并在上下文不足时返回 "I don't know"，与朴素提示相比，将幻觉率降低了 40-60%。更复杂的模式添加引用、置信度分数和结构化提取。

### 步骤 4：反映真实世界的评估

SQuAD 使用 **精确匹配（Exact Match, EM）** 和 **词元级 F1**。EM 是归一化后的严格匹配（小写、去掉标点、去掉冠词）——要么预测完全匹配，要么得 0 分。F1 在预测与参考之间的词元重叠上计算，给予部分分数。两者都低估了释义："June 29, 2007" 与 "June 29th, 2007" 通常得 0 EM（序数打破归一化），但由于重叠词元仍然从 F1 中获得实质性分数。

对于生产问答：

- **答案准确性（Answer accuracy）。** LLM 评判或人工评判，因为度量无法捕捉语义等价。
- **引用准确性（Citation accuracy）。** 引用的段落是否真正支持答案？在生成引用和检索段落之间使用字符串匹配自动检查。
- **拒绝校准（Refusal calibration）。** 当答案不在检索段落中时，系统是否正确地说了 "I don't know"？测量错误置信率。
- **检索召回率（Retrieval recall）。** 在评估阅读器之前，测量检索器是否将正确的段落放入前 `k` 个中。阅读器无法修复缺失的段落。

### RAGAS：2026 生产评估框架

`RAGAS` 专为 RAG 系统构建，是 2026 年的默认交付框架。它在不需要黄金参考的情况下对四个维度评分：

- **忠实度（Faithfulness）。** 答案中的每个主张是否来自检索到的上下文？通过基于 NLI 的蕴涵测量。你的主要幻觉度量。
- **答案相关性（Answer relevance）。** 答案是否回答了问题？通过从答案生成假设问题并与真实问题比较来测量。
- **上下文精确率（Context precision）。** 在检索到的块中，有多少实际上相关？低精确率 = 提示中有噪音。
- **上下文召回率（Context recall）。** 检索的集合是否包含所有需要的信息？低召回率 = 阅读器无法成功。

免参考评分让你可以在没有策划的黄金答案的情况下评估实时生产流量。在精确匹配度量无用的开放式问题上，在顶层叠加 LLM 作为评判。

`pip install ragas`。插入你的检索器 + 阅读器。每个查询获得四个标量。对回归发出警报。

## 使用它（Use It）

2026 年技术栈。

| 使用场景 | 推荐 |
|---------|-------------|
| 给定段落，查找答案跨度 | `deepset/roberta-base-squad2` |
| 在固定语料库上，不接受闭卷 | RAG：密集检索器 + LLM 阅读器 |
| 在文档存储上的实时查询 | RAG 与混合（BM25 + 密集）检索器 + 重排序器（第 14 课） |
| 对话式问答（后续问题） | 带有对话历史的 LLM + 每轮 RAG |
| 高度事实性、受监管领域 | 在权威语料库上的抽取式；永远不要单独的生成式 |

抽取式问答在 2026 年不时髦，因为带有 LLM 的 RAG 处理更多情况。当需要逐字引用时，它仍然交付：法律研究、监管合规、审计工具。

## 发布它（Ship It）

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 练习（Exercises）

1. **简单（Easy）。** 在 10 个 Wikipedia 段落上设置上述 SQuAD 抽取式管道。手工制作 10 个问题。测量答案正确的频率。如果段落和问题干净，你应该看到 7-9 个正确。
2. **中等（Medium）。** 添加拒绝分类器。当 top 检索分数低于阈值（比如 0.3 余弦）时，返回 "I don't know" 而不是调用阅读器。在保留集上调优阈值。
3. **困难（Hard）。** 在你选择的 10000 文档语料库上构建 RAG 管道。使用 RRF 融合实现混合检索（BM25 + 密集）（见第 14 课）。在有和没有混合步骤的情况下测量答案准确性。记录哪些问题类型受益最大。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| 抽取式问答（Extractive QA） | 找到答案跨度 | 在给定段落中预测答案的起始和结束索引。 |
| 开放域问答（Open-domain QA） | 在语料库上的问答 | 没有给定段落；必须检索然后回答。 |
| RAG | 检索然后生成 | 检索增强生成。检索器 + 阅读器管道。 |
| SQuAD | 规范性基准 | 斯坦福问答数据集。EM + F1 度量。 |
| 幻觉（Hallucination） | 编造答案 | 不被检索上下文支持的阅读器输出。 |
| 拒绝校准（Refusal calibration） | 知道何时闭嘴 | 当无法回答时，系统正确地说 "I don't know"。 |

## 延伸阅读（Further Reading）

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 基准论文。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，问答的规范密集检索器。
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — 命名 RAG 的论文。
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 全面 RAG 综述。
