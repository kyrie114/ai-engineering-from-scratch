# 文本摘要（Text Summarization）

> 抽取式摘要告诉你文档说了什么。生成式摘要告诉你作者想表达什么。不同的任务，不同的陷阱。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 11 (Machine Translation)
**Time:** ~75 minutes

## 问题（The Problem）

你的信息流中出现了一篇 2000 字的新闻文章。你需要 120 字来概括它。你可以从文章中挑选最重要的三句话（抽取式），也可以用自己的话重写内容（生成式）。两者都叫摘要。它们是完全不同的问题。

抽取式摘要是一个排序问题。给每个句子打分，返回前 `k` 个。输出始终语法正确，因为是逐字摘录。风险是遗漏分布在文章中的内容。

生成式摘要是一个生成问题。一个 transformer 基于输入产生新文本。输出流利且压缩，但可能编造源语言中不存在的事实。风险是自信的捏造。

本课构建两者，并指出每种方式专有的失败模式。

## 概念（The Concept）

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**抽取式（Extractive）。** 将文章视为一个图，节点是句子，边是相似度。在图上运行 PageRank（或类似算法）来根据句子与其余部分的连接程度对它们评分。最高分的句子就是摘要。规范的实现是 TextRank（Mihalcea and Tarau, 2004）。

**生成式（Abstractive）。** 在文档-摘要对上微调一个 transformer 编码器-解码器（BART, T5, Pegasus）。推理时，模型通过交叉注意力逐词元读取文档并生成摘要。Pegasus 尤其使用间隙句预训练目标，使其无需太多微调就能出色完成摘要。

使用 ROUGE（Recall-Oriented Understudy for Gisting Evaluation）评估。ROUGE-1 和 ROUGE-2 对一元语法（unigram）和二元语法（bigram）重叠评分。ROUGE-L 对最长公共子序列评分。越高越好，但 40 ROUGE-L 是 "好"，50 是 "卓越"。每篇论文报告所有三个。使用 `rouge-score` 包。

```figure
summarize-collapse
```

## 构建它（Build It）

### 步骤 1：TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

两件事值得注意。相似度函数使用对数归一化词重叠，这是原始 TextRank 变体。TF-IDF 向量的余弦（cosine）也可以。阻尼系数（damping factor）0.85 和迭代次数是 PageRank 默认值。

### 步骤 2：使用 BART 进行生成式摘要

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 在 CNN/DailyMail 语料库上微调。开箱即用生成新闻风格摘要。对于其他领域（科学论文、对话、法律），使用相应的 Pegasus 检查点或在你的目标数据上微调。

### 步骤 3：ROUGE 评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

始终使用词干提取（stemming）。没有它，"running" 和 "run" 算作不同的词，ROUGE 会低估。

### ROUGE 之外（2026 摘要评估）

ROUGE 作为主导的摘要度量已经存在二十年，在 2026 年单独使用是不够的。一项对 NLG 论文的大规模元分析显示：

- **BERTScore**（上下文嵌入相似度）通过 2023 年获得关注，现在在大多数摘要论文中与 ROUGE 一起报告。
- **BARTScore** 将评估视为生成：通过预训练的 BART 在给定源语言的情况下对摘要的可能性评分。
- **MoverScore**（上下文嵌入上的推土机距离（Earth Mover's Distance））在 2025 年摘要基准测试中位居榜首，因为它比 ROUGE 更好地捕捉语义重叠。
- **FactCC** 和 **基于 QA 的忠实度（QA-based faithfulness）** 在 2021-2023 年很常见，现在通常被 **G-Eval** 替代（一个用思维链推理评分连贯性、一致性、流畅性、相关性的 GPT-4 提示链）。
- **G-Eval** 和类似的 LLM 评判方法与人类判断约 80% 一致，当评分标准设计良好时。

生产建议：报告 ROUGE-L 用于遗留比较，BERTScore 用于语义重叠，G-Eval 用于连贯性和事实性。根据 50-100 个人工标注的摘要进行校准。

### 步骤 4：事实性问题

生成式摘要容易出现幻觉。抽取式摘要的幻觉风险低得多，因为输出是从源语言逐字摘录的，尽管如果源语言句子被脱节、过时或断章取义，它们仍然可能误导。这是生产系统对合规相关内容仍然偏好抽取式方法的单一最大原因。

要命名的幻觉类型：

- **实体交换（Entity swap）。** 源语言说 "John Smith"。摘要说 "John Brown"。
- **数字漂移（Number drift）。** 源语言说 "25,000"。摘要说 "25 million"。
- **极性翻转（Polarity flip）。** 源语言说 "rejected the offer"。摘要说 "accepted the offer"。
- **事实编造（Fact invention）。** 源语言没有提到 CEO。摘要说 CEO 批准了。

有效的评估方法：

- **FactCC。** 在源语言句子和摘要句子之间的蕴涵（entailment）上训练的二元分类器。预测事实/非事实。
- **基于 QA 的事实性（QA-based factuality）。** 向 QA 模型询问答案在源语言中的问题。如果摘要支持不同的答案，标记。
- **实体级 F1。** 比较源语言与摘要中的命名实体。仅出现在摘要中的实体值得怀疑。

对于任何事实性重要的面向用户的内容（新闻、医疗、法律、金融），抽取式是更安全的选择。生成式需要事实性检查。

## 使用它（Use It）

2026 年技术栈：

| 使用场景 | 推荐 |
|---------|-------------|
| 新闻，3-5 句摘要，英语 | `facebook/bart-large-cnn` |
| 科学论文 | `google/pegasus-pubmed` 或调优的 T5 |
| 多文档、长文 | 任何具有 32k+ 上下文的 LLM，提示 |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式，天生的低幻觉风险 | TextRank 或 `sumy` 的 LSA / LexRank |

具有长上下文的 LLM 在 2026 年当计算不是约束时，通常优于专门的模型。权衡是成本和可复现性；专门的模型给出更一致的输出。

## 发布它（Ship It）

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## 练习（Exercises）

1. **简单（Easy）。** 在 5 篇新闻文章上运行 TextRank。将前 3 句话与参考摘要进行比较。测量 ROUGE-L。对于 CNN/DailyMail 风格的文章，你应该看到 30-45 ROUGE-L。
2. **中等（Medium）。** 实现实体级事实性：从源语言和摘要中提取命名实体（spaCy），计算源语言实体在摘要中的召回率和摘要实体对源语言的精确率。高精确率和低召回率意味着安全但简洁；低精确率意味着幻觉实体。
3. **困难（Hard）。** 在 50 篇 CNN/DailyMail 文章上比较 BART-large-CNN 与一个 LLM（Claude 或 GPT-4）。报告 ROUGE-L、事实性（按实体 F1）和每摘要成本。记录每个在哪里获胜。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| 抽取式（Extractive） | 挑选句子 | 从源语言逐字返回句子。永远不会产生幻觉。 |
| 生成式（Abstractive） | 重写 | 基于源语言生成新文本。可能产生幻觉。 |
| ROUGE | 摘要度量 | 系统输出与参考之间的 n-gram / LCS 重叠。 |
| TextRank | 基于图的抽取式 | 句子相似度图上的 PageRank。 |
| 事实性（Factuality） | 它正确吗 | 摘要主张是否被源语言支持。 |
| 幻觉（Hallucination） | 编造内容 | 摘要中源语言不支持的内容。 |

## 延伸阅读（Further Reading）

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 抽取式规范性论文。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART 论文。
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus 和间隙句目标。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE 论文。
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — 事实性领域论文。
