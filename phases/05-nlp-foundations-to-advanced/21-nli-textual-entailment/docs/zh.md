# 自然语言推理——文本蕴涵（Natural Language Inference — Textual Entailment）

> "t 蕴涵 h" 意味着人类阅读 t 会得出 h 是真的。NLI 是预测蕴涵/矛盾/中性的任务。表面无聊，生产中至关重要。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment Analysis), Phase 5 · 13 (Question Answering)
**Time:** ~60 minutes

## 问题（The Problem）

你构建了一个摘要器。它产生了一个摘要。你怎么知道摘要不包含幻觉？

你构建了一个聊天机器人。它回答 "yes。"你怎么知道答案被检索到的段落支持？

你需要按主题对 10,000 篇新闻文章进行分类。你没有训练标签。你能复用一个模型吗？

所有三个问题都归结为自然语言推理（Natural Language Inference）。NLI 询问：给定一个前提 `t` 和一个假设 `h`，`h` 是被 `t` 蕴涵的、矛盾的，还是中性的（无关的）？

- **幻觉检查（Hallucination check）：** `t` = 源文档，`h` = 摘要主张。不是蕴涵 = 幻觉。
- **有依据的问答（Grounded QA）：** `t` = 检索到的段落，`h` = 生成的答案。不是蕴涵 = 捏造。
- **零样本分类（Zero-shot classification）：** `t` = 文档，`h` = 口化的标签（"This is about sports"）。蕴涵 = 预测的标签。

一个任务，三个生产用途。这就是为什么每个 RAG 评估框架在底层都搭载了一个 NLI 模型。

## 概念（The Concept）

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**三个标签。**

- **蕴涵（Entailment）。** `t` → `h`。"The cat is on the mat" 蕴涵 "There is a cat。"
- **矛盾（Contradiction）。** `t` → ¬`h`。"The cat is on the mat" 矛盾 "There is no cat。"
- **中性（Neutral）。** 两者都没有推断。"The cat is on the mat" 对 "The cat is hungry。" 是中性的。

**不是逻辑蕴涵。** NLI 是*自然语言*推理——一个典型的人类读者会推断什么，不是严格的逻辑。"John walked his dog" 在 NLI 中蕴涵 "John has a dog"，但严格的一阶逻辑只有在你公理化占有的情况下才会承认它。

**数据集。**

- **SNLI**（2015）。570k 人工标注对，图像说明作为前提。狭窄领域。
- **MultiNLI**（2017）。433k 对，跨 10 种类型。2026 年的标准训练语料库。
- **ANLI**（2019）。对抗性 NLI。人类编写专门设计来打破现有模型的例子。更难。
- **DocNLI、ConTRoL**（2020-21）。文档长度前提。测试多跳和长距离推理。

**架构。** 一个 transformer 编码器（BERT、RoBERTa、DeBERTa）读取 `[CLS] premise [SEP] hypothesis [SEP]`。`[CLS]` 表示喂养一个 3 路 softmax。在 MNLI 上训练，在保留基准上评估，对分布内对获得 90%+ 的准确性。

**通过 NLI 实现零样本。** 给定一个文档和候选标签，将每个标签变成一个假设（"This text is about sports"）。为每个计算蕴涵概率。选取最大值。这就是 Hugging Face 的 `zero-shot-classification` 管道背后的机制。

```figure
nli-router
```

## 构建它（Build It）

### 步骤 1：运行预训练 NLI 模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

对于生产 NLI，`facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli` 是开放默认值。DeBERTa-v3 在排行榜上名列前茅。

### 步骤 2：零样本分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

默认模板是 "This example is about {label}。"使用 `hypothesis_template` 自定义。不需要训练数据。不需要微调。开箱即用。

### 步骤 3：用于 RAG 的忠实度检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这是 RAGAS 忠实度的核心。将生成的答案分解为原子主张。根据检索到的上下文检查每个主张。报告被支持的比例。

### 步骤 4：手写 NLI 分类器（概念性）

见 `code/main.py` 中的 stdlib-only 玩具：前提和假设通过词汇重叠 + 否定检测进行比较。无法与 transformer 模型竞争——但它展示了任务的形状：两个文本输入，3 路标签输出，损失 = `{entail, contradict, neutral}` 上的交叉熵。

## 陷阱

- **仅假设快捷方式。** 模型可以仅从假设中预测标签，在 SNLI 上约 60% 准确，因为 "not"、"nobody"、"never" 与矛盾相关。检测标签泄露的强基线。
- **词汇重叠启发式。** 子序列启发式（"每个子序列都被蕴涵"）通过 SNLI 但失败于 HANS/ANLI。使用对抗基准。
- **文档长度退化。** 单句 NLI 模型在文档长度前提上下降 20+ F1。对于长上下文使用 DocNLI 训练的模型。
- **零样本模板敏感性。** "This example is about {label}" 与 "{label}" 与 "The topic is {label}" 可以摆动 10+ 点的准确性。调优模板。
- **领域不匹配。** MNLI 在通用英语上训练。法律、医疗和科学文本需要领域特定的 NLI 模型（例如 SciNLI、MedNLI）。

## 使用它（Use It）

2026 年技术栈：

| 使用场景 | 模型 |
|---------|-------|
| 通用 NLI | `microsoft/deberta-v3-large-mnli` |
| 快速 / 边缘 | `cross-encoder/nli-deberta-v3-base` |
| 零样本分类（轻量级） | `facebook/bart-large-mnli` |
| 文档级 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多语言 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG 中的幻觉检测 | RAGAS / DeepEval 内部的 NLI 层 |

2026 年的元模式：NLI 是文本理解的胶带。每当你需要 "A 支持 B 吗？" 或 "A 与 B 矛盾吗？"——在找另一个 LLM 调用之前先找 NLI。

## 发布它（Ship It）

保存为 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## 练习（Exercises）

1. **简单（Easy）。** 在 20 个手工制作的前提-假设-标签三元组上运行 `facebook/bart-large-mnli`，涵盖所有三个类别。测量准确性。添加对抗性 "子序列启发式" 陷阱（"I did not eat the cake" 与 "I ate the cake"）看看它是否打破。
2. **中等（Medium）。** 将零样本模板 `"This text is about {label}"` 与 `"The topic is {label}"` 和 `"{label}"` 在 100 个 AG News 标题上比较。报告准确性摆动。
3. **困难（Hard）。** 构建一个 RAG 忠实度检查器：原子主张分解 + 每个主张的 NLI。在 50 个具有黄金上下文的 RAG 生成答案上评估。与人工标签比较假阳性和假阴性率。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| NLI | 自然语言推理 | 前提-假设关系的 3 路分类。 |
| RTE |  Recognizing Textual Entailment | NLI 的旧名称；相同的任务。 |
| 蕴涵（Entailment） | "t 蕴涵 h" | 给定 t，典型读者会得出 h 是真的。 |
| 矛盾（Contradiction） | "t 排除 h" | 给定 t，典型读者会得出 h 是假的。 |
| 中性（Neutral） | "未决定" | 从 t 到 h 都没有推断。 |
| 零样本分类 | NLI 作为分类器 | 将标签口化为假设，选取最大蕴涵。 |
| 忠实度（Faithfulness） | 答案被支持了吗？ | 检索上下文与生成答案之间的 NLI。 |

## 延伸阅读（Further Reading）

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI。
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI。
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — ANLI 基准。
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI 作为分类器。
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — 2026 年 NLI 主力。
