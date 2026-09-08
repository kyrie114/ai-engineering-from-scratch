# 指代消解（Coreference Resolution）

> "She called him. He did not answer. The doctor was at lunch。"三个引用，两个人，没有人被命名。指代消解弄清楚谁是谁。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 07 (POS & Parsing)
**Time:** ~60 minutes

## 问题（The Problem）

从 300 字的文章中提取 Apple Inc. 的每一个提及。当文章说 "Apple" 时很容易。当它说 "the company"、"they"、"Cupertino's technology giant" 或 "Jobs's firm" 时很难。如果不将这些提及解析到同一个实体，你的 NER 管道会错过 60-80% 的提及。

指代消解将每一个引用同一真实世界实体的表达式链接到一个聚类中。它是表面 NLP（NER、解析）和下游语义（IE、QA、摘要、KG）之间的粘合剂。

2026 年为什么重要：

- **摘要：** "The CEO announced..." 与 "Tim Cook announced..."——摘要应该命名 CEO。
- **问答：** "Who did she call?" 需要解析 "she。"
- **信息提取：** 一个知识图谱中有 "PER1 founded Apple" 和 "Jobs founded Apple" 作为独立条目是错误的。
- **多文档 IE：** 跨文章合并关于同一事件的提及是跨文档指代消解。

## 概念（The Concept）

![Coreference clustering: mentions → entities](../assets/coref.svg)

**任务。** 输入：一个文档。输出：提及（span）的聚类，其中每个聚类引用一个实体。

**提及类型。**

- **命名实体。** "Tim Cook"
- **名词性（Nominal）。** "the CEO"、"the company"
- **代词性（Pronominal）。** "he"、"she"、"they"、"it"
- **同位语（Appositive）。** "Tim Cook, Apple's CEO,"

**架构。**

1. **基于规则（Hobbs, 1978）。** 使用语法规则的基于语法树-的代词消解。好的基线。在代词上出奇地难以击败。
2. **提及对分类器。** 对于每一对提及（m_i、m_j），预测它们是否指代相同。通过传递闭包聚类。标准 2016 年之前。
3. **提及排名。** 对于每个提及，排名候选先行词（包括 "no antecedent"）。选取顶部。
4. **基于跨度的端到端（Lee et al., 2017）。** Transformer 编码器。枚举长度上限内的所有候选跨度。预测提及分数。预测每个跨度的先行词概率。贪婪聚类。现代默认。
5. **生成式（2024+）。** 提示 LLM："List every pronoun in this text and its antecedent。"在简单情况下工作良好，在长文档和罕见先行词上挣扎。

**评估度量。** 五个标准度量（MUC、B³、CEAF、BLANC、LEA），因为没有单一度量捕捉聚类质量。报告前三个的平均值作为 CoNLL F1。2026 年在 CoNLL-2012 上的最先进水平：约 83 F1。

**已知困难情况。**

- 指代在前几页引入的实体的限定描述。
- 桥接回指（"the wheels" → 之前提到的汽车）。
- 中文和日语等语言中的零回指。
- 指代前先行词（cataphora）："When **she** walked in, Mary smiled。"

```figure
coref-links
```

## 构建它（Build It）

### 步骤 1：预训练神经指代消解（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在较长的文档上，你会得到类似这样的结果：
- Cluster 1: [Apple, The company, they]
- Cluster 2: [new products]

### 步骤 2：基于规则的代词解析器（教学）

见 `code/main.py` 中的 stdlib-only 实现：

1. 提取提及：命名实体（大写跨度）、代词（字典查找）、限定描述（"the X"）。
2. 对于每个代词，查看前 K 个提及并按以下评分：
   - 性别/数量一致（启发式）
   - 新近度（更近获胜）
   - 句法角色（主语优先）
3. 链接得分最高的先行词。

无法与神经模型竞争。但它展示了搜索空间和端到端模型必须做出的决策。

### 步骤 3：使用 LLM 进行指代消解

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

两个失败模式需要注意。首先，LLM 过度合并（"him" 和 "her" 引用两个不同的人）。其次，LLM 在长文档中静默删除提及。始终使用跨度偏移检查验证。

### 步骤 4：评估

标准的 conll-2012 脚本计算 MUC、B³、CEAF-φ4 并报告平均值。对于内部评估，从你的标注测试集上的跨度级精确率和召回率开始，然后添加提及链接 F1。

## 陷阱

- **单例爆炸。** 一些系统将每个提及报告为自己的聚类。B³ 很宽容。MUC 惩罚这一点。始终检查所有三个度量。
- **长上下文中的代词。** 超过 2,000 词元的文档性能下降约 15 F1。小心分块。
- **性别假设。** 硬编码的性别规则在非二元先行词、组织、动物上失败。使用学习模型或中性评分。
- **LLM 在长文档上的漂移。** 单个 API 调用无法可靠地跨 50+ 段落聚类提及。使用滑动窗口 + 合并。

## 使用它（Use It）

2026 年技术栈：

| 情况 | 选择 |
|-----------|------|
| 英语，单文档 | `en_coreference_web_trf`（spaCy-experimental）或 AllenNLP 神经指代消解 |
| 多语言 | 在 OntoNotes 或多语言 CoNLL 上训练的 SpanBERT / XLM-R |
| 跨文档事件指代消解 | 专门的端到端模型（2025-26 SOTA） |
| 快速 LLM 基线 | GPT-4o / Claude 配合结构化输出指代消解提示 |
| 生产对话系统 | 基于规则的备用 + 神经主要 + 关键槽位人工审查 |

2026 年交付的集成模式：首先运行 NER，运行指代消解，将指代消解聚类合并到 NER 实体中。下游任务看到每个聚类一个实体，而不是每个提及一个实体。

## 发布它（Ship It）

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 练习（Exercises）

1. **简单（Easy）。** 在 5 个手工制作的段落上运行 `code/main.py` 中的基于规则的解析器。与真实情况比较提及链接准确性。
2. **中等（Medium）。** 在新闻文章上使用预训练的神经指代消解模型。将聚类与你自己的手工标注比较。它在哪里失败了？
3. **困难（Hard）。** 构建一个指代消解增强的 NER 管道：首先 NER，然后通过指代消解聚类合并。与仅 NER 在 100 篇文章上比较实体覆盖率提升。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| 提及（Mention） | 一个引用 | 引用实体的文本跨度（名称、代词、名词短语）。 |
| 先行词（Antecedent） | "it" 引用的是什么 | 后来提及与其指代相同的较早提及。 |
| 聚类（Cluster） | 实体的提及 | 都引用同一真实世界实体的提及集。 |
| 回指（Anaphora） | 向后引用 | 后来提及引用较早（"he" → "John"）。 |
| 指代前指（Cataphora） | 向前引用 | 较早提及引用较晚（"When he arrived, John..."）。 |
| 桥接（Bridging） | 隐式引用 | "I bought a car. The wheels were bad。"（ THAT car 的车轮。） |
| CoNLL F1 | 排行榜上的数字 | MUC、B³、CEAF-φ4 F1 分数的平均值。 |

## 延伸阅读（Further Reading）

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 规范教科书章节。
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — 基于跨度的端到端。
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 改善指代消解的预训练。
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — 基准。
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — 基于规则的经典。
