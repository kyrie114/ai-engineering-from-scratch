# 实体链接与消歧（Entity Linking & Disambiguation）

> NER 找到了 "Paris。"实体链接决定：Paris, France？Paris Hilton？Paris, Texas？Paris（the Trojan prince）？没有链接，你的知识图谱保持歧义。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 24 (Coreference Resolution)
**Time:** ~60 minutes

## 问题（The Problem）

一个句子读作："Jordan beat the press。"你的 NER 将 "Jordan" 标记为 PERSON。好。但是*哪个* Jordan？

- Michael Jordan（篮球）？
- Michael B. Jordan（演员）？
- Michael I. Jordan（Berkeley ML 教授——是的，这个混淆在 ML 论文中是真实的）？
- Jordan（国家）？
- Jordan（希伯来语名字）？

实体链接（Entity linking, EL）将每个提及解析为知识库中的唯一条目：Wikidata、Wikipedia、DBpedia 或你的领域 KB。两个子任务：

1. **候选生成。** 给定 "Jordan"，哪些 KB 条目是合理的？
2. **消歧。** 给定上下文，哪个候选是正确的？

两个步骤都是可学习的。两者都有基准。组合管道已经稳定了十年——改变的是消歧器的质量。

## 概念（The Concept）

![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**候选生成。** 给定提及表面形式（"Jordan"），在别名索引中查找候选。Wikipedia 别名字典覆盖大多数命名实体："JFK" → John F. Kennedy、Jacqueline Kennedy、JFK 机场、JFK（电影）。典型索引每个提及返回 10-30 个候选。

**消歧：三种方法。**

1. **先验 + 上下文（Milne & Witten, 2008）。** `P(entity | mention) × context-similarity(entity, text)`。工作良好，快速，无需训练。
2. **基于嵌入的（ESS / REL / Blink）。** 编码提及 + 上下文。编码每个候选的描述。选取最大余弦。2020-2024 默认。
3. **生成式（GENRE, 2021; LLM-based, 2023+）。** 逐字符解码实体的规范名称。约束到有效实体名称的 trie，因此输出保证是有效的 KB id。现代后代是 REL-GEN 和具有结构化输出的 LLM 提示 EL。

**端到端 vs 管道。** 现代模型（ELQ、BLINK、ExtEnD、GENRE）在一个通道中运行 NER + 候选生成 + 消歧。管道系统在生产中仍然占主导地位，因为你可以交换组件。

### 两种测量

- **提及召回（候选生成）。** 正确 KB 条目出现在候选列表中的黄金提及比例。整个管道的下限。
- **消歧准确性 / F1。** 给定正确的候选，top-1 正确的频率。

始终报告两者。一个在 80% 候选召回率上具有 99% 消歧的系统是一个 80% 的管道。

```figure
gx-entity-linking
```

## 构建它（Build It）

### 步骤 1：从 Wikipedia 重定向构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia 别名数据：约 1800 万（别名，实体）对。从 Wikidata 转储下载。存储为倒排索引。

### 步骤 2：基于上下文的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard 重叠是一个玩具。用嵌入上的余弦相似度替换（见 `code/main.py` 步骤 2 的 transformer 版本）。

### 步骤 3：基于嵌入的（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

在索引时，每个 KB 实体嵌入一次。在查询时，提及 + 上下文嵌入一次，点积针对候选池，选取最大值。

### 步骤 4：生成式实体链接（概念）

GENRE 逐字符解码实体的 Wikipedia 标题。约束解码（见第 20 课）确保只有有效标题才能输出。与 KB 支持的 trie 紧密集成。现代后代是 REL-GEN 和具有结构化输出的 LLM 提示 EL。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

结合白名单（Outlines `choice`），这是 2026 年交付的最简单的 EL 管道。

### 步骤 5：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是标准 EL 基准：1,393 篇路透社文章，34k 提及，Wikipedia 实体。报告 KB 内准确性（`P@1`）和 KB 外 NIL 检测率。

## 陷阱

- **NIL 处理。** 一些提及不在 KB 中（新兴实体、冷僻人物）。系统必须预测 NIL 而不是猜测错误的实体。单独测量。
- **提及边界错误。** 上游 NER 遗漏部分跨度（"Bank of America" 被标记为仅 "Bank"）。EL 召回下降。
- **流行度偏见。** 训练好的系统过度预测频繁实体。ML 论文中 "Michael I. Jordan" 的提及通常链接到篮球 Jordan。
- **跨语言 EL。** 将中文文本中的提及映射到英语 Wikipedia 实体。需要多语言编码器或翻译步骤。
- **KB 过时。** 新公司、事件、人物不在去年的 Wikipedia 转储中。生产管道需要刷新循环。

## 使用它（Use It）

2026 年技术栈：

| 情况 | 选择 |
|-----------|------|
| 通用英语 + Wikipedia | BLINK 或 REL |
| 跨语言，KB = Wikipedia | mGENRE |
| LLM 友好，每天提及少 | 提示 Claude/GPT-4 配合候选列表 + 约束 JSON |
| 领域特定 KB（医疗、法律） | 具有 KB 感知检索的定制 BERT + 在领域 AIDA 风格集上微调 |
| 极低延迟 | 仅精确匹配先验（Milne-Witten 基线） |
| 研究 SOTA | GENRE / ExtEnD / 生成式 LLM-EL |

2026 年交付的生产模式：NER → 核心消解 → 对每个提及进行 EL → 将聚类折叠为一个每个聚类的规范实体。输出：文档中每个实体一个 KB id，而不是每个提及一个。

## 发布它（Ship It）

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 练习（Exercises）

1. **简单（Easy）。** 在 `code/main.py` 的 10 个歧义提及（Paris、Jordan、Apple）上实现先验+上下文消歧器。手工标记正确的实体。测量准确性。
2. **中等（Medium）。** 用句子 transformer 编码 50 个歧义提及。嵌入每个候选的描述。将基于嵌入的消歧与 Jaccard 上下文重叠比较。
3. **困难（Hard）。** 构建一个 1k 实体领域 KB（例如你公司的员工 + 产品）。端到端实现 NER + EL。在 100 个保留句子上测量精确率和召回率。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| 实体链接（EL） | 链接到 Wikipedia | 将提及映射到唯一的 KB 条目。 |
| 候选生成 | 它可能是什么？ | 为提及返回一个合理的 KB 条目短名单。 |
| 消歧 | 挑选正确的 | 使用上下文对候选评分，选取赢家。 |
| 别名索引 | 查找表 | 从表面形式 → 候选实体的映射。 |
| NIL | 不在 KB 中 | 没有 KB 条目匹配的显式预测。 |
| KB | 知识库 | Wikidata、Wikipedia、DBpedia 或你的领域 KB。 |
| AIDA-CoNLL | 基准 | 1,393 篇路透社文章，具有黄金实体链接。 |

## 延伸阅读（Further Reading）

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 基础先验+上下文方法。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — 基于嵌入的主力。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — 具有约束解码的生成式 EL。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — 基准论文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — 开放生产栈。
