# 多语言 NLP（Multilingual NLP）

> 一个模型，100+ 种语言，大多数零训练数据。跨语言迁移（Cross-lingual transfer）是 2020 年代的实际奇迹。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 04 (GloVe, FastText, Subword), Phase 5 · 11 (Machine Translation)
**Time:** ~45 minutes

## 问题（The Problem）

英语有数十亿个标注示例。乌尔都语（Urdu）有数千个。迈蒂利语（Maithili）几乎没有。任何为全球受众服务的实用 NLP 系统必须在任务特定训练数据不存在的长尾语言上工作。

多语言模型通过同时在许多语言上训练一个模型来解决这个问题。共享表示让模型将在高资源语言中学到的技能迁移到低资源语言。在英语情感分析上微调模型，它在乌尔都语上开箱即用地产生相当好的情感预测。这就是零样本跨语言迁移（zero-shot cross-lingual transfer），它重塑了 NLP 如何交付给世界。

本课命名权衡、规范模型，以及一个让多语言工作新手团队陷入困境的决定：为迁移选择源语言。

## 概念（The Concept）

![Cross-lingual transfer via shared multilingual embedding space](../assets/multilingual.svg)

**共享词汇（Shared vocabulary）。** 多语言模型使用在所有目标语言文本上训练的 SentencePiece 或 WordPiece 分词器。词汇是共享的：相同的子词单元在相关语言中代表相同的词素。英语和意大利语中的 `anti-` 获得相同的词元。

**共享表示（Shared representation）。** 跨多种语言在掩码语言模型上预训练的 transformer 学习到不同语言中语义相似的句子产生相似的隐藏状态。mBERT、XLM-R 和 NLLB 都表现出这一点。"cat" 在英语中的嵌入聚集在法语 "chat" 和西班牙语 "gato" 附近，完整句子嵌入也是如此。

**零样本迁移（Zero-shot transfer）。** 用一种语言（通常是英语）的标注数据微调模型。推理时，在任何其他支持的语言上运行它。不需要目标语言标签。对于类型学相关的语言结果强劲，对于距离远的语言较弱。

**少样本微调（Few-shot fine-tuning）。** 添加 100-500 个目标语言的标注示例。分类任务的准确性跃升至英语基线的 95-98%。这是多语言 NLP 中单一最成本有效的杠杆。

## 模型（The models）

| 模型（Model） | 年份（Year） | 覆盖范围（Coverage） | 备注（Notes） |
|-------|------|----------|-------|
| mBERT | 2018 | 104 种语言 | 在 Wikipedia 上训练。第一个实用的多语言 LM。低资源弱。 |
| XLM-R | 2019 | 100 种语言 | 在 CommonCrawl 上训练（比 Wikipedia 大得多）。设定跨语言基线。Base 270M，Large 550M。 |
| XLM-V | 2023 | 100 种语言 | 带有 100 万词元词汇表的 XLM-R（对比 250k）。低资源更好。 |
| mT5 | 2020 | 101 种语言 | 多语言生成的 T5 架构。 |
| NLLB-200 | 2022 | 200 种语言 | Meta 的翻译模型；包括 55 种低资源语言。 |
| BLOOM | 2022 | 46 种语言 + 13 种编程 | 多语言训练的开放 176B LLM。 |
| Aya-23 | 2024 | 23 种语言 | Cohere 的多语言 LLM。阿拉伯语、印地语、斯瓦希里语强劲。 |

按使用案例选择。分类工作使用 XLM-R-base 作为理智默认。生成任务调用 mT5 或 NLLB，取决于翻译与开放生成。LLM 风格的工作与 Aya-23 或 Claude 配对，使用显式多语言提示。

## 源语言决策（2026 研究）

大多数团队默认英语作为微调源。最近的研究（2026）表明这通常是错的。

语言相似性比原始语料库规模更好地预测迁移质量。对于斯拉夫语目标，德语或俄语通常击败英语。对于印度语目标，印地语通常击败英语。**qWALS** 相似度度量（2026，基于语言结构图集特征）量化了这一点。**LANGRANK**（Lin 等人，ACL 2019）是一种独立的早期方法，它从语言相似性、语料库规模和遗传相关性的组合中排名候选源语言。

实用规则：如果你的目标语言有一个类型学上接近的高资源亲戚，首先尝试在该语言上微调，然后与英语微调比较。

```figure
n5-crosslingual-bridge
```

## 构建它（Build It）

### 步骤 1：零样本跨语言分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，相同的 API。在 NLI 数据上训练的 XLM-R 通过蕴涵技巧很好地迁移到分类。

### 步骤 2：多语言嵌入空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译在嵌入空间中接近。不同的英语句子更远。这使得跨语言检索、聚类和相似性工作。

### 步骤 3：少样本微调策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对于 100-500 个目标语言示例，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全默认值。更高的学习率会导致多语言对齐崩溃，你会得到一个只有英语的模型。

## 实际有效的评估

- **保留集上的每种语言的准确性。** 不是聚合。聚合隐藏了长尾。
- **与单语言基线基准测试。** 对于有足够数据的语言，从头开始训练的单语言模型有时会击败多语言模型。测试它。
- **实体级测试。** 目标语言中的命名实体。多语言模型对于远离拉丁语的文字通常有弱分词。
- **跨语言一致性。** 两种语言中的相同含义应该产生相同的预测。测量差距。

## 使用它（Use It）

2026 年技术栈：

| 任务（Task） | 推荐（Recommended） |
|-----|-------------|
| 分类，100 种语言 | 微调的 XLM-R-base（约 270M） |
| 零样本文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句子嵌入 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻译，200 种语言 | `facebook/nllb-200-distilled-600M`（见第 11 课） |
| 多语言生成 | Claude, GPT-4, Aya-23, mT5-XXL |
| 低资源语言 NLP | XLM-V 或在相关高资源语言上的领域特定微调 |

始终为微调目标语言预留预算，如果性能很重要。零样本是一个起点，不是最终答案。

### 低资源语言的词元化税（什么会出错）

多语言模型在所有语言之间共享一个分词器。该词汇表在一个由英语、法语、西班牙语、中文、德语主导的语料库上训练。对于任何不在主导集合中的语言，三种税 silently 复合：

- **生育税（Fertility tax）。** 低资源语言文本每个词被分词成比英语多得多的词元。一个印地语句子可能需要同等英语句子的 3-5 倍词元。这 3-5 倍消耗你的上下文窗口、训练效率和延迟。
- **变体恢复税（Variant recovery tax）。** 每个拼写错误、变音符号变体、Unicode 归一化不匹配或大小写变化都成为嵌入空间中的一个冷启动无关序列。模型无法学习母语者视为明显的正字法对应关系。
- **容量溢出税（Capacity spillover tax）。** 税 1 和 2 消耗上下文位置、层深度和嵌入维度。对于实际推理剩余的东西系统性地小于高资源语言从同一个模型得到的东西。

实际症状：你的模型在印地语上正常训练，损失曲线看起来正确，评估困惑度看起来合理，生产输出微妙地错误。形态在句子中间崩溃。罕见屈折保持不可恢复。**你无法通过数据扩展来解决损坏的分词器。**

缓解措施：选择对你的目标语言有良好覆盖的分词器（XLM-V 的 100 万词元词汇表是一个直接修复）；在训练之前验证保留目标文本上的词元化生育率；对真正长尾文字使用字节级回退（SentencePiece `byte_fallback=True`，GPT-2 风格字节级 BPE），使任何东西都不会 OOV。

## 发布它（Ship It）

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 练习（Exercises）

1. **简单（Easy）。** 在英语、法语、印地语和阿拉伯语上，每种语言 10 个句子，运行零样本分类管道。报告每种语言的准确性。你应该看到强大的法语，不错的印地语，可变的阿拉伯语。
2. **中等（Medium）。** 使用 `paraphrase-multilingual-MiniLM-L12-v2` 在混合语言语料库上构建跨语言检索器。用英语查询，检索任何语言的文档。测量 recall@5。
3. **困难（Hard）。** 为印地语分类任务比较英语源和印地语源微调。在两种方案下使用 500 个目标语言示例进行少样本微调。报告哪种源产生更好的印地语准确性以及好多少。这就是 LANGRANK 论文的缩影。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| 多语言模型（Multilingual model） | 一个模型，多种语言 | 跨语言共享词汇表和参数。 |
| 跨语言迁移（Cross-lingual transfer） | 在一种语言上训练，在另一种语言上运行 | 在源语言上微调，在没有目标语言标签的情况下在目标语言上评估。 |
| 零样本（Zero-shot） | 没有目标语言标签 | 不在目标语言上微调的情况下迁移。 |
| 少样本（Few-shot） | 小的目标标签 | 用于微调的 100-500 个目标语言示例。 |
| mBERT | 第一个多语言 LM | 在 Wikipedia 上预训练的 104 语言 BERT。 |
| XLM-R | 标准跨语言基线 | 在 CommonCrawl 上预训练的 100 语言 RoBERTa。 |
| NLLB | Meta 的 200 语言 MT | No Language Left Behind。包括 55 种低资源语言。 |

## 延伸阅读（Further Reading）

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R 论文。
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — 开启跨语言迁移研究线的分析论文。
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200 论文。
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya，Cohere 的多语言 LLM。
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK 源语言论文。
