# 命名实体识别（Named Entity Recognition）

> 把名称抽出来。听起来简单，直到你遇到模糊边界、嵌套实体和领域术语。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word Embeddings)
**Time:** ~75 minutes

## 问题（The Problem）

"Apple 起诉 Google，因其 iPhone 搜索交易。" 五个实体：Apple (ORG)、Google (ORG)、iPhone (PRODUCT)、search deal（可能）、US (GPE)。一个优秀的命名实体识别系统会提取所有这些实体并赋予正确的类型。糟糕的系统会漏掉 iPhone，混淆 Apple（公司）与 Apple（水果），并把 "US" 标记为 PERSON。

NER 是每个结构化提取管线的核心。简历解析、合规日志扫描、医疗记录匿名化、搜索查询理解、聊天机器人响应 grounding、法律合同提取。你几乎看不到它，但你总是依赖它。

本课沿着经典路径（规则系统、HMM、CRF）走到现代路径（BiLSTM-CRF，再到 transformers）。每一步都解决前一步的具体局限。这种模式就是本课的核心。

## 概念（The Concept）

**BIO 标注**（或 BILOU）将实体提取变为序列标注问题。将每个词元标记为 `B-TYPE`（实体开头）、`I-TYPE`（实体内部）或 `O`（不在任何实体内）。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多词元实体链式标注：`New B-GPE`、`York I-GPE`、`City I-GPE`。理解 BIO 的模型可以提取任意跨度。

架构演进：

- **规则系统。** 正则表达式 + 名称词典查找。对已知实体精度高，对新实体零覆盖。
- **HMM。** 隐马尔可夫模型（Hidden Markov Model）。给定标签的词元发射概率、标签到标签的转移概率。Viterbi 解码。在标注数据上训练。
- **CRF。** 条件随机场（Conditional Random Field）。类似于 HMM 但是判别式，因此可以混合任意特征（词形、大小写、相邻词元）。在 2026 年仍是低资源部署的经典生产主力。
- **BiLSTM-CRF。** 神经特征替代人工特征。LSTM 双向读取句子，CRF 层在顶部强制一致的标签序列。
- **基于 Transformer。** 微调 BERT，加上 token classification 头。精度最高。计算量最大。

```figure
ner-bio-tagging
```

## 构建它（Build It）

### 步骤 1：BIO 标注辅助函数（BIO tagging helpers）

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 步骤 2：人工特征（hand-crafted features）

对于经典（非神经）NER，特征是关键。有用的特征：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 返回 `xXxxxx`。`word_shape("USA-2024")` 返回 `XXX-dddd`。大小写模式是命名实体的强信号。

### 步骤 3：简单的规则系统 + 词典基线（a simple rule-based + dictionary baseline）

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产级词典有从 Wikipedia 和 DBpedia 爬取的数百万条目。覆盖良好。消歧（`Apple` 公司还是 Apple 水果）糟糕。这就是为什么统计模型胜出。

### 步骤 4：CRF 步骤（草图，非完整实现）（the CRF step (sketch, not full impl)）

没有概率论基础，50 行代码写完整 CRF 并不直观。使用 `sklearn-crfsuite`：

```python
import sklearn_crfsuite


def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 是 L1 和 L2 正则化。`all_possible_transitions=True` 让模型学习非法序列（如 `O` 后面的 `I-ORG`）是不可能的，这就是 CRF enforcing BIO 一致性的方式，无需你手动编写约束。

### 步骤 5：BiLSTM-CRF 增加了什么（what a BiLSTM-CRF adds）

特征变为学习得到。输入：词元嵌入（GloVe 或 fastText）。LSTM 从左到右和从右到左读取。拼接的隐藏状态通过 CRF 输出层。CRF 仍然强制标签序列一致性；LSTM 用学习到的特征替代人工特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

对于 CRF 层，使用 `torchcrf.CRF`（pip install pytorch-crf）。相比人工特征 CRF 的提升是可测量的，但除非你有数万条标注句子，否则提升没有你预期的那么大。

## 使用它（Use It）

spaCy 开箱即用提供生产级 NER。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意 `iPhone` 被标记为 `ORG` 而非 `PRODUCT` — spaCy 小模型的 product-entity 覆盖较弱。大模型（`en_core_web_lg`）表现更好。transformer 模型（`en_core_web_trf`）更好。

Hugging Face 用于基于 BERT 的 NER：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 将连续的 B-X、I-X 词元合并为一个跨度。不使用它的话，你会得到词元级标签，需要自己合并。

### 基于 LLM 的 NER（2026 年方案）（LLM-based NER (the 2026 option)）

零样本和少样本 LLM NER 在许多领域已与微调模型竞争，当标注数据稀缺时表现明显更好。

- **零样本提示。** 给 LLM 一份实体类型列表和一个示例 schema。要求 JSON 输出。开箱即用；在新领域上精度中等。
- **ZeroTuneBio 风格提示。** 将任务分解为候选提取 → 含义解释 → 判断 → 复查。多阶段提示（非单次）在生物医学 NER 上显著提升精度。同样的模式也适用于法律、金融和科学领域。
- **使用 RAG 的动态提示。** 每次推理调用时从一小部分标注种子集中检索最相似的标注示例；动态构建少样本提示。在 2026 年基准测试中，这使 GPT-4 生物医学 NER F1 比静态提示提升 11-12%。
- **按实体类型分解。** 对于长文档，一次调用提取所有实体类型会随着长度增长而降低召回率。按实体类型分别运行一次提取。推理成本更高，精度显著提升。这是临床笔记和法律合同的标准模式。

截至 2026 年的生产建议：在收集训练数据之前，先用 LLM 零样本基线。通常 F1 足够好，你永远不需要微调。

### 经典 NER 仍然胜出的场景（Where classical NER still wins）

即使有 LLM 可用，在以下场景经典 NER 仍然胜出：

- 延迟预算低于 50 毫秒。
- 你有数千条标注示例，需要 98%+ F1。
- 领域拥有稳定本体，预训练 CRF 或 BiLSTM 可以很好地迁移。
- 监管约束要求本地部署、非生成式模型。

### 经典 NER 失效的场景（Where it falls apart）

- **领域偏移。** 在 CoNLL 上训练的 NER 在法律合同上的表现不如词典。在你的领域上微调。
- **嵌套实体。** "Bank of America Tower" 同时是 ORG 和 FACILITY。标准 BIO 无法表示重叠跨度。你需要嵌套 NER（多轮次或基于跨度的模型）。
- **长实体。** "United States Federal Deposit Insurance Corporation." 词元级模型有时会将其拆分。使用 `aggregation_strategy` 或后处理。
- **稀疏类型。** 医疗 NER 标签如 DRUG_BRAND、ADVERSE_EVENT、DOSE。通用模型完全不知道。Scispacy 和 BioBERT 是起点。

## 交付物（Ship It）

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## 练习（Exercises）

1. **简单。** 实现 `bio_to_spans`（`spans_to_bio` 的逆操作）并在 10 个句子上验证往返一致性。
2. **中等。** 在 CoNLL-2003 英文 NER 数据集上训练上述 sklearn-crfsuite CRF。使用 `seqeval` 报告每个实体的 F1。典型结果：~84 F1。
3. **困难。** 在领域特定的 NER 数据集（医疗、法律或金融）上微调 `distilbert-base-cased`。与 spaCy 小模型比较。记录数据泄漏检查并写下你的意外发现。

## 关键术语（Key Terms）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| NER | 提取名称 | 将词元跨度标记为类型（PERSON、ORG、GPE、DATE……）。 |
| BIO | 标注方案 | `B-X` 开头，`I-X` 延续，`O` 表示外部。 |
| BILOU | 改进的 BIO | 添加 `L-X`（最后一个）、`U-X`（单元）以获得更干净的边界。 |
| CRF | 结构化分类器 | 对标签之间的转移建模，而不仅仅是发射。强制有效序列。 |
| Nested NER | 重叠实体 | 一个跨度是它的子跨度的不同实体。BIO 无法表达这一点。 |
| Entity-level F1 | 正确的 NER 指标 | 预测跨度必须与真实跨度完全匹配。词元级 F1 高估准确率。 |

## 拓展阅读（Further Reading）

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF 论文。规范。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 引入了成为标准的 token-classification 模式。
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — `Doc.ents` 和 `Span` 每个属性的实用参考。
- [seqeval](https://github.com/chakki-works/seqeval) — 正确的指标库。始终使用它。
