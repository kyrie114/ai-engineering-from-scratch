# 词性标注与句法分析（POS Tagging and Syntactic Parsing）

> 语法曾一度不再流行。直到每个 LLM 管线都需要验证结构化提取，语法又回来了。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## 问题（The Problem）

第 01 课承诺词形还原需要词性标注。不知道 `running` 是动词，词形还原器无法将其还原为 `run`。不知道 `better` 是形容词，它无法还原为 `good`。

这个承诺掩盖了一个完整的子领域。词性标注为词元分配语法类别。句法分析恢复句子的树结构：哪个词修饰哪个词，哪个动词支配哪个论元。经典 NLP 花了二十年时间完善两者。然后深度学习将它们 collapse 为一个预训练 transformer 上的词元分类任务，研究界随之转移。

不是应用界。每个结构化提取管线仍在底层使用词性和依存树。LLM 生成的 JSON 根据语法约束进行验证。问答系统使用依存分析分解查询。机器翻译质量评估器检查分析树的对齐。

值得了解。本课介绍标签集、基线和停止从零实现、转而调用 spaCy 的节点。

## 概念（The Concept）

**词性标注**为每个词元标记语法类别。**宾州树库（Penn Treebank，PTB）**标签集是英语默认。36 个标签，包含普通读者觉得繁琐的区别：`NN` 单数名词、`NNS` 复数名词、`NNP` 专有名词单数、`VBD` 过去式动词、`VBZ` 第三人称单数现在时动词，等等。**通用依存（Universal Dependencies，UD）**标签集更粗（17 个标签）且语言无关；它成为跨语言工作的默认。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法分析**生成一棵树。两种主要风格：

- **成分分析。** 名词短语、动词短语、介词短语相互嵌套。输出是非终端类别（NP、VP、PP）的树，词作为叶子。
- **依存分析。** 每个词有一个它依赖的 head 词，标记为语法关系。输出是一棵树，其中每条边都是（head、dependent、relation）三元组。

依存分析在 2010 年代胜出，因为它可以干净地跨语言泛化，特别是对于自由语序语言。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

```figure
pos-tagger
```

```figure
dependency-arcs
```

## 构建它（Build It）

### 步骤 1：最频繁标签基线（most-frequent-tag baseline）

最笨但有效的词性标注器。对每个词，预测它在训练中最常出现的标签。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown 语料库上，这个基线达到约 85% 准确率。不算好，但是严肃模型不应低于的下限。

### 步骤 2：二元隐马尔可夫模型标注器（bigram HMM tagger）

建模序列的联合概率：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两个表：转移概率（给定前一个标签的标签）、发射概率（给定标签的词元）。通过计数加上 Laplace 平滑估计两者。使用 Viterbi 解码（在标签格上的动态规划）。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Brown 上的二元 HMM 达到约 93% 准确率。从 85% 到 93% 的提升主要来自转移概率 — 模型学到 `DET NOUN` 常见而 `NOUN DET` 罕见。

### 步骤 3：为什么现代标注器更优（why modern taggers beat this）

转移 + 发射概率是局部的。它们无法捕捉 `saw` 在 "I bought a saw" 中是名词但在 "I saw the movie" 中是动词。带有任意特征（后缀、词形、前一个词和后一个词、词本身）的 CRF 达到约 97%。BiLSTM-CRF 或 transformer 达到约 98%+。

这个任务的上限由标注者不一致决定。人类标注者在宾州树库上约 97% 一致。超过 98% 的模型可能过度拟合测试集。

### 步骤 4：依存分析草图（dependency parsing sketch）

完整的依存分析从零实现超出范围；规范教科书处理在 Jurafsky 和 Martin 的著作中。两个需要了解的经典家族：

- **基于转移的分析器**（arc-eager、arc-standard）像 shift-reduce 分析器一样工作：它们读取词元，将其 shift 到栈上，并应用创建弧的 reduce 操作。贪心解码速度快。经典实现是 MaltParser。现代神经版本：Chen 和 Manning 的基于转移的分析器。
- **基于图的分析器**（Eisner 算法、Dozat-Manning biaffine）为每个可能的 head-dependent 边打分并选择最大生成树。更慢但更准确。

对于大多数应用工作，调用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下往上读取 `dep` 列，句子的语法结构就显现出来了。

## 使用它（Use It）

每个生产级 NLP 库都将其作为标准管线的一部分提供：

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）。快速、准确、与分词 + NER + 词形还原集成。`token.tag_`（宾州）、`token.pos_`（UD）、`token.dep_`（依存关系）。
- **Stanford NLP（stanza）**。Stanford 的 CoreNLP 继任者。在 60+ 种语言上达到最先进水平。
- **trankit**。基于 transformer，UD 准确率高。
- **NLTK**。`pos_tag`。可用但慢且老旧。适合教学。

### 这在 2026 年仍然重要的场景（Where this still matters in 2026）

- **词形还原。** 第 01 课需要词性标注才能正确词形还原。始终需要。
- **从 LLM 输出中结构化提取。** 验证生成的句子是否遵守语法约束（如主谓一致、必需修饰语）。
- **基于方面的情感分析。** 依存分析告诉你哪个形容词修饰哪个名词。
- **查询理解。** "movies directed by Wes Anderson starring Bill Murray" 通过分析分解为结构化约束。
- **跨语言迁移。** UD 标签和依存关系是语言无关的，使新语言的零样本结构化分析成为可能。
- **低计算管线。** 如果你不能部署 transformer，词性标注 + 依存分析 + 词典能让你走得很远。

## 交付物（Ship It）

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (saw, book, can) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## 练习（Exercises）

1. **简单。** 使用最频繁标签基线在一个小型标注语料库（如 NLTK 的 Brown 子集）上，在保留句子上测量准确率。验证约 85% 的结果。
2. **中等。** 训练上述二元 HMM 并报告每个标签的精确率/召回率。HMM 最常混淆哪些标签？
3. **困难。** 使用 spaCy 的依存分析从 1000 句样本中提取主谓宾三元组。在 50 个人工标注的三元组上评估。记录提取失败的位置（通常是被动语态、并列结构和省略主语）。

## 关键术语（Key Terms）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| POS tag | 词的词性 | 语法类别。PTB 有 36 个；UD 有 17 个。 |
| Penn Treebank | 标准标签集 | 英语专用。细粒度动词时态和名词数。 |
| Universal Dependencies | 多语言标签集 | 比 PTB 更粗；语言中立；跨语言工作默认。 |
| Dependency parse | 句子树 | 每个词有一个 head，每条边有一个语法关系。 |
| Viterbi | 动态规划 | 给定发射和转移，寻找最高概率标签序列。 |

## 拓展阅读（Further Reading）

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — 词性和句法分析的规范教科书处理。
- [Universal Dependencies project](https://universaldependencies.org/) — 每个多语言解析器使用的跨语言标签集和树库集合。
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — `Token` 上暴露的每个属性的实用参考。
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 将神经解析器带入主流的论文。
