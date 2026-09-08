# 文本处理 — 分词、词干提取、词元化（Text Processing — Tokenization, Stemming, Lemmatization）

> 语言是连续的。模型是离散的。预处理是桥梁。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## 问题（The Problem）

模型无法直接读取 "The cats were running."。它读取的是整数。

每个 NLP 系统都从相同的三个问题开始。一个词从哪里开始。这个词的根是什么。当需要时，我们如何将 "run"、"running"、"ran" 视为同一个词；当不需要时，又如何将它们视为不同词。

如果分词器出错，模型就会从垃圾数据中学习。如果你的分词器将 `don't` 视为一个词元，却将 `do n't` 视为两个，训练分布就会分裂。如果你的词干提取器将 `organization` 和 `organ` 压缩为同一个词干，主题建模就会失效。如果你的词元化器需要词性上下文，而你却没有传入，动词就会被当作名词处理。

本课将从零开始构建这三个预处理步骤，然后展示 NLTK 和 spaCy 如何完成相同的工作，以便你了解其中的权衡。

## 概念（The Concept）

三个操作。每个都有自己的职责和失败模式。

**分词（Tokenization）** 将一个字符串拆分为词元。"词元"（token）一词故意保持模糊，因为正确的粒度取决于任务。经典 NLP 用词级别。Transformer 用子词。没有空格的语种用字符级别。

**词干提取（Stemming）** 用规则砍掉后缀。速度快、攻击性强、简单粗暴。`running -> run`。`organization -> organ`。第二个就是失败模式。

**词元化（Lemmatization）** 利用语法知识将词还原为词典形式。速度较慢、准确、需要查找表或形态分析器。`ran -> run`（需要知道 "ran" 是 "run" 的过去式）。`better -> good`（需要知道比较级形式）。

经验法则。速度优先且能容忍噪声时用词干提取（搜索索引、粗略分类）。含义优先时用词元化（问答、语义搜索、任何用户会阅读的内容）。

```figure
edit-distance
```

## 构建它（Build It）

### 步骤1：正则表达式分词器（Step 1: a regex word tokenizer）

最简单的实用分词器在非字母数字字符处拆分，同时将标点作为独立的词元保留。不完美，不是最终方案，但一行代码就能运行。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三种模式按优先级排列。带可选内部撇号的词（`don't`、`it's`）。纯数字。任何单个非空白非字母数字字符作为独立词元（标点）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

需要注意的失败模式。`3pm` 拆分为 `['3', 'pm']`，因为我们在字母运行和数字运行之间交替。对于大多数任务来说已经足够好。URL、电子邮件、话题标签都会出错。生产环境需要在通用模式之前添加特定模式。

### 步骤2：Porter 词干提取器（仅第1a步）（Step 2: a Porter stemmer (step 1a only)）

完整的 Porter 算法有五个阶段的规则。仅第1a步就覆盖了最常见的英语后缀，并展示了这一模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

从上到下阅读规则。`ies -> i` 规则解释了为什么 `ponies -> poni` 而不是 `pony`。真正的 Porter 算法有第1b步会修正它。规则之间存在竞争。先出现的规则获胜。顺序比任何单个规则都重要。

### 步骤3：基于查找表的词元化器（Step 3: a lookup-based lemmatizer）

真正的词元化需要形态学。一个便于教学的版本使用小型词元表和一个回退方案。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个案例是关键的教学时刻。`watched` 不在我们的词元表中，我们的回退方案只处理 `ing`。真正的词元化涵盖 `ed`、不规则动词、比较级形容词、有发音变化的复数（`children -> child`）。这就是为什么生产系统使用 WordNet、spaCy 的形态分析器或完整的形态分析器。

### 步骤4：组合流水线（Step 4: pipe them together）

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺少的部分是词性标注器。第5阶段 · 第07课（词性标注）会构建一个。目前，将所有词默认为 `NOUN` 并承认这一局限性。

## 使用它（Use It）

NLTK 和 spaCy 提供了生产级版本。每款只需几行代码。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 处理缩写、Unicode 和你的正则表达式遗漏的边缘情况。`PorterStemmer` 运行全部五个阶段。`WordNetLemmatizer` 需要将 NLTK 的宾州树库词性标签转换为 WordNet 的缩写集。上面的转换代码正是大多数教程跳过的部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 将整个流水线隐藏在 `nlp(text)` 后面。分词、词性标注和词元化全部运行。比 NLTK 更快，开箱即用更准确。代价是你无法轻松替换各个组件。

### 如何选择（When to pick which）

| 场景（Situation） | 选择（Pick） |
|-----------|------|
| 教学、研究、需要交换组件 | NLTK |
| 生产环境、多语言、速度优先 | spaCy |
| Transformer 流水线（你无论如何都会用模型的词元器分词） | 使用 `tokenizers` / `transformers` 并跳过经典预处理 |

### 没人警告你的两种失败模式（The two failure modes nobody warns you about）

大多数教程讲完算法就停了。有两件事会坑到一个真正的预处理流水线，而且几乎从未被涵盖。

**可复现性漂移（Reproducibility drift）。** NLTK 和 spaCy 在不同版本之间会改变分词和词元化器的行为。spaCy 2.x 产生 `['do', "n't"]` 的结果，在 3.x 可能变成 `["don't"]`。你的模型是在一个分布上训练的。推理现在在另一个分布上运行。准确率 quietly 下降，没人知道为什么。在 `requirements.txt` 中锁定库版本。编写一个预处理回归测试，冻结 20 个样本句子的预期分词结果。每次升级时运行。

**训练/推理不匹配（Training / inference mismatch）。** 训练时使用激进的预处理（小写、停用词移除、词干提取），部署时使用原始用户输入，观察性能骤降。这是最常见的生产 NLP 失败。如果训练时进行了预处理，推理时必须运行完全相同的函数。将预处理作为一个函数随模型包一起交付，而不是作为 notebook 中的一个单元格，让服务团队重写。

## 交付使用（Ship It）

一个可复用的提示，帮助工程师在不阅读三本教科书的情况下选择预处理策略。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: 为 NLP 任务推荐分词、词干提取和词元化方案。
phase: 5
lesson: 01
---

你为经典 NLP 预处理提供建议。给定任务描述，你输出：

1. 分词选择（正则表达式、NLTK word_tokenize、spaCy 或 transformer 词元器）。解释原因。
2. 是否词干提取、词元化、两者都用，还是都不用。解释原因。
3. 具体的库调用。命名函数。引用 NLTK 的词性标签转换代码。
4. 用户应该测试的一种失败模式。

拒绝为面向用户的文本推荐词干提取。拒绝在未提供词性标签的情况下推荐词元化。将非英语输入标记为需要不同流水线。
```

## 练习（Exercises）

1. **简单。** 扩展 `tokenize` 以将 URL 保留为单个词元。测试：`tokenize("Visit https://example.com today.")` 应该产生一个 URL 词元。
2. **中等。** 实现 Porter 第1b步。如果一个词包含元音并以 `ed` 或 `ing` 结尾，则移除它。处理双辅音规则（`hopping -> hop`，不是 `hopp`）。
3. **困难。** 构建一个以 WordNet 作为查找表但回退到你的 Porter 词干提取器的词元化器。在带标签的语料库上测量准确率，与纯 WordNet 和纯 Porter 对比。

## 关键术语（Key Terms）

| 术语（Term） | 人们的说法（What people say） | 实际含义（What it actually means） |
|------|-----------------|-----------------------|
| 词元（Token） | 一个词 | 模型消耗的任何单元。可以是词、子词、字符或字节。 |
| 词干（Stem） | 词根 | 基于规则的后缀剥离结果。不一定是真正的词。 |
| 词元化（Lemma） | 词典形式 | 你会去查找的形式。需要语法上下文才能正确计算。 |
| 词性标签（POS tag） | 词性 | 如 NOUN、VERB、ADJ 这样的类别。准确词元化需要它。 |
| 形态学（Morphology） | 词形规则 | 词如何根据时态、数、格改变形式。词元化依赖它。 |

## 延伸阅读（Further Reading）

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，五页，至今仍是最清晰的解释。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 真实流水线如何接线。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 你还没想到的分词边缘情况。
