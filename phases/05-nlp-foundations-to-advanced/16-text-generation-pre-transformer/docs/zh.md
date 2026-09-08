# 转换器之前的文本生成——N 元语法语言模型（Text Generation Before Transformers — N-gram Language Models）

> 如果一个词令人惊讶，模型就是坏的。困惑度（Perplexity）将惊讶变为一个数字。平滑（Smoothing）使其保持有限。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## 问题（The Problem）

在 transformer 之前，在 RNN 之前，在词嵌入（word embeddings）之前，语言模型通过计算一个词在前一个 `n-1` 个词之后出现的频率来预测下一个词。计数 "the cat" → "sat" 47 次，"the cat" → "jumped" 12 次，"the cat" → "refrigerator" 0 次。归一化以获得概率分布。

这就是 N 元语法（N-gram）语言模型。它在 1980 年到 2015 年间运行了每个语音识别器、每个拼写检查器和每个基于短语的机器翻译系统。当你需要廉价的设备端语言模型时，它仍然在运行。

有趣的问题是对未见过的 N 元语法该怎么办。原始计数模型对任何未见过的内容赋予零概率，这是灾难性的，因为句子很长，几乎每个长句子都包含至少一个未见过的序列。五十年的平滑（Smoothing）研究解决了这个问题。Kneser-Ney 平滑（Kneser-Ney smoothing）是结果，现代深度学习继承了其实证传统。

## 概念（The Concept）

![N-gram model: count, smooth, generate](../assets/ngram.svg)

### 预测游戏

在这种机器出现之前，一个实验定义了什么是语言模型。遮住一个英语句子的下一个字母。要求某人一次猜一个字母，直到他们猜对。写下猜测次数。对几百个字母重复。

猜测次数不是琐事。它们是文本的无损重新编码：将猜测序列交给第二个相同的猜测者，他们可以重建每个字母，因为在每个位置他们都知道哪些猜测最先到来。一个可以用更少符号重新编码的消息每个符号携带的信息更少，所以猜测次数统计为英语的熵设定了一个上限。

Shannon 在 1951 年运行了它，得到了一个至今仍支配该领域的数字。一个 27 符号字母表（26 个字母加空格）每个字母可以携带 `log2(27) ≈ 4.75` 比特。具有 100 个字母上下文的人类猜测者在每个字母 0.6 到 1.3 比特之间。英语大约四分之三是强制性的走法。模型必须学习的结构在任何模型能够学习它之前就被测量了。

此后的每个语言模型都是这个游戏的机械玩家，本课中的每个评估数字都是游戏得分：

- **交叉熵损失（Cross-entropy loss）** 是模型每个符号所需的平均比特数。训练 LM 从字面上最小化它在猜测游戏中的得分。
- **困惑度（Perplexity）** 是 `2^bits`（或 `e^nats`）：模型在猜测之后仍然面临的并行因子。对 27 个符号的均匀猜测是困惑度 27；一个每个字母 1 比特的玩家有困惑度 2。
- **上下文长度（Context length）是玩家的记忆。** 三元语法（trigram）模型以两个词元的记忆玩游戏。Transformer 以 100K 词元玩同样的游戏。规则从未改变；玩家变得更好了。

一个单位切换需要注意：游戏按字母以比特（`log2`）计分，而下面的 N 元语法公式按词词元以纳特（自然对数）计分——并且由于纳特中的困惑度 `e^H` 等于比特中的 `2^H`，这两个视图是相同测量以不同单位表示。

```figure
prediction-game
```

**N 元语法概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定 `n`（通常 3 用于三元语法，4 用于 4 元语法）。从计数计算：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 任何在训练中未见过的 N 元语法获得零概率。2007 年在 Brown 语料库上的研究发现，即使 4 元语法模型也有 30% 的held-out 4 元语法在训练中未见过。你无法在任何真实文本上进行评估，如果没有平滑。

**平滑方法，按复杂程度排序：**

1. **Laplace（加一）。** 给每个计数加 1。简单，在稀有事件上很糟糕。
2. **Good-Turing。** 基于频率的频率，将概率质量从高频事件重新分配到未见过的。
3. **插值（Interpolation）。** 用可调权重组合 N 元语法、(n-1) 元语法等的估计。
4. **回退（Backoff）。** 如果 N 元语法计数为零，回退到 (n-1) 元语法。Katz 回退对此进行归一化。
5. **绝对折扣（Absolute discounting）。** 从所有计数中减去一个固定折扣 `D`，重新分配到未见过。
6. **Kneser-Ney。** 绝对折扣加上低阶模型的 clever 选择：使用延续概率（continuation probability，一个词出现在多少上下文中）而不是原始频率。

Kneser-Ney 的洞察很深。"San Francisco" 是一个常见的二元语法。一元语法 "Francisco" 主要出现在 "San" 之后。朴素的绝对折扣给 "Francisco" 高的一元语法概率（因为计数很高）。Kneser-Ney 注意到 "Francisco" 只出现在一个上下文中，并相应地降低其延续概率。结果：一个以 "Francisco" 结尾的新二元语法获得适当的低概率。

**评估：困惑度（Perplexity）。** 在保留测试集上每个词的平均负对数似然的指数。越低越好。困惑度 100 意味着模型像从 100 个词中均匀选择一样困惑。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

```figure
ngram-backoff
```

## 构建它（Build It）

### 步骤 1：三元语法计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是一个分词句子的列表。输出是 N 元语法计数和上下文计数。`<s>` 和 `</s>` 是句子边界。

### 步骤 2：Laplace 平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

给每个计数加 1。平滑但过度分配概率质量给未见过的事件，伤害稀有已知事件。

### 步骤 3：Kneser-Ney（二元语法，插值）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

三个移动部件。`continuation_prob` 捕捉 "这个词出现在多少不同的上下文中？"（Kneser-Ney 的创新）。`lambda_prev` 是折扣释放的质量，用于加权回退。最终概率是折扣的主要项加上加权延续项。

### 步骤 4：使用采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率采样。始终根据种子产生不同的输出。对于类束搜索的输出，在每一步选取 argmax（贪心）并添加一个小的随机性旋钮（temperature）。

### 步骤 5：困惑度

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。对于 Brown 语料库，一个调优良好的 4 元语法 KN 模型达到约 140 的困惑度。同一测试集上的 transformer LM 达到 15-30。差距大约是 10 倍。这就是领域继续前进的原因。

## 使用它（Use It）

- **经典 NLP 教学。** 你能得到的最清晰的平滑、MLE 和困惑度暴露。
- **KenLM。** 生产 N 元语法库。用于在延迟很重要的语音和机器翻译系统中作为重新评分器。
- **设备端自动补全。** 键盘中的三元语法模型。仍然。
- **基线。** 在声明神经 LM 良好之前，始终计算 N 元语法 LM 困惑度。如果你的 transformer 没有以很大优势击败 KN，有些东西是错的。

## 发布它（Ship It）

保存为 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 练习（Exercises）

1. **简单（Easy）。** 在 1000 句莎士比亚语料库上训练三元语法 LM。生成 20 个句子。它们将在局部上合理但全局上不连贯。这是规范演示。
2. **中等（Medium）。** 在保留的莎士比亚分割上为你的 KN 模型实现困惑度。与 Laplace 比较。你应该看到 KN 将困惑度降低 30-50%。
3. **困难（Hard）。** 构建一个三元语法拼写纠正器：给定一个拼写错误的词和它的上下文，生成更正并按 LM 下的上下文概率排序。在 Birkbeck 拼写语料库（公开）上评估。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| N 元语法（N-gram） | 词序列 | `n` 个连续词元的序列。 |
| 平滑（Smoothing） | 避免零 | 重新分配概率质量，使未见过的事件获得非零概率。 |
| 困惑度（Perplexity） | LM 质量度量 | 保留数据上的 `exp(-average log-prob)`。越低越好。 |
| 回退（Backoff） | 回退到更短的上下文 | 如果三元语法计数为零，使用二元语法。Katz 回退形式化了这一点。 |
| Kneser-Ney | 最好的 N 元语法平滑 | 低阶模型的绝对折扣 + 延续概率。 |
| 延续概率（Continuation probability） | KN 特有 | `P(w)` 由 `w` 出现的上下文数量加权，而不是原始计数。 |
| 文本熵（Entropy of text） | 每符号的信息 | 给定上下文下编码下一个符号所需的平均比特数。Shannon 1951 年对印刷英语的估计，上下文最多 100 个字母：0.6-1.3 比特/字母，在任何模型存在之前测量。 |

## 延伸阅读（Further Reading）

- [Shannon (1951). Prediction and Entropy of Printed English](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf) — 定义每个语言模型仍然优化的目标的猜测游戏实验。
- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — N 元语法 LM 和平滑的规范论述。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — 确立 Kneser-Ney 为最佳 N 元语法平滑器的论文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — 原始 KN 论文。
- [KenLM](https://kheafield.com/code/kenlm/) — 快速生产 N 元语法 LM，在 2026 年仍用于延迟敏感应用。
