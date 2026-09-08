# GloVe、FastText 与子词嵌入（GloVe, FastText, and Subword Embeddings）

> Word2Vec 为每个词训练一个嵌入。GloVe 分解了共现矩阵。FastText 嵌入了片段。BPE 桥接到 Transformer。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec from Scratch)
**Time:** ~45 minutes

## 问题（The Problem）

Word2Vec 留下了两个未解决的问题。

首先，有一条并行研究路线直接分解共现矩阵（LSA、HAL），而不是做在线 skip-gram 更新。Word2Vec 的迭代方法从根本上更好，还是两者差异只是处理计数的不同方式造成的？**GloVe** 回答了这个问题：使用 thoughtfully 选择的损失进行矩阵分解匹配或超过 Word2Vec，且训练成本更低。

其次，两种方法都对从未见过的词没有解决方案。`Zoomer-approved`、`dogecoin`、任何上周创造的新词、稀有词根的每一种屈折形式。**FastText** 通过嵌入字符 n 元组修复了这个问题：一个词是其片段的（包括词素）之和，因此即使词表外的词也能获得合理的向量。

第三，一旦 Transformer 出现，问题再次转移。词级词汇表最多约有一百万个条目；真实语言比这更开放。**字节对编码（BPE）** 及其亲属通过学习覆盖所有内容的频繁子词单元词汇表解决了这个问题。每个现代 LLM 的每个现代分词器都是子词分词器。

本课涵盖全部三种，然后解释何时应该使用哪一种。

## 概念（The Concept）

**GloVe（全局向量）。** 构建词-词共现矩阵 `X`，其中 `X[i][j]` 是词 `j` 在词 `i` 的上下文中出现的频率。训练向量使 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。对损失加权使频繁对不会主导。完成。

**FastText。** 一个词是其字符 n 元组加上词本身的总和。`where` 变为 `<wh, whe, her, ere, re>, <where>`。词向量是这些组件向量的总和。像 Word2Vec 一样训练。好处：未见过的词（`whereupon`）从已知的 n 元组组合而成。

**BPE（字节对编码）。** 从单个字节（或字符）的词汇表开始。统计语料库中的每一对相邻词。将最频繁的对合并为一个新词元。重复 `k` 次迭代。结果：一个 `k + 256` 个词元的词汇表，其中频繁序列（`ing`、`tion`、`the`）是单个词元，稀有词被拆分为熟悉的片段。每个句子都被分词为某种东西。

```figure
n5-subword-merge
```

## 构建它（Build It）

### GloVe：分解共现矩阵（GloVe: factorize the co-occurrence matrix）

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

两个值得命名的移动部件。加权函数 `f(x) = (x/x_max)^alpha` 降低非常频繁的对（如 `(the, and)`）的权重，使它们不会主导损失。最终嵌入是 `W`（中心）和 `W_tilde`（上下文）表的总和。将两者相加是一个已发表的技巧，往往比只使用一个表现更好。

### FastText：子词感知嵌入（FastText: subword-aware embeddings）

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词由其 n 元组集（通常 3 到 6 个字符）表示。词向量是其 n 元组向量的总和。对于 skip-gram 训练，将其插入 Word2Vec 使用单个向量的位置。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对于一个未见过的词，只要它的某些 n 元组是已知的，你仍然可以得到一个向量。`whereupon` 与 `where` 共享 `<wh`、`her`、`ere` 和 `<where`，因此两个词落在相近的位置。

### BPE：学习的子词词汇表（BPE: learned subword vocabulary）

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一次迭代合并最常见的相邻对。足够迭代后，频繁子串（`low`、`est`、`tion`）成为单个词元，稀有词被干净地拆分。

真正的 GPT / BERT / T5 分词器学习 3 万-10 万次合并。结果：任何文本都被分词为已知 ID 的有界长度序列，永远不会出现 OOV。

## 使用它（Use It）

实际上，你很少自己训练这些。你加载预训练的检查点。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

对于 Transformer 时代的 BPE 风格子词分词：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 前缀标记词边界（GPT-2 约定）。每个现代分词器都是 BPE 变体、WordPiece（BERT）或 SentencePiece（T5、LLaMA）。

### 如何选择（When to pick which）

| 场景（Situation） | 选择（Pick） |
|-----------|------|
| 预训练通用词向量，不需要 OOV 容忍 | GloVe 300d |
| 预训练通用词向量，必须处理拼写错误/新词/形态丰富语言 | FastText |
| 进入 Transformer 的任何东西（训练或推理） | 模型自带的任何分词器。绝不交换。 |
| 从零开始训练自己的语言模型 | 先在语料库上训练 BPE 或 SentencePiece 分词器 |
| 使用线性模型的生产文本分类 | 仍然使用 TF-IDF。第02课。 |

## 交付使用（Ship It）

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: 为新语言模型或文本流水线选择分词方法。
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

给定任务和数据集描述，你输出：

1. 分词策略（词级、BPE、WordPiece、SentencePiece、字节级）。一句话原因。
2. 词汇表大小目标（例如，纯英语 LM 用 32k，多语言用 64k-100k）。
3. 带确切训练命令的库调用。命名库。引用参数。
4. 一个可复现性陷阱。分词器-模型不匹配是单最常见的静默生产 bug；指出哪对必须一起使用。

当用户对预训练 LLM 进行微调时，拒绝推荐训练自定义分词器。拒绝为任何面向生产推理的模型推荐词级分词。将非英语/多脚本语料库标记为需要带字节回退的 SentencePiece。
```

## 练习（Exercises）

1. **简单。** 运行 `char_ngrams("playing")` 和 `char_ngrams("played")`。计算两个 n 元组集合的 Jaccard 重叠。你应该看到大量共享片段（`pla`、`lay`、`play`），这就是 FastText 在形态变体之间迁移效果好的原因。
2. **中等。** 扩展 `learn_bpe` 以跟踪词汇表增长。绘制每语料库字符的词元数作为合并次数的函数。你应该看到初始的快速压缩，渐近接近约 2-3 个字符每词元。
3. **困难。** 在莎士比亚的完整作品上训练 1k 合并的 BPE。比较常见词与罕见专有名词的分词。合并前后测量平均每词词元数。写下什么让你惊讶。

## 关键术语（Key Terms）

| 术语（Term） | 人们的说法（What people say） | 实际含义（What it actually means） |
|------|-----------------|-----------------------|
| 共现矩阵 | 词-词频率表 | `X[i][j]` = 词 `j` 在词 `i` 的窗口中出现的频率。 |
| 子词 | 词的片段 | 字符 n 元组（FastText）或学习到的词元（BPE/WordPiece/SentencePiece）。 |
| BPE | 字节对编码 | 迭代合并最频繁的相邻对，直到词汇表达到目标大小。 |
| OOV | 词表外 | 模型从未见过的词。Word2Vec/GloVe 失败。FastText 和 BPE 处理它。 |
| 字节级 BPE | 原始字节上的 BPE | GPT-2 的方案。词汇表从 256 个字节开始，因此永远不会出现 OOV。 |

## 延伸阅读（Further Reading）

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe 论文，七页，至今仍是损失的最佳推导。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — 将 BPE 引入现代 NLP 的论文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — BPE、WordPiece 和 SentencePiece 在实践中实际如何不同。
