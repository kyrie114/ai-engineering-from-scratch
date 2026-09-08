# 子词分词——BPE, WordPiece, Unigram, SentencePiece（Subword Tokenization — BPE, WordPiece, Unigram, SentencePiece）

> 词分词器（Word tokenizers）在未见过词上卡住。字符分词器（Character tokenizers）使序列长度爆炸。子词分词器（Subword tokenizers）取中间值。每个现代 LLM 都基于其中之一交付。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 minutes

## 问题（The Problem）

你的词汇表有 50,000 个词。用户输入 "untokenizable"。你的分词器返回 `[UNK]`。模型现在关于这个词没有信号。更糟糕：语料库中第 90 百分位的文档有 40 个稀有词，这意味着每个文档 40 比特的丢失信息。

子词分词解决了这个问题。常见词保持单个词元。稀有词分解为有意义的片段：`untokenizable` → `un`、`token`、`izable`。训练数据覆盖一切，因为任何字符串最终都是字节序列。

2026 年的每个前沿 LLM 都基于三种算法之一（BPE、Unigram、WordPiece）交付，包装在三个库之一（tiktoken、SentencePiece、HF Tokenizers）中。你不能在不挑选一个的情况下交付语言模型。

## 概念（The Concept）

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE（Byte-Pair Encoding）。** 从字符级词汇表开始。计数每个相邻对。将最频繁的对合并为一个新词元。重复直到达到目标词汇表大小。主导算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**字节级 BPE。** 相同的算法但基于原始字节（256 个基础词元）而不是 Unicode 字符。保证零 `[UNK]` 词元——任何字节序列都可以编码。GPT-2 使用 50,257 个词元（256 个字节 + 50,000 个合并 + 1 个特殊）。

**Unigram。** 从一个巨大的词汇表开始。给每个词元分配一个一元语法概率。迭代地剪除以最小增加语料库对数似然的词元。推理时概率化：可以采样分词（通过子词正则化进行数据增强很有用）。T5、mBART、ALBERT、XLNet、Gemma 使用它。

**WordPiece。** 合并最大化训练语料库似然而不是原始频率的对。BERT、DistilBERT、ELECTRA 使用它。

**SentencePiece 与 tiktoken。** SentencePiece 是直接基于原始 Unicode 文本*训练*词汇表的库（BPE 或 Unigram），将空白编码为 `▁`。tiktoken 是 OpenAI 的快速*编码器*，针对预建的词汇表；它不训练。

经验法则：

- **训练新词汇表：** SentencePiece（多语言，无预分词）或 HF Tokenizers。
- **针对 GPT 词汇表的快速推理：** tiktoken（cl100k_base、o200k_base）。
- **两者兼有：** HF Tokenizers——一个库，训练 + 服务。

```figure
bpe-merge
```

## 构建它（Build It）

### 步骤 1：从零开始的 BPE

见 `code/main.py`。循环：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

算法编码的三个事实。`</w>` 标记词尾，使 "low"（后缀）和 "lower"（前缀）保持不同。频率加权使高频对早期获胜。合并列表是有序的——推理按训练顺序应用合并。

### 步骤 2：使用学习的合并进行编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素的 O(n·|merges|)。生产实现（tiktoken、HF Tokenizers）使用带优先队列的合并秩查找，并在线性时间内运行。

### 步骤 3：实践中的 SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：不需要预分词，空白编码为 `▁`，`character_coverage` 控制稀有字符保留与映射到 `<unk>` 的积极性。

### 步骤 4：OpenAI 兼容词汇表的 tiktoken

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

仅编码。快速（Rust 后端）。与 GPT-4/5 分词完全匹配，用于字节计数、成本估计、上下文窗口预算。

## 2026 年仍然交付的陷阱

- **分词器漂移（Tokenizer drift）。** 在词汇表 A 上训练，针对词汇表 B 部署。词元 ID 不同；模型输出垃圾。在 CI 中检查 `tokenizer.json` 哈希。
- **空白歧义（Whitespace ambiguity）。** BPE "hello" 与 " hello" 产生不同的词元。始终显式指定 `add_special_tokens` 和 `add_prefix_space`。
- **多语言训练不足（Multilingual undertraining）。** 英语主导的语料库产生的词汇表将非拉丁文字分词为多 5-10 倍的词元。相同的提示在日语/阿拉伯语上 GPT-3.5 上花费 5-10 倍。o200k_base 部分修复了它。
- **Emoji 分割（Emoji splits）。** 一个单独的 emoji 可以占用 5 个词元。在预算上下文时检查 emoji 处理。

## 使用它（Use It）

2026 年技术栈：

| 情况（Situation） | 选择（Pick） |
|-----------|------|
| 从头训练单语言模型 | HF Tokenizers（BPE） |
| 训练多语言模型 | SentencePiece（Unigram，`character_coverage=0.9995`） |
| 提供 OpenAI 兼容 API | tiktoken（`o200k_base` 用于 GPT-4+） |
| 领域特定词汇（代码、数学、蛋白质） | 在领域语料库上训练自定义 BPE，与基础词汇表合并 |
| 边缘推理，小模型 | Unigram（较小的词汇表效果更好） |

词汇表大小是一个扩展决策，不是常数。粗略启发式：<1B 参数 32k，1-10B 50-100k，多语言/前沿 200k+。

## 发布它（Ship It）

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## 练习（Exercises）

1. **简单（Easy）。** 在 `code/main.py` 的小语料库上训练 500 合并 BPE。对三个保留词进行编码。多少个产生正好 1 个词元 vs >1 个词元？
2. **中等（Medium）。** 在 100 个英语 Wikipedia 句子上比较 `cl100k_base`、`o200k_base` 和你以 vocab=32k 训练的 SentencePiece BPE 之间的词元计数。报告每个的压缩比。
3. **困难（Hard）。** 用 BPE、Unigram 和 WordPiece 训练相同的语料库。在每种方法用于小型情感分类器时测量下游准确性。选择是否移动超过 1 点 F1 的指针？

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | 最频繁字符对的贪婪合并，直到目标词汇表大小。 |
| 字节级 BPE | 没有未知词元 | 基于原始 256 字节的 BPE；GPT-2 / Llama 使用它。 |
| Unigram | 概率分词器 | 使用对数似然从大的候选集中剪枝；T5、Gemma 使用。 |
| SentencePiece | 空白那个 | 在原始文本上训练 BPE/Unigram 的库；空白编码为 `▁`。 |
| tiktoken | 快速那个 | OpenAI 的 Rust 支持的预建词汇表 BPE 编码器。无训练。 |
| 合并列表（Merge list） | 神奇的数字 | 有序的 `(a, b) → ab` 合并列表；推理按顺序应用。 |
| 字符覆盖（Character coverage） | 多稀有是太稀有？ | 分词器必须覆盖的语料库中字符的比例；~0.9995 典型。 |

## 延伸阅读（Further Reading）

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE 论文。
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — Unigram 论文。
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — 库。
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — 简洁参考。
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — 手册 + 编码列表。
