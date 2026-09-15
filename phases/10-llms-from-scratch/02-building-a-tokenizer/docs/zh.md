# 从零构建分词器（Building a Tokenizer from Scratch）

> 第 01 课给你的是玩具。这一课给你的是武器。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 01 (Tokenizers: BPE, WordPiece, SentencePiece)
**Time:** ~90 minutes

## 学习目标（Learning Objectives）

- 构建生产级 BPE 分词器，处理 Unicode、空白归一化和特殊 token
- 实现字节级回退，使分词器能编码任意输入（包括 emoji、CJK 和代码），且不产生未知 token
- 添加预分词正则模式，在应用 BPE 合并之前按词边界切开文本
- 在语料上训练自定义分词器，并在多语言文本上对照 tiktoken 评估压缩比

## 问题（The Problem）

第 01 课的 BPE 分词器能处理英语。现在把日语扔给它。或者 emoji。或者混用制表符和空格的 Python 代码。

它会崩。

不是因为 BPE 错了——而是因为实现不完整。生产级分词器要处理任意编码的原始字节，在切分前归一化 Unicode，管理永远不会被合并的特殊 token，把预分词与子词切分串起来，并且快到不会拖垮处理 15 万亿 token 的训练流水线。

GPT-2 的分词器有 50,257 个 token。Llama 3 有 128,256。GPT-4 大约 100,000。这些不是玩具数字。这些词表背后的合并表是在数百 GB 文本上训练出来的，而外围机械——归一化、预分词、特殊 token 注入、聊天模板格式化——才把能处理 "hello world" 的分词器，和能处理整个互联网的分词器分开。

你将构建这套机械。

## 概念（The Concept）

### 完整流水线（The Full Pipeline）

生产级分词器不是一种算法。它是五阶段流水线，每一阶段解决不同的问题。

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每一阶段都有明确职责：

| 阶段 | 做什么 | 为什么重要 |
|-------|-------------|----------------|
| Normalize | NFKC Unicode，可选小写，可选去掉重音 | "fi" 连字（U+FB01）变成 "fi"（两个字符）。没有这一步，同一个词会得到不同 token。 |
| Pre-Tokenize | 在 BPE 之前把文本切成块 | 阻止 BPE 跨词边界合并。"the cat" 永远不该产出 token "e c"。 |
| BPE Merge | 对字节序列应用学到的合并规则 | 核心压缩。把原始字节变成子词 token。 |
| Special Tokens | 注入 [BOS]、[EOS]、[PAD]、聊天模板标记 | 这些 token 有固定 ID。它们从不参与 BPE 合并。模型需要它们来表达结构。 |
| ID Mapping | 把 token 字符串转成整数 ID | 模型看到的是整数，不是字符串。 |

### 字节级 BPE（Byte-Level BPE）

第 01 课的分词器在 UTF-8 字节上操作。这是正确的选择。但我们跳过了一件重要的事：这些字节不是合法 UTF-8 时会发生什么？

字节级 BPE 的解法是：把每一个可能的字节值（0-255）都当成合法 token。你的基础词表恰好 256 项。任何文件——文本、二进制、损坏的——都能分词，且不会产生未知 token。

GPT-2 加了一个技巧：把每个字节映射到可打印的 Unicode 字符，让词表保持人类可读。字节 0x20（空格）在他们的映射里变成字符 "G"。这纯粹是外观。算法并不在乎。

真正的威力：字节级 BPE 能处理地球上每一种语言。汉字每个是 3 个 UTF-8 字节。日文可以是 3–4 字节。阿拉伯文、天城文、emoji——全都只是字节序列。BPE 算法在这些字节序列里找模式，方式和它在英语 ASCII 字节里找模式完全一样。

### 预分词（Pre-Tokenization）

在 BPE 碰到文本之前，你需要先把它切成块。这能阻止合并算法创造跨越词边界的 token。

GPT-2 用正则模式切分文本：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

这个模式按缩写（"don't" 变成 "don" + "'t"）、可选前导空格的词、数字、标点和空白来切。前导空格保持贴在词上——所以 "the cat" 变成 [" the", " cat"]，而不是 ["the", " ", "cat"]。

Llama 使用 SentencePiece，完全跳过正则。它把原始字节流当成一条长序列，让 BPE 算法自己找边界。更简单，但给了 BPE 更多自由去创造跨词 token。

这个选择很重要。GPT-2 的正则阻止分词器去学"一个词末尾的 the 和下一个词开头的 the 应该合并"。SentencePiece 允许这一点，有时压缩更高效，但 token 更难解释。

### 特殊 token（Special Tokens）

每个生产级分词器都会为结构标记预留 token ID：

| Token | 用途 | 使用者 |
|-------|---------|---------|
| `[BOS]` / `<s>` | 序列开始 | Llama 3, GPT |
| `[EOS]` / `</s>` | 序列结束 | 所有模型 |
| `[PAD]` | 批对齐填充 | BERT, T5 |
| `[UNK]` | 未知 token（字节级 BPE 消除了它） | BERT, WordPiece |
| `<\|im_start\|>` | 聊天消息边界开始 | ChatGPT, Qwen |
| `<\|im_end\|>` | 聊天消息边界结束 | ChatGPT, Qwen |
| `<\|user\|>` | 用户轮次标记 | Llama 3 |
| `<\|assistant\|>` | 助手轮次标记 | Llama 3 |

特殊 token 永远不会被 BPE 切开。它们在合并算法运行之前被精确匹配，替换成固定 ID，周围文本再正常分词。

### 聊天模板（Chat Templates）

这是大多数人搞混、大多数实现会崩的地方。

当你向聊天模型发送消息时，API 接受一份消息列表：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型看不到 JSON。它看到的是一条扁平的 token 序列。聊天模板用特殊 token 把消息转成那条扁平序列。每个模型的做法都不同：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

模板搞错了，模型就会输出垃圾。它是在一种精确格式上训练的。任何偏差——少一个换行、换了一个 token、多一个空格——都会把输入推到训练分布之外。

### 速度（Speed）

Python 对生产分词来说太慢。

tiktoken（OpenAI）用 Rust 写成，带 Python 绑定。HuggingFace tokenizers 也是 Rust。SentencePiece 是 C++。它们相对纯 Python 能快 10–100 倍。

换个角度：为 Llama 3 预训练分词 15 万亿 token，按每秒 100 万 token（较快的 Python）需要 174 天。按每秒 1 亿 token（Rust），需要 1.7 天。

你用 Python 构建是为了理解算法。在生产中，你会用编译实现，只碰 Python 包装层。

```figure
weight-tying
```

## 构建它（Build It）

### 步骤 1：字节级编码（Step 1: Byte-Level Encoding）

地基。把任意字符串转成字节序列，把每个字节映射到可打印字符以便展示，再把过程反过来。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

在多语言文本上测试，看看字节数：

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" 是 5 字节。"你好" 是 6 字节（每个字符 3 字节）。火焰 emoji 是 4 字节。字节级分词器不在乎是什么语言。字节就是字节。

### 步骤 2：带正则的预分词器（Step 2: Pre-Tokenizer with Regex）

用 GPT-2 正则模式把文本切成块。每一块由 BPE 独立分词。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 模块支持 Unicode 属性转义（`\p{L}` 表示字母，`\p{N}` 表示数字）。标准库 `re` 模块不支持，所以我们回退到 ASCII 字符类。生产级多语言分词器应安装 `regex`。

试一下：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

前导空格保持贴在词上。缩写在撇号处切开。标点变成自己的块。BPE 永远不会跨这些边界合并 token。

### 步骤 3：在字节序列上做 BPE（Step 3: BPE on Byte Sequences）

第 01 课的核心算法，但现在独立地作用在预分词后的块上。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### 步骤 4：特殊 token 处理（Step 4: Special Token Handling）

特殊 token 需要精确匹配和固定 ID。它们完全绕过 BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### 步骤 5：完整分词器类（Step 5: Full Tokenizer Class）

把所有东西串起来：归一化、按特殊 token 切开、预分词、BPE 合并、映射到 ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 步骤 6：多语言测试（Step 6: Multilingual Test）

真正的测试。把英语、中文、emoji 和代码扔给它。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

汉字每个产出 3 个字节。emoji 产出 4 个字节。这些都不会让分词器崩溃。都不会产生未知 token。这就是字节级 BPE 的威力。

## 用起来（Use It）

### 对比真实分词器（Comparing Real Tokenizers）

加载 Llama 3、GPT-4 和 Mistral 的真实分词器。看每一款如何处理同一段多语言段落。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

你会看到同一段文本有不同的 token 数。Llama 3 的 128K 词表在合并常见模式时更激进。GPT-4 的 100K 居中。Mistral 的 32K 产出更多 token，但嵌入层更小。

权衡永远一样：更大的词表意味着更短的序列，但更多参数。

## 交付（Ship It）

本课产出一个用于构建和调试生产级分词器的提示。见 `outputs/prompt-tokenizer-builder.md`。

## 练习（Exercises）

1. **简单：** 添加 `get_token_bytes(id)` 方法，显示任意 token ID 的原始字节。用它检查最常见的合并 token 实际代表什么。
2. **中等：** 实现 Llama 风格的预分词器：按空白和数字切开，但保留前导空格。在同一语料上把它的词表与 GPT-2 正则方案对比。
3. **困难：** 添加聊天模板方法，接受 `{"role": ..., "content": ...}` 消息列表，产出 Llama 3 聊天格式的正确 token 序列。对照 HuggingFace 实现做测试。

## 关键术语（Key Terms）

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| Byte-level BPE | "在字节上工作的分词器" | 基础词表为 256 个字节值的 BPE——处理任意输入且没有未知 token |
| Pre-tokenization | "BPE 之前先切开" | 基于正则或规则的切分，阻止 BPE 跨词边界合并 |
| NFKC normalization | "Unicode 清理" | 规范分解后再做兼容组合——"fi" 连字变成 "fi"，全角 "A" 变成 "A" |
| Chat template | "消息如何变成 token" | 把角色/内容消息列表转成扁平 token 序列的精确格式——因模型而异，必须匹配训练格式 |
| Special tokens | "控制 token" | 绕过 BPE 的预留 token ID——[BOS]、[EOS]、[PAD]、聊天标记——合并前精确匹配 |
| Fertility | "每个词多少 token" | 输出 token 数与输入词数之比——GPT-4 英语约 1.3，韩语 2–3，越高意味着浪费越多上下文 |
| tiktoken | "OpenAI 分词器" | 带 Python 绑定的 Rust BPE 实现——比纯 Python 快 10–100 倍 |
| Merge table | "词表" | 训练时学到的字节对合并有序列表——这就是分词器学到的知识 |

## 延伸阅读（Further Reading）

- [OpenAI tiktoken source](https://github.com/openai/tiktoken) —— GPT-3.5/4 使用的 Rust BPE 实现
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) —— 支持 BPE、WordPiece、Unigram 的 Rust 分词库
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783) —— 128K 词表与分词器训练细节
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) —— 语言无关分词
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py) —— 原始的字节到 Unicode 映射
