# BERT — Masked Language Modeling（BERT — 掩码语言模型）

> GPT 预测下一个词。BERT 预测缺失的词。一句话的差异——以及半个十年以来所有与嵌入相关的一切。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer)（完整 Transformer）, Phase 5 · 02 (Text Representation)（文本表示）
**Time:** ~45 minutes（约 45 分钟）

## The Problem（问题）

2018 年，每一个 NLP 任务——情感分析、NER、问答、蕴含——都在自己的标签数据上从头训练自己的模型。没有预训练的“理解英语”检查点可以微调。ELMo（2018）表明你可以用双向 LSTM 预训练上下文嵌入；它有帮助但没有推广开来。

BERT（Devlin et al. 2018）问了一个问题：如果我们拿一个 transformer 编码器，在互联网上的每个句子上训练它，并强迫它从上下文的两侧预测缺失的词会怎样？然后你在你的下游任务上微调一个头。参数效率是一个启示。

结果：在 18 个月内，BERT 及其变体（RoBERTa、ALBERT、ELECTRA）主导了当时存在的每一个 NLP 排行榜。到 2020 年，地球上的每个搜索引擎、内容审核管线和语义搜索系统内部都有一个 BERT。

2026 年，encoder-only 模型仍然是分类、检索和结构化提取的正确工具——它们每个 token 的运行速度比解码器快 5–10 倍，它们的嵌入是每个现代检索栈的骨干。ModernBERT（2024 年 12 月）用 2026 年的原语把架构推进到 8K 上下文，包含 Flash Attention + RoPE + GeGLU。

## The Concept（概念）

![Masked language modeling: pick tokens, mask them, predict originals（掩码语言模型：挑选 token，掩码它们，预测原始值）](../assets/bert-mlm.svg)

### 训练信号

拿一个句子：`the quick brown fox jumps over the lazy dog`。

随机掩码 15% 的 token：

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

训练模型在掩码位置预测原始 token。因为编码器是双向的，在位置 1 预测 `[MASK]` 可以使用位置 2+ 的 `brown fox jumps`。这是 GPT 做不到的事情。

### BERT 掩码规则

在选中的 15% 用于预测的 token 中：

- 80% 被替换为 `[MASK]`。
- 10% 被替换为一个随机 token。
- 10% 保持不变。

为什么不总是 `[MASK]`？因为 `[MASK]` 在推理时永远不会出现。在 100% 的掩码位置训练模型期望 `[MASK]` 会在预训练和微调之间造成分布偏移。10% 随机 + 10% 保持原样让模型保持诚实。

### Next Sentence Prediction (NSP) — 以及为什么它被放弃了

原始 BERT 还在 NSP 上训练：给定两个句子 A 和 B，预测 B 是否跟随 A。RoBERTa（2019）消融了它，并表明 NSP 有害无益。现代编码器跳过它。

### 2026 年的变化：ModernBERT

2024 年的 ModernBERT 论文用 2026 年的原语重建了块：

| Component（组件） | Original BERT (2018)（原始 BERT） | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| Positional（位置） | Learned absolute（可学习绝对） | RoPE |
| Activation（激活） | GELU | GeGLU |
| Normalization（归一化） | LayerNorm | Pre-norm RMSNorm |
| Attention（注意力） | Full dense（全稠密） | Alternating local (128) + global（交替局部 + 全局） |
| Context length（上下文长度） | 512 | 8192 |
| Tokenizer（分词器） | WordPiece | BPE |

而且与 2018 年的栈不同，它是原生 Flash-Attention 的。在序列长度 8K 时，推理速度比 DeBERTa-v3 快 2–3 倍，同时 GLUE 分数更好。

### 2026 年仍然选择编码器的用例

| Task（任务） | Why encoder beats decoder（为什么编码器优于解码器） |
|------|---------------------------|
| 检索 / 语义搜索嵌入 | 双向上下文 = 更好的每个 token 嵌入质量 |
| 分类（情感、意图、毒性） | 一次前向传播；没有生成开销 |
| NER / token 标注 | 逐位置输出，原生双向 |
| Zero-shot 蕴含（NLI） | 编码器上的分类头 |
| RAG 的 Reranker | 交叉编码器评分，比 LLM rerankers 快 10 倍 |

```figure
transformer-residual
```

## Build It（动手实现）

### Step 1: 掩码逻辑

见 `code/main.py`。函数 `create_mlm_batch` 接收一个 token ID 列表、词表大小和掩码概率。返回 input IDs（应用掩码后）和标签（只在掩码位置，其他地方是 -100 —— PyTorch 的忽略索引约定）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original（否则：保持原样）
    return input_ids, labels
```

### Step 2: 在微型语料库上运行 MLM 预测

在一个 20 个词的词表、200 个句子上训练一个 2 层编码器 + MLM 头。没有梯度——我们做前向传播健全性检查。完整训练需要 PyTorch。

### Step 3: 比较掩码类型

展示三种规则如何使模型在没有 `[MASK]` 的情况下仍然可用。在未掩码句子和掩码句子上预测。两者都应该产生合理的 token 分布，因为模型在训练中看到了两种模式。

### Step 4: 微调头

用玩具情感数据集上的分类头替换 MLM 头。只有头训练；编码器冻结。这是每个 BERT 应用遵循的模式。

## Use It（实际应用）

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**嵌入模型是微调过的 BERT。** `sentence-transformers` 模型如 `all-MiniLM-L6-v2` 是用对比损失训练的 BERT。编码器是一样的。损失变了。

**Cross-encoder rerankers 也是微调过的 BERT。** 在 `[CLS] query [SEP] doc [SEP]` 上的对分类。查询和文档之间的双向注意力正是赋予 cross-encoder 比 biencoder 质量优势的原因。

**2026 年什么时候不选 BERT。** 任何生成性任务。编码器没有合理的方式自回归地产生 token。还有：在 1B 参数以下，小解码器可以以更多灵活性匹配质量（Phi-3-Mini、Qwen2-1.5B）。

## Ship It（交付）

见 `outputs/skill-bert-finetuner.md`。这个 skill 为一个新的分类或提取任务规划 BERT 微调（骨干选择、头规范、数据、评估、停止）。

## Exercises（练习）

1. **Easy（简单）。** 运行 `code/main.py`，打印 10,000 个 token 上的掩码分布。确认约 15% 被选中，其中约 80% 变成 `[MASK]`。
2. **Medium（中等）。** 实现 whole-word masking：如果一个词被分成子词，一起掩码所有子词或不掩码。测量这是否在 500 个句子的语料库上提高了 MLM 准确率。
3. **Hard（困难）。** 在一个公共数据集的 10,000 个句子上训练一个微型（2 层，d=64）BERT。在 SST-2 情感任务上微调 `[CLS]` token。与匹配参数的 decoder-only 基线比较——哪个赢？

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| MLM | "Masked language modeling"（掩码语言模型） | 训练信号：随机将 15% 的 token 替换为 `[MASK]`，预测原始值。 |
| Bidirectional（双向） | "Looks both ways"（双向看） | 编码器注意力没有因果掩码——每个位置看到每个其他位置。 |
| `[CLS]` | "The pooler token"（池化 token） | 一个特殊 token，前置在每个序列前；它的最终嵌入用作句子级表示。 |
| `[SEP]` | "Segment separator"（段分隔符） | 分隔配对序列（例如 query/doc、句子 A/B）。 |
| NSP | "Next sentence prediction"（下一句预测） | BERT 的第二预训练任务；在 RoBERTa 中显示无用，2019 年后被放弃。 |
| Fine-tuning（微调） | "Adapt to a task"（适应任务） | 大部分冻结编码器；在顶部训练一个小头用于下游任务。 |
| Cross-encoder（交叉编码器） | "A reranker"（一个 reranker） | 一个同时接收 query 和 doc 作为输入、输出相关性分数的 BERT。 |
| ModernBERT | "2024 refresh"（2024 年刷新） | 用 RoPE、RMSNorm、GeGLU、交替局部/全局注意力、8K 上下文重建的编码器。 |

## Further Reading（延伸阅读）

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 原始论文。
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — 如何正确训练 BERT；杀死了 NSP。
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — 替换 token 检测在匹配计算下优于 MLM。
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT 论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 规范编码器参考。
