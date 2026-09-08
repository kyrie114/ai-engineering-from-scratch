# T5, BART — Encoder-Decoder Models（T5、BART — 编码器-解码器模型）

> 编码器理解。解码器生成。把它们放回一起，你会得到一个为 input → output 任务打造的模型：翻译、摘要、改写、转录。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer)（完整 Transformer）, Phase 7 · 06 (BERT)（BERT）, Phase 7 · 07 (GPT)（GPT）
**Time:** ~45 minutes（约 45 分钟）

## The Problem（问题）

Decoder-only GPT 和 encoder-only BERT 各自把 2017 年架构剥离以适应不同目标。但许多任务天生就是 input-output：

- 翻译：英语 → 法语。
- 摘要：5,000 token 的文章 → 200 token 的摘要。
- 语音识别：audio tokens → text tokens。
- 结构化提取：prose → JSON。

对于这些，encoder-decoder 是最干净的适配。编码器为源产生一个密集表示。解码器生成输出，在每一步 cross-attend 到那个表示。训练是输出侧的 shift-by-one。和 GPT 同样的损失，只是以编码器输出为条件。

两篇论文定义了现代剧本：

1. **T5**（Raffel et al. 2019）。"Text-to-Text Transfer Transformer。" 每个 NLP 任务都被重新框定为 text-in, text-out。单一架构、单一词表、单一损失。在掩码跨度预测上预训练（在输入中 corrupt spans，在输出中解码它们）。
2. **BART**（Lewis et al. 2019）。"Bidirectional and Auto-Regressive Transformer。" Denoising autoencoder：以多种方式 corrupt 输入（shuffle、mask、delete、rotate），要求解码器重建原始值。

2026 年，encoder-decoder 格式在输入结构重要的地方继续存在：

- Whisper（speech → text）。
- Google 的翻译栈。
- 一些具有 distinct context-and-edit 结构的代码补全/修复模型。
- Flan-T5 和变体用于结构化推理任务。

Decoder-only 抢了风头，但 encoder-decoder 从未消失。

## The Concept（概念）

![Encoder-decoder with cross-attention（带交叉注意力的编码器-解码器）](../assets/encoder-decoder.svg)

### 前向循环

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

关键的是，编码器为每个输入运行一次。解码器自回归地运行，但在每一步 cross-attend 到*相同的*编码器输出。缓存编码器输出是长输入的一个免费加速。

### T5 预训练 — 跨度损坏

在输入上选随机跨度（平均长度 3 token，15% 总计）。用唯一的 sentinel 替换每个跨度：`<extra_id_0>`、`<extra_id_1>` 等。解码器只输出带有 sentinel 前缀的损坏跨度：

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

比预测整个序列更便宜的信号。在 T5 论文的消融中与 MLM（BERT）和 prefix-LM（UniLM）竞争。

### BART 预训练 — 多噪声 denoising

BART 尝试五种 noising 函数：

1. Token masking。
2. Token deletion。
3. Text infilling（mask 一个跨度，解码器插入正确的长度）。
4. Sentence permutation。
5. Document rotation。

结合 text infilling + sentence permutation 产生了最好的下游数字。解码器总是重建原始值。BART 的输出是完整序列，不只是损坏的跨度——所以预训练计算比 T5 高。

### 推理

和 GPT 同样的自回归生成。Greedy / beam / top-p 采样适用。Beam search（宽度 4–5）是翻译和摘要的标准，因为输出分布比对话更窄。

### 2026 年何时选择每个变体

| Task（任务） | Encoder-decoder?（编码器-解码器？） | Why（为什么） |
|------|------------------|-----|
| 翻译 | Yes, usually（通常是） | 清晰的源序列；固定的输出分布；beam search 有效 |
| Speech-to-text | Yes (Whisper)（是） | 输入模态与输出不同；编码器塑造音频特征 |
| Chat / reasoning（聊天/推理） | No, decoder-only（否，仅解码器） | 没有持久的"input"——对话就是序列 |
| Code completion（代码补全） | Usually no（通常否） | Decoder-only 长上下文获胜；代码模型如 Qwen 2.5 Coder 是 decoder-only |
| Summarization（摘要） | Either works（两者皆可） | BART、PEGASUS 打败早期 decoder-only 基线；现代 decoder-only LLM 匹配它们 |
| Structured extraction（结构化提取） | Either（两者皆可） | T5 很干净，因为 "text → text" 吸收任何输出格式 |

自 ~2022 年以来的趋势：decoder-only 接管 encoder-decoder 曾经拥有的任务，因为 (a) instruction-tuned decoder-only LLM 通过 prompting 推广到任何东西，(b) 一个架构比两个更容易扩展，(c) RLHF 假设一个解码器。Encoder-decoder 在输入模态不同（speech、images）或 beam search 质量重要的地方坚守。

```figure
encoder-decoder
```

## Build It（动手实现）

见 `code/main.py`。我们实现 T5 风格的跨度损坏，用于一个 toy 语料库——这是这节课最有用的单个片段，因为它出现在每个 encoder-decoder 预训练配方中。

### Step 1: 跨度损坏

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标格式是 T5 约定：`<sent0> span0 <sent1> span1 ...`。损坏的输入在跨度位置交错未更改的 token 和 sentinel token。

### Step 2: 验证往返

给定损坏的输入和目标，重建原始句子。如果你的损坏是可逆的，前向传播就是定义良好的。这是一个健全性检查——真实训练从不这样做，但这个测试很便宜，能捕获跨度记账中的 off-by-one bug。

### Step 3: BART noising

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合其中两个并展示结果。

## Use It（实际应用）

HuggingFace 参考：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 技巧：任务名称进入输入文本。同一个模型处理数十个任务，因为每个任务都是 text-in, text-out。2026 年，这个模式已经被 instruction-tuned decoder-only 模型推广，但 T5 首先规范了它。

## Ship It（交付）

见 `outputs/skill-seq2seq-picker.md`。这个 skill 在给定 input-output 结构、延迟和质量目标的情况下，为一个新任务在 encoder-decoder 和 decoder-only 之间选择。

## Exercises（练习）

1. **Easy（简单）。** 运行 `code/main.py`，将跨度损坏应用于一个 30 token 的句子，验证连接非 sentinel 源 token 和解码目标跨度能重建原始值。
2. **Medium（中等）。** 实现 BART 的 `text_infill` 噪声：用单个 `<mask>` token 替换随机跨度，解码器必须推断正确的跨度长度加上内容。展示一个示例。
3. **Hard（困难）。** 在一个微型英语 → pig-Latin 语料库（200 对）上微调 `flan-t5-small`。在一个保留的 50 对集合上测量 BLEU。与在相同数据上使用相同计算微调 `Llama-3.2-1B` 比较。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Encoder-decoder（编码器-解码器） | "Seq2seq transformer" | 两个栈：用于输入的双向编码器，用于输出带交叉注意力的因果解码器。 |
| Cross-attention（交叉注意力） | "Where source talks to target"（源与目标对话的地方） | 解码器的 Q × 编码器的 K/V。编码器信息进入解码器的唯一地方。 |
| Span corruption（跨度损坏） | "T5's pretraining trick"（T5 的预训练技巧） | 用 sentinel token 替换随机跨度；解码器输出跨度。 |
| Denoising objective（denoising 目标） | "BART's game"（BART 的游戏） | 对输入应用噪声函数，训练解码器重建干净序列。 |
| Sentinel token（哨兵 token） | "The `<extra_id_N>` placeholder"（`<extra_id_N>` 占位符） | 在源中标记损坏跨度并在目标中重新标记它们的特殊 token。 |
| Flan | "Instruction-tuned T5"（指令微调 T5） | 在 >1,800 个任务上微调的 T5；使 encoder-decoder 在指令遵循上具有竞争力。 |
| Beam search（束搜索） | "Decoding strategy"（解码策略） | 在每一步保留 top-k 部分序列；翻译/摘要的标准。 |
| Teacher forcing（教师强制） | "Training-time input"（训练时输入） | 在训练期间，将真实的先前输出 token 输入解码器，而不是采样的那个。 |

## Further Reading（延伸阅读）

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART。
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper，2026 年规范 encoder-decoder。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — 参考实现。
