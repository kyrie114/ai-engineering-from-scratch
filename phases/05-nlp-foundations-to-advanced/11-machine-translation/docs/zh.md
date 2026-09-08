# 机器翻译（Machine Translation）

> 翻译这项任务为 NLP 研究付了三十年经费，而且至今仍在持续。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 10 (Attention Mechanism), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 minutes

## 问题（The Problem）

一个模型读取一种语言的句子，输出另一种语言的句子。长度不同，词序不同，有些源语言词映射到多个目标词，反之亦然。习语拒绝一对一映射。"I miss you" 在法语里是 "tu me manques"——字面意思是 "you are lacking to me"。没有任何词级对齐能在这场转换中存活。

机器翻译这项任务迫使 NLP 发明了 编码器-解码器（encoder-decoder）、注意力机制（attention）、transformer，以及最终整个 LLM 范式。每一步前进都因为翻译质量可测量，而且人类与机器之间的差距始终顽固。

本课跳过历史，教授 2026 年的工作流程：预训练的多语言 编码器-解码器（encoder-decoder，NLLB-200 或 mBART）、子词分词（subword tokenization）、束搜索（beam search）、BLEU 与 chrF 评估，以及仍然悄悄进入生产环境的几种失败模式。

## 概念（The Concept）

![MT pipeline: tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

现代机器翻译是在平行文本上训练的 transformer 编码器-解码器。编码器以源语言的分词读取输入。解码器逐个子词生成目标文本，通过交叉注意力（cross-attention，第 10 课）使用编码器的输出。解码使用束搜索以避免贪心解码陷阱。输出经过去分词（detokenize）、去真实化（detruecase）并与参考译文评分。

三个实际操作选择驱动真实世界机器翻译质量。

- **分词器（Tokenizer）。** 在混合语言语料库上训练的 SentencePiece BPE。跨语言共享词汇表是 NLLB 实现零-shot 语言对的关键。
- **模型大小（Model size）。** NLLB-200 distilled 600M 可以在一台笔记本电脑上运行。NLLB-200 3.3B 是公布的默认生产模型。54.5B 是研究天花板。
- **解码（Decoding）。** 通用内容束宽（beam width）4-5。长度惩罚以避免输出过短。需要术语一致性时使用约束解码（constrained decoding）。

```figure
seq2seq-alignment
```

## 构建它（Build It）

### 步骤 1：预训练机器翻译调用

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

这里有三件事重要。`src_lang` 告诉分词器应用哪种文字和分词。`forced_bos_token_id` 告诉解码器生成哪种语言。两者都是 NLLB 特有的技巧；mBART 和 M2M-100 使用自己的约定，它们不可互换。

### 步骤 2：BLEU 与 chrF

BLEU 测量输出与参考译文之间的 n-gram 重叠。四个参考 n-gram 大小（1-4），精确率的几何均值，对过短输出的惩罚。分数在 [0, 100] 范围内。常用。令人沮丧的是难以解释：30 BLEU 是 "可用"；40 是 "好"；50 是 "卓越"；1 BLEU 以下的差异是噪音。

chrF 测量字符级 F 分数（F-score）。对于形态丰富的语言更敏感，BLEU 会低估匹配。通常与 BLEU 一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

始终使用 `sacrebleu`。它规范化分词，因此分数在论文之间可比。自己写 BLEU 计算是误导性基准产生的方式。

### 三层评估层次（2026）

现代机器翻译评估使用三种互补的度量家族。至少与两个一起发布。

- **启发式（Heuristic）**（BLEU, chrF）。快速，基于参考，可解释，对释义不敏感。用于遗留比较和回归检测。
- **学习的（Learned）**（COMET, BLEURT, BERTScore）。在人类判断上训练的神经模型；比较翻译与源和参考的语义相似度。COMET 自 2023 年以来与机器翻译研究的关联度最高，是 2026 年质量优先的生产默认。
- **LLM 作为评判（LLM-as-judge）**（免参考）。提示一个大型模型根据流畅度、充分性（adequacy）、语气、文化适当性对翻译评分。当评分标准设计良好时，GPT-4 作为评判与人类协议约 80% 一致。用于没有参考的开放式内容。

实用的 2026 技术栈：`sacrebleu` 用于 BLEU 和 chrF，`unbabel-comet` 用于 COMET，提示 LLM 用于最终面向人类的信号。在生产数据上信任之前，根据 50-100 个人工标注的样本校准每个度量。

免参考度量（COMET-QE, BLEURT-QE, LLM-as-judge）让你在没有参考的情况下评估翻译，这对于长尾语言对很重要，因为不存在参考翻译。

### 步骤 3：生产中什么会出错

上面的工作流程 80% 的时间会流利地翻译，剩下的 20% 会静默失败。已命名的失败模式：

- **幻觉（Hallucination）。** 模型编造源语言中没有的内容。在不熟悉的领域词汇中常见。症状：输出流利，但声称源语言没有陈述的事实。缓解：术语上的约束解码（constrained decoding），受监管内容的人工审查，监控比输入长得多的输出。
- **目标语言错误生成（Off-target generation）。** 模型翻译成错误的语言。NLLB 在稀有语言对上对此特别敏感。缓解：验证 `forced_bos_token_id`，始终使用语言 ID 模型检查输出来解码。
- **术语漂移（Terminology drift）。** "Sign up" 在文档 1 变成 "s'inscrire"，在文档 2 变成 "créer un compte"。对于 UI 文本和面向用户的字符串，一致性比原始质量更重要。缓解：词汇表约束解码或后期编辑字典。
- **正式性不匹配（Formality mismatch）。** 法语的 "tu" 与 "vous"，日语礼貌级别。模型选择训练中更常见的形式。对于面向客户的内容，这通常是错的。缓解：如果模型支持，用正式性词元（formality token）作为提示前缀，或仅在正式语料库上微调一个小模型。
- **短输入的长度爆炸（Length explosion on short input）。** 非常短的输入句子经常产生过长的翻译，因为长度惩罚在约 5 个源词元以下急剧下降。缓解：与源长度成比例的硬最大长度限制。

### 步骤 4：领域微调

预训练模型是通才。法律、医疗或游戏对话翻译从领域平行数据的微调中明显受益。这个配方并不稀奇：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千个高质量平行示例胜过几十万个嘈杂的网络抓取样本。训练数据的质量是单一最大的生产杠杆。

## 使用它（Use It）

2026 年机器翻译生产栈：

| 使用场景 | 推荐的起点 |
|---------|---------------------------|
| 任意到任意，200 种语言 | `facebook/nllb-200-distilled-600M`（笔记本电脑）或 `nllb-200-3.3B`（生产环境） |
| 以英语为中心，高质量，50 种语言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短运行，廉价推理，英法/德/西 | Helsinki-NLP / Marian 模型 |
| 延迟关键的浏览器端 | ONNX 量化 Marian（约 50 MB） |
| 最大质量，愿意付费 | GPT-4 / Claude / Gemini 配合翻译提示 |

截至 2026 年，LLM 在几种语言对上已经优于专门的机器翻译模型，特别是在习语内容和长上下文上。权衡是按词元成本和延迟。当上下文长度、风格一致性或通过提示的领域适应性比吞吐量更重要时，选择 LLM。

## 发布它（Ship It）

保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## 练习（Exercises）

1. **简单（Easy）。** 使用 `nllb-200-distilled-600M` 将一个 5 句英语段落翻译成法语，再回译成英语。测量回译与原文的接近程度。你应该看到语义保留，但用词漂移。
2. **中等（Medium）。** 使用 `fasttext lid.176` 或 `langdetect` 实现翻译输出的语言 ID 检查。集成到机器翻译调用中，以便在返回之前捕获目标语言错误的生成。
3. **困难（Hard）。** 在你选择的 5000 对领域语料库上微调 `nllb-200-distilled-600M`。在保留集上测量微调前后的 BLEU。报告哪些句子改进了，哪些退化了。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| BLEU | 翻译分数 | 带简洁性惩罚的 n-gram 精确率。[0, 100]。 |
| chrF | 字符 F 分数 | 字符级 F 分数。对形态丰富的语言更敏感。 |
| NMT | 神经机器翻译 | 在平行文本上训练的 transformer 编码器-解码器。2017 年以后的默认。 |
| NLLB | No Language Left Behind | Meta 的 200 种语言机器翻译模型家族。 |
| 约束解码（Constrained decoding） | 受控输出 | 强制特定词元或 n-gram 在输出中出现 / 不出现。 |
| 幻觉（Hallucination） | 编造内容 | 模型输出不被源语言支持的内容。 |

## 延伸阅读（Further Reading）

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB 论文。
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — 为什么 `sacrebleu` 是报告 BLEU 的唯一正确方式。
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF 论文。
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — 实用微调教程。
