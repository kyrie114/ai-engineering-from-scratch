# 序列到序列模型（Sequence-to-Sequence Models）

> 两个 RNN 假装是翻译器。它们遇到的瓶颈是注意力存在的原因。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 08 (CNNs + RNNs for Text), Phase 3 · 11 (PyTorch Intro)
**Time:** ~75 minutes

## 问题（The Problem）

分类将一个可变长度序列映射到单个标签。翻译将一个可变长度序列映射到另一个可变长度序列。输入和输出属于不同词汇表，可能不同语言，长度不一定相等。

seq2seq 架构（Sutskever、Vinyals、Le，2014）用一个刻意简单的配方解决了这个问题。两个 RNN。一个读取源句子并产生固定大小的上下文向量。另一个读取该向量并逐个词元生成目标句子。和你第 08 课写的代码一样，只是拼接方式不同。

这值得学习有两个原因。首先，上下文向量瓶颈是 NLP 中最有教学用途的失败。它促成了注意力和 transformers 所做的一切。其次，训练配方（教师强制、计划采样、推理时的束搜索）仍然适用于包括 LLM 在内的每个现代生成系统。

## 概念（The Concept）

**编码器。** 读取源句子的 RNN。其最后隐藏状态是上下文向量 — 整个输入的固定大小摘要。据称不会丢失任何东西。

**解码器。** 从上下文向量初始化的另一个 RNN。在每个步，它取先前生成的词元作为输入，并产生目标词汇表上的分布。采样或 argmax 选取下一个词元。反馈回去。重复直到产生 `<EOS>` 词元或达到最大长度。

**训练：** 每个解码器步的交叉熵损失，对序列求和。通过两个网络的标准反向传播 through time。

**教师强制。** 训练期间，解码器在步 `t` 的输入是步 `t-1` 的真实词元，而非解码器自己的先前预测。这稳定训练；没有它，早期错误级联，模型永远学不会。推理时，你必须使用模型自己的预测，所以总是存在训练/推理分布差距。这个差距称为暴露偏差（exposure bias）。

**瓶颈。** 编码器学到的关于源的全部内容必须被压缩进那一个上下文向量。长句子丢失细节。罕见词被模糊。重排（chat noir 与 black cat）必须被记忆，而非计算。

注意力（第 10 课）通过让解码器查看编码器的每个隐藏状态（而不仅是最后一个）来修复这个问题。这就是全部卖点。

```figure
lstm-gates
```

## 构建它（Build It）

### 步骤 1：编码器（an encoder）

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 的形状是 `[batch, seq_len, hidden_dim]` — 每个输入位置一个隐藏状态。`hidden` 的形状是 `[1, batch, hidden_dim]` — 最后一步。第 08 课说"对输出池化用于分类。"这里我们保留最后隐藏状态作为上下文向量，并忽略每步输出。

### 步骤 2：解码器（a decoder）

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器一次被调用一步。输入：一批单个词元和当前隐藏状态。输出：下一个词元的词汇表 logits 和更新后的隐藏状态。

### 步骤 3：带教师强制的训练循环（training loop with teacher forcing）

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

两个值得命名的旋钮。`ignore_index=0` 跳过填充词元的损失。`teacher_forcing_ratio` 是每一步使用真实词元而非模型预测的概率。从 1.0（完全教师强制）开始，在训练过程中 anneal 到约 0.5 以缩小暴露偏差差距。

### 步骤 4：推理循环（贪心）（inference loop (greedy)）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪心解码在每一步选取最高概率词元。它会走偏：一旦你提交一个词元，你无法撤回。束搜索（beam search）保留 top-`k` 部分序列存活，在最后选取最高分的完整序列。束宽 3-5 是标准的。

### 步骤 5：瓶颈演示（the bottleneck, demonstrated）

在玩具复制任务上训练模型：源 `[a, b, c, d, e]`，目标 `[a, b, c, d, e]`。增加序列长度。观察准确率。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个 GRU 隐藏状态无法无损记忆 40 词元输入。信息在编码器的每一步都存在，但解码器只看到最后状态。注意力直接修复这个问题。

## 使用它（Use It）

PyTorch 有 `nn.Transformer` 和基于 `nn.LSTM` 的 seq2seq 模板。Hugging Face 的 `transformers` 库提供完整的编码器-解码器模型（BART、T5、mBART、NLLB），在数十亿词元上训练。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代编码器-解码器为 transformers 抛弃了 RNNs。高层形状（编码器、解码器、逐个词元生成）与 2014 年 seq2seq 论文相同。每个块内部的机制不同。

### 何时仍然使用基于 RNN 的 seq2seq（When to still reach for RNN-based seq2seq）

几乎从不，对于新项目。特定例外：

- 流式翻译，你一次消费一个词元且有界内存。
- 端侧文本生成，transformer 内存成本 prohibitive。
- 教学。理解编码器-解码器瓶颈是理解为什么 transformers 胜出的最快路径。

### 暴露偏差及其缓解方法（Exposure bias and its mitigations）

- **计划采样。** 在训练期间 anneal 教师强制比率，使模型学会从自己的错误中恢复。
- **最小风险训练。** 在句子级 BLEU 分数上训练，而非词元级交叉熵。更接近你真正想要的。
- **强化学习微调。** 用指标奖励序列生成器。现代 LLM RLHF 中使用。

所有三个仍然适用于基于 transformer 的生成。

## 交付物（Ship It）

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 练习（Exercises）

1. **简单。** 实现玩具复制任务。在输入-输出对（目标等于源）上训练 GRU seq2seq。在长度 5、10、20 上测量准确率。复现瓶颈。
2. **中等。** 添加束宽为 3 的束搜索解码。在小并行语料库上与贪心解码比较 BLEU。记录束搜索在哪里胜出（通常是最后词元）以及在哪里没有区别。
3. **困难。** 在 10k 对复述数据集上微调 `facebook/bart-base`。将微调模型的束宽 4 输出与基础模型在保留输入上的输出比较。报告 BLEU 并挑选 10 个定性示例。

## 关键术语（Key Terms）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Encoder | 输入 RNN | 读取源。产生每步隐藏状态和最终上下文向量。 |
| Decoder | 输出 RNN | 从上下文向量初始化。逐个生成目标词元。 |
| Context vector | 摘要 | 最终编码器隐藏状态。固定大小。注意力解决的瓶颈。 |
| Teacher forcing | 使用真实词元 | 训练时喂入真实的前一个词元。稳定学习。 |
| Exposure bias | 训练/测试差距 | 在真实词元上训练的模型从未练习过从自己的错误中恢复。 |
| Beam search | 更好的解码 | 在每一步保留 top-k 部分序列，而不是贪心提交。 |

## 拓展阅读（Further Reading）

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 原始 seq2seq 论文。四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — 引入了 GRU 和编码器-解码器框架。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 注意力论文。本课后立即阅读。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 可构建的 seq2seq + 注意力代码。
