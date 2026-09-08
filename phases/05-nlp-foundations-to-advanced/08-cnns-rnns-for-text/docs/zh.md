# 用于文本的 CNN 和 RNN（CNNs and RNNs for Text）

> 卷积学习 n-gram。循环网络记住。两者都在注意力面前被超越。在受限硬件上两者仍然重要。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 11 (PyTorch Intro), Phase 5 · 03 (Word Embeddings), Phase 4 · 02 (Convolutions from Scratch)
**Time:** ~75 minutes

## 问题（The Problem）

TF-IDF 和 Word2Vec 产生忽略词序的扁平向量。基于它们的分类器无法区分 `dog bites man` 和 `man bites dog`。词序有时携带信号。

两类架构在 transformers 到来之前填补了这一空白。

**用于文本的卷积网络（TextCNN）。** 在词嵌入序列上应用一维卷积。宽度为 3 的过滤器是可学习的三元组检测器：它跨越三个词并输出一个分数。堆叠不同宽度（2、3、4、5）以检测多尺度模式。最大池化到固定大小的表示。扁平、并行、快速。

**循环网络（RNN、LSTM、GRU）。** 一次处理一个词元，维护一个向前携带信息的隐藏状态。顺序、有记忆、灵活的输入长度。从 2014 年到 2017 年主导序列建模，然后注意力出现了。

本课构建两者，然后命名导致注意力出现的失败。

## 概念（The Concept）

**TextCNN**（Kim，2014）。词元被嵌入。宽度为 `k` 的一维卷积在嵌入的连续 `k`-gram 上滑动一个过滤器，产生特征图。全局最大池化在该图上选取最强激活。从几个过滤器宽度的最大池化输出中拼接。馈送到分类头。

为什么它有效。过滤器是可学习的 n-gram。最大池化是位置不变的，所以 "not good" 在评论开头或中间触发相同的特征。三个宽度各 100 个过滤器给你 300 个学习到的 n-gram 检测器。训练是并行的；没有顺序依赖。

**RNN。** 在每个时间步 `t`，隐藏状态 `h_t = f(W * x_t + U * h_{t-1} + b)`。跨时间共享 `W`、`U`、`b`。时间步 `T` 的隐藏状态是整个前缀的摘要。对于分类，在 `h_1 ... h_T` 上池化（最大、平均或最后一个）。

普通 RNN 遭受梯度消失。**LSTM** 添加门控，决定忘记什么、存储什么、输出什么，通过长序列稳定梯度。**GRU** 将 LSTM 简化为两个门；参数更少，表现相似。

**双向 RNNs** 运行一个 RNN 向前和一个向后，拼接隐藏状态。每个词元的表示看到左右上下文。对于标注任务至关重要。

```figure
rnn-unroll
```

## 构建它（Build It）

### 步骤 1：PyTorch 中的 TextCNN（TextCNN in PyTorch）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 将 `[batch, seq_len, embed_dim]` 重塑为 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 将中间轴视为通道。池化输出是固定大小的，无论输入长度如何。

### 步骤 2：LSTM 分类器（LSTM classifier）

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

对序列最大池化，而非取最后状态。对于分类，最大池化通常优于取最后隐藏状态，因为长序列末尾的信息往往主导最后状态。

### 步骤 3：梯度消失演示（直觉）（the vanishing gradient demo (intuition)）

没有门控的普通 RNN 无法学习长距离依赖。考虑一个玩具任务：预测词元 `A` 是否出现在序列的任何位置。如果 `A` 在位置 1 且序列长 100 个词元，损失中的梯度必须通过 99 次循环权重的乘法反向传播。如果权重小于 1，梯度消失。如果大于 1，梯度爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTMs 通过单元状态修复这个问题，单元状态以仅加法交互（忘记门对其进行乘法缩放，但梯度仍沿"高速公路"流动）贯穿网络。GRUs 用更少参数做类似的事情。两者都让你在 100+ 步序列中稳定训练。

### 步骤 4：为什么这仍然不够（why this still was not enough）

即使有了 LSTMs，三个问题仍然存在。

1. **顺序瓶颈。** 在长度为 1000 的序列上训练 RNN 需要 1000 个串行前向/反向步骤。无法在时间上并行化。
2. **编码器-解码器设置中的固定大小上下文向量。** 解码器只看到编码器的最后隐藏状态，在整个输入上压缩。长输入丢失细节。第 09 课直接涵盖这一点。
3. **远距离依赖精度上限。** LSTMs 优于普通 RNN 但仍难以在 200+ 步上传播特定信息。

注意力解决了所有三个问题。Transformers 完全抛弃了循环。第 10 课是 pivot。

## 使用它（Use It）

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 是生产就绪的。训练代码是标准的。

Hugging Face 提供预训练嵌入，你可以作为输入层插入：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

适用时使用约束清单。

- **边缘 / 端侧推理。** 带 GloVe 嵌入的 TextCNN 比 transformer 小 10-100 倍。如果部署目标是手机，这就是你的栈。
- **流式 / 在线分类。** RNN 一次处理一个词元；transformers 需要完整序列。对于实时传入文本，LSTMs 仍然胜出。
- **用于基线的微小模型。** 在新任务上快速迭代。在 CPU 上 5 分钟训练一个 TextCNN。
- **有限数据的序列标注。** BiLSTM-CRF（第 06 课）对于 1k-10k 标注句子仍然是生产级 NER 架构。

其他所有情况都交给 transformer。

## 交付物（Ship It）

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 练习（Exercises）

1. **简单。** 在 3 类玩具数据集（你自创数据）上训练 TextCNN。验证过滤器宽度（2、3、4）平均 F1 优于单一宽度（3）。
2. **中等。** 为 LSTM 分类器实现最大池化、平均池化和最后状态池化。在小数据集上比较；记录哪种池化胜出并假设原因。
3. **困难。** 构建 BiLSTM-CRF NER 标注器（结合第 06 课和本课）。在 CoNLL-2003 上训练。与第 06 课的 CRF 基线和 BERT 微调比较。报告训练时间、内存和 F1。

## 关键术语（Key Terms）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| TextCNN | 用于文本的 CNN | 词嵌入上全局最大池化的一维卷积堆叠。Kim (2014)。 |
| RNN | 循环网络 | 在每个时间步更新的隐藏状态：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | 门控循环网络 | 添加输入 / 遗忘 / 输出门 + 一个单元状态。在长序列中稳定训练。 |
| GRU | 更简单的 LSTM | 两个门而非三个。相似准确率，更少参数。 |
| Bidirectional | 双向 | 前向 + 后向循环网络拼接。每个词元看到其上下文的两侧。 |
| Vanishing gradient | 训练信号消失 | 普通 RNN 中重复乘以小于 1 的权重使早期步骤梯度几乎为零。 |

## 拓展阅读（Further Reading）

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 论文。八页。易读。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 论文。意外地清晰。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 让 LSTM 对所有人都易懂的图。
