# OCR 与文档理解（OCR & Document Understanding）

> OCR 是一条三段式流水线——检测文本框、识别字符、再做版面。每个现代 OCR 系统都会重排或合并这些阶段。

**Type:** Learn + Use
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 06 (Detection), Phase 7 Lesson 02 (Self-Attention)
**Time:** ~45 minutes

## 学习目标（Learning Objectives）

- 梳理经典 OCR 流水线（检测 -> 识别 -> 版面）与现代端到端替代方案（Donut、Qwen-VL-OCR）
- 为序列到序列的 OCR 训练实现 CTC（Connectionist Temporal Classification）损失
- 不经训练，直接用 PaddleOCR 或 EasyOCR 做生产级文档解析
- 分清 OCR、版面解析与文档理解——并按任务挑对工具

## 问题所在（The Problem）

满是文字的图片无处不在：收据、发票、证件、扫描书、表单、白板、路牌、截图。从中提取结构化数据——不只是字符，而是“这是总金额”——是应用视觉里价值最高的问题之一。

这个领域分成三层技能：

1. **狭义 OCR（OCR proper）**：把像素变成文本。
2. **版面解析（layout parsing）**：把 OCR 输出分组为区域（标题、正文、表格、页眉）。
3. **文档理解（document understanding）**：从版面中提取结构化字段（“invoice_total = $42.50”）。

每一层都有经典与现代两种路线，而“我想要图片里的文字”和“我要这张收据上的总金额”之间的鸿沟，比大多数团队意识到的大得多。

## 核心概念（The Concept）

### 经典流水线（The classical pipeline）

```mermaid
flowchart LR
    IMG["图片"] --> DET["文本检测<br/>（DB, EAST, CRAFT）"]
    DET --> BOX["词/行<br/>边界框"]
    BOX --> CROP["裁剪每个区域"]
    CROP --> REC["识别<br/>（CRNN + CTC）"]
    REC --> TXT["文本字符串"]
    TXT --> LAY["版面<br/>排序"]
    LAY --> OUT["按阅读顺序排列的文本"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **文本检测**输出逐行或逐词的四边形。
- **识别**把每个区域裁成固定高度，跑一个 CNN + BiLSTM + CTC 得到字符序列。
- **版面**重建阅读顺序（拉丁文自上而下、从左到右；阿拉伯语、日语则不同）。

### 一段话讲完 CTC（CTC in one paragraph）

OCR 识别要从定长特征图产生变长序列。CTC（Graves 等，2006）让你无需字符级对齐就能训练它。模型在每个时间步输出一个（词表 + blank）上的分布；CTC 损失对所有能经过合并重复、去掉 blank 之后还原成目标文本的对齐路径求边缘化。

```
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```

CTC 是 CRNN 能在 2015 年奏效的原因，而且 2026 年大多数生产级 OCR 模型仍用它来训练。

### 现代端到端模型（Modern end-to-end models）

- **Donut**（Kim 等，2022）——ViT 编码器 + 文本解码器；读入图片，直接输出 JSON。没有文本检测器，没有版面模块。
- **TrOCR**——ViT + transformer 解码器，做行级 OCR。
- **Qwen-VL-OCR / InternVL**——为 OCR 任务微调过的完整视觉语言模型；2026 年在复杂文档上准确率最好。
- **PaddleOCR**——经典 DB + CRNN 流水线打包成成熟的生产套件；仍是开源界的主力。

端到端模型需要更多数据和算力，但跳过了多级流水线的误差累积。

### 版面解析（Layout parsing）

对结构化文档，运行一个版面检测器（LayoutLMv3、DocLayNet）给每个区域打标签：标题（Title）、段落（Paragraph）、插图（Figure）、表格（Table）、脚注（Footnote）。阅读顺序于是变成“按版面顺序遍历区域，再拼接起来”。

对表单，使用**键值抽取（Key-Value extraction）**模型（视觉富文档用 Donut，普通扫描件用 LayoutLMv3）。它们接收图片 + 检测出的文本 + 位置，预测结构化的键值对。

### 评估指标（Evaluation metrics）

- **字符错误率（Character Error Rate, CER）**——Levenshtein 距离 / 参考文本长度。越低越好。生产目标：干净扫描件上 < 2%。
- **词错误率（Word Error Rate, WER）**——词粒度上的同样指标。
- **结构化字段的 F1**——针对键值任务；衡量 `{invoice_total: 42.50}` 是否正确出现。
- **JSON 编辑距离**——针对端到端文档解析；Donut 论文引入了归一化树编辑距离。

```figure
cv3-ctc-collapse
```

## 动手构建（Build It）

### 第 1 步：CTC 损失 + 贪心解码器（Step 1: CTC loss + greedy decoder）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax over vocab including blank at index 0
    targets:        (N, S) int targets (no blanks)
    input_lengths:  (N,) per-sample time steps used
    target_lengths: (N,) per-sample target length
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (blanks removed, repeats merged)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss` 在可用时会用高效的 CuDNN 实现。贪心解码器比束搜索简单，CER 通常相差不到 1%。

### 第 2 步：微型 CRNN 识别器（Step 2: Tiny CRNN recogniser）

面向行级 OCR 的极简 CNN + BiLSTM。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

定高输入（CNN 把高度最大池化到 1）。宽度就是 CTC 的时间维度。

### 第 3 步：合成 OCR（Step 3: Synthetic OCR）

生成白底黑字的数字字符串，做端到端冒烟测试。

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

真实的 OCR 数据集会加上字体、噪声、旋转、模糊和色彩。流水线与上面完全一样。

### 第 4 步：训练速写（Step 4: Training sketch）

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

在这份简单合成数据上，损失应在 200 步内从 ~3 降到 ~0.2。

## 实际使用（Use It）

三条生产路线：

- **PaddleOCR**——成熟、快、多语言。一行用法：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。
- **EasyOCR**——Python 原生、多语言、PyTorch 骨干。
- **Tesseract**——经典；当现代模型吃力时，对老旧扫描文档仍然有用。

端到端文档解析用 Donut 或 VLM：

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

对结构可重复的收据、发票和表单，微调 Donut。对任意文档或需要推理的 OCR，Qwen-VL-OCR 这类 VLM 是当前默认。

## 交付成果（Ship It）

本课产出：

- `outputs/prompt-ocr-stack-picker.md` —— 一个提示词，根据文档类型、语言和结构挑选 Tesseract / PaddleOCR / Donut / VLM-OCR。
- `outputs/skill-ctc-decoder.md` —— 一个技能，从零编写贪心和束搜索 CTC 解码器，包含长度归一化。

## 练习（Exercises）

1. **（简单）** 用 5 位随机数字字符串训练 TinyCRNN 500 步。报告留出集上的 CER。
2. **（中等）** 把贪心解码换成束搜索（beam_width=5）。报告 CER 差值。束搜索在哪类输入上占优？
3. **（困难）** 在 20 张收据上用 PaddleOCR 抽取明细行，并对 {item_name, price} 对相对手工标注的真值计算 F1。

## 关键术语（Key Terms）

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| OCR | “从像素到文字” | 把图像区域变成字符序列 |
| CTC | “免对齐损失” | 无需逐时间步标签就能训练序列模型的损失；对所有对齐做边缘化 |
| CRNN | “经典 OCR 模型” | 卷积特征提取器 + BiLSTM + CTC；2015 年的基线至今仍在生产中使用 |
| Donut | “端到端 OCR” | ViT 编码器 + 文本解码器；从图片直接输出 JSON |
| 版面解析（Layout parsing） | “找区域” | 在文档中检测并标注标题/表格/插图/段落区域 |
| 阅读顺序（Reading order） | “文本顺序” | 把识别出的区域排成一句话的顺序；对拉丁文很简单，对混合版面不简单 |
| CER / WER | “错误率” | 字符或词粒度上的 Levenshtein 距离 / 参考长度 |
| VLM-OCR | “会读的 LLM” | 为 OCR 任务训练或提示的视觉语言模型；当前在复杂文档上是 SOTA |

## 延伸阅读（Further Reading）

- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717) —— 最早的 CNN+RNN+CTC 架构
- [CTC (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) —— CTC 原始论文；满满都是算法思想
- [Donut (Kim et al., 2022)](https://arxiv.org/abs/2111.15664) —— 无 OCR 的文档理解 transformer
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) —— 开源生产级 OCR 套件
