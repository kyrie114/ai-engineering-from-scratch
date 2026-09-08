# Audio Classification — From k-NN on MFCCs to AST and BEATs（音频分类——从 MFCC 上的 k-NN 到 AST 和 BEATs）

> 从"狗叫 vs 警笛"到"这是什么语言"都是音频分类。特征是梅尔特征。架构每个十年都在演进。评估指标始终是 AUC、F1 和每类召回率。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 3 · 06 (CNNs), Phase 5 · 08 (CNNs & RNNs for Text)
**Time:** ~75 minutes

## 问题（The Problem）

你拿到一段 10 秒的音频。你想知道："这是什么？"城市声音（警笛、电钻、狗叫）、语音指令（是/否/停止）、语言 ID（en/es/ar）、说话人情绪（愤怒/中立）或环境音（室内/室外、嘈杂）。这些都属于*音频分类*，2026 年的基线架构已成熟：对数梅尔 → CNN 或 Transformer → softmax。

核心难点不在网络，而在数据。音频数据集有严重的类别不平衡、强烈的领域偏移（干净 vs 嘈杂）以及标签噪声（谁决定了"城市嘈杂声"和"餐厅噪声"的边界？）。问题的 80% 在于数据整理、增强和评估，而不是把 CNN 换成 Transformer。

## The Concept（概念）


将每个片段的 MFCC 展平，计算与标注库的余弦相似度，返回前 K 个的多数投票。在干净的小数据集（Speech Commands、ESC-50）上出奇地强。无需 GPU 即可运行。

将 `(T, n_mels)` 的对数梅尔视为一张图像。应用 ResNet-18 或 VGG 风格网络。对时间轴做全局平均池化。对类别做 softmax。在大多数 2026 年 kaggle 竞赛中仍是基线。

**Audio Spectrogram Transformer, AST (2021-2024)（音频语谱图 Transformer）.** Patchify the log-mel (e.g. 16×16 patches), add position embeddings, feed to a ViT. State of the art on AudioSet (mAP 0.485) for supervised learning.

在数百万小时数据上进行自监督预训练。用你原本所需的监督数据的 1–10% 在你的任务上微调。2026 年这是非语音音频的默认起点。BEATs-iter3 在 AudioSet 上以 1/4 的计算量超过 AST 1–2 个 mAP。

取 Whisper 的编码器，去掉解码器，接一个线性分类器。在语言 ID 和简单事件分类上接近 SOTA，无需音频增强。"免费午餐"基线。

### Class imbalance is the real challenge（类别不平衡才是真正的挑战）

ESC-50：50 个类别，每类 40 个片段——平衡，容易。UrbanSound8K：10 个类别，不平衡 10:1。AudioSet：632 个类别，长尾比 100,000:1。有效技术：

  训练时平衡采样（评估时不用）。
  Mixup：线性插值两个片段（及其标签）作为增强。
  SpecAugment：掩码随机时间和频率带。简单；至关重要。

### Evaluation（评估）

- Multiclass exclusive (Speech Commands): top-1 accuracy, top-5 accuracy.
  互斥多类（Speech Commands）：top-1 准确率，top-5 准确率。
- Multiclass multi-label (AudioSet, UrbanSound-style): mean average precision (mAP).
  多标签多类（AudioSet、UrbanSound 风格）：平均精度均值（mAP）。
  严重不平衡：每类召回率 + macro F1。

2026 年你应该知道的数字：

|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs paper (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR leaderboard 2026 |

```figure
mfcc-pipeline
```

## Build It（动手实现）

### Step 1: featurize（特征提取）

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### Step 2: fixed-length summary（固定长度摘要）

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单但强大：跨时间的均值 + 方差为一个 13 系数 MFCC 生成一个 26 维固定嵌入。瞬间运行。直到 2017 年还在 ESC-50 上击败最先进的 NN 基线。

### Step 3: k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### Step 4: upgrade to CNN on log-mels（升级为对数梅尔上的 CNN）

In PyTorch:

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

300 万个参数。在单张 RTX 4090 上约 10 分钟即可在 ESC-50 上训练完成。准确率 80% 以上。

### Step 5: the 2026 default — fine-tune BEATs（2026 年默认方案——微调 BEATs）

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

For BEATs, use `microsoft/BEATs-base` via the `beats` library; the transformers API is the same shape.

## Use It（实际应用）

2026 年栈：

|-----------|-----------|
|                              | MFCC 均值上的 k-NN（你的基线）+ 音频增强 |
| Medium dataset (1K–100K) | BEATs or AST fine-tune |
|                           | 微调 BEATs 或 AST |
|                       | 从头训练或微调 Whisper 编码器 |
| Real-time, edge | 40-MFCC CNN, quantized to int8 (KWS-style) |
|                  | 40-MFCC CNN，量化为 int8（KWS 风格） |
| Multi-label (AudioSet) | BEATs-iter3 with BCE loss + mixup + SpecAugment |
|                         | BEATs-iter3 + BCE 损失 + mixup + SpecAugment |
| Language ID | MMS-LID, SpeechBrain VoxLingua107 baseline |

决策规则：**从冻结的骨干网络开始，而不是从零开始的新模型**。微调一个 BEATs head 能在几小时内达到 95% 的 SOTA，而不是几周。

## Ship It（交付成果）

保存为 `outputs/skill-classifier-designer.md`。为给定音频分类任务选择架构、增强策略、类别平衡策略和评估指标。

## Exercises（练习）

   运行 `code/main.py`。它在 4 类合成数据集（不同音高的纯音）上训练 k-NN MFCC 基线。报告混淆矩阵。
2. **Medium（中等）.** Replace `summarize` with [mean, var, skew, kurtosis]. Does 4-moment pooling beat mean+var on the same synthetic dataset?
   用 [mean, var, skew, kurtosis] 替换 summarize。4 阶矩池化在相同合成数据集上是否优于 mean+var？
3. **Hard（困难）.** Using `torchaudio`, train a 2D CNN on ESC-50 fold 1. Report 5-fold cross-validation accuracy. Add SpecAugment (time mask = 20, freq mask = 10) and report the delta.

## Key Terms（关键术语）

|------|-----------------|-----------------------|
|           | 音频领域的 ImageNet | Google 的 200 万片段、632 类弱标注 YouTube 数据集。 |
|        | 小型分类基准 | 50 个类别 × 40 个环境音片段。 |
| AST | Audio Spectrogram Transformer | ViT on log-mel patches; 2021 SOTA. |
|     | 音频语谱图 Transformer | 在 log-mel patch 上的 ViT；2021 年 SOTA。 |
| BEATs | Self-supervised audio | Microsoft model, iter3 leads AudioSet as of 2026. |
|       | 自监督音频 | 微软模型，截至 2026 年 iter3 在 AudioSet 上领先。 |
| Mixup | Pair augmentation | x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2. |
|       | 对增强 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | Mask-based augmentation | Zero-out random time and frequency bands of the spectrogram. |
|             | 基于掩码的增强 | 将语谱图的随机时间和频率带置零。 |
| mAP | Main multi-label metric | Mean average precision across classes and thresholds. |
|     | 主要多标签指标 | 跨类别和阈度的平均精度均值。 |

## Further Reading（延伸阅读）

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — the architecture of record from 2021–2024.
  Gong, Chung, Glass (2021).《AST：音频语谱图 Transformer》——2021–2024 年的记录架构。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) —— 2024+ 默认方案。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) —— 主导音频增强。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) —— 持续存在的小型基准。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 632-class YouTube taxonomy; still the gold standard.
  Gemmeke et al. (2017).《AudioSet》——632 类 YouTube 分类体系；仍是黄金标准。