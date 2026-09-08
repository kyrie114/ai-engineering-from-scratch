# Speaker Recognition & Verification（说话人识别与验证）

> ASR asks "what did they say?" Speaker recognition asks "who said it?" The math looks the same — embeddings plus cosine — but every production decision hinges on a single EER number.
> ASR 问的是"他们说了什么？"说话人识别问的是"是谁说的？"数学看起来一样——嵌入加余弦——但每一个生产决策都取决于一个 EER 数字。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 22 (Embedding Models)
**Time:** ~45 minutes

## The Problem（问题）

A user says a passphrase. You want to know: is this the person they claim to be (*verification*, 1:1), or is it the first person in your enrollment bank (*identification*, 1:N)? Or neither — is this an unknown speaker (*open-set*)?
用户说一句口令。你想知道：他是不是自称的那个人（验证，1:1），还是你注册库里的第一个人（识别，1:N）？或者都不是——这是一个未知说话人（开集）？

Pre-2018: GMM-UBM + i-vectors. Reasonable EER but fragile to channel shift (phone vs laptop) and emotion. 2018–2022: x-vectors (TDNN backbone trained with angular margin). 2022+: ECAPA-TDNN and WavLM-large embeddings. By 2026 the field is dominated by three models and one metric.
2018 年之前：GMM-UBM + i-vectors。EER 尚可，但对信道偏移（手机 vs 笔记本）和情绪脆弱。2018–2022 年：x-vectors（用角边距训练的 TDNN 主干）。2022 年之后：ECAPA-TDNN 和 WavLM-large 嵌入。到 2026 年，该领域由三个模型和一个指标主导。

The metric is **EER** — Equal Error Rate. Set your decision threshold so False Accept Rate = False Reject Rate. The crossover is EER. Used in every paper, every leaderboard, every procurement call.
这个指标是 **EER**——等错误率。设置决策阈值使错误接受率等于错误拒绝率，这个交叉点就是 EER。每篇论文、每个排行榜、每次采购电话都在用它。

## The Concept（概念）

![Enrollment + verification pipeline with embedding + cosine + EER](../assets/speaker-verification.svg)

**The pipeline（流水线）.** Enrollment: record 5–30 seconds of the target speaker; compute a fixed-dimension embedding (192-d for ECAPA-TDNN, 256-d for WavLM-large). Verification: get the test utterance embedding; compute cosine similarity; compare to a threshold.
注册：录制 5–30 秒目标说话人的音频；计算一个固定维度的嵌入（ECAPA-TDNN 为 192 维，WavLM-large 为 256 维）。验证：获取测试话语嵌入；计算余弦相似度；与阈值比较。

**ECAPA-TDNN (2020, still dominant 2026)（ECAPA-TDNN（2020 年，2026 年仍占主导）).** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network. 1D conv blocks with squeeze-excitation, multi-head attention pooling, followed by a linear layer to 192-d. Trained on VoxCeleb 1+2 (2,700 speakers, 1.1M utterances) with Additive Angular Margin loss (AAM-softmax).
强调通道注意力、传播与聚合——时延神经网络。带 squeeze-excitation 的 1D 卷积块、多头注意力池化，后接一个线性层到 192 维。在 VoxCeleb 1+2（2,700 个说话人，110 万条话语）上用加性角边距损失（AAM-softmax）训练。

**WavLM-SV (2022+)（WavLM-SV（2022 年之后）).** Fine-tune a pretrained WavLM-large SSL backbone with AAM loss. Higher quality but slower — 300+ MB vs 15 MB.
用 AAM 损失微调一个预训练的 WavLM-large SSL 主干。质量更高但更慢——300+ MB 对 15 MB。

**x-vector (baseline)（x-vector（基线）).** TDNN + statistics pooling. Classic; still useful on CPU / edge.
TDNN + 统计池化。经典；在 CPU / 边缘设备上仍然有用。

**AAM-softmax.** Standard softmax with added margin `m` in the angular space: `cos(θ + m)` for the correct class. Forces inter-class angular separation. Typical `m=0.2`, scale `s=30`.
标准 softmax，在角空间中加入边距 `m`：正确类为 `cos(θ + m)`。强制类间角分离。典型值 `m=0.2`，缩放 `s=30`。

### Scoring（评分）

- **Cosine** between enrollment and test embeddings. Threshold-based decision.
  注册与测试嵌入之间的余弦。基于阈值的决策。
- **PLDA (Probabilistic LDA).** Project embeddings into a latent space where same-speaker vs different-speaker has a closed-form likelihood ratio. Added on top of cosine for +10–20% EER reduction. Standard pre-2020; now used only in closed-set setups.
  将嵌入投影到一个隐空间，其中同说话人 vs 不同说话人有闭式似然比。在余弦基础上叠加可降低 10–20% 的 EER。2020 年之前的标准；现在仅在闭集设置中使用。
- **Score normalization.** `S-norm` or `AS-norm`: normalize each score against a cohort of imposter means and stds. Essential for cross-domain eval.
  得分归一化。`S-norm` 或 `AS-norm`：用一个 impostor 均值和标准差群体对每个得分做归一化。跨域评估必不可少。

### Numbers you should know (2026)（2026 年应知的数字）

| Model | VoxCeleb1-O EER | Params | Throughput (A100) |
|-------|-----------------|--------|-------------------|
| x-vector (classic) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### Diarization（说话人分离）

"Who spoke when" in a multi-speaker clip. Pipeline: VAD → segment → embed each segment → cluster (agglomerative or spectral) → smooth boundaries. Modern stack: `pyannote.audio` 3.1, which bundles speaker segmentation + embedding + clustering behind one call. 2026 SOTA DER on AMI is ~15% (down from 23% in 2022).
多说话人片段中"谁在什么时候说话"。流水线：VAD → 分段 → 对每个片段做嵌入 → 聚类（凝聚或谱聚类） → 平滑边界。现代栈：`pyannote.audio` 3.1，它将说话人分割 + 嵌入 + 聚类打包在一个调用中。2026 年在 AMI 上的 SOTA DER 约为 15%（从 2022 年的 23% 下降）。

```figure
sp-eer-crossover
```

## Build It（动手实现）

### Step 1: toy embedding from MFCC statistics（基于 MFCC 统计的玩具嵌入）

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

Not SOTA by a mile — for teaching only. `code/main.py` uses this as a proof-of-concept on synthetic speaker data.
离 SOTA 差得远——仅供教学。`code/main.py` 在合成说话人数据上用这个做概念验证。

### Step 2: cosine similarity + threshold（余弦相似度 + 阈值）

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### Step 3: EER from similarity pairs（从相似度对计算 EER）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

Returns (eer, threshold_at_eer). Report both.
返回 (eer, threshold_at_eer)。两者都报告。

### Step 4: production with SpeechBrain（用 SpeechBrain 做生产级）

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### Step 5: diarize with pyannote（用 pyannote 做说话人分离）

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## Use It（实际应用）

The 2026 stack:
2026 年栈：

| Situation | Pick |
|-----------|------|
| Closed-set 1:1 verification, edge | ECAPA-TDNN + cosine threshold |
|                                   | 闭集 1:1 验证，边缘 |
| Open-set verification, cloud | WavLM-SV + AS-norm |
|                               | 开集验证，云端 |
| Diarization (meetings, podcasts) | `pyannote/speaker-diarization-3.1` |
|                                   | 说话人分离（会议、播客） |
| Anti-spoofing (replay / deepfake detection) | AASIST or RawNet2 |
|                                                 | 反欺骗（重放 / deepfake 检测） |
| Tiny embedded (KWS + enrollment) | Titanet-Small (NeMo) |
|                                     | 微型嵌入（关键词检测 + 注册） |

## Pitfalls（陷阱）

- **Channel mismatch（信道不匹配).** Model trained on VoxCeleb (web video) ≠ phone-call audio. Always evaluate on target channel.
  在 VoxCeleb（网络视频）上训练的模型 ≠ 电话音频。始终在目标信道上评估。
- **Short utterances（短话语).** EER degrades sharply below 3 seconds of test audio.
  测试音频低于 3 秒时 EER 急剧恶化。
- **Enrollment with noise（带噪声的注册).** One noisy enrollment poisons the anchor. Use ≥3 clean samples and average.
  一次带噪声的注册会污染锚点。使用 ≥3 条干净样本并平均。
- **Fixed threshold across conditions（跨条件固定阈值).** Always tune the threshold on a held-out dev set from the target domain.
  始终在目标域的保留开发集上调优阈值。
- **Cosine on non-normalized embeddings（对非归一化嵌入做余弦).** L2-normalize first; otherwise the magnitude dominates.
  先做 L2 归一化；否则幅度占主导。

## Ship It（交付成果）

Save as `outputs/skill-speaker-verifier.md`. Pick model, enrollment protocol, threshold-tuning plan, and fraud safeguards.
保存为 `outputs/skill-speaker-verifier.md`。为给定场景选择模型、注册协议、阈值调优计划和防欺诈保护措施。

## Exercises（练习）

1. **Easy（简单）.** Run `code/main.py`. Builds synthetic "speakers" (different tone profiles), enrolls, computes EER on a 100-pair trial list.
   运行 `code/main.py`。构建合成"说话人"（不同的音色轮廓），注册，在 100 对试验列表上计算 EER。
2. **Medium（中等）.** Use SpeechBrain ECAPA on 30 VoxCeleb1 utterances (5 speakers × 6 each). Compute EER with cosine vs PLDA.
   在 30 条 VoxCeleb1 话语上使用 SpeechBrain ECAPA（5 个说话人 × 每人 6 条）。用余弦 vs PLDA 计算 EER。
3. **Hard（困难）.** Build the full enroll → diarize → verify pipeline with `pyannote.audio`. Evaluate DER on AMI dev set.
   用 `pyannote.audio` 构建完整的 注册 → 分离 → 验证 流水线。在 AMI 开发集上评估 DER。

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| EER | The headline metric | Threshold where False Accept = False Reject. |
|     | 头条指标 | 错误接受等于错误拒绝的阈值。 |
| Verification | 1:1 | "Is this Alice?" |
|              | 1:1 | "这是 Alice 吗？" |
| Identification | 1:N | "Who is speaking?" |
|                | 1:N | "谁在说话？" |
| Open-set | Unknown possible | Test set can contain unenrolled speakers. |
|          | 允许未知 | 测试集可以包含未注册的说话人。 |
| Enrollment | Registering | Computing a speaker's reference embedding. |
|            | 注册 | 计算一个说话人的参考嵌入。 |
| AAM-softmax | The loss | Softmax with additive angular margin; forces cluster separation. |
|             | 损失 | 带加性角边距的 softmax；强制簇间分离。 |
| PLDA | Classic scoring | Probabilistic LDA; likelihood-ratio scoring on top of embeddings. |
|     | 经典评分 | 概率 LDA；在嵌入之上做似然比评分。 |
| DER | Diarization metric | Diarization Error Rate — miss + false alarm + confusion. |
|     | 说话人分离指标 | 说话人分离错误率——漏检 + 误报 + 混淆。 |

## Further Reading（延伸阅读）

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — the classic deep-embedding paper.
  Snyder et al. (2018).《X-Vectors：说话人识别的鲁棒 DNN 嵌入》——经典深度嵌入论文。
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — dominant architecture 2020–2026.
  Desplanques et al. (2020).《ECAPA-TDNN》——2020–2026 年的主导架构。
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — SSL backbone for SV and diarization.
  Chen et al. (2022).《WavLM：全栈语音处理的大规模自监督预训练》——说话人验证和分离的 SSL 主干。
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — production diarization + embedding stack.
  Bredin et al. (2023).《pyannote.audio 3.1》——生产级说话人分离 + 嵌入栈。
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — current EER standings across models.
  VoxCeleb 排行榜（2026 年更新）——各模型当前的 EER 排名。
