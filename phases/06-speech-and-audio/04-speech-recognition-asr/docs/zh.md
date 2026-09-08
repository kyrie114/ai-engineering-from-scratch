# 语音识别（ASR）—— CTC、RNN-T、注意力（Speech Recognition (ASR) — CTC, RNN-T, Attention）

> 语音识别是每个时间步的音频分类，由了解英语和静默的序列模型粘合在一起。CTC、RNN-T 和注意力是三种实现方式。选一种并理解为什么。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 08 (CNNs & RNNs for Text), Phase 5 · 10 (Attention)
**Time:** ~45 minutes

## 问题（The Problem）

你有一段 10 秒 16 kHz 的音频。你想要一个字符串："turn on the kitchen lights"。挑战在于结构层面：音频帧与字符并非一一对应。单词"okay"可能持续 200 ms 或 1200 ms。沉默打断语句。某些音素比其他音素长。输出 token 的数量无法事先知道。

三种方案解决了这个问题：

1. **CTC（连接主义时间分类）。** 输出每帧 token 概率，包括一个特殊的 blank。解码时折叠重复和 blank。非自回归，快速。wav2vec 2.0、MMS 使用。
2. **RNN-T（循环神经网络转导器）。** 联合网络根据编码器帧和先前的 token 预测下一个 token。可流式传输。Google 端侧 ASR、NVIDIA Parakeet 使用。
3. **注意力编码器-解码器。** 编码器将音频压缩为隐状态，解码器交叉注意力自回归生成 token。Whisper、SeamlessM4T 使用。

2026 年，LibriSpeech test-clean 上的 SOTA WER 是 1.4%（Parakeet-TDT-1.1B，NVIDIA）和 1.58%（Whisper-Large-v3-turbo）。差异微小；部署差异巨大。

## 概念（The Concept）

![Three ASR formulations: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**CTC 直觉。** 让编码器输出 `T` 帧级分布，覆盖 `V+1` 个 token（V 个字符 + blank）。对于长度为 `U < T` 的目标字符串 `y`，任何折叠为 `y` 的帧对齐都计入。CTC 损失对所有此类对齐求和。推理：每帧 argmax，折叠重复，移除 blank。

优点：非自回归，可流式传输，零前瞻。缺点：*条件独立性假设*——每帧预测相互独立，因此没有内部语言模型。可通过束搜索或浅融合加入外部 LM 修复。

**RNN-T 直觉。** 添加一个预测器网络嵌入 token 历史，以及一个连接器将预测器状态与编码器帧组合为 `V+1`（`+1` 是 null / 不发出）上的联合分布。显式建模了 CTC 忽略的条件依赖。可流式传输，因为每一步仅依赖过去的帧和过去的 token。

优点：可流式 + 内部 LM。缺点：训练更复杂且更耗内存（3D loss lattice）；RNN-T loss kernel 本身就是一个完整的库类别。

**注意力编码器-解码器。** 编码器（6–32 个 transformer 层）作用于 log-mel 帧。解码器（6–32 个 transformer 层）交叉注意力编码器输出以自回归生成 token。没有对齐约束——注意力可以看音频的任何位置。除非限制注意力（分块 Whisper-Streaming，2024），否则不可流式传输。

优点：离线 ASR 质量最高，用标准 seq2seq 工具易于训练。缺点：自回归延迟与输出长度成正比；不经工程无法流式传输。

### WER：唯一的数字

词错误率 = `(S + D + I) / N`，其中 S=替换，D=删除，I=插入，N=参考词数。对应词级别的 Levenshtein 编辑距离。越低越好。WER 高于 20% 通常无法使用；低于 5% 对朗读语音达到人类水平。2026 年标准基准数字：

| Model | LibriSpeech test-clean | LibriSpeech test-other | Size |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B params |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

这些都是基于编码器-解码器或 RNN-T。纯 CTC 系统（wav2vec 2.0）在 test-clean 上约为 1.8–2.1%。

```figure
ctc-collapse
```

## 构建它（Build It）

### 步骤 1：贪心 CTC 解码（greedy CTC decode）

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：折叠连续重复，移除 blank。示例：`a a _ _ a b b _ c` → `a a b c`。

### 步骤 2：束搜索 CTC（beam-search CTC）

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产环境使用带 LM 融合的前缀树束搜索；这是概念骨架。

### 步骤 3：WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 步骤 4：使用 Whisper 推理（inference against Whisper）

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026 年最强通用 ASR 的一行代码。在 24 GB GPU 上以约 20 倍实时运行。

### 步骤 5：使用 Parakeet 或 wav2vec 2.0 流式传输（streaming with Parakeet or wav2vec 2.0）

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

流式 ASR 需要分块编码器注意力和状态传递；使用支持它的库（Parakeet 用 NeMo，`transformers` pipeline 带 `chunk_length_s`）。

## 使用它（Use It）

2026 年的栈：

| Situation | Pick |
|-----------|------|
| English, offline, max quality | Whisper-large-v3-turbo |
| Multilingual, robust | SeamlessM4T v2 |
| Streaming, low latency | Parakeet-TDT-1.1B or Riva |
| Edge, mobile, <500 ms latency | Whisper-Tiny quantized or Moonshine (2024) |
| Long-form | Whisper with VAD-based chunking (WhisperX) |
| Domain-specific (medical, legal) | Fine-tune wav2vec 2.0 + domain LM fusion |

## 2026 年仍然存在的陷阱（Pitfalls that still ship in 2026）

- **没有 VAD。** 在静默上运行 Whisper 会产生幻觉（"Thanks for watching!"）。始终用 VAD 做门控。
- **字符 vs 词 vs subword WER。** 在归一化（小写、去除标点）*之后*报告词级 WER。
- **语言 ID 漂移。** Whisper 的自动 LID 会将嘈杂片段误路由到日语或威尔士语；知道时强制 `language="en"`。
- **长片段不进行分块。** Whisper 有 30 秒窗口。更长的音频使用 `chunk_length_s=30, stride=5`。

## 交付它（Ship It）

保存为 `outputs/skill-asr-picker.md`。为给定部署目标选择模型、解码策略、分块和 LM 融合。

## 练习（Exercises）

1. **简单（Easy）。** 运行 `code/main.py`。它贪心解码一个手工制作的 CTC 输出，并计算与参考的 WER。
2. **中等（Medium）。** 正确实现第 2 步中的前缀树束搜索（考虑 blank 合并规则）。在 10 示例合成数据集上与贪心解码比较。
3. **困难（Hard）。** 在 [LibriSpeech test-clean](https://www.openslr.org/12) 上使用 `whisper-large-v3-turbo`。计算前 100 个话语的 WER。与已发表数字比较。

## 关键术语（Key Terms）

| 术语（Term） | 人们常说的（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| CTC | 带 blank 的损失 | 所有帧到 token 对齐的边缘分布；非 AR。 |
| RNN-T | 流式损失 | CTC + 下一个 token 预测器；处理词序。 |
| 注意力编码器-解码器 | Whisper 风格 | 编码器 + 交叉注意力解码器；最佳离线质量。 |
| WER | 你报告的数字 | 词级别的 `(S+D+I)/N`。 |
| Blank | 空 | CTC 中的特殊 token，表示"本帧不输出"。 |
| LM 融合 | 外部语言模型 | 在束搜索期间加入加权的 LM 对数概率。 |
| VAD | 静音门控 | 语音活动检测器；修剪非语音。 |

## 延伸阅读（Further Reading）

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) —— CTC 论文。
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) —— RNN-T 论文。
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) —— 2022 年规范论文；v3-turbo 扩展于 2024 年。
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) —— 2026 年 Open ASR Leaderboard 领先者。
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) —— 25+ 模型的实时基准。
