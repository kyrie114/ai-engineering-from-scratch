# Neural Audio Codecs — EnCodec, SNAC, Mimi, DAC and the Semantic-Acoustic Split（神经音频编码器——EnCodec、SNAC、Mimi、DAC 和语义-声学拆分）

> 2026 audio generation is almost all tokens. EnCodec, SNAC, Mimi, and DAC turn continuous waveforms into discrete sequences that a transformer can predict. The semantic-vs-acoustic token split — first-codebook as semantic, rest as acoustic — is the most important architectural shift since the Transformer for audio.
> 2026 年音频生成几乎全是 token。EnCodec、SNAC、Mimi 和 DAC 把连续波形变成 transformer 可以预测的离散序列。语义 vs 声学 token 拆分——第一个码本作为语义，其余作为声学——是自 Transformer 以来音频领域最重要的架构转变。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 10 · 11 (Quantization), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 minutes

## The Problem（问题）

Language models work on discrete tokens. Audio is continuous. If you want an LLM-style model for speech / music — MusicGen, Moshi, Sesame CSM, VibeVoice, Orpheus — you first need a **neural audio codec**: a learned encoder that discretizes audio into a small vocabulary of tokens, and a matching decoder that reconstructs the waveform.
语言模型使用离散 token。音频是连续的。如果你想要一个用于语音 / 音乐的 LLM 风格模型——MusicGen、Moshi、Sesame CSM、VibeVoice、Orpheus——你首先需要一个**神经音频编码器**：一个将音频离散化为小型 token 词表的学习编码器，以及一个重建波形的匹配解码器。

Two families have emerged:
已出现两个家族：

1. **Reconstruction-first codecs（重建优先编码器）. ** — EnCodec, DAC. Optimize perceptual audio quality. Tokens are "acoustic" — they capture everything including speaker identity, timbre, background noise.
   ——EnCodec、DAC。优化感知音频质量。Token 是"声学的"——它们捕获一切，包括说话人身份、音色、背景噪声。
2. **Semantic-first codecs（语义优先编码器）. ** — Mimi (Kyutai), SpeechTokenizer. Force the first codebook to encode linguistic / phonetic content (often by distilling from WavLM). Subsequent codebooks are acoustic detail.
   ——Mimi (Kyutai)、SpeechTokenizer。强制第一个码本编码语言 / 语音内容（通常通过从 WavLM 蒸馏）。后续码本是声学细节。

The 2024-2026 insight: **a pure reconstruction codec gives you blurry speech when you try to generate from text.** The LLM over codec tokens has to learn both language structure AND acoustic structure in the same codebook, which doesn't scale. Separating them — semantic codebook 0, acoustic codebooks 1-N — is what makes Moshi and Sesame CSM work.
2024-2026 年的洞见：**纯重建编码器在你尝试从文本生成时会给你模糊的语音。** 在编码器 token 上的 LLM 必须在同一个码本中学习语言结构 AND 声学结构，这无法扩展。把它们分开——语义码本 0，声学码本 1-N——这就是 Moshi 和 Sesame CSM 能工作的原因。

## The Concept（概念）

![Four codec landscape: EnCodec, DAC, SNAC (multi-scale), Mimi (semantic+acoustic)](../assets/codec-comparison.svg)

### The core trick: Residual Vector Quantization (RVQ)（核心技巧：残差向量量化 (RVQ)）

Rather than one big codebook (which would need millions of codes for good quality), all modern audio codecs use **RVQ**: a cascade of small codebooks. The first codebook quantizes the encoder output; the second quantizes the residual; etc. Each codebook is 1024 codes. 8 codebooks = effective vocabulary of 1024^8 = 10^24.
与其使用一个大码本（这需要数百万个代码才能获得良好质量），所有现代音频编码器都使用 **RVQ**：一小串码本。第一个码本量化编码器输出；第二个量化残差；等等。每个码本有 1024 个代码。8 个码本 = 1024^8 = 10^24 的有效词表。

At inference time, the decoder sums all chosen codes per frame to reconstruct.
推理时，解码器对每帧求和所有选中的代码以重建。

### The four codecs that matter in 2026（2026 年重要的四个编码器）

**EnCodec (Meta, 2022)（EnCodec (Meta, 2022)）. ** The baseline. Encoder-decoder over waveform, RVQ bottleneck. 24 kHz, 32 codebooks possible, default 4 codebooks @ 1.5 kbps. Uses `1D conv + transformer + 1D conv` architecture. Used by MusicGen.
基线。波形上的编码器-解码器，RVQ 瓶颈。24 kHz，最多 32 个码本，默认 4 个码本 @ 1.5 kbps。使用 `1D conv + transformer + 1D conv` 架构。MusicGen 使用。

**DAC (Descript, 2023)（DAC (Descript, 2023)）. ** RVQ with L2-normalized codebooks, periodic activation functions, improved losses. Highest reconstruction fidelity of any open codec — sometimes indistinguishable from original speech with 12 codebooks. 44.1 kHz full-band.
带 L2 归一化码本的 RVQ，周期性激活函数，改进的损失。任何开源编码器中最高的重建保真度——用 12 个码本时有时与原始语音无法区分。44.1 kHz 全频带。

**SNAC (Hubert Siuzdak, 2024)（SNAC (Hubert Siuzdak, 2024)）. ** Multi-scale RVQ — the coarse codebooks operate at a lower frame rate than fine ones. Effectively models audio hierarchically: a coarse "sketch" at ~12 Hz plus detail at 50 Hz. Used by Orpheus-3B because the hierarchical structure maps well onto LM-based generation.
多尺度 RVQ——粗码本以比精细码本更低的帧率运行。有效地分层建模音频：约 12 Hz 的粗"草图"加 50 Hz 的细节。Orpheus-3B 使用，因为分层结构很好地映射到基于 LM 的生成。

**Mimi (Kyutai, 2024)（Mimi (Kyutai, 2024)）. ** The 2026 game-changer. 12.5 Hz frame rate (extremely low), 8 codebooks @ 4.4 kbps. Codebook 0 is **distilled from WavLM** — trained to predict WavLM's speech-content features. Codebooks 1-7 are acoustic residuals. This split powers Moshi (Lesson 15) and Sesame CSM.
2026 年游戏改变者。12.5 Hz 帧率（极低），8 个码本 @ 4.4 kbps。码本 0 是**从 WavLM 蒸馏**——训练为预测 WavLM 的语音内容特征。码本 1-7 是声学残差。这个拆分为 Moshi（第 15 课）和 Sesame CSM 提供动力。

### Frame rates matter for language modeling（帧率对语言建模很重要）

Lower frame rate = shorter sequence = faster LM.
更低帧率 = 更短序列 = 更快 LM。

| Codec | Frame rate | 1 s = N frames | Good for |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | music, general audio |
| DAC-44.1k | 86 Hz | 86 | high-fidelity music |
| SNAC-24k (coarse) | ~12 Hz | 12 | AR-LM efficient |
| Mimi | 12.5 Hz | 12.5 | streaming speech |

At 12.5 Hz, a 10-second utterance is only 125 codec frames — a transformer can easily predict them.
在 12.5 Hz，一段 10 秒的话语只有 125 个编码帧——一个 transformer 可以轻松预测它们。

### Semantic vs acoustic tokens（语义 vs 声学 token）

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **Semantic token (codebook 0 in Mimi)（语义 token（Mimi 中的码本 0）). ** Encodes what was said — phonemes, words, content. Distilled from WavLM via an auxiliary prediction loss.
  编码说了什么——音素、词、内容。通过辅助预测损失从 WavLM 蒸馏。
- **Acoustic tokens (codebooks 1-7)（声学 token（码本 1-7）). ** Encode timbre, speaker identity, prosody, background noise, fine detail.
  编码音色、说话人身份、韵律、背景噪声、精细细节。

An AR LM predicts the semantic token first (conditioned on text), then predicts acoustic tokens (conditioned on semantic + speaker reference). This factorization is why modern TTS can zero-shot-clone voices: the semantic model handles content; the acoustic model handles timbre.
一个 AR LM 首先预测语义 token（以文本为条件），然后预测声学 token（以语义 + 说话人参考为条件）。这种因子分解是现代 TTS 能零样本克隆声音的原因：语义模型处理内容；声学模型处理音色。

### 2026 reconstruction quality (bits per sec, lower bitrate is better)（2026 年重建质量（比特每秒，更低比特率更好））

| Codec | Bitrate | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

Traditional codecs like Opus still win per bit on perceptual quality. Neural codecs win on **discrete tokens** (which Opus does not produce) and **generative-model quality** (what the LM can do with those tokens).
像 Opus 这样的传统编码器在每比特感知质量上仍然获胜。神经编码器在**离散 token**（Opus 不产生）和**生成模型质量**（LM 用这些 token 能做什么）上获胜。

```figure
rvq-codec-cascade
```

## Build It（动手实现）

### Step 1: encode with EnCodec（用 EnCodec 编码）

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

`n_codebooks=8` at 6 kbps. Each code is 0-1023 (10-bit).
6 kbps 时 `n_codebooks=8`。每个代码是 0-1023（10 位）。

### Step 2: decode and measure reconstruction（解码并测量重建）

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### Step 3: the semantic-acoustic split (Mimi-style)（语义-声学拆分（Mimi 风格））

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

Semantic codebook 0 is WavLM-aligned. You can train a text-to-semantic transformer — much smaller vocabulary than going direct-to-audio. Then a separate acoustic-to-waveform decoder conditions on a speaker reference.
语义码本 0 是 WavLM 对齐的。你可以训练一个文本到语义的 transformer——比直接到音频的词表小得多。然后一个独立的声学到波形解码器以说话人参考为条件。

### Step 4: why AR LM over codec tokens works（为什么在编码器 token 上的 AR LM 有效）

For a 10 s speech clip at Mimi's 12.5 Hz × 8 codebooks:
对于 Mimi 的 12.5 Hz × 8 码本下 10 秒语音片段：

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 tokens is a trivial context for a transformer. A 256M-parameter transformer can generate 10 seconds of speech in milliseconds on a modern GPU.
1000 个 token 对 transformer 来说是小菜一碟。一个 2.56 亿参数的 transformer 在现代 GPU 上可以在毫秒级生成 10 秒语音。

## Use It（实际应用）

Map problem → codec:
问题 → 编码器：

| Task | Codec |
|------|-------|
| General music generation | EnCodec-24k |
|                           | 通用音乐生成 |
| Highest-fidelity reconstruction | DAC-44.1k |
|                                | 最高保真重建 |
| AR LM over speech (TTS) | SNAC or Mimi |
|                         | 语音上的 AR LM（TTS） |
| Streaming full-duplex speech | Mimi (12.5 Hz) |
|                               | 流式全双工语音 |
| Sound-effect library with text | EnCodec + T5 condition |
|                                | 带文本的音效库 |
| Fine-grained audio editing | DAC + inpainting |
|                            | 细粒度音频编辑 |

Rule of thumb: **if you're building a generative model, start with Mimi or SNAC. if you're building a compression pipeline, use Opus.**
经验法则：**如果你正在构建生成模型，从 Mimi 或 SNAC 开始。如果你正在构建压缩流水线，使用 Opus。**

## Pitfalls（陷阱）

- **Too many codebooks（太多码本）. ** Adding codebooks increases fidelity linearly but LM sequence length linearly too. Stop at 8-12.
  添加码本线性增加保真度，但 LM 序列长度也线性增加。停在 8-12。
- **Frame-rate mismatch（帧率不匹配）. ** Training LM on 12.5 Hz Mimi then fine-tuning on 50 Hz EnCodec fails silently.
  在 12.5 Hz Mimi 上训练 LM，然后在 50 Hz EnCodec 上微调，会静默失败。
- **Assuming all codebooks equal（假设所有码本相等）. ** In Mimi, codebook 0 carries content; losing it destroys intelligibility. Losing codebook 7 is barely noticeable.
  在 Mimi 中，码本 0 承载内容；丢失它会破坏可懂度。丢失码本 7 几乎不可察觉。
- **Using reconstruction quality as the only metric（仅用重建质量作为唯一指标）. ** A codec can have great reconstruction but be useless for LM-based generation if the semantic structure is bad.
  一个编码器可能有很好的重建，但如果语义结构不好，对于基于 LM 的生成毫无用处。

## Ship It（交付成果）

Save as `outputs/skill-codec-picker.md`. Pick a codec for a given generative or compression task.
保存为 `outputs/skill-codec-picker.md`。为给定的生成或压缩任务选择编码器。

## Exercises（练习）

1. **Easy（简单）.** Run `code/main.py`. It implements a toy scalar + residual quantizer and measures reconstruction error as you add codebooks.
   运行 `code/main.py`。它实现了一个玩具标量 + 残差量化器，并在你添加码本时测量重建误差。
2. **Medium（中等）.** Install `encodec` and compare 1, 4, 8, 32 codebooks on a held-out speech clip. Plot PESQ or MSE vs bitrate.
   安装 `encodec` 并在保留的语音片段上比较 1、4、8、32 个码本。绘制 PESQ 或 MSE vs 比特率。
3. **Hard（困难）.** Load Mimi. Encode a clip. Replace codebook 0 with random integers; decode. Then replace codebook 7 similarly. Compare the two corruptions — codebook 0 corruption should destroy intelligibility; codebook 7 corruption should barely change anything.
   加载 Mimi。编码一个片段。用随机整数替换码本 0；解码。然后类似地替换码本 7。比较两种损坏——码本 0 损坏应该破坏可懂度；码本 7 损坏应该几乎什么都不改变。

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| RVQ | Residual quantization | Cascade of small codebooks; each quantizes the previous residual. |
|     | 残差量化 | 小码本级联；每个量化前一个残差。 |
| Frame rate | Codec speed | How many token-frames per second. Lower = faster LM. |
|            | 编码器速度 | 每秒多少 token 帧。越低 = LM 越快。 |
| Semantic codebook | Codebook 0 (Mimi) | Codebook distilled from SSL features; encodes content. |
|                    | 码本 0 (Mimi) | 从 SSL 特征蒸馏的码本；编码内容。 |
| Acoustic codebooks | Everything else | Timbre, prosody, noise, fine detail. |
|                     | 其他所有 | 音色、韵律、噪声、精细细节。 |
| PESQ / ViSQOL | Perceptual quality | Objective metrics correlating with MOS. |
|               | 感知质量 | 与 MOS 相关的客观指标。 |
| EnCodec | Meta codec | The RVQ baseline; used by MusicGen. |
|         | Meta 编码器 | RVQ 基线；MusicGen 使用。 |
| Mimi | Kyutai codec | 12.5 Hz frame rate; semantic-acoustic split; powers Moshi. |
|      | Kyutai 编码器 | 12.5 Hz 帧率；语义-声学拆分；为 Moshi 提供动力。 |

## Further Reading（延伸阅读）

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — the RVQ baseline.
  Défossez 等 (2023).《EnCodec》——RVQ 基线。
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — highest-fidelity open.
  Kumar 等 (2023).《Descript Audio Codec (DAC)》——最高保真开源。
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — multi-scale RVQ.
  Siuzdak (2024).《SNAC》——多尺度 RVQ。
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — semantic-acoustic split, WavLM distillation.
  Kyutai (2024).《Mimi 编码器》——语义-声学拆分，WavLM 蒸馏。
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) — the two-stage semantic/acoustic paradigm.
  Borsos 等 (2023).《AudioLM》——两阶段语义/声学范式。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — the original streamable RVQ codec.
  Zeghidour 等 (2021).《SoundStream》——原始可流式 RVQ 编码器。
