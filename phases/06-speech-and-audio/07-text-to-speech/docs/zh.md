# Text-to-Speech (TTS) — From Tacotron to F5 and Kokoro（文本转语音 (TTS)——从 Tacotron 到 F5 和 Kokoro）

> ASR inverts speech to text; TTS inverts text to speech. The 2026 stack is three parts: text → tokens, tokens → mel, mel → waveform. Each part has a default model that fits in a laptop.
> ASR 把语音转成文本；TTS 把文本转成语音。2026 年的栈分三部分：文本 → token，token → mel，mel → 波形。每一部分都有一个可以装在笔记本上的默认模型。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 09 (Seq2Seq), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## The Problem（问题）

You have a string: "Please remind me to water the plants at 6 pm." You need a 3-second audio clip that sounds natural, has correct prosody (pauses, stress), pronounces "plants" with the right vowel, and runs in under 300 ms on a CPU for a live voice assistant. You also need to swap voices, handle code-switched input ("remind me at 6 pm, daijoubu?"), and not embarrass yourself on names.
你有一段文本："Please remind me to water the plants at 6 pm." 你需要一段 3 秒的音频片段，听起来自然，有正确的韵律（停顿、重音），"plants" 的元音发音正确，并且在 CPU 上运行时间低于 300 毫秒，用于实时语音助手。你还需要换声音、处理代码混合输入（"remind me at 6 pm, daijoubu?"），并且在名字上不出丑。

Modern TTS pipelines look like this:
现代 TTS 流水线如下：

1. **Text frontend（文本前端）.** Normalize text (dates, numbers, emails), convert to phonemes or subword tokens, predict prosody features.
   归一化文本（日期、数字、邮箱），转换为音素或子词 token，预测韵律特征。
2. **Acoustic model（声学模型）.** Text → mel spectrogram. Tacotron 2 (2017), FastSpeech 2 (2020), VITS (2021), F5-TTS (2024), Kokoro (2024).
   文本 → 梅尔语谱图。Tacotron 2 (2017)、FastSpeech 2 (2020)、VITS (2021)、F5-TTS (2024)、Kokoro (2024)。
3. **Vocoder（声码器）.** Mel → waveform. WaveNet (2016), WaveRNN, HiFi-GAN (2020), BigVGAN (2022), neural codec vocoders in 2024+.
   梅尔 → 波形。WaveNet (2016)、WaveRNN、HiFi-GAN (2020)、BigVGAN (2022)、2024 年及以后的神经编码声码器。

In 2026 the acoustic + vocoder split blurs with end-to-end diffusion and flow-matching models. But the mental model of three parts still holds for debugging.
2026 年，端到端扩散和流匹配模型让声学 + 声码器的划分变得模糊。但三部分的心智模型在调试时仍然成立。

## The Concept（概念）

![Tacotron, FastSpeech, VITS, F5/Kokoro side-by-side](../assets/tts.svg)

**Tacotron 2 (2017)（Tacotron 2 (2017)）. ** Seq2seq: char-embedding → BiLSTM encoder → location-sensitive attention → autoregressive LSTM decoder emits mel frames. Slow (AR), wobbly on long text. Still cited as a baseline.
Seq2seq：字符嵌入 → BiLSTM 编码器 → 位置敏感注意力 → 自回归 LSTM 解码器输出梅尔帧。慢（自回归），长文本上不稳定。仍被引用为基线。

**FastSpeech 2 (2020)（FastSpeech 2 (2020)）. ** Non-autoregressive. Duration predictor outputs how many mel frames each phoneme gets. 1-pass, 10× faster than Tacotron. Loses some naturalness (monotonic alignment) but ships everywhere.
非自回归。时长预测器输出每个音素获得多少梅尔帧。1 遍，比 Tacotron 快 10 倍。失去一些自然度（单调对齐）但到处都在用。

**VITS (2021)（VITS (2021)）. ** Jointly trains encoder + flow-based duration + HiFi-GAN vocoder end-to-end with variational inference. High quality, single model. Dominant open-source TTS 2022–2024. Variants: YourTTS (multi-speaker zero-shot), XTTS v2 (2024, Coqui).
联合训练编码器 + 基于流的时长 + HiFi-GAN 声码器，端到端使用变分推断。高质量，单模型。2022–2024 年主导开源 TTS。变体：YourTTS（多说话人零样本）、XTTS v2 (2024, Coqui)。

**F5-TTS (2024)（F5-TTS (2024)）. ** Diffusion transformer over flow matching. Natural prosody, zero-shot voice cloning with 5 seconds of reference audio. Top of the 2026 open-source TTS leaderboards. 335M params.
流匹配上的扩散 transformer。自然韵律，用 5 秒参考音频做零样本声音克隆。2026 年开源 TTS 排行榜榜首。3.35 亿参数。

**Kokoro (2024)（Kokoro (2024)）. ** Small (82M), CPU-runnable, best-in-class English TTS for real-time use. Closed-vocabulary English-only, apache-2.0.
小（8200 万），CPU 可运行，实时英语 TTS 同类最佳。闭词表仅英语，apache-2.0 协议。

**OpenAI TTS-1-HD, ElevenLabs v2.5, Google Chirp-3（OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3）. ** Commercial state of the art. ElevenLabs v2.5 emotion tags ("[whispered]", "[laughing]") and character voices dominate audiobook production in 2026.
商业 SOTA。ElevenLabs v2.5 的情感标签（"[whispered]"、"[laughing]"）和角色声音在 2026 年主导有声书制作。

### Vocoder evolution（声码器演进）

| Era | Vocoder | Latency | Quality |
|-----|---------|---------|---------|
| 2016 | WaveNet | offline only | SOTA at release |
| 2018 | WaveRNN | ~realtime | good |
| 2020 | HiFi-GAN | 100× realtime | near-human |
| 2022 | BigVGAN | 50× realtime | generalizes across speakers/langs |
| 2024 | SNAC, DAC (neural codecs) | integrated with AR models | discrete tokens, bit-efficient |

By 2026 most "TTS" models are end-to-end from text to waveform; the mel spectrogram is an internal representation.
到 2026 年，大多数"TTS" 模型都是端到端从文本到波形；梅尔语谱图是一个内部表示。

### Evaluation（评估）

- **MOS (Mean Opinion Score)（MOS（平均意见分）). ** 1–5 scale, crowd-sourced. Still the gold standard; painfully slow.
  1–5 分，众包。仍是黄金标准； painfully slow。
- **CMOS (Comparative MOS)（CMOS（比较平均意见分）). ** A-vs-B preference. Tighter confidence intervals per annotation.
  A vs B 偏好。每次标注的置信区间更紧。
- **UTMOS, DNSMOS（UTMOS、DNSMOS）. ** Reference-free neural MOS predictors. Used for leaderboards.
  无参考神经 MOS 预测器。用于排行榜。
- **CER (Character Error Rate) via ASR（通过 ASR 计算 CER（字符错误率）). ** Run TTS output through Whisper, compute CER against the input text. Proxy for intelligibility.
  将 TTS 输出通过 Whisper，与输入文本计算 CER。可懂度的代理指标。
- **SECS (Speaker Embedding Cosine Similarity)（SECS（说话人嵌入余弦相似度）). ** Voice-cloning quality.
  声音克隆质量。

2026 numbers on LibriTTS test-clean:
2026 年在 LibriTTS test-clean 上的数字：

| Model | UTMOS | CER (via Whisper) | Size |
|-------|-------|-------------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

```figure
sp-tts-stack
```

## Build It（动手实现）

### Step 1: phonemize input（音素化输入）

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

Phonemes are the universal bridge. Avoid feeding raw text to anything below VITS-level quality.
音素是通用桥梁。避免把原始文本喂给任何低于 VITS 质量的模型。

### Step 2: run Kokoro (2026 CPU default)（运行 Kokoro（2026 年 CPU 默认））

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

Runs offline, single file, 82M params.
离线运行，单文件，8200 万参数。

### Step 3: run F5-TTS with voice cloning（用 F5-TTS 做声音克隆）

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

Pass a 5-second reference clip + its transcript; F5 clones prosody and timbre.
传入一段 5 秒的参考片段 + 它的转录文本；F5 克隆韵律和音色。

### Step 4: HiFi-GAN vocoder from scratch（从零实现 HiFi-GAN 声码器）

Too big to fit in a tutorial script, but the shape is:
太大不适合放在教程脚本里，但结构是：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

Training: adversarial (discriminator on short windows) + mel-spectrogram reconstruction loss + feature-matching loss. Commoditized — use pretrained checkpoints from `hifi-gan` repo or nvidia-NeMo.
训练：对抗（短窗口上的判别器）+ 梅尔语谱图重建损失 + 特征匹配损失。已商品化——使用 `hifi-gan` 仓库或 nvidia-NeMo 的预训练 checkpoint。

### Step 5: the full pipeline (pseudocode)（完整流水线（伪代码））

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## Use It（实际应用）

The 2026 stack:
2026 年栈：

| Situation | Pick |
|-----------|------|
| Real-time English voice assistant | Kokoro (CPU) or XTTS v2 (GPU) |
|                                    | 实时英语语音助手 |
| Voice cloning from 5 s reference | F5-TTS |
|                                   | 从 5 秒参考做声音克隆 |
| Commercial character voices | ElevenLabs v2.5 |
|                            | 商业角色声音 |
| Audiobook narration | ElevenLabs v2.5 or XTTS v2 + fine-tune |
|                    | 有声书旁白 |
| Low-resource language | Train VITS on 5–20 h target-lang data |
|                       | 低资源语言 |
| Expressive / emotion tags | ElevenLabs v2.5 or StyleTTS 2 fine-tune |
|                           | 表现力 / 情感标签 |

Open-source leader as of 2026: **F5-TTS for quality, Kokoro for efficiency**. Don't reach for Tacotron unless you are a historian.
截至 2026 年的开源领先者：**F5-TTS 追求质量，Kokoro 追求效率**。除非你是历史学家，否则不要碰 Tacotron。

## Pitfalls（陷阱）

- **No text normalizer（没有文本归一器）. ** "Dr. Smith" reads as "Doctor" or "Drive"? "2026" as "twenty twenty six" or "two zero two six"? Normalize BEFORE phonemizer.
  "Dr. Smith" 读成 "Doctor" 还是 "Drive"？"2026" 读成 "twenty twenty six" 还是 "two zero two six"？在 phonemizer 之前先归一化。
- **OOV proper nouns（OOV 专有名词）. ** "Ghumare" → "ghyu-mair"? Ship a fallback grapheme-to-phoneme model for unknown tokens.
  "Ghumare" → "ghyu-mair"？为未知 token 配备一个回退的字素到音素模型。
- **Clipping（削波）. ** Vocoder output rarely clips, but mel scaling mismatch at inference can overshoot ±1.0. Always `np.clip(wav, -1, 1)`.
  声码器输出很少削波，但推理时的梅尔缩放不匹配可能超出 ±1.0。始终执行 `np.clip(wav, -1, 1)`。
- **Sample-rate mismatch（采样率不匹配）. ** Kokoro outputs 24 kHz; your downstream pipeline expects 16 kHz → resample or get aliasing.
  Kokoro 输出 24 kHz；你的下游流水线期望 16 kHz → 重采样或得到混叠。

## Ship It（交付成果）

Save as `outputs/skill-tts-designer.md`. Design a TTS pipeline for a given voice, latency, and language target.
保存为 `outputs/skill-tts-designer.md`。为给定的声音、延迟和语言目标设计 TTS 流水线。

## Exercises（练习）

1. **Easy（简单）.** Run `code/main.py`. Builds a phoneme dictionary from a toy vocab, estimates duration per phoneme, and prints a fake "mel" schedule.
   运行 `code/main.py`。从小词汇表构建音素字典，估计每个音素的时长，并打印一个假的"mel" 计划。
2. **Medium（中等）.** Install Kokoro, synthesize the same sentence at voice `af_bella` and `am_adam`. Compare audio durations and subjective quality.
   安装 Kokoro，用声音 `af_bella` 和 `am_adam` 合成同一句话。比较音频时长和主观质量。
3. **Hard（困难）.** Record a 5-second reference clip of yourself. Use F5-TTS to clone it. Report SECS between reference and cloned output.
   录制一段你自己的 5 秒参考片段。用 F5-TTS 克隆它。报告参考和克隆输出之间的 SECS。

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Phoneme | Sound unit | Abstract sound class; 39 in English (ARPABet). |
|        | 声音单元 | 抽象声音类别；英语中有 39 个（ARPABet）。 |
| Duration predictor | How long each phoneme lasts | Non-AR model output; integer frames per phoneme. |
|                    | 每个音素持续多久 | 非自回归模型输出；每个音素的整数帧数。 |
| Vocoder | Mel → waveform | Neural net mapping mel-spec to raw samples. |
|         | 梅尔 → 波形 | 神经网络将梅尔谱映射到原始采样。 |
| HiFi-GAN | Standard vocoder | GAN-based; dominant 2020–2024. |
|          | 标准声码器 | 基于 GAN；2020–2024 年主导。 |
| MOS | Subjective quality | 1–5 mean opinion score from human raters. |
|    | 主观质量 | 人类评分者的 1–5 平均意见分。 |
| SECS | Voice-clone metric | Cosine similarity between target and output speaker embedding. |
|      | 声音克隆指标 | 目标和输出说话人嵌入之间的余弦相似度。 |
| F5-TTS | 2024 open-source SOTA | Flow-matching diffusion; zero-shot cloning. |
|        | 2024 年开源 SOTA | 流匹配扩散；零样本克隆。 |
| Kokoro | CPU English leader | 82M-param model, Apache 2.0. |
|        | CPU 英语领先者 | 8200 万参数模型，Apache 2.0 协议。 |

## Further Reading（延伸阅读）

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — the seq2seq baseline.
  Shen et al. (2017).《Tacotron 2》——seq2seq 基线。
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — end-to-end flow-based.
  Kim, Kong, Son (2021).《VITS》——端到端基于流。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — current open-source SOTA.
  Chen et al. (2024).《F5-TTS》——当前开源 SOTA。
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — the vocoder that still ships in 2026.
  Kong, Kim, Bae (2020).《HiFi-GAN》——2026 年仍在使用的声码器。
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — 2024 CPU-friendly English TTS.
  HuggingFace 上的 Kokoro-82M——2024 年 CPU 友好的英语 TTS。
