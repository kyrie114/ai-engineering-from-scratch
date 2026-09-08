# Audio Transformers — Whisper Architecture（音频 Transformer — Whisper 架构）

> 音频是频率随时间的图像。Whisper 是一个吃 mel spectrogram 并说话回来的 ViT。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer)（完整 Transformer）, Phase 7 · 08 (Encoder-Decoder)（编码器-解码器）, Phase 7 · 09 (ViT)（视觉 Transformer）
**Time:** ~45 minutes（约 45 分钟）

## The Problem（问题）

Whisper（OpenAI，Radford et al. 2022）之前，最先进的自动语音识别（ASR）意味着 wav2vec 2.0 和 HuBERT —— 自监督特征提取器加一个微调头。高质量，昂贵的数据管线，领域脆弱。多语言语音识别需要每个语言家族单独的模型。

Whisper 做了三个赌注：

1. **在 everything 上训练。** 680,000 小时从互联网爬取的弱标签音频，覆盖 97 种语言。没有干净的学术语料库。没有音素标签。
2. **多任务单一模型。** 一个解码器联合训练转录、翻译、voice activity detection、language ID 和 timestamping，通过 task token。
3. **标准 encoder-decoder transformer。** 编码器消耗 log-mel spectrogram。解码器自回归地产生 text tokens。没有 vocoder，没有 CTC，没有 HMM。

结果：Whisper large-v3 在 accents、noise 和零干净标签数据的语言上都很 robust。它是 2026 年每个开源语音助手和大多数商业语音助手的默认 speech front-end。

## The Concept（概念）

![Whisper pipeline: audio → mel → encoder → decoder → text（Whisper 流水线：音频 → mel → 编码器 → 解码器 → 文本）](../assets/whisper.svg)

### Step 1 — resample + window（重采样 + 窗口）

16 kHz 的音频。Clip/pad 到 30 秒。计算 log-mel spectrogram：80 mel bins，10 ms stride → ~3,000 帧 × 80 特征。这是 Whisper 看到的“input image”。

### Step 2 — 卷积 stem

两个 kernel 3、stride 2 的 Conv1D 层把 3,000 帧减少到 1,500。在不增加大量参数的情况下减半序列长度。

### Step 3 — 编码器

一个 24 层（对于 large）的 transformer 编码器，超过 1,500 个 timestep。正弦位置编码，自注意力，GELU FFN。产生 1,500 × 1,280 隐藏状态。

### Step 4 — 解码器

一个 24 层 transformer 解码器。它自回归地从一个 BPE 词表产生 token，该词表是 GPT-2 的超集，带有一些 audio-specific 特殊 token。

### Step 5 — task token

解码器提示以告诉模型做什么的控制 token 开头：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型在这个约定上训练。你通过前缀控制任务。2026 年 instruction-tuning 的等价物，但应用于 speech。

### Step 6 — 输出

Beam search（宽度 5）带一个 log-prob 阈值。当 `<|notimestamps|>` token 缺失时，时间戳每 0.02 秒音频预测一次。

### Whisper 尺寸

| Model（模型） | Params（参数） | Layers（层） | d_model | Heads（头） | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB（4 层解码器） |

Large-v3-turbo（2024）把解码器从 32 层减少到 4 层。8 倍更快解码，<1 WER point 回归。那个解码速度解锁是为什么 Whisper-turbo 是 2026 年实时语音助手的默认。

### Whisper 不做什么

- 没有 diarization（谁在说话）。和 pyannote 配对用于那个。
- 没有原生实时流式 —— 30 秒窗口是固定的。现代包装器（`faster-whisper`、`WhisperX`）通过 VAD + overlap  bolt on streaming。
- 没有超出 30 秒的长形式上下文，没有外部分块。在实践中效果很好，因为人类语音很少需要长范围上下文进行转录。

### 2026 landscape

| Task（任务） | Model（模型） | Notes（注释） |
|------|-------|-------|
| English ASR | Whisper-turbo, Moonshine | Moonshine 在边缘快 4 倍 |
| Multilingual ASR | Whisper-large-v3 | 97 种语言 |
| Streaming ASR | faster-whisper + VAD | 可实现 150 ms 延迟目标 |
| TTS | Piper, XTTS-v2, Kokoro | Encoder-decoder 模式，但 Whisper-shaped |
| Audio + language | AudioLM, SeamlessM4T | 一个 transformer 中的 text token + audio token |

```figure
n5-mel-decode
```

## Build It（动手实现）

见 `code/main.py`。我们不训练 Whisper —— 我们构建 log-mel spectrogram 流水线 + task-token prompt formatter。那些是你在生产中实际接触的部分。

### Step 1: synthesize audio（合成音频）

生成一个 1 秒 440 Hz 正弦波，采样率 16 kHz。16,000 个样本。

### Step 2: log-mel spectrogram (simplified)（简化）

完整 mel spectrogram 需要 FFT。我们做一个简化的 framing + per-frame energy 版本，展示流水线而不需要 `librosa`：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

Frame = 25 ms，hop = 10 ms。匹配 Whisper 的窗口。Per-frame energy 代表 pedagogy 的 mel bins。

### Step 3: pad to 30 s（填充到 30 秒）

Whisper 总是处理 30 秒的块。把 spectrogram pad（或 clip）到 3,000 帧。

### Step 4: build the prompt tokens（构建提示 token）

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是整个任务控制表面。一个 4 token 前缀。

## Use It（实际应用）

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快，OpenAI 兼容：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026 年何时选择 Whisper：**

- 用一个模型做多语言 ASR。
- 嘈杂、多样音频的鲁棒转录。
- 研究 / prototype ASR —— 最快的起点。

**何时选择其他东西：**

- 边缘超低延迟流式 —— Moonshine 在匹配质量下击败 Whisper。
- 需要 <200 ms 的实时对话式 AI —— 专用流式 ASR。
- Speaker diarization —— Whisper 不做这个；bolt on pyannote。

## Ship It（交付）

见 `outputs/skill-asr-configurator.md`。这个 skill 为一个新的语音应用选择 ASR 模型、解码参数和预处理流水线。

## Exercises（练习）

1. **Easy（简单）。** 运行 `code/main.py`。确认 1 秒 16 kHz 信号带 10 ms hop 的帧数约 100 帧。30 秒：约 3,000 帧。
2. **Medium（中等）。** 用 `numpy.fft` 构建完整 log-mel spectrogram。验证 80 mel bins 在数值误差内匹配 `librosa.feature.melspectrogram(n_mels=80)`。
3. **Hard（困难）。** 实现流式推理：把音频切成 10 秒窗口带 2 秒 overlap，在每个块上运行 Whisper，合并 transcript。在 5 分钟播客样本上测量 word-error rate vs single-pass。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Mel spectrogram（Mel spectrogram） | "Audio image"（音频图像） | 2D 表示：一个轴上的频率 bins，一个轴上的时间帧；每个 cell 的对数缩放能量。 |
| Log-mel | "What Whisper sees"（Whisper 看到的） | 通过 log 的 mel spectrogram；近似人类对响度的感知。 |
| Frame（帧） | "One time slice"（一个时间切片） | 25 ms 的样本窗口；10 ms stride 重叠。 |
| Task token（任务 token） | "Prompt prefix for speech"（语音的提示前缀） | 特殊 token 如 `<|transcribe|>` / `<|translate|>` 在解码器提示中。 |
| Voice activity detection (VAD)（语音活动检测） | "Find the speech"（找到语音） | 在 ASR 之前去除沉默的门；大幅降低成本。 |
| CTC | "Connectionist Temporal Classification"（连接ist 时序分类） | 无对齐训练的经典 ASR 损失；Whisper *不*使用它。 |
| Whisper-turbo | "Small decoder, full encoder"（小解码器，完整编码器） | large-v3 编码器 + 4 层解码器；8 倍更快解码。 |
| Faster-whisper | "The production wrapper"（生产包装器） | CTranslate2 重新实现；int8 量化；比 OpenAI 参考快 4 倍。 |

## Further Reading（延伸阅读）

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper 论文。
- [OpenAI Whisper repo](https://github.com/openai/whisper) — 参考代码 + 模型权重。阅读 `whisper/model.py` 查看 Conv1D stem + encoder + decoder 自顶向下约 400 行。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — Steps 5-6 中描述的 beam-search + task-token 逻辑在这里；500 行，完全可读。
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — 前身；在某些设置中仍然是 SOTA 特征。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 生产包装器，比参考快 4 倍。
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024 边缘友好 ASR，Whisper-shaped 但更小。
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — 规范微调配方，包括 mel spectrogram preprocessor 和 token-timestamp handling。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完整实现（encoder、decoder、cross-attention、generation），匹配课程的架构图。
