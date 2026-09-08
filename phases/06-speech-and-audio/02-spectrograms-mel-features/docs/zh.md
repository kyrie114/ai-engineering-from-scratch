# 语谱图、梅尔标度与音频特征（Spectrograms, Mel Scale & Audio Features）

> 神经网络不擅长消费原始波形，它们消费语谱图。它们消费梅尔语谱图效果更好。2026 年每个 ASR、TTS 和音频分类器的成败都取决于这个单一的预处理选择。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 01 (Audio Fundamentals)
**Time:** ~45 minutes

## 问题（The Problem）

取一个 10 秒 16 kHz 的片段。那是 160,000 个浮点数，都在 `[-1, 1]` 范围内，几乎与"狗叫"或"猫这个词"的标签完全不相关。原始波形包含信息，但模型无法轻易提取的形式。相隔 100 ms 的两个相同音素具有完全不同的原始采样。

语谱图解决了这个问题。它压缩了人类感知忽略的时间细节（微秒级抖动），并保留了感知关注的结构（哪些频率有能量，在 ~10–25 ms 的时间窗口内）。

梅尔语谱图走得更远。人类以对数方式感知音高：100 Hz 与 200 Hz 之间的听感差距和 1000 Hz 与 2000 Hz 之间的听感差距"相同"。梅尔标度将频率轴扭曲以匹配这一点。梅尔标度的语谱图是 2010 年至 2026 年语音 ML 中最重要的特征。

## 概念（The Concept）

![Waveform to STFT to mel spectrogram to MFCC ladder](../assets/mel-features.svg)

**短时傅里叶变换（STFT）（Short-Time Fourier Transform）。** 将波形切成重叠的帧（典型：25 ms 窗，10 ms hop = 16 kHz 下 400 采样 / 160 采样）。将每帧乘以窗函数（Hann 是默认；Hamming 略有不同的权衡）。对每帧做 FFT。将幅度谱堆叠成形状为 `(n_frames, n_freq_bins)` 的矩阵。这就是你的语谱图。

**对数幅度（Log-magnitude）。** 原始幅度跨越 5-6 个数量级。取 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 以压缩动态范围。每个生产流水线都使用对数幅度，而非原始幅度。

**梅尔标度（Mel scale）。** 频率 `f`（Hz）映射到梅尔 `m` 通过 `m = 2595 * log10(1 + f / 700)`。映射在 1 kHz 以下大致线性，以上大致对数。覆盖 0–8 kHz 的 80 个梅尔 bin 是标准 ASR 输入。

**梅尔滤波器组（Mel filterbank）。** 一组在梅尔标度上等间距的三角滤波器。每个滤波器是相邻 FFT bin 的加权和。将 STFT 幅度与滤波器组矩阵相乘，一次 matmul 就得到梅尔语谱图。

**对数梅尔语谱图（Log-mel spectrogram）。** `log(mel_spec + 1e-10)`。Whisper 的输入。Parakeet 的输入。SeamlessM4T 的输入。2026 年的通用音频前端。

**MFCC。** 取对数梅尔语谱图，应用 DCT（type II），保留前 13 个系数。去相关特征并进一步压缩。直到约 2015 年都是主导特征，当时 CNN/Transformer 在原始对数梅尔上追平。仍然用于说话人识别（x-vectors、ECAPA）。

**分辨率权衡（Resolution trade）。** 更大的 FFT = 更好的频率分辨率但更差的时间分辨率。25 ms / 10 ms 是音频 ML 的默认值；50 ms / 12.5 ms 用于音乐；5 ms / 2 ms 用于瞬态检测（鼓声、爆破音）。

```figure
spectrogram-window
```

## 构建它（Build It）

### 步骤 1：分帧波形（frame the waveform）

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

使用 `frame_len=400, hop=160` 的 10 秒 16 kHz 片段产生 998 帧。

### 步骤 2：Hann 窗（Hann window）

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

在 FFT 前逐元素相乘。消除因在非零端点截断而引起的频谱泄漏。

### 步骤 3：STFT 幅度（STFT magnitude）

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产环境使用 `torch.stft` 或 `librosa.stft`（FFT 支持，向量化）。这里的循环是教学性的；它在 `code/main.py` 中的短片段上运行。

### 步骤 4：梅尔滤波器组（mel filterbank）

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

覆盖 0–8 kHz 的 80 个梅尔 bin，使用 `n_fft=400`，给出 `(80, 201)` 矩阵。将 `(n_frames, 201)` STFT 幅度与转置相乘，得到 `(n_frames, 80)` 梅尔语谱图。

### 步骤 5：对数梅尔（log-mel）

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见替代方案：`librosa.power_to_db`（参考归一化的 dB），`10 * log10(power + eps)`。Whisper 使用更复杂的 clip + 归一化例程（见 Whisper 的 `log_mel_spectrogram`）。

### 步骤 6：MFCC（MFCCs）

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每个对数梅尔帧应用 DCT，保留前 13 个系数。这就是你的 MFCC 矩阵。第一个系数通常被丢弃（它编码整体能量）。

## 使用它（Use It）

2026 年的栈：

| Task | Features |
|------|----------|
| ASR (Whisper, Parakeet, SeamlessM4T) | 80 个对数梅尔，10 ms hop，25 ms 窗 |
| TTS acoustic model (VITS, F5-TTS, Kokoro) | 80 个梅尔，5–12 ms hop 以获得精细时间控制 |
| Audio classification (AST, PANNs, BEATs) | 128 个对数梅尔，10 ms hop |
| Speaker embedding (ECAPA-TDNN, WavLM) | 80 个对数梅尔或原始波形 SSL |
| Music (MusicGen, Stable Audio 2) | EnCodec 离散 token（非梅尔） |
| Keyword spotting | 40 个 MFCC，用于微型设备 |

经验法则：**如果你不做音乐，从 80 个对数梅尔开始**。任何偏离都需要举证。

## 2026 年仍然存在的陷阱（Pitfalls that still ship in 2026）

- **梅尔数量不匹配（Mel count mismatch）。** 训练用 80 个梅尔，推理用 128 个。静默失败。在两端记录特征形状。
- **上游采样率不匹配（Sample-rate mismatch upstream）。** 在 22.05 kHz 下计算的梅尔与 16 kHz 下不同。在特征提取前修复 SR。
- **dB vs log（dB 与对数）。** Whisper 期望对数梅尔，而非 dB 梅尔。某些 HF 流水线会自动检测；你的自定义代码不会。
- **归一化漂移（Normalization drift）。** 训练时逐语句归一化，推理时全局归一化。使 WER 翻倍的生产 bug。
- **填充泄漏（Leakage from padding）。** 对片段末尾做零填充会在尾帧产生平坦频谱。对称填充或复制填充。

## 交付它（Ship It）

保存为 `outputs/skill-feature-extractor.md`。该技能根据给定的模型目标选择特征类型、梅尔数量、帧/hop 和归一化。

## 练习（Exercises）

1. **简单（Easy）。** 运行 `code/main.py`。它合成一个 chirp（频率从 200 Hz 扫到 4000 Hz）并打印每帧的 argmax 梅尔 bin。可选绘图并确认它匹配扫频。
2. **中等（Medium）。** 使用 `n_mels` 在 `{40, 80, 128}` 和 `frame_len` 在 `{200, 400, 800}` 中重新运行。测量跨时间轴的尖峰带宽。哪种组合最能解析 chirp？
3. **困难（Hard）。** 实现 `power_to_db` 并在 AudioMNIST 上比较微小 CNN 分类器的 ASR 准确率，分别使用 (a) 原始对数梅尔，(b) `ref=max` 的 dB 梅尔，(c) MFCC-13 + delta + delta-delta。报告 top-1 准确率。

## 关键术语（Key Terms）

| 术语（Term） | 人们常说的（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| 帧（Frame） | 一个切片 | 送入一个 FFT 的 25 ms 波形块。 |
| Hop | 步幅 | 连续帧之间的采样数；10 ms 是 ASR 默认值。 |
| 窗函数（Window） | Hann/Hamming 的东西 | 逐点乘数，将帧边缘 taper 到零。 |
| STFT | 语谱图生成器 | 分帧 + 加窗 FFT；产生时间 × 频率矩阵。 |
| Mel | 弯曲的频率 | 对数感知标度；`m = 2595·log10(1 + f/700)`。 |
| 滤波器组（Filterbank） | 矩阵 | 将 STFT 投影到梅尔 bin 的三角滤波器。 |
| 对数梅尔（Log-mel） | Whisper 的输入 | `log(mel_spec + eps)`；2026 年标准化。 |
| MFCC | 老派特征 | 对数梅尔的 DCT；13 个系数，去相关。 |

## 延伸阅读（Further Reading）

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) —— MFCC 论文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) —— 原始的梅尔标度。
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) —— 阅读参考实现。
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) —— `mfcc`、`melspectrogram` 和 hop/window 的参考。
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) —— Parakeet + Canary 模型的生产级流水线。
