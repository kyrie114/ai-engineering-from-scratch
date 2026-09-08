# 音频基础——波形、采样、傅里叶变换（Audio Fundamentals — Waveforms, Sampling, Fourier Transform）

> 波形是原始信号，语谱图是表示方式，梅尔特征是对机器学习友好的形式。每条现代 ASR 和 TTS 流水线都要走过这条阶梯，而第一级就是理解采样和傅里叶变换。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Vectors & Matrices), Phase 1 · 14 (Probability Distributions)
**Time:** ~45 minutes

## 问题（The Problem）

麦克风输出的是压力随时间变化的信号，而你的神经网络消费的是张量。在这两者之间有一整套约定，一旦违反就会产生静默 bug：模型训练正常，但 WER 翻倍；或者 TTS 带出嘶声；又或者语音克隆系统记住的是麦克风而不是说话人。

语音系统中的每个 bug 都能追溯到以下三个问题之一：

1. 数据以什么采样率录制，模型又期望什么采样率？
2. 信号是否存在混叠？
3. 你是在操作原始采样，还是在操作频率表示？

把这些搞对，Phase 6 剩下的内容就都能驾驭；搞错了，即使 Whisper-Large-v4 也会输出垃圾。

## 概念（The Concept）

![Waveform, sampling, DFT, and frequency bins visualized](../assets/audio-fundamentals.svg)

**波形（Waveform）。** 一个取值范围为 `[-1.0, 1.0]` 的一维浮点数组。按采样序号索引。要转换为秒，除以采样率：`t = n / sr`。一段 16 kHz 下的 10 秒音频是 160,000 个浮点数的数组。

**采样率（sr）（Sampling rate）。** 每秒采样的次数。2026 年常见的采样率：

| Rate | Use |
|------|-----|
| 8 kHz | Telephony, legacy VOIP. Nyquist at 4 kHz kills consonants. Avoid for ASR. |
|      | 电话、旧 VOIP。4 kHz 的奈奎斯特频率会抹掉辅音。ASR 请避免。 |
| 16 kHz | ASR standard. Whisper, Parakeet, SeamlessM4T v2 all consume 16 kHz. |
|        | ASR 标准。Whisper、Parakeet、SeamlessM4T v2 都使用 16 kHz。 |
| 22.05 kHz | TTS vocoder training for older models. |
|            | 旧模型 TTS 声码器训练。 |
| 24 kHz | Modern TTS (Kokoro, F5-TTS, xTTS v2). |
|        | 现代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD audio, music. |
|           | CD 音频、音乐。 |
| 48 kHz | Film, pro audio, high-fidelity TTS (VALL-E 2, NaturalSpeech 3). |
|        | 电影、专业音频、高保真 TTS（VALL-E 2、NaturalSpeech 3）。 |

**奈奎斯特-香农（Nyquist-Shannon）。** 采样率为 `sr` 的信号最多可以无歧义地表示 `sr/2` 以下的频率。`sr/2` 这个边界就是*奈奎斯特频率*。超过奈奎斯特频率的能量会被*混叠*——折叠到较低频率——并污染信号。下采样前务必先做低通滤波。

**位深（Bit depth）。** 16 位 PCM（有符号 int16，范围 ±32,767）是通用交换格式。音乐用 24 位，内部 DSP 用 32 位 float。`soundfile` 等库读取 int16，但会返回 `[-1, 1]` 范围内的 float32 数组。

**傅里叶变换（Fourier Transform）。** 任何有限信号都是不同频率正弦波的和。离散傅里叶变换（DFT）对 `N` 个采样计算 `N` 个复数系数——每个频率 bin 一个。`bin k` 对应频率 `k · sr / N` Hz。幅度是该频率的振幅，角度是相位。

**FFT。** 快速傅里叶变换：当 `N` 是 2 的幂时，DFT 的 `O(N log N)` 算法。每个音频库都在底层使用 FFT。16 kHz 下的 1024 采样 FFT 给出 512 个可用频率 bin，覆盖 0–8 kHz，分辨率为 15.6 Hz。

**分帧 + 窗函数（Framing + window）。** 我们不对整个片段做 FFT。我们把它切成重叠的*帧*（通常 25 ms，10 ms  hop），将每帧乘以窗函数（Hann、Hamming）以消除边缘不连续，然后对每帧做 FFT。这就是短时傅里叶变换（STFT）。第 02 课从这里继续。

```figure
mel-scale
```

## 构建它（Build It）

### 步骤 1：读取片段并绘制波形（read a clip and plot the waveform）

`code/main.py` 仅使用 stdlib 的 `wave` 模块以保持演示无依赖。生产环境你会使用 `soundfile` 或 `torchaudio.load`（两者都返回 `(waveform, sr)` 元组）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 步骤 2：从第一性原理合成正弦波（synthesize a sine wave from first principles）

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

440 Hz 正弦波（音乐会 A）在 16 kHz 下 1 秒是 16,000 个浮点数。使用 `wave.open(..., "wb")` 以 16 位 PCM 编码写入。

### 步骤 3：手动计算 DFT（compute the DFT by hand）

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)`——对于 `N=256` 验证正确性还可以，对真实音频无用。真实代码调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 步骤 4：找到主频率（find the dominant frequency）

幅度峰值索引 `k_star` 对应频率 `k_star * sr / N`。在 440 Hz 正弦波上运行此代码应在 bin `440 * N / sr` 处返回峰值。

### 步骤 5：演示混叠（demonstrate aliasing）

以 10 kHz 采样 7 kHz 正弦波（奈奎斯特 = 5 kHz）。7 kHz 音调高于奈奎斯特，折叠到 `10 − 7 = 3 kHz`。FFT 峰值出现在 3 kHz。这是经典的混叠演示，也是每个 DAC/ADC 都配备 brick-wall 低通滤波器的原因。

## 使用它（Use It）

2026 年你会实际部署的栈：

| Task | Library | Why |
|------|---------|-----|
| Read/write WAV/FLAC/OGG | `soundfile` (libsndfile wrapper) | Fastest, stable, returns float32. |
| Resample | `torchaudio.transforms.Resample` or `librosa.resample` | Correct anti-aliasing built in. |
| STFT / Mel | `torchaudio` or `librosa` | GPU-friendly; PyTorch ecosystem. |
| Real-time streaming | `sounddevice` or `pyaudio` | Cross-platform PortAudio bindings. |
| Inspect a file | `ffprobe` or `soxi` | CLI, fast, reports sr/channels/codec. |

决策规则：**先匹配采样率，再匹配其他任何东西**。Whisper 期望 16 kHz 单声道 float32。给它 44.1 kHz 立体声，你会得到看起来像模型 bug 的垃圾输出。

## 交付它（Ship It）

保存为 `outputs/skill-audio-loader.md`。该技能帮助你检查音频输入是否匹配下游模型的期望，并在不匹配时正确重采样。

## 练习（Exercises）

1. **简单（Easy）。** 在 16 kHz 下合成 1 秒的 220 Hz + 440 Hz + 880 Hz 混合音。运行 DFT。确认三个峰值在预期的 bin 处。
2. **中等（Medium）。** 以 48 kHz 录制 3 秒你的声音 WAV。使用 `torchaudio.transforms.Resample`（带抗混叠）下采样到 16 kHz，然后使用朴素抽取（每第 3 个采样）下采样到 16 kHz。对两者做 FFT。混叠出现在哪里？
3. **困难（Hard）。** 仅使用 `math` 和第 3 步的 DFT 从头构建 STFT。帧大小 400，hop 160，Hann 窗。使用 `matplotlib.pyplot.imshow` 绘制幅度。这就是第 02 课的语谱图。

## 关键术语（Key Terms）

| 术语（Term） | 人们常说的（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| 采样率（Sample rate） | 每秒采样次数 | ADC 测量信号的频率，单位为 Hz。 |
| 奈奎斯特定理（Nyquist） | 你能表示的最高频率 | `sr/2`；高于它的能量会混叠回较低 bin。 |
| 位深（Bit depth） | 每个采样的分辨率 | `int16` = 65,536 个电平；`float32` = `[-1, 1]` 内的 24 位精度。 |
| DFT | 序列的傅里叶变换 | `N` 个采样 → `N` 个复数频率系数。 |
| FFT | 快速 DFT | 要求 `N` 为 2 的幂的 `O(N log N)` 算法。 |
| Bin | 频率列 | `k · sr / N` Hz；分辨率 = `sr / N`。 |
| STFT | 底层的语谱图 | 随时间分帧 + 加窗 FFT。 |
| 混叠（Aliasing） | 奇怪的频率幽灵 | 高于奈奎斯特的能量镜像到较低 bin。 |

## 延伸阅读（Further Reading）

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) —— 采样定理背后的论文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) —— 免费、经典的 DSP 教材。
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) —— 带代码的实用教程。
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) —— 解释为什么真实世界音频不是干净正弦波的参考。
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) —— 10 分钟讲清频率 bin 直觉。
