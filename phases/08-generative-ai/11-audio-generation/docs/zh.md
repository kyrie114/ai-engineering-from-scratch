# Audio Generation（音频生成）

> 音频是一个 16-48 kHz 的 1-D 信号。一个五秒的片段是 80-240k 个样本。没有 Transformer 直接关注那个序列。2026 年每一个生产音频模型的解决方案是相同的：一个神经编解码器（Encodec、SoundStream、DAC）将音频压缩成 50-75 Hz 的离散令牌，一个 Transformer 或扩散模型生成令牌。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Audio Features)（Phase 6 第 02 课 音频特征）, Phase 6 · 04 (ASR)（Phase 6 第 04 课 自动语音识别）, Phase 8 · 06 (DDPM)（Phase 8 第 06 课 DDPM）
**Time:** ~45 minutes（约 45 分钟）

## The Problem（问题）

三个音频生成任务：

1. **Text-to-speech（文本到语音）。** 给定文本，产生语音。干净语音是窄带的，具有强烈的语音结构 —— 被令牌上的 Transformer 很好地解决。VALL-E（微软）、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **Music generation（音乐生成）。** 给定一个提示（文本、旋律、和弦进行、流派），产生音乐。宽得多分布。MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **Audio effects / sound design（音频效果/音效设计）。** 给定一个提示，产生环境声音或 Foley。AudioGen、AudioLDM 2、Stable Audio Open。

所有三个都运行在相同的基质上：神经音频编解码器 + 令牌 AR 或扩散生成器。

## The Concept（概念）

![Audio generation: codec tokens + transformer or diffusion（音频生成：编解码器令牌 + Transformer 或扩散）](../assets/audio-generation.svg)

### Neural audio codecs（神经音频编解码器）

Encodec（Meta，2022）、SoundStream（Google，2021）、Descript Audio Codec（DAC，2023）。一个卷积编码器将波形压缩为每个时间步向量；残差向量量化（RVQ）将每个向量转换为 K 个码本索引的级联。解码器反转它。24 kHz 音频在 2 kbps 使用 8 个 RVQ 码本在 75 Hz = 600 个令牌/秒。

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### Two generative paradigms on top（顶部的两种生成范式）

**Token-autoregressive（令牌自回归）。** 将 RVQ 令牌展平为序列，运行仅解码器 Transformer。MusicGen 使用“延迟并行”以每流偏移并行发射 K 个码本流。VALL-E 从文本提示 + 3 秒语音样本生成语音令牌。

**Latent diffusion（潜空间扩散）。** 将编解码器令牌打包为连续潜空间或用分类扩散建模它们。Stable Audio 2.5 在连续音频潜空间上使用流匹配。AudioLDM 2 使用文本到 mel 到音频扩散。

2024-2026 趋势：流匹配在音乐上获胜（更快的推理，更干净的样本），而令牌 AR 仍然主导语音，因为它自然地因果并且流式传输良好。

## Production landscape（生产格局）

| System（系统） | Task（任务） | Backbone（主干） | Latency（延迟） |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + neural vocoder | ~300ms first token |
| OpenAI GPT-4o audio | Full-duplex speech | End-to-end multimodal AR | ~200ms |
| NaturalSpeech 3 | TTS | Latent flow matching | Non-streaming |
| Stable Audio 2.5 | Music / SFX | DiT + flow matching on audio latents | ~10s for 1-minute clip |
| Suno v4 | Full songs | Undisclosed; token-AR suspected | ~30s per song |
| Udio v1.5 | Full songs | Undisclosed | ~30s per song |
| MusicGen 3.3B | Music | Token-AR on Encodec 32kHz | Real-time |
| AudioCraft 2 | Music + SFX | Flow matching | ~5s for 5s clip |
| Riffusion v2 | Music | Spectrogram diffusion | ~10s |

```figure
score-matching
```

## Build It（动手实现）

`code/main.py` 模拟了核心思想：在从两个不同“风格”（风格 A 交替高低令牌，风格 B 单调斜坡）生成的合成“音频令牌”序列上训练一个微型下一个令牌 Transformer。以风格为条件并采样。

### Step 1: synthetic audio tokens（合成音频令牌）

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating（“类语音”：交替）
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp（“类音乐”：斜坡）
    return [(i * 3) % vocab_size for i in range(length)]
```

### Step 2: train a tiny token predictor（训练一个微型令牌预测器）

一个以风格为条件的大括号风格预测器。要点是模式：编解码器令牌 → 交叉熵训练 → 自回归采样。

### Step 3: sample conditionally（有条件地采样）

给定风格令牌和起始令牌，从预测分布中采样下一个令牌。继续 20-40 个令牌。

## Pitfalls（陷阱）

- **Codec quality caps output quality（编解码器质量上限输出质量）。** 如果编解码器不能忠实地表示声音，再多的生成器质量也没有帮助。DAC 是当前的开放最佳。
- **RVQ error accumulation（RVQ 错误累积）。** 每个 RVQ 层对前一个的残差建模。第 1 层的错误传播。在更高层上用温度 0 采样有帮助。
- **Musical structure（音乐结构）。** 30 秒的令牌在 75 Hz 是 20k+ 令牌。对 Transformer 很难。MusicGen 使用滑动窗口 + 提示延续；Stable Audio 使用较短的片段 + 交叉淡化。
- **Artifacts at boundaries（边界伪影）。** 在生成的片段之间交叉淡化需要仔细的重叠添加。
- **Clean-data appetite（干净数据胃口）。** 音乐生成器需要数万小时的许可音乐。Suno / Udio RIAA 诉讼（2024）将这一点公之于众。
- **Voice cloning ethics（语音克隆伦理）。** 一个 3 秒样本加一个文本提示足以让 VALL-E / XTTS / ElevenLabs 克隆一个声音。每一个生产模型都需要滥用检测 + 选择退出列表。

## Use It（实际应用）

| Task（任务） | 2026 stack（2026 栈） |
|------|------------|
| Commercial TTS（商业 TTS） | ElevenLabs, OpenAI TTS, or Azure Neural |
| Voice cloning (consent-verified)（语音克隆（同意验证）） | XTTS v2 (open) or ElevenLabs Pro |
| Background music, fast（背景音乐，快速） | Stable Audio 2.5 API, Suno, or Udio |
| Music with lyrics（带歌词的音乐） | Suno v4 or Udio v1.5 |
| Sound effects / Foley（音效/Foley） | AudioCraft 2, ElevenLabs SFX, or Stable Audio Open |
| Real-time voice agent（实时语音代理） | GPT-4o realtime or Gemini Live |
| Open-weights music research（开放权重音乐研究） | MusicGen 3.3B, Stable Audio Open 1.0, AudioLDM 2 |
| Dubbing / translation（配音/翻译） | HeyGen, ElevenLabs Dubbing |

## Ship It（交付）

保存为 `outputs/skill-audio-brief.md`。该技能接收一个音频简报（任务、时长、风格、声音、许可），并输出：模型 + 托管、提示格式（流派标签、风格描述符、结构标记）、编解码器 + 生成器 + 声码器链、种子协议和评估计划（MOS / CLAP score / CER for TTS / user A/B）。

## Exercises（练习）

1. **Easy（简单）。** 运行 `code/main.py` 并明确设置风格。验证生成的序列与风格的模式匹配。
2. **Medium（中等）。** 添加延迟并行解码：模拟 2 个必须保持 1 步偏移的令牌流。训练一个联合预测器。
3. **Hard（困难）。** 使用 HuggingFace transformers 在本地运行 MusicGen-small。用三个不同的提示生成一个 10 秒片段；A/B 测试风格遵循。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Codec（编解码器） | "Neural compression"（神经压缩） | 音频的编码器/解码器；典型输出是 50-75 Hz 令牌。 |
| RVQ | "Residual VQ"（残差 VQ） | K 个量化器的级联；每一个对前一个的残差建模。 |
| Token（令牌） | "One codec symbol"（一个编解码器符号） | 码本中的离散索引；典型 1024 或 2048。 |
| Delayed parallel（延迟并行） | "Offset codebooks"（偏移码本） | 以交错偏移发射 K 个令牌流以减少序列长度。 |
| Flow matching（流匹配） | "The 2024 win for audio"（2024 年音频胜利） | 比扩散更直的路径；更快的采样。 |
| Voice prompt（语音提示） | "3-second sample"（3 秒样本） | 引导克隆声音的说话人嵌入或令牌前缀。 |
| Mel spectrogram（Mel 频谱图） | "The visual"（可视化） | 对数幅度感知频谱图；被许多 TTS 系统使用。 |
| Vocoder（声码器） | "Mel to wave"（Mel 到波形） | 将 Mel 频谱图转换回音频的神经组件。 |

## Production note: audio is a streaming problem（生产笔记：音频是一个流式问题）

音频是用户期望*在生成时*到达的输出模态，而不是一次性。在生产术语中，这意味着 TPOT 很重要（每输出令牌时间），因为用户的收听速度是目标吞吐量 —— 而不是他们的阅读速度。对于在约 75 个令牌/秒（Encodec）令牌化的 16kHz 音频，服务器必须每个用户生成 ≥75 个令牌/秒以保持播放平滑。

两个建筑后果：

- **Flow-matching audio models cannot stream trivially（流匹配音频模型不能轻易流式传输）。** Stable Audio 2.5 和 AudioCraft 2 在一个通道中渲染固定片段长度。要流式传输，你块片段并在边界重叠 —— 想想滑动窗口扩散 —— 增加 100-300ms 的延迟开销 vs 编解码器 AR 模型。

如果产品是“实时语音聊天”或“实时音乐延续”，选择编解码器 AR 路径。如果它是“在提交时渲染一个 30 秒片段”，流匹配在质量和总延迟上获胜。

## Further Reading（延伸阅读）

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) —— 编解码器标准。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) —— 第一个广泛使用的神经音频编解码器。
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) —— DAC。
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) —— VALL-E。
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) —— MusicGen。
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) —— AudioLDM 2。
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) —— 2025 年流匹配文本到音乐。
