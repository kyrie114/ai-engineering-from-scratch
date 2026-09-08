# Music Generation — MusicGen, Stable Audio, Suno, and the Licensing Earthquake（音乐生成——MusicGen、Stable Audio、Suno 和版权地震）

> 2026 music generation: Suno v5 and Udio v4 dominate commercial; MusicGen, Stable Audio Open, and ACE-Step lead open-source. The technical problem is mostly solved. The legal problem (Warner Music $500M settlement, UMG settlement) reshaped the field in 2025-2026.
> 2026 年音乐生成：Suno v5 和 Udio v4 主导商业市场；MusicGen、Stable Audio Open 和 ACE-Step 领先开源。技术问题基本已解决。法律问题（Warner Music 5 亿美元和解、UMG 和解）在 2025-2026 年重塑了该领域。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 4 · 10 (Diffusion Models)
**Time:** ~75 minutes

## The Problem（问题）

Text → a 30-second to 4-minute music clip, with lyrics, vocals, and structure. Three sub-problems:
文本 → 一段 30 秒到 4 分钟的音乐片段，包含歌词、人声和结构。三个子问题：

1. **Instrumental generation（器乐生成）. ** Text like "lo-fi hip-hop drums with warm keys" → audio. MusicGen, Stable Audio, AudioLDM.
   像 "lo-fi hip-hop drums with warm keys" 这样的文本 → 音频。MusicGen、Stable Audio、AudioLDM。
2. **Song generation (with vocals + lyrics)（歌曲生成（带人声 + 歌词）). ** "Country song about rainy Texas nights" → full song. Suno, Udio, YuE, ACE-Step.
   "Country song about rainy Texas nights" → 完整歌曲。Suno、Udio、YuE、ACE-Step。
3. **Conditional / controllable（条件 / 可控）. ** Extend an existing clip, regenerate a bridge, swap genre, stem-separate, or inpaint. Udio's inpainting + stem separation is the 2026 feature to match.
   扩展现有片段，重新生成桥段，换风格，分离音轨，或修复。Udio 的修复 + 音轨分离是 2026 年需要匹配的功能。

## The Concept（概念）

![Music generation: token-LM vs diffusion, the 2026 model map](../assets/music-generation.svg)

### Token LM over neural-codec tokens（神经编码器 token 上的 token LM）

Meta's **MusicGen** (2023, MIT) and many derivatives: condition on text/melody embeddings, autoregressively predict EnCodec tokens (32 kHz, 4 codebooks), decode with EnCodec. 300M - 3.3B params. Strong baseline; struggles past 30 seconds.
Meta 的 **MusicGen** (2023, MIT) 及许多衍生模型：以文本/旋律嵌入为条件，自回归预测 EnCodec token（32 kHz，4 个码本），用 EnCodec 解码。3 亿 - 33 亿参数。强基线；超过 30 秒会吃力。

**ACE-Step** (open-source, 4B XL released April 2026) extends this for full-song lyric-conditioned generation. The open community's closest thing to Suno.
**ACE-Step**（开源，4B XL 于 2026 年 4 月发布）将其扩展为带歌词条件的全歌曲生成。开源社区最接近 Suno 的东西。

### Diffusion over mels or latents（在 mel 或隐空间上的扩散）

**Stable Audio (2023)** and **Stable Audio Open (2024)**: latent diffusion on compressed audio. Excels at loops, sound design, ambient textures. Not great at structured full songs.
**Stable Audio (2023)** 和 **Stable Audio Open (2024)**：压缩音频上的隐空间扩散。擅长循环、声音设计、氛围纹理。在结构化完整歌曲上表现不佳。

**AudioLDM / AudioLDM2**: text-to-audio via T2I-style latent diffusion, generalized to music, sound effects, speech.
**AudioLDM / AudioLDM2**：通过 T2I 风格隐空间扩散做文本到音频，泛化到音乐、音效、语音。

### Hybrid (production) — Suno, Udio, Lyria（混合（生产级）——Suno、Udio、Lyria）

Closed weights. Likely AR codec LM + diffusion-based vocoder with specialized voice / drum / melody heads. Suno v5 (2026) is the ELO 1293 quality leader. Udio v4 adds inpainting + stem separation (bass, drums, vocals separate downloads).
闭源权重。可能是 AR 编码 LM + 基于扩散的声码器，带有专门的人声 / 鼓 / 旋律头。Suno v5 (2026) 是 ELO 1293 质量领导者。Udio v4 增加了修复 + 音轨分离（贝斯、鼓、人声分开下载）。

### Evaluation（评估）

- **FAD (Fréchet Audio Distance)（FAD（Fréchet 音频距离）). ** Embedding-level distance between generated vs real audio distribution using VGGish or PANNs features. Lower is better. MusicGen small: 4.5 FAD on MusicCaps; SOTA ~3.0.
  用 VGGish 或 PANNs 特征计算生成与真实音频分布之间的嵌入级距离。越低越好。MusicGen small：在 MusicCaps 上 FAD 为 4.5；SOTA 约 3.0。
- **Musicality (subjective)（音乐性（主观）). ** Human preference. Suno v5 ELO 1293 leads.
  人类偏好。Suno v5 ELO 1293 领先。
- **Text-audio alignment（文本-音频对齐）. ** CLAP score between prompt and output.
  提示和输出之间的 CLAP 分数。
- **Musicality artifacts（音乐性伪影）. ** Off-beat transitions, vocal-phrase drift, loss of structure past 30 s.
  抢拍过渡、人声短语漂移、超过 30 秒后结构丢失。

## 2026 model map（2026 年模型地图）

| Model | Params | Length | Vocals | License |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 s | no | MIT |
| Stable Audio Open | 1.2B | 47 s | no | Stability non-commercial |
| ACE-Step XL (Apr 2026) | 4B | &gt; 2 min | yes | Apache-2.0 |
| YuE | 7B | &gt; 2 min | yes, multilingual | Apache-2.0 |
| Suno v5 (closed) | ? | 4 min | yes, ELO 1293 | commercial |
| Udio v4 (closed) | ? | 4 min | yes + stems | commercial |
| Google Lyria 3 (closed) | ? | real-time | yes | commercial |
| MiniMax Music 2.5 | ? | 4 min | yes | commercial API |

## The legal landscape (2025-2026)（法律格局 (2025-2026)）

- **Warner Music vs Suno settlement（Warner Music 与 Suno 和解）. ** $500M. WMG now has oversight of AI-likeness, music rights, and user-generated tracks on Suno. Similar UMG settlement on Udio.
  5 亿美元。WMG 现在对 Suno 上的 AI 相似性、音乐权利和用户生成曲目拥有监督权。Udio 上有类似的 UMG 和解。
- **EU AI Act** + **California SB 942**: AI-generated music must be disclosed.
  欧盟 AI 法案 + 加利福尼亚州 SB 942：AI 生成音乐必须披露。
- **Riffusion / MusicGen** under MIT have no compliance baggage but also no commercial vocals.
   MIT 下的 Riffusion / MusicGen 没有合规负担，但也没有商业人声。

Safe-to-ship patterns:
可安全发货的模式：

1. Generate instrumental only (MusicGen, Stable Audio Open, MIT/CC0 outputs).
   仅生成器乐（MusicGen、Stable Audio Open、MIT/CC0 输出）。
2. Use commercial APIs (Suno, Udio, ElevenLabs Music) with per-generation license.
   使用商业 API（Suno、Udio、ElevenLabs Music）并附带每次生成的许可证。
3. Train on owned or licensed catalog (most enterprises end up here).
   在自有或授权目录上训练（大多数企业最终会这样做）。
4. Tag generations with watermarks + metadata.
   用水印 + 元数据标记生成内容。

```figure
sp-codec-tokens
```

## Build It（动手实现）

### Step 1: generate with MusicGen（用 MusicGen 生成）

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

Three sizes: `small` (300M, fast), `medium` (1.5B), `large` (3.3B). Small is enough for "does the idea land."
三种尺寸：`small` (3 亿，快)、`medium` (15 亿)、`large` (33 亿)。Small 足够回答"这个想法可行吗"。

### Step 2: melody conditioning（旋律条件）

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody takes a chromagram and preserves the tune while swapping timbre. Useful for "give me this melody as a string quartet."
MusicGen-melody 接收一个色度图并保留旋律，同时交换音色。适用于"把这个旋律给我做成弦乐四重奏"。

### Step 3: FAD evaluation（FAD 评估）

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

Computes VGGish-embedding distance. Useful for genre-level regression tests; not a substitute for human listeners.
计算 VGGish 嵌入距离。适用于流派级回归测试；不能替代人类听众。

### Step 4: adding to the LLM-music workflow（加入 LLM-音乐工作流）

Combine with the ideas from Lessons 7-8:
结合第 7-8 课的想法：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## Use It（实际应用）

| Goal | Stack |
|------|-------|
| Instrumental sound design | Stable Audio Open |
|                           | 器乐声音设计 |
| Game / adaptive music | Google Lyria RealTime (closed) |
|                       | 游戏 / 自适应音乐 |
| Full songs with vocals (commercial) | Suno v5 or Udio v4 with explicit license |
|                                    | 带人声的完整歌曲（商业） |
| Full songs with vocals (open) | ACE-Step XL or YuE |
|                               | 带人声的完整歌曲（开源） |
| Short ad jingle | MusicGen melody-conditioned on a hummed reference |
|                | 短广告 jingle |
| Music-video background | MusicGen + Stable Video Diffusion |
|                       | 音乐视频背景 |

## Pitfalls that still ship in 2026（2026 年仍然上线的坑）

- **Copyright-laundering prompts（版权洗白提示）. ** "Song in the style of Taylor Swift" — commercial Suno/Udio filter these now, open models do not. Add your own filter list.
  "Song in the style of Taylor Swift"——商业 Suno/Udio 现在会过滤这些，开源模型不会。添加你自己的过滤列表。
- **Repetition / drift past 30 s（超过 30 秒的重复 / 漂移）. ** AR models loop. Crossfade multiple generations, or use ACE-Step for structural coherence.
  AR 模型循环。交叉淡入淡出多个生成，或使用 ACE-Step 保持结构连贯性。
- **Tempo drift（速度漂移）. ** Models wander off the BPM. Use BPM tags in the prompt and post-filter with librosa's `beat_track`.
  模型会偏离 BPM。在提示中使用 BPM 标签并用 librosa 的 `beat_track` 做后过滤。
- **Vocal intelligibility（人声可懂度）. ** Suno is excellent; open models are often mushy on words. If lyrics matter, use a commercial API or fine-tune.
  Suno 很棒；开源模型在 words 上常常模糊。如果歌词重要，使用商业 API 或微调。
- **Mono output（单声道输出）. ** Open models generate mono or fake-stereo. Upgrade with a proper stereo reconstruction (ezst, Cartesia's stereo diffusion).
  开源模型生成单声道或假立体声。用适当的立体声重建升级（ezst、Cartesia 的立体声扩散）。

## Ship It（交付成果）

Save as `outputs/skill-music-designer.md`. Pick model, license strategy, length / structure plan, and disclosure metadata for a music-gen deployment.
保存为 `outputs/skill-music-designer.md`。为音乐生成部署选择模型、许可证策略、长度 / 结构计划和披露元数据。

## Exercises（练习）

1. **Easy（简单）.** Run `code/main.py`. It produces a "generative" chord progression + drum pattern as ASCII symbols — a music-gen cartoon. Play it back via any MIDI renderer if you want.
   运行 `code/main.py`。它生成一个"生成式"和弦进行 + 鼓模式，以 ASCII 符号表示——一个音乐生成卡通。如果需要，可以通过任何 MIDI 渲染器回放。
2. **Medium（中等）.** Install `audiocraft`, generate 10-second clips across 4 genre prompts with MusicGen-small, measure FAD against a reference genre set.
   安装 `audiocraft`，用 MusicGen-small 在 4 个流派提示上生成 10 秒片段，对照参考流派集测量 FAD。
3. **Hard（困难）.** Using ACE-Step (or MusicGen-melody), generate three variations of the same tune with different timbre prompts. Compute CLAP similarity to the prompt to verify alignment.
   用 ACE-Step（或 MusicGen-melody），用不同的音色提示生成同一旋律的三个变体。计算与提示的 CLAP 相似度以验证对齐。

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| FAD | Audio FID | Fréchet distance between embedding distributions of real vs generated. |
|     | 音频 FID | 真实与生成嵌入分布之间的 Fréchet 距离。 |
| Chromagram | Melody as pitches | 12-dim per-frame vector; input to melody conditioning. |
|            | 旋律作为音高 | 每帧 12 维向量；旋律条件的输入。 |
| Stems | Instrument tracks | Separated bass / drums / vocals / melody as WAV. |
|       | 乐器音轨 | 分离的贝斯 / 鼓 / 人声 / 旋律，以 WAV 格式。 |
| Inpainting | Regen a section | Mask a time window; model regenerates just that. |
|             | 重新生成一个段落 | 掩码一个时间窗口；模型只重新生成该部分。 |
| CLAP | Text-audio CLIP | Contrastive audio-text embedding; eval text-audio alignment. |
|     | 文本-音频 CLIP | 对比音频-文本嵌入；评估文本-音频对齐。 |
| EnCodec | Music codec | Meta's neural codec used by MusicGen; 32 kHz, 4 codebooks. |
|         | 音乐编码器 | Meta 的神经编码器，MusicGen 使用；32 kHz，4 个码本。 |

## Further Reading（延伸阅读）

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — the open autoregressive benchmark.
  Copet 等 (2023).《MusicGen》——开源自回归基准。
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — the sound-design default.
  Evans 等 (2024).《Stable Audio Open》——声音设计默认选择。
- [ACE-Step](https://github.com/ace-step/ACE-Step) — open 4B full-song generator, April 2026.
  ACE-Step——开源 4B 全歌曲生成器，2026 年 4 月。
- [Suno v5 platform docs](https://suno.com) — the commercial quality leader.
  Suno v5 平台文档——商业质量领导者。
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — latent diffusion for music + sound effects.
  AudioLDM2——用于音乐 + 音效的隐空间扩散。
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — Nov 2025 precedent.
  WMG-Suno 和解报道——2025 年 11 月先例。
