# Audio Evaluation — WER, MOS, UTMOS, MMAU, FAD, and the Open Leaderboards（音频评估——WER、MOS、UTMOS、MMAU、FAD 和公开排行榜）

> You cannot ship what you cannot measure. This lesson names the 2026 metrics for every audio task: ASR (WER, CER, RTFx), TTS (MOS, UTMOS, SECS, WER-on-ASR-round-trip), audio-language (MMAU, LongAudioBench), music (FAD, CLAP), and speaker (EER). Plus the leaderboards where you compare.
> 你无法发货你无法测量的东西。这节课列出了 2026 年每个音频任务的指标：ASR（WER、CER、RTFx）、TTS（MOS、UTMOS、SECS、ASR 往返 WER）、音频语言（MMAU、LongAudioBench）、音乐（FAD、CLAP）和说话人（EER）。外加你用来比较的排行榜。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 (Model Evaluation)
**Time:** ~60 minutes

## The Problem（问题）

Every audio task has multiple metrics, each measuring a different axis. Using the wrong metric is how you ship a model that looks great on your dashboard and terribly in production. The 2026 canonical list:
每个音频任务都有多个指标，每个测量不同的轴。使用错误的指标就是你如何发货一个在你的仪表板上看起来很棒但在生产中很糟糕的模型。2026 年规范列表：

| Task | Primary | Secondary |
|------|---------|-----------|
| ASR | WER | CER · RTFx · first-token latency |
|     | WER | CER · RTFx · 首 token 延迟 |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
|     | MOS / UTMOS | SECS · ASR 往返 WER · CER · TTFA |
| Voice cloning | SECS (ECAPA cosine) | MOS · CER |
|               | SECS (ECAPA 余弦) | MOS · CER |
| Speaker verification | EER | minDCF · FAR / FRR at operating point |
|                      | EER | minDCF · 工作点的 FAR / FRR |
| Diarization | DER | JER · speaker confusion |
|              | DER | JER · 说话人混淆 |
| Audio classification | top-1 · mAP | macro F1 · per-class recall |
|                        | top-1 · mAP | 宏 F1 · 每类召回 |
| Music generation | FAD | CLAP · listening panel MOS |
|                  | FAD | CLAP · 听力小组 MOS |
| Audio language model | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
|                       | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| Streaming S2S | latency P50/P95 | WER · MOS |
|               | 延迟 P50/P95 | WER · MOS |

## The Concept（概念）

![Audio evaluation matrix — metrics vs tasks vs 2026 leaderboards](../assets/eval-landscape.svg)

### ASR metrics（ASR 指标）

**WER (Word Error Rate)（WER（词错误率）). ** `(S + D + I) / N`. Lowercase, strip punctuation, normalize numbers before scoring. Use `jiwer` or OpenAI's `whisper_normalizer`. < 5% = human-parity read speech.
  `(S + D + I) / N`。小写，去除标点，评分前归一化数字。使用 `jiwer` 或 OpenAI 的 `whisper_normalizer`。< 5% = 人类水平朗读语音。

**CER (Character Error Rate)（CER（字符错误率）). ** Same formula, character-level. Used for tone languages (Mandarin, Cantonese) where word segmentation is ambiguous.
  相同公式，字符级。用于词分割模糊的声调语言（普通话、粤语）。

**RTFx (inverse real-time factor)（RTFx（实时因子倒数）). ** Audio seconds processed per wall-clock second. Higher is better. Parakeet-TDT hits 3380×. Whisper-large-v3 is ~30×.
  每秒处理的音频秒数。越高越好。Parakeet-TDT 达到 3380×。Whisper-large-v3 约 30×。

**First-token latency（首 token 延迟）. ** Wall-clock from audio input to first transcript token. Critical for streaming. Deepgram Nova-3: ~150 ms.
  从音频输入到第一个转录 token 的挂钟时间。对流式至关重要。Deepgram Nova-3：约 150 毫秒。

### TTS metrics（TTS 指标）

**MOS (Mean Opinion Score)（MOS（平均意见分）). ** 1-5 human rating. Gold standard but slow. Collect 20+ listeners per sample, 100+ samples per model.
  1-5 人类评分。黄金标准但慢。每个样本收集 20+ 个听众，每个模型 100+ 个样本。

**UTMOS (2022-2026)（UTMOS（2022-2026）). ** Learned MOS predictor. Correlates ~0.9 with human MOS on standard benchmarks. F5-TTS: UTMOS 3.95; ground truth: 4.08.
  学习的 MOS 预测器。在标准基准上与人类 MOS 相关约 0.9。F5-TTS：UTMOS 3.95；真实值：4.08。

**SECS (Speaker Encoder Cosine Similarity)（SECS（说话人编码器余弦相似度）). ** For voice cloning. ECAPA embedding cosine between reference and cloned output. > 0.75 = recognizable clone.
  用于声音克隆。参考和克隆输出之间的 ECAPA 嵌入余弦。> 0.75 = 可识别的克隆。

**WER-on-ASR-round-trip（ASR 往返 WER）. ** Run Whisper over TTS output, compute WER against the input text. Catches intelligibility regressions. 2026 SOTA: < 2% CER.
  在 TTS 输出上运行 Whisper，与输入文本计算 WER。捕获可懂度退化。2026 SOTA：< 2% CER。

**TTFA (time-to-first-audio)（TTFA（首音频时间）). ** Wall-clock latency. Kokoro-82M: ~100 ms; F5-TTS: ~1 s.
  挂钟延迟。Kokoro-82M：约 100 毫秒；F5-TTS：约 1 秒。

### Voice-cloning-specific（声音克隆特有）

**SECS + MOS + CER** as a triple. Cloning that scores high SECS but low MOS means timbre-right-but-unnatural; the opposite means natural voice but wrong speaker.
**SECS + MOS + CER** 作为三元组。高 SECS 但低 MOS 的克隆表示音色对但不自然；相反表示自然声音但错误的说话人。

### Speaker verification（说话人验证）

**EER (Equal Error Rate)（EER（等错误率）). ** The threshold where False Accept Rate equals False Reject Rate. ECAPA on VoxCeleb1-O: 0.87%.
  错误接受率等于错误拒绝率的阈值。ECAPA 在 VoxCeleb1-O 上：0.87%。

**minDCF (min Detection Cost)（minDCF（最小检测成本）). ** Weighted cost at a chosen operating point (often FAR=0.01). More production-relevant than EER.
  所选工作点的加权成本（通常 FAR=0.01）。比 EER 更与生产相关。

### Diarization（说话人分离）

**DER (Diarization Error Rate)（DER（说话人分离错误率）). ** `(FA + Miss + Confusion) / total_speaker_time`. Missed speech + false-alarm speech + speaker-confusion, each as a fraction. AMI meetings: DER ~10-20% is realistic. pyannote 3.1 + Precision-2 commercial: <10% DER on well-recorded audio.
  `(FA + Miss + Confusion) / total_speaker_time`。漏检语音 + 误报警语音 + 说话人混淆，各为一个分数。AMI 会议：DER 约 10-20% 是现实的。pyannote 3.1 + Precision-2 商业：在录得好的音频上 DER < 10%。

**JER (Jaccard Error Rate)（JER（Jaccard 错误率）). ** Alternative to DER, robust to short-segment bias.
  DER 的替代，对短段偏差鲁棒。

### Audio classification（音频分类）

Multi-label: **mAP (mean Average Precision)** over all classes. AudioSet: 0.548 mAP for BEATs-iter3.
多标签：所有类别上的 **mAP（平均精度均值）**。AudioSet：BEATs-iter3 的 mAP 为 0.548。

Multi-class exclusive: **top-1, top-5 accuracy**. Speech Commands v2: 99.0% top-1 (Audio-MAE).
多类独占：**top-1、top-5 准确率**。Speech Commands v2：top-1 99.0%（Audio-MAE）。

Imbalanced: **macro F1** + **per-class recall**. Report per-class — aggregate accuracy hides which classes fail.
不平衡：**宏 F1** + **每类召回**。报告每类——聚合准确率隐藏了哪些类失败。

### Music generation（音乐生成）

**FAD (Fréchet Audio Distance)（FAD（Fréchet 音频距离）). ** Distance between VGGish-embedding distributions of real vs generated audio. MusicGen-small on MusicCaps: 4.5. MusicLM: 4.0. Lower better.
  真实与生成音频的 VGGish 嵌入分布之间的距离。MusicGen-small 在 MusicCaps 上：4.5。MusicLM：4.0。越低越好。

**CLAP Score（CLAP 分数）. ** Text-audio alignment score using CLAP embeddings. > 0.3 = reasonable alignment.
  使用 CLAP 嵌入的文本-音频对齐分数。> 0.3 = 合理的对齐。

**Listening panel MOS（听力小组 MOS）. ** Still the final word for consumer-grade music. Suno v5 ELO 1293 on TTS Arena (from paired human preferences).
  对于消费级音乐仍然是最终判决。Suno v5 ELO 1293 在 TTS Arena 上（来自配对人类偏好）。

### Audio-language benchmarks（音频语言基准）

**MMAU (Massive Multi-Audio Understanding)（MMAU（大规模多音频理解）). ** 10k audio-QA pairs.
  1 万个音频 QA 对。

**MMAU-Pro（MMAU-Pro）. ** 1800 hard items, four categories: speech / sound / music / multi-audio. Random chance 25% on 4-way. Gemini 2.5 Pro overall ~60%; multi-audio ~22% across all models.
  1800 个难题，四个类别：语音 / 声音 / 音乐 / 多音频。4 选 1 随机概率 25%。Gemini 2.5 Pro 整体约 60%；所有模型多音频约 22%。

**LongAudioBench（LongAudioBench）. ** Multi-minute clips with semantic queries. Audio Flamingo Next beats Gemini 2.5 Pro.
  带语义查询的多分钟片段。Audio Flamingo Next 击败 Gemini 2.5 Pro。

**AudioCaps / Clotho（AudioCaps / Clotho）. ** Captioning benchmarks. SPICE, CIDEr, FENSE metrics.
  字幕基准。SPICE、CIDEr、FENSE 指标。

### Streaming speech-to-speech（流式语音到语音）

**Latency P50 / P95 / P99（延迟 P50 / P95 / P99）. ** Wall-clock from end-of-user-speech to first audible response. Moshi: 200 ms; GPT-4o Realtime: 300 ms.
  从用户语音结束到第一个可听见响应的挂钟时间。Moshi：200 毫秒；GPT-4o Realtime：300 毫秒。

**WER / MOS** on the output.
输出上的 **WER / MOS**。

**Barge-in responsiveness（抢话响应性）. ** Time from user interrupt to assistant mute. Target < 150 ms.
  从用户打断到助手静音的时间。目标 < 150 毫秒。

### The 2026 leaderboards（2026 年排行榜）

| Leaderboard | Tracks | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | English + multilingual + long-form | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
|                               | 英语 + 多语言 + 长格式 |
| TTS Arena (HF) | English TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
|                | 英语 TTS |
| Artificial Analysis Speech | TTS + STT, ELO from paired votes | `artificialanalysis.ai/speech` |
|                            | TTS + STT，来自配对投票的 ELO |
| MMAU-Pro | LALM reasoning | `mmaubenchmark.github.io` |
|          | LALM 推理 |
| SpeakerBench / VoxSRC | Speaker recognition | `voxsrc.github.io` |
|                       | 说话人识别 |
| MMAU music subset | Music LALM | (within MMAU) |
|                   | 音乐 LALM |
| HEAR benchmark | Self-supervised audio | `hearbenchmark.com` |
|               | 自监督音频 |

```figure
sp-wer-align
```

## Build It（动手实现）

### Step 1: WER with normalization（带归一化的 WER）

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### Step 2: TTS round-trip WER（TTS 往返 WER）

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### Step 3: SECS for voice cloning（声音克隆的 SECS）

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### Step 4: FAD for music generation（音乐生成的 FAD）

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### Step 5: EER for speaker verification (same code as Lesson 6)（说话人验证的 EER（与第 6 课代码相同））

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## Use It（实际应用）

Pair every deploy with a fixed eval harness that runs on every model update. Three cardinal rules:
为每次部署搭配一个在每次模型更新时运行的固定评估工具。三条基本规则：

1. **Normalize before scoring（评分前归一化）. ** Lowercase, punctuation-strip, number-expand. Report the normalization rule.
   小写，去除标点，数字扩展。报告归一化规则。
2. **Report distributions, not averages（报告分布，而非平均值）. ** P50/P95/P99 for latency. Per-class recall for classification. Per-category for MMAU.
   延迟的 P50/P95/P99。分类的每类召回。MMAU 的每类别。
3. **Run one canonical public benchmark（运行一个规范的公开基准）. ** Even if your production data differs, reporting on Open ASR / TTS Arena / MMAU lets reviewers compare apples-to-apples.
   即使你的生产数据不同，在 Open ASR / TTS Arena / MMAU 上报告也让审阅者可以 apples-to-apples 比较。

## Pitfalls（陷阱）

- **UTMOS extrapolation（UTMOS 外推）. ** Trained on VCTK-style clean speech; scores noisy / cloned / emotional audio poorly.
  在 VCTK 风格干净语音上训练；对嘈杂 / 克隆 / 情感语音评分差。
- **MOS panel bias（MOS 小组偏见）. ** 20 Amazon Mechanical Turk workers ≠ 20 target users. Pay for a domain panel if stakes are high.
  20 个 Amazon Mechanical Turk 工人 ≠ 20 个目标用户。如果赌注高，花钱找一个领域小组。
- **FAD depends on reference set（FAD 依赖参考集）. ** Compare against the same reference distribution across models.
  跨模型与同一参考分布比较。
- **Aggregate WER（聚合 WER）. ** A 5% WER overall can hide 30% WER on accented speech. Report by demographic slice.
  整体 5% WER 可能隐藏带口音语音的 30% WER。按人群切片报告。
- **Public benchmark saturation（公开基准饱和）. ** Most frontier models are near the ceiling on standard benchmarks. Build an in-house held-out set that reflects your traffic.
  大多数前沿模型在标准基准上接近上限。构建一个反映你流量的内部保留集。

## Ship It（交付成果）

Save as `outputs/skill-audio-evaluator.md`. Pick metrics, benchmarks, and reporting format for any audio model release.
保存为 `outputs/skill-audio-evaluator.md`。为任何音频模型发布选择指标、基准和报告格式。

## Exercises（练习）

1. **Easy（简单）.** Run `code/main.py`. Compute WER / CER / EER / SECS / FAD-ish / MMAU-ish on toy inputs.
   运行 `code/main.py`。在玩具输入上计算 WER / CER / EER / SECS / FAD-ish / MMAU-ish。
2. **Medium（中等）.** Build a TTS round-trip WER harness. Run your Kokoro or F5-TTS output through Whisper. Compute WER over 50 prompts. Flag prompts with WER > 10%.
   构建一个 TTS 往返 WER 工具。把你的 Kokoro 或 F5-TTS 输出通过 Whisper。在 50 个提示上计算 WER。标记 WER > 10% 的提示。
3. **Hard（困难）.** Score your Lesson 10 LALM choice on MMAU-Pro speech + multi-audio subsets (50 items each). Report per-category accuracy and compare with the published number.
   在 MMAU-Pro 语音 + 多音频子集（各 50 项）上给你的第 10 课 LALM 选择打分。报告每类准确率并与已发布数字比较。

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| WER | ASR score | `(S+D+I)/N` at word level after normalization. |
|     | ASR 分数 | 归一化后词级的 `(S+D+I)/N`。 |
| CER | Character WER | For tone languages or char-level systems. |
|     | 字符 WER | 用于声调语言或字符级系统。 |
| MOS | Human opinion | 1-5 rating; 20+ listeners × 100 samples. |
|    | 人类意见 | 1-5 评分；20+ 听众 × 100 样本。 |
| UTMOS | ML MOS predictor | Learned model; correlates ~0.9 with human MOS. |
|       | ML MOS 预测器 | 学习模型；与人类 MOS 相关约 0.9。 |
| SECS | Voice-clone similarity | ECAPA cosine between reference and clone. |
|     | 声音克隆相似度 | 参考和克隆之间的 ECAPA 余弦。 |
| EER | Speaker verif score | Threshold where FAR = FRR. |
|     | 说话人验证分数 | FAR = FRR 的阈值。 |
| DER | Diarization score | (FA + Miss + Confusion) / total. |
|     | 说话人分离分数 | (FA + Miss + Confusion) / 总计。 |
| FAD | Music-gen quality | Fréchet distance on VGGish embeddings. |
|     | 音乐生成质量 | VGGish 嵌入上的 Fréchet 距离。 |
| RTFx | Throughput | Audio seconds per wall-clock second. |
|     | 吞吐量 | 每秒挂钟的音频秒数。 |

## Further Reading（延伸阅读）

- [jiwer](https://github.com/jitsi/jiwer) — WER/CER library with normalization utilities.
  jiwer——带归一化工具的 WER/CER 库。
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — learned MOS predictor.
  UTMOS (Saeki 等 2022)——学习的 MOS 预测器。
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — the music-gen standard.
  Fréchet 音频距离 (Kilgour 等 2019)——音乐生成标准。
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026 live rankings.
  Open ASR 排行榜——2026 年实时排名。
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — human-vote TTS leaderboard.
  TTS Arena——人类投票 TTS 排行榜。
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM reasoning leaderboard.
  MMAU-Pro 基准——LALM 推理排行榜。
- [HEAR benchmark](https://hearbenchmark.com/) — audio SSL benchmarks.
  HEAR 基准——音频 SSL 基准。
