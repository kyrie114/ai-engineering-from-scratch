# Voice Cloning & Voice Conversion（声音克隆与声音转换）

> Voice cloning reads your text in someone else's voice. Voice conversion rewrites your voice into someone else's while preserving what you said. Both hang on the same decomposition: separate speaker identity from content.
> 声音克隆用别人的声音朗读你的文本。声音转换把你的声音改写成别人的声音，同时保留你说的内容。两者都依赖于同一个分解：把说话人身份和内容分开。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## The Problem（问题）

In 2026, a 5-second audio clip is enough to produce a high-quality clone of anyone's voice with a consumer GPU. ElevenLabs, F5-TTS, OpenVoice v2, VoiceBox all ship zero-shot or few-shot cloning. The technology is a blessing (accessibility TTS, dubbing, assistive voices) and a weapon (scam calls, political deepfakes, IP theft).
2026 年，一段 5 秒的音频片段就足以用消费级 GPU 克隆任何人的高质量声音。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox 都提供零样本或少样本克隆。这项技术既是福音（无障碍 TTS、配音、辅助声音）也是武器（诈骗电话、政治 deepfake、知识产权盗窃）。

Two closely-related tasks:
两个紧密相关的任务：

- **Voice cloning (TTS-side)（声音克隆（TTS 侧）): ** text + 5-second reference voice → audio in that voice.
  文本 + 5 秒参考声音 → 用该声音生成的音频。
- **Voice conversion (speech-side)（声音转换（语音侧）): ** source audio (person A saying X) + reference voice of person B → audio of B saying X.
  源音频（A 说 X）+ B 的参考声音 → B 说 X 的音频。

Both factor a waveform into (content, speaker, prosody) and recombine content from one source with speaker from another.
两者都把波形分解为（内容、说话人、韵律），并将一个来源的内容与另一个来源的说话人重新组合。

Key constraint you now ship under in 2026: **watermarking and consent gates are legally required in the EU (AI Act, enforceable August 2026) and in California (AB 2905, effective 2025)**. Your pipeline must emit an inaudible watermark and refuse non-consensual clones.
2026 年你必须遵守的关键约束：**欧盟（AI 法案，2026 年 8 月可执行）和加利福尼亚州（AB 2905，2025 年生效）法律规定必须加水印和同意门控**。你的流水线必须发出不可听见的水印并拒绝非授权克隆。

## The Concept（概念）

![Voice cloning vs conversion: factorize, swap speaker, recombine](../assets/voice-cloning.svg)

**Zero-shot cloning（零样本克隆）. ** Pass a 5-second clip to a model that has been trained on thousands of speakers. The speaker encoder maps the clip to a speaker embedding; the TTS decoder conditions on that embedding plus text.
把一段 5 秒的片段传给一个在数千个说话人上训练过的模型。说话人编码器把片段映射为说话人嵌入；TTS 解码器以该嵌入加文本为条件。

Used by: F5-TTS (2024), YourTTS (2022), XTTS v2 (2024), OpenVoice v2 (2024).
使用者：F5-TTS (2024)、YourTTS (2022)、XTTS v2 (2024)、OpenVoice v2 (2024)。

**Few-shot fine-tuning（少样本微调）. ** Record 5-30 minutes of the target voice. LoRA-fine-tune a base model for an hour. Quality leaps from "okay" to "indistinguishable". Coqui and ElevenLabs both support this pattern; community uses it with F5-TTS.
录制 5-30 分钟目标声音。对基础模型做一小时的 LoRA 微调。质量从"还行"飞跃到" indistinguishable"。Coqui 和 ElevenLabs 都支持这种模式；社区用 F5-TTS 这样做。

**Voice conversion (VC)（声音转换 (VC)）. ** Two families:
两个家族：

- **Recognition-synthesis（识别-合成）. ** Run ASR-like model to extract content representation (e.g., soft phoneme posteriors, PPGs), then resynthesize with target speaker embedding. Robust to language and accent. Used by KNN-VC (2023), Diff-HierVC (2023).
  运行类似 ASR 的模型提取内容表示（例如软音素后验、PPG），然后用目标说话人嵌入重新合成。对语言和口音鲁棒。使用者：KNN-VC (2023)、Diff-HierVC (2023)。
- **Disentanglement（解耦）. ** Train an autoencoder that separates content, speaker, and prosody in latent space at the bottleneck. Swap speaker embedding at inference. Lower quality but faster. Used by AutoVC (2019), VITS-VC variants.
  训练一个自编码器，在瓶颈处的隐空间分离内容、说话人和韵律。推理时交换说话人嵌入。质量较低但更快。使用者：AutoVC (2019)、VITS-VC 变体。

**Neural codec-based cloning (2024+)（基于神经编码器的克隆（2024 年及以后）). ** VALL-E, VALL-E 2, NaturalSpeech 3, VoiceBox — treat audio as discrete tokens from SoundStream / EnCodec, train a large autoregressive or flow-matching model over codec tokens. Quality comparable to ElevenLabs on short prompts.
VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox——把音频视为 SoundStream / EnCodec 的离散 token，在编码器 token 上训练一个大型自回归或流匹配模型。在短提示上的质量与 ElevenLabs 相当。

### The ethics bit, not a bolt-on（伦理部分，不是附加组件）

**Watermarking（水印）. ** PerTh (Perth) and SilentCipher (2024) embed a ~16-32 bit ID imperceptibly in the audio. Survives re-encoding, streaming, and common edits. Production-ready open source.
PerTh (Perth) 和 SilentCipher (2024) 在音频中嵌入一个约 16-32 位的 ID，人耳不可察觉。在重编码、流媒体和常见编辑中存活。生产级开源。

**Consent gates（同意门控）. ** Must pair every cloned output with a verifiable consent record. "I, Rohit, on 2026-04-22, authorize this voice for X purpose." Store in a tamper-evident log.
每个克隆输出必须配上一个可验证的同意记录。"我，Rohit，于 2026-04-22，授权此声音用于 X 目的。"存储在防篡改日志中。

**Detection（检测）. ** AASIST, RawNet2, and Wav2Vec2-AASIST ship as detectors. ASVspoof 2025 challenge published EERs of 0.8–2.3% for state-of-the-art detectors against ElevenLabs, VALL-E 2, and Bark outputs.
AASIST、RawNet2 和 Wav2Vec2-AASIST 作为检测器发货。ASVspoof 2025 挑战赛发布了针对 ElevenLabs、VALL-E 2 和 Bark 输出的 SOTA 检测器的 EER，为 0.8–2.3%。

### Numbers (2026)（数字 (2026)）

| Model | Zero-shot? | SECS (target sim) | WER (intel.) | Params |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | Yes | 0.72 | 2.1% | 335M |
| XTTS v2 | Yes | 0.65 | 3.5% | 470M |
| OpenVoice v2 | Yes | 0.70 | 2.8% | 220M |
| VALL-E 2 | Yes | 0.77 | 2.4% | 370M |
| VoiceBox | Yes | 0.78 | 2.1% | 330M |

SECS > 0.70 is generally indistinguishable from the target for most listeners.
对大多数听众来说，SECS > 0.70 通常与目标无法区分。

```figure
sp-voice-factorize
```

## Build It（动手实现）

### Step 1: decompose with recognition-synthesis (code-only demo in main.py)（用识别-合成分解（仅在 main.py 中的代码演示））

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

Conceptually simple; implementation mass is in `tts_model` and speaker encoder.
概念上简单；实现重量在 `tts_model` 和说话人编码器中。

### Step 2: zero-shot clone with F5-TTS（用 F5-TTS 做零样本克隆）

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

Reference transcript must exactly match the audio; mismatch breaks alignment.
参考转录文本必须与音频完全匹配；不匹配会破坏对齐。

### Step 3: voice conversion with KNN-VC（用 KNN-VC 做声音转换）

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC runs WavLM to extract per-frame embeddings for source and target pool, then replaces each source frame with its nearest neighbor in the pool. Non-parametric, works with a minute of target speech.
KNN-VC 运行 WavLM 为源和目标池提取每帧嵌入，然后用池中的最近邻替换每个源帧。非参数化，用一分钟的目标语音即可工作。

### Step 4: embed a watermark（嵌入水印）

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

~32 bits of payload, detectable after MP3 re-encode and light noise.
约 32 位有效载荷，在 MP3 重编码和轻度噪声后可检测。

### Step 5: consent gate（同意门控）

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## Use It（实际应用）

The 2026 stack:
2026 年栈：

| Situation | Pick |
|-----------|------|
| 5-sec zero-shot clone, open-source | F5-TTS or OpenVoice v2 |
|                                  | 5 秒零样本克隆，开源 |
| Commercial production cloning | ElevenLabs Instant Voice Clone v2.5 |
|                               | 商业生产级克隆 |
| Voice conversion (rewriting) | KNN-VC or Diff-HierVC |
|                               | 声音转换（重写） |
| Many-speaker fine-tune | StyleTTS 2 + speaker adapter |
|                        | 多说话人微调 |
| Cross-lingual cloning | XTTS v2 or VALL-E X |
|                       | 跨语言克隆 |
| Deepfake detection | Wav2Vec2-AASIST |
|                    | Deepfake 检测 |

## Pitfalls（陷阱）

- **Misaligned reference transcript（对齐错误的参考转录）. ** F5-TTS and similar require the reference text to match the reference audio exactly, punctuation included.
  F5-TTS 及类似模型要求参考文本与参考音频完全匹配，包括标点。
- **Reverberant reference（混响参考）. ** Echo kills the clone. Record dry, close-mic.
  回声会毁掉克隆。录制干声，近距离麦克风。
- **Emotional mismatch（情感不匹配）. ** Training reference "cheerful" produces cheerful clones of everything. Match reference emotion to target use.
  训练参考"cheerful" 会把所有内容都克隆成 cheerful。让参考情感匹配目标用途。
- **Language leakage（语言泄露）. ** Cloning an English speaker then asking the model to speak French often carries the accent anyway; use cross-lingual models (XTTS, VALL-E X).
  克隆一个英语说话人然后让模型说法语，口音往往还是会带过去；使用跨语言模型（XTTS、VALL-E X）。
- **No watermark（无水印）. ** Legally unshippable in EU from Aug 2026.
  2026 年 8 月起在欧盟无法合法发货。

## Ship It（交付成果）

Save as `outputs/skill-voice-cloner.md`. Design a cloning or conversion pipeline with consent gate + watermark + quality target.
保存为 `outputs/skill-voice-cloner.md`。设计一个带同意门控 + 水印 + 质量目标的克隆或转换流水线。

## Exercises（练习）

1. **Easy（简单）.** Run `code/main.py`. Demonstrates the speaker-embedding swap by computing the cosine between two "speakers" pre and post swap.
   运行 `code/main.py`。通过计算交换前后两个"说话人"之间的余弦来演示说话人嵌入交换。
2. **Medium（中等）.** Use OpenVoice v2 to clone your own voice. Measure SECS between reference and clone. Measure CER via Whisper.
   用 OpenVoice v2 克隆你自己的声音。测量参考和克隆之间的 SECS。通过 Whisper 测量 CER。
3. **Hard（困难）.** Apply SilentCipher watermark to 20 clones, run them through 128 kbps MP3 encode+decode, detect the payload. Report bit-accuracy.
   对 20 个克隆应用 SilentCipher 水印，让它们经过 128 kbps MP3 编码+解码，检测有效载荷。报告比特准确率。

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Zero-shot clone | 5 seconds is enough | Pretrained model + speaker embedding; no training. |
|                 | 5 秒就够 | 预训练模型 + 说话人嵌入；无需训练。 |
| PPG | Phonetic posteriorgram | Per-frame ASR posteriors used as language-agnostic content rep. |
|    | 音素后验gram | 每帧 ASR 后验，用作语言无关内容表示。 |
| KNN-VC | Nearest-neighbor conversion | Replace each source frame with nearest target-pool frame. |
|        | 最近邻转换 | 用目标池中最近邻帧替换每个源帧。 |
| Neural codec TTS | VALL-E style | AR model over EnCodec/SoundStream tokens. |
|                  | VALL-E 风格 | 在 EnCodec/SoundStream token 上的自回归模型。 |
| Watermark | Inaudible signature | Bits embedded in audio, survive re-encode. |
|           | 不可听签名 | 嵌入音频的比特，经重编码后存活。 |
| SECS | Cloning fidelity | Cosine between target and clone speaker embeddings. |
|     | 克隆保真度 | 目标和克隆说话人嵌入之间的余弦。 |
| AASIST | Deepfake detector | Anti-spoof model; detects synthesized speech. |
|        | Deepfake 检测器 | 反欺骗模型；检测合成语音。 |

## Further Reading（延伸阅读）

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — open-source SOTA zero-shot cloning.
  Chen et al. (2024).《F5-TTS》——开源 SOTA 零样本克隆。
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) and [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — neural-codec TTS.
  Baevski 等 / Microsoft (2023).《VALL-E》和 VALL-E 2 (2024)——神经编码 TTS。
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — disentanglement-based voice conversion.
  Qian 等 (2019).《AutoVC》——基于解耦的声音转换。
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — retrieval-based VC.
  Baas、Waubert de Puiseau、Kamper (2023).《KNN-VC》——基于检索的 VC。
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — production-ready 32-bit audio watermark.
  SilentCipher (2024)——音频水印——生产级 32 位音频水印。
- [ASVspoof 2025 results](https://www.asvspoof.org/) — detector vs synthesizer arms race, updated 2026.
  ASVspoof 2025 结果——检测器 vs 合成器军备竞赛，2026 年更新。
