# Voice Anti-Spoofing & Audio Watermarking — ASVspoof 5, AudioSeal, WaveVerify（语音反欺骗与音频水印——ASVspoof 5、AudioSeal、WaveVerify）

> Voice cloning shipped faster than defenses. 2026 production voice systems need two things: a detector (AASIST, RawNet2) that classifies real vs fake speech, and a watermark (AudioSeal) that survives compression and editing. Ship both or do not ship voice cloning.
> 声音克隆的发布速度超过了防御。2026 年生产级语音系统需要两样东西：一个分类真实 vs 伪造语音的检测器（AASIST、RawNet2），以及一个能经受压缩和编辑的水印（AudioSeal）。两者都发货，否则不要发货声音克隆。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 08 (Voice Cloning)
**Time:** ~75 minutes

## The Problem（问题）

Three related defenses:
三个相关的防御：

1. **Anti-spoofing / deepfake detection（反欺骗 / deepfake 检测）. ** Given an audio clip, is it synthetic or real? ASVspoof benchmarks (ASVspoof 2019 → 2021 → 5) are the gold standard.
   给定一个音频片段，它是合成的还是真实的？ASVspoof 基准（ASVspoof 2019 → 2021 → 5）是黄金标准。
2. **Audio watermarking（音频水印）. ** Embed an imperceptible signal in generated audio that a detector can extract later. AudioSeal (Meta) and WavMark are the open options.
   在生成的音频中嵌入一个不可感知的信号，检测器以后可以提取。AudioSeal (Meta) 和 WavMark 是开源选项。
3. **Authenticated provenance（认证溯源）. ** Cryptographic signing of audio files + metadata. C2PA / Content Authenticity Initiative.
   音频文件的密码学签名 + 元数据。C2PA / 内容真实性倡议。

Detection handles adversaries who don't cooperate. Watermarking handles compliance — AI-generated audio should be identifiable as such. Both are required in 2026.
检测处理不合作的对手。水印处理合规——AI 生成的音频应该能被识别为如此。2026 年两者都是必需的。

## The Concept（概念）

![Anti-spoofing vs watermarking vs provenance — three defense layers](../assets/spoofing-watermark.svg)

### ASVspoof 5 — the 2024-2025 benchmark（ASVspoof 5——2024-2025 基准）

Biggest change from prior editions:
与之前版本最大的变化：

- **Crowdsourced data（众包数据）. ** (not studio clean) — realistic conditions.
  （不是演播室干净）——真实条件。
- **~2000 speakers（约 2000 个说话人）. ** (vs ~100 before).
  （之前约 100）。
- **32 attack algorithms（32 种攻击算法）. ** TTS + voice conversion + adversarial perturbation.
  TTS + 声音转换 + 对抗扰动。
- **Two tracks（两条赛道）. ** Countermeasure (CM) standalone detection; Spoofing-robust ASV (SASV) for biometric systems.
  对策（CM）独立检测；防欺骗 ASV（SASV）用于生物识别系统。

State-of-the-art on ASVspoof 5: ~7.23% EER. On the older ASVspoof 2019 LA: 0.42% EER. Real-world deployment: expect 5-10% EER on in-the-wild clips.
ASVspoof 5 上的 SOTA：约 7.23% EER。在更早的 ASVspoof 2019 LA 上：0.42% EER。真实世界部署：在真实场景片段上预期 5-10% EER。

### AASIST and RawNet2 — detection model families（AASIST 和 RawNet2——检测模型家族）

**AASIST** (2021, updated through 2026)（AASIST（2021，更新至 2026）). ** Graph-attention on spectral features. Current SOTA on ASVspoof 5 countermeasure task.
   光谱特征上的图注意力。ASVspoof 5 对策任务上的当前 SOTA。

**RawNet2（RawNet2）. ** Convolutional front-end over raw waveform + TDNN backbone. Simpler baseline; still competitive with fine-tuning.
   原始波形上的卷积前端 + TDNN 主干。更简单的基线；微调后仍然有竞争力。

**NeXt-TDNN + SSL features（NeXt-TDNN + SSL 特征）. ** 2025 variant: ECAPA-style + WavLM features + focal loss. Achieves the 0.42% EER on ASVspoof 2019 LA.
  2025 变体：ECAPA 风格 + WavLM 特征 + focal loss。在 ASVspoof 2019 LA 上达到 0.42% EER。

### AudioSeal — the 2024 watermark default（AudioSeal——2024 年水印默认）

Meta's **AudioSeal** (Jan 2024, v0.2 Dec 2024)（Meta 的 **AudioSeal**（2024 年 1 月，v0.2 2024 年 12 月）). Key design:
关键设计：

- **Localized（定位）. ** Detects the watermark per-frame at 16 kHz sample resolution (1/16000 s).
  以 16 kHz 采样分辨率（1/16000 秒）每帧检测水印。
- **Generator + detector jointly trained（生成器 + 检测器联合训练）. ** Generator learns to embed inaudible signal; detector learns to find it through augmentations.
  生成器学习嵌入不可听信号；检测器学习通过增强找到它。
- **Robust（鲁棒）. ** Survives MP3 / AAC compression, EQ, speed-shift ±10%, noise mix +10 dB SNR.
  经受 MP3 / AAC 压缩、EQ、速度偏移 ±10%、噪声混合 +10 dB SNR。
- **Fast（快）. ** Detector runs at 485× realtime; 1000× faster than WavMark.
  检测器以 485 倍实时运行；比 WavMark 快 1000 倍。
- **Capacity（容量）. ** 16-bit payload (can encode model ID, generation timestamp, user ID) embeddable in each utterance.
  16 位有效载荷（可以编码模型 ID、生成时间戳、用户 ID）可嵌入每次话语。

### WavMark（WavMark）

The pre-AudioSeal open baseline. Invertible neural network, 32 bits/sec. Problems:
AudioSeal 之前的开源基线。可逆神经网络，32 位/秒。问题：

- Synchronization brute-force is slow.
  同步暴力破解很慢。
- Can be removed by Gaussian noise or MP3 compression.
  可以被高斯噪声或 MP3 压缩去除。
- Not real-time friendly.
  对实时不友好。

### WaveVerify (July 2025)（WaveVerify（2025 年 7 月）)

Addresses AudioSeal's weaknesses — specifically temporal manipulations (reversal, speed). Uses FiLM-based generator + Mixture-of-Experts detector. Competitive with AudioSeal on standard attacks; handles temporal edits.
解决 AudioSeal 的弱点——特别是时间操作（反转、速度）。使用基于 FiLM 的生成器 + 专家混合检测器。在标准攻击上与 AudioSeal 竞争；处理时间编辑。

### The gap adversaries exploit（对手利用的缺口）

From AudioMarkBench: "under pitch shift, all watermarks show Bit Recovery Accuracy below 0.6, indicating near-complete removal." **Pitch-shift is the universal attack.** No 2026 watermark is fully robust to aggressive pitch modification. This is why you need detection (AASIST) alongside watermarking.
来自 AudioMarkBench："在音高偏移下，所有水印的比特恢复准确率都低于 0.6，表示几乎完全去除。"**音高偏移是通用攻击。** 2026 年没有水印能完全鲁棒于激进的音高修改。这就是为什么你需要检测（AASIST）配合水印。

### C2PA / Content Authenticity Initiative（C2PA / 内容真实性倡议）

Not an ML technique — a manifest format. Audio files carry cryptographically signed metadata about creation tool, author, date. Audobox / Seamless use it. Good for provenance; does nothing if a bad actor re-encodes and strips metadata.
不是 ML 技术——一个清单格式。音频文件携带关于创建工具、作者、日期的密码学签名元数据。Audobox / Seamless 使用它。对溯源有好处；如果坏演员重新编码并剥离元数据，它什么都不做。

```figure
v4-audio-watermark
```

## Build It（动手实现）

### Step 1: a simple spectral-feature detector (toy)（简单的光谱特征检测器（玩具））

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

Synthetic speech often has unusually flat high-frequency energy. Production detectors use AASIST, not this. But the intuition holds.
合成语音通常有不寻常的平坦高频能量。生产级检测器使用 AASIST，不是这个。但直觉成立。

### Step 2: AudioSeal embed + detect（AudioSeal 嵌入 + 检测）

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### Step 3: evaluation — EER（评估——EER）

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### Step 4: the production integration（生产集成）

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

Every generation ships: (1) watermark, (2) signed manifest, (3) retention-policy-compliant audit log.
每次生成都发货：(1) 水印，(2) 签名清单，(3) 符合保留策略的审计日志。

## Use It（实际应用）

| Use case | Defense |
|----------|---------|
| Shipping TTS / voice cloning | AudioSeal embed on every output (non-negotiable) |
|                                   | 发货 TTS / 声音克隆 |
| Biometric voice unlock | AASIST + ECAPA ensemble; liveness challenge |
|                          | 生物识别语音解锁 |
| Call-center fraud detection | AASIST on 20% sample of incoming calls |
|                              | 呼叫中心欺诈检测 |
| Podcast authenticity | C2PA signing on upload, AudioSeal if AI-generated |
|                      | 播客真实性 |
| Research / training detectors | ASVspoof 5 train/dev/eval sets |
|                               | 研究 / 训练检测器 |

## Pitfalls（陷阱）

- **Watermark without detector ever running（水印从不运行检测器）. ** Pointless. Ship the detector in your CI.
  无意义。在 CI 中发货检测器。
- **Detection without calibration（检测不带校准）. ** AASIST trained on ASVspoof LA overfits; real-world accuracy drops. Calibrate on your domain.
  在 ASVspoof LA 上训练的 AASIST 过拟合；真实世界精度下降。在你的领域校准。
- **Pitch-shift gap（音高偏移缺口）. ** Aggressive pitch shift removes most watermarks. Have a detection fallback.
  激进的音高偏移去除大多数水印。有一个检测回退。
- **Metadata strip-and-rehost（元数据剥离和重新托管）. ** C2PA is trivially bypassable by re-encoding. Always add cryptographic + perceptual (watermark) defense together.
  C2PA 通过重新编码轻易绕过。始终一起添加密码学 + 感知（水印）防御。
- **Liveness as detection（活性作为检测）. ** Ask user to say a random phrase. Prevents replay attacks but not real-time cloning.
  要求用户说一个随机短语。防止重放攻击但不能防止实时克隆。

## Ship It（交付成果）

Save as `outputs/skill-spoof-defender.md`. Pick detection model, watermark, provenance manifest, and operational playbook for a voice-gen deployment.
保存为 `outputs/skill-spoof-defender.md`。为语音生成部署选择检测模型、水印、溯源清单和操作手册。

## Exercises（练习）

1. **Easy（简单）.** Run `code/main.py`. Toy detector + toy watermark embed/detect on synthetic audio.
   运行 `code/main.py`。合成音频上的玩具检测器 + 玩具水印嵌入/检测。
2. **Medium（中等）.** Install `audioseal`, embed a 16-bit payload in a TTS output, re-decode. Corrupt the audio with noise and measure Bit Recovery Accuracy.
   安装 `audioseal`，在 TTS 输出中嵌入一个 16 位有效载荷，重新解码。用噪声损坏音频并测量比特恢复准确率。
3. **Hard（困难）.** Fine-tune a RawNet2 or AASIST on ASVspoof 2019 LA. Measure EER. Test on a held-out set of F5-TTS-generated clips — see how OOD detection degrades.
   在 ASVspoof 2019 LA 上微调 RawNet2 或 AASIST。测量 EER。在保留的 F5-TTS 生成片段集上测试——看看 OOD 检测如何退化。

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| ASVspoof | The benchmark | Biennial challenge; 2024 = ASVspoof 5. |
|        | 基准 | 双年挑战；2024 = ASVspoof 5。 |
| CM (countermeasure) | Detector | Classifier: real speech vs synthetic / converted. |
|                      | 检测器 | 分类器：真实语音 vs 合成 / 转换。 |
| SASV | Speaker verif + CM | Integrated biometric + spoof detection. |
|     | 说话人验证 + CM | 集成生物识别 + 欺骗检测。 |
| AudioSeal | Meta watermark | Localized, 16-bit payload, 485× faster than WavMark. |
|           | Meta 水印 | 定位，16 位有效载荷，比 WavMark 快 485 倍。 |
| Bit Recovery Accuracy | Watermark survival | Fraction of payload bits recovered after attack. |
|                        | 水印存活 | 攻击后恢复的有效载荷比特比例。 |
| C2PA | Provenance manifest | Cryptographic metadata about creation / authorship. |
|     | 溯源清单 | 关于创建 / 作者身份的密码学元数据。 |
| AASIST | Detector family | Graph-attention-based anti-spoofing SOTA. |
|        | 检测器家族 | 基于图注意力的反欺骗 SOTA。 |

## Further Reading（延伸阅读）

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — the current benchmark.
  Todisco 等 (2024).《ASVspoof 5》——当前基准。
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — the watermark default.
  Defossez 等 (2024).《AudioSeal》——水印默认。
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — MoE detector for temporal attacks.
  Chen 等 (2025).《WaveVerify》——用于时间攻击的 MoE 检测器。
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — the SOTA detection backbone.
  Jung 等 (2022).《AASIST》——SOTA 检测主干。
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — robustness evaluation.
  AudioMarkBench (2024)——鲁棒性评估。
- [C2PA specification](https://c2pa.org/specifications/specifications/) — provenance manifest format.
  C2PA 规范——溯源清单格式。
