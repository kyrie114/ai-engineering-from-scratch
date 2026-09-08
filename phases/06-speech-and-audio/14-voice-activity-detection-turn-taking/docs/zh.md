# Voice Activity Detection & Turn-Taking — Silero, Cobra, and the Flush Trick（语音活动检测与轮次切换——Silero、Cobra 和 Flush 技巧）

> Every voice agent lives or dies on two decisions: is the user speaking now, and are they done? VAD answers the first. Turn-detection (VAD + silence-hangover + semantic endpoint model) answers the second. Get either wrong and your assistant either cuts users off or never shuts up.
> 每个语音 agent 都生死于两个决策：用户现在在说话吗？他们说完了吗？VAD 回答第一个。轮次检测（VAD + 静默悬停 + 语义端点模型）回答第二个。任何一个决策错了，你的助手要么打断用户，要么没完没了。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 11 (Real-Time Audio), Phase 6 · 12 (Voice Assistant)
**Time:** ~45 minutes

## The Problem（问题）

Three distinct decisions a voice agent makes on every 20 ms chunk:
语音 agent 在每个 20 毫秒块上做出三个不同的决策：

1. **Is this frame speech?（这一帧是语音吗？). ** — VAD. Binary, per-frame.
   ——VAD。二值，每帧。
2. **Has the user started a new utterance?（用户开始新的话语了吗？). ** — onset detection.
   ——起始检测。
3. **Has the user finished?（用户完成了吗？). ** — end-pointing (turn-end).
   ——端点检测（轮次结束）。

The naive answer (energy threshold) fails on any noise — traffic, keyboards, crowd babble. The 2026 answer: Silero VAD (open, deep-learned) + a turn-detection model (semantic endpointing) + a VAD-calibrated silence hangover.
 naive 答案（能量阈值）在任何噪声下都会失败——交通、键盘、人群嘈杂。2026 年答案：Silero VAD（开源，深度学习）+ 一个轮次检测模型（语义端点）+ 一个 VAD 校准的静默悬停。

## The Concept（概念）

![VAD cascade: energy → Silero → turn-detector → flush trick](../assets/vad-turn-taking.svg)

### The three-tier VAD cascade（三层 VAD 级联）

**Tier 1: energy gate（第 1 层：能量门）. ** Cheapest. Threshold RMS at -40 dBFS. Filters obvious silence but fires on any noise above the threshold.
最便宜。RMS 阈值 -40 dBFS。过滤明显的静默，但对阈值以上的任何噪声都会触发。

**Tier 2: Silero VAD（第 2 层：Silero VAD） (2020-2026, MIT). 1M parameters. Trained on 6000+ languages. Runs in ~1 ms per 30 ms chunk on a single CPU thread. 87.7% TPR at 5% FPR. The open-source default.
（2020-2026，MIT）。100 万参数。在 6000+ 种语言上训练。在单个 CPU 线程上每 30 毫秒块运行约 1 毫秒。5% FPR 下 TPR 87.7%。开源默认。

**Tier 3: semantic turn detector（第 3 层：语义轮次检测器）. ** LiveKit's turn-detection model (2024-2026) or your own small classifier. Distinguishes "pause mid-sentence" from "done talking." Uses linguistic context (intonation + recent words), not just silence.
LiveKit 的轮次检测模型 (2024-2026) 或你自己的小型分类器。区分"句子中间暂停"和"说完了"。使用语言上下文（语调 + 最近词），而不仅仅是静默。

### Key parameters and their defaults（关键参数及其默认值）

- **Threshold（阈值）. ** Silero outputs a probability; classify speech at > 0.5 (default) or > 0.3 (sensitive). Lower threshold = fewer first-word clips, more false positives.
  Silero 输出一个概率；在 > 0.5（默认）或 > 0.3（敏感）处分类为语音。更低阈值 = 更少首词截断，更多误报。
- **Minimum speech duration（最短语音时长）. ** Reject speech shorter than 250 ms — usually coughs or chair noise.
  拒绝短于 250 毫秒的语音——通常是咳嗽或椅子噪声。
- **Silence hangover (end-pointing)（静默悬停（端点）). ** After VAD returns to 0, wait 500-800 ms before declaring end-of-turn. Too short → interrupt user. Too long → feels sluggish.
  VAD 返回 0 后，等待 500-800 毫秒再声明轮次结束。太短 → 打断用户。太长 → 感觉迟钝。
- **Pre-roll buffer（预滚动缓冲）. ** Keep 300-500 ms of audio before VAD fires. Prevents "hey" being clipped.
  在 VAD 触发前保留 300-500 毫秒音频。防止"hey"被截断。

### The flush trick (Kyutai 2025)（Flush 技巧 (Kyutai 2025)）

Streaming STT models have a look-ahead delay (500 ms for Kyutai STT-1B, 2.5 s for STT-2.6B). Normally you'd wait that long after end-of-speech for the transcript. Flush trick: when VAD fires end-of-speech, **send a flush signal to the STT** that forces immediate output. STT processes at ~4× realtime, so the 500 ms buffer finishes in ~125 ms.
流式 STT 模型有前向延迟（Kyutai STT-1B 500 毫秒，STT-2.6B 2.5 秒）。通常你会在语音结束后等待那么长时间得到转录。Flush 技巧：当 VAD 触发语音结束时，**向 STT 发送一个 flush 信号**，强制立即输出。STT 以约 4 倍实时处理，因此 500 毫秒缓冲在约 125 毫秒内完成。

End-to-end: 125 ms VAD + flush STT = conversational latency.
端到端：125 毫秒 VAD + flush STT = 对话延迟。

### 2026 VAD comparison（2026 年 VAD 比较）

| VAD | TPR @ 5% FPR | Latency | License |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | commercial |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Silero is the right default. Cobra is the compliance / accuracy upgrade. Energy-only VAD has no place in 2026 production.
Silero 是正确的默认。Cobra 是合规 / 精度升级。纯能量 VAD 在 2026 年生产中没有位置。

```figure
sp-vad-cascade
```

## Build It（动手实现）

### Step 1: the energy gate（能量门）

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### Step 2: Silero VAD in Python（Python 中的 Silero VAD）

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### Step 3: turn-end state machine（轮次结束状态机）

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### Step 4: the flush trick skeleton（Flush 技巧骨架）

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT (Kyutai, Deepgram, AssemblyAI) must support flush for this to work. Whisper streaming does not — it's block-based and always waits for chunks.
STT（Kyutai、Deepgram、AssemblyAI）必须支持 flush 才能工作。Whisper streaming 不支持——它是基于块的，总是等待块。

## Use It（实际应用）

| Situation | VAD choice |
|-----------|-----------|
| Open, fast, general | Silero VAD |
|                           | 开源、快速、通用 |
| Commercial call center | Cobra VAD |
|                        | 商业呼叫中心 |
| On-device (phone) | Silero VAD ONNX |
|                   | 设备端（手机） |
| Research / diarization | pyannote segmentation |
|                         | 研究 / 说话人分离 |
| Zero-dependency fallback | WebRTC VAD (legacy) |
|                           | 零依赖回退 |
| Need turn-ending quality | Silero + LiveKit turn-detector layered |
|                           | 需要轮次结束质量 |

Rule of thumb: never ship energy-only VAD unless you really have no other option.
经验法则：除非你真的没有其他选择，否则永远不要发货纯能量 VAD。

## Pitfalls（陷阱）

- **Fixed threshold（固定阈值）. ** Works in quiet, fails in noisy. Either calibrate on-device or switch to Silero.
  安静时有效，嘈杂时失败。要么在设备上校准，要么切换到 Silero。
- **Too-short silence hangover（太短的静默悬停）. ** Agent interrupts mid-sentence. 500-800 ms is the sweet spot for conversational speech.
  Agent 在句子中间打断。500-800 毫秒是对话语音的最佳点。
- **Too-long hangover（太长悬停）. ** Feels sluggish. A/B test with target users.
  感觉迟钝。与目标用户做 A/B 测试。
- **No pre-roll buffer（无预滚动缓冲）. ** First 200-300 ms of user audio lost. Always keep a rolling pre-roll.
  用户音频的前 200-300 毫秒丢失。始终保留一个滚动的预滚动。
- **Ignoring semantic endpointing（忽略语义端点）. ** "Hmm, let me think..." contains long pauses. Users hate being cut off mid-thought. Use LiveKit's turn-detector or similar.
  "Hmm，让我想想..."包含长停顿。用户讨厌在思考中被切断。使用 LiveKit 的轮次检测器或类似产品。

## Ship It（交付成果）

Save as `outputs/skill-vad-tuner.md`. Pick VAD model, threshold, hangover, pre-roll, and turn-detection strategy for a workload.
保存为 `outputs/skill-vad-tuner.md`。为工作负载选择 VAD 模型、阈值、悬停、预滚动和轮次检测策略。

## Exercises（练习）

1. **Easy（简单）.** Run `code/main.py`. It simulates a speech + silence + speech + coughs sequence and tests three VAD tiers.
   运行 `code/main.py`。它模拟一个语音 + 静默 + 语音 + 咳嗽序列，并测试三个 VAD 层。
2. **Medium（中等）.** Install `silero-vad`, process a 5-min recording, tune threshold to minimize both first-word clips and false triggers. Report precision/recall.
   安装 `silero-vad`，处理一个 5 分钟的录音，调优阈值以最小化首词截断和误触发。报告精确率/召回率。
3. **Hard（困难）.** Build a mini turn-detector: Silero VAD + a 3-layer MLP on the last 10 words' embeddings (use sentence-transformers). Train on a hand-labeled turn-end dataset. Beat Silero-only by 10% F1.
   构建一个迷你轮次检测器：Silero VAD + 一个在最后 10 个词嵌入上的 3 层 MLP（使用 sentence-transformers）。在手工标注的轮次结束数据集上训练。F1 比仅 Silero 提高 10%。

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| VAD | Voice detector | Binary per-frame: is this speech? |
|     | 语音检测器 | 每帧二值：这是语音吗？ |
| Turn detection | End-pointing | VAD + silence-hangover + semantic endpoint. |
|                 | 端点检测 | VAD + 静默悬停 + 语义端点。 |
| Silence hangover | Wait-after-speech | Time to wait before declaring turn end; 500-800 ms. |
|                  | 说话后等待 | 声明轮次结束前等待的时间；500-800 毫秒。 |
| Pre-roll | Pre-speech buffer | Keep 300-500 ms audio before VAD fires. |
|          | 语音前缓冲 | 在 VAD 触发前保留 300-500 毫秒音频。 |
| Flush trick | Kyutai hack | VAD → flush-STT → 125 ms instead of 500 ms delay. |
|             | Kyutai 技巧 | VAD → flush-STT → 125 毫秒代替 500 毫秒延迟。 |
| Semantic endpoint | "Did they mean to stop?" | ML classifier that looks at words, not just silence. |
|                   | "他们是想停下来吗？" | 看词而不仅仅是静默的 ML 分类器。 |
| TPR @ FPR 5% | ROC point | Standard VAD benchmark; 87.7% for Silero, 50% WebRTC. |
|              | ROC 点 | 标准 VAD 基准；Silero 87.7%，WebRTC 50%。 |

## Further Reading（延伸阅读）

- [Silero VAD](https://github.com/snakers4/silero-vad) — the reference open VAD.
  Silero VAD——参考开源 VAD。
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — commercial accuracy leader.
  Picovoice Cobra VAD——商业精度领导者。
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — the sub-200 ms engineering trick.
  Kyutai——Unmute + flush 技巧——亚 200 毫秒工程技巧。
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — semantic endpointing in production.
  LiveKit——轮次检测——生产级语义端点。
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — the legacy baseline.
  WebRTC VAD——遗留基线。
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — diarization-grade segmentation.
  pyannote 分割——说话人分离级分割。
