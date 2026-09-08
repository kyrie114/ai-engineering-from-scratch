# Real-Time Audio Processing（实时音频处理）

> Batch pipelines process a file. Real-time pipelines process the next 20 milliseconds before the next 20 arrive. Every conversational AI, broadcast studio, and telephony bot lives and dies by this latency budget.
> 批处理流水线一次处理一个文件。实时流水线在下 20 毫秒到达之前处理当前这 20 毫秒。每个对话式 AI、广播工作室和电话机器人都在这个延迟预算下生生死死。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## The Problem（问题）

You want a voice assistant that feels alive. Human conversational turn-taking latency is ~230 ms (silence-to-response). Anything above 500 ms feels robotic; above 1500 ms feels broken. The budget for a full **hear → understand → respond → speak** loop in 2026 is:
你想要一个感觉活着的语音助手。人类对话轮换延迟约为 230 毫秒（静默到响应）。超过 500 毫秒感觉像机器人；超过 1500 毫秒感觉坏了。2026 年完整的**听 → 理解 → 响应 → 说**循环的预算是：

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

Moshi (Kyutai, 2024) clocked 200 ms full-duplex. GPT-4o-realtime (2024) clocks ~320 ms. Cascaded pipelines in 2022 shipped at 2500 ms. The 10× improvement came from three techniques: (1) streaming everywhere, (2) asynchronous pipelining with partial results, (3) interruptible generation.
Moshi（Kyutai，2024）全双工 200 毫秒。GPT-4o-realtime（2024）约 320 毫秒。2022 年的级联流水线发货时为 2500 毫秒。10 倍改进来自三个技术：(1) 处处流式传输，(2) 带部分结果的异步流水线，(3) 可中断生成。

## The Concept（概念）

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**Frame / chunk / window（帧 / 块 / 窗口）. ** Real-time audio flows as fixed-size blocks. Common choice: 20 ms (320 samples at 16 kHz). Everything downstream must keep up with this cadence.
实时音频以固定大小的块流动。常见选择：20 毫秒（16 kHz 下 320 个采样）。下游所有部分都必须跟上这个节奏。

**Ring buffer（环形缓冲区）. ** Fixed-size circular buffer. Producer thread writes new frames, consumer thread reads. Prevents allocations in the hot path. Size ≈ maximum-latency × sample-rate; a 2-second 16 kHz ring = 32,000 samples.
固定大小的循环缓冲区。生产者线程写入新帧，消费者线程读取。防止热路径中的分配。大小 ≈ 最大延迟 × 采样率；2 秒 16 kHz 环 = 32,000 个采样。

**VAD (Voice Activity Detection)（VAD（语音活动检测）). ** Gates downstream work when nobody is speaking. Silero VAD 4.0 (2024) runs <1 ms per 30 ms frame on CPU. `webrtcvad` is the older alternative.
当没人说话时门控下游工作。Silero VAD 4.0 (2024) 在 CPU 上每 30 毫秒帧运行不到 1 毫秒。`webrtcvad` 是更老的替代品。

**Streaming ASR（流式 ASR）. ** Models that emit partial transcripts as audio arrives. Parakeet-CTC-0.6B in streaming mode (NeMo, 2024) does 2–5% WER at 320 ms latency. Whisper-Streaming (Macháček et al., 2023) chunks Whisper for near-streaming at ~2 s latency.
随着音频到达输出部分转录的模型。流式模式下的 Parakeet-CTC-0.6B（NeMo，2024）在 320 毫秒延迟下实现 2–5% WER。Whisper-Streaming（Macháček 等，2023）把 Whisper 分块以实现约 2 秒延迟的近流式传输。

**Interruption（打断）. ** When the user speaks while the assistant is talking, you must (a) detect the barge-in, (b) stop the TTS, (c) discard the remaining LLM output. All within 100 ms, or the user perceives deaf assistant.
当用户在助手说话时说话，你必须 (a) 检测到抢话，(b) 停止 TTS，(c) 丢弃剩余的 LLM 输出。全部在 100 毫秒内，否则用户会感知到聋助手。

**WebRTC Opus transport（WebRTC Opus 传输）. ** 20 ms frames, 48 kHz, adaptive bitrate 8–128 kbps. Standard for browser and mobile. LiveKit, Daily.co, Pion are the 2026 stacks for building voice apps.
20 毫秒帧，48 kHz，自适应比特率 8–128 kbps。浏览器和移动端的标准。LiveKit、Daily.co、Pion 是 2026 年构建语音应用的栈。

**Jitter buffer（抖动缓冲区）. ** Network packets arrive out of order / late. The jitter buffer reorders and smooths; too small → audible gaps, too large → latency. 60–80 ms typical.
网络包乱序 / 迟到。抖动缓冲区重排序和平滑；太小 → 可听见的间隙，太大 → 延迟。典型 60–80 毫秒。

### Common gotchas（常见陷阱）

- **Thread contention（线程争用）. ** Python's GIL + heavy models can starve the audio thread. Use a C-callback audio library (sounddevice, PortAudio) and keep Python off the hot path.
  Python 的 GIL + 重型模型会饿死音频线程。使用带 C 回调的音频库（sounddevice、PortAudio）并让 Python 远离热路径。
- **Sample-rate conversion latency（采样率转换延迟）. ** Resampling inside the pipeline adds 5–20 ms. Either resample upfront or use a zero-latency resampler (PolyPhase, `soxr_hq`).
  流水线内部的重新采样增加 5–20 毫秒。要么提前重采样，要么使用零延迟重采样器（PolyPhase、`soxr_hq`）。
- **TTS priming（TTS 预热）. ** Even fast TTS like Kokoro has a 100–200 ms warm-up on first request. Cache model + warm it with a dummy run before the first real turn.
  即使像 Kokoro 这样快的 TTS，在第一次请求时也有 100–200 毫秒的预热。缓存模型 + 在第一次真实轮次前用虚拟运行预热。
- **Echo cancellation（回声消除）. ** Without AEC, TTS output re-enters the mic and triggers ASR on the bot's own voice. WebRTC AEC3 is the open-source default.
  没有 AEC，TTS 输出会重新进入麦克风并触发 ASR 对机器人自己的声音进行识别。WebRTC AEC3 是开源默认选择。

```figure
nyquist-aliasing
```

## Build It（动手实现）

### Step 1: ring buffer（环形缓冲区）

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf))]
    def level(self):
        return len(self.buf)
```

Capacity determines max buffering latency. 32,000 samples at 16 kHz = 2 s.
容量决定最大缓冲延迟。16 kHz 下 32,000 个采样 = 2 秒。

### Step 2: VAD gate（VAD 门控）

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

Replace with Silero VAD in production:
生产环境中替换为 Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Step 3: streaming ASR（流式 ASR）

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Step 4: interruption handler（打断处理器）

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

Hinges on async I/O and cancellable TTS streaming. WebRTC peerconnection.stop() on the audio track is the canonical way.
关键在于异步 I/O 和可取消的 TTS 流式传输。WebRTC peerconnection.stop() 在音频轨道上是规范做法。

## Use It（实际应用）

The 2026 stack:
2026 年栈：

| Layer | Pick |
|-------|------|
| Transport | LiveKit (WebRTC) or Pion (Go) |
|         | 传输 |
| VAD | Silero VAD 4.0 |
|    | VAD |
| Streaming ASR | Parakeet-CTC-0.6B or Whisper-Streaming |
|               | 流式 ASR |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
|                 | LLM 首 token |
| Streaming TTS | Kokoro or ElevenLabs Turbo v2.5 |
|               | 流式 TTS |
| Echo cancel | WebRTC AEC3 |
|             | 回声消除 |
| End-to-end native | OpenAI Realtime API or Moshi |
|                   | 端到端原生 |

## Pitfalls（陷阱）

- **Buffering 500 ms to be safe（缓冲 500 毫秒求安全）. ** The buffer *is* your latency floor. Shrink it.
  缓冲区*就是*你的延迟下限。缩小它。
- **Not pinning threads（不固定线程）. ** Audio callback on a priority-lower-than-UI thread = glitches under load.
  音频回调在优先级低于 UI 的线程上 = 负载下出现故障。
- **TTS chunks too small（TTS 块太小）. ** Sub-200 ms chunks make vocoder artifacts audible. 320 ms chunks are the sweet spot.
  小于 200 毫秒的块会让声码器伪影可听见。320 毫秒块是最佳点。
- **No jitter buffer（无抖动缓冲区）. ** Real networks are jittery; without smoothing you get pops.
  真实网络是抖动的；没有平滑就会有 pop 声。
- **Single-shot error handling（单次错误处理）. ** Audio pipelines must be crash-proof. One exception kills the session.
  音频流水线必须是防崩溃的。一个异常就会杀死会话。

## Ship It（交付成果）

Save as `outputs/skill-realtime-designer.md`. Design a real-time audio pipeline with concrete latency budgets per stage.
保存为 `outputs/skill-realtime-designer.md`。设计一个每阶段有具体延迟预算的实时音频流水线。

## Exercises（练习）

1. **Easy（简单）.** Run `code/main.py`. Simulates a ring buffer + energy VAD; prints stage latencies for a fake 10-second stream.
   运行 `code/main.py`。模拟环形缓冲区 + 能量 VAD；打印一个假 10 秒流的阶段延迟。
2. **Medium（中等）.** Using `sounddevice`, build a passthrough loop that processes your mic in 20 ms frames and prints VAD state at each frame.
   使用 `sounddevice`，构建一个直通循环，以 20 毫秒帧处理你的麦克风并在每帧打印 VAD 状态。
3. **Hard（困难）.** Build a full duplex echo test with `aiortc`: browser → WebRTC → Python → WebRTC → browser. Measure glass-to-glass latency with a 1 kHz pulse.
   用 `aiortc` 构建一个全双工回声测试：browser → WebRTC → Python → WebRTC → browser。用 1 kHz 脉冲测量玻璃到玻璃延迟。

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Ring buffer | The circular queue | Fixed-size, lock-free (or SPSC-locked) FIFO for audio frames. |
|             | 循环队列 | 用于音频帧的固定大小、无锁（或 SPSC 锁）FIFO。 |
| VAD | Silence gate | Model or heuristic marking speech vs non-speech. |
|    | 静默门 | 标记语音 vs 非语音的模型或启发式方法。 |
| Streaming ASR | Real-time STT | Emits partial text as audio arrives; bounded lookahead. |
|                | 实时 STT | 随着音频到达输出部分文本；有界前向。 |
| Jitter buffer | Network smoother | Queue reordering out-of-order packets; 60–80 ms typical. |
|               | 网络平滑器 | 重排序乱序包的队列；典型 60–80 毫秒。 |
| AEC | Echo cancellation | Subtracts speaker-to-mic feedback path. |
|    | 回声消除 | 减去扬声器到麦克风的反馈路径。 |
| Barge-in | User interrupt | System detects user speech mid-TTS; must cancel playback. |
|          | 用户打断 | 系统在 TTS 中途检测到用户语音；必须取消播放。 |
| Full duplex | Simultaneous both ways | User and bot can talk at the same time; Moshi is full duplex. |
|              | 同时双向 | 用户和机器人可以同时说话；Moshi 是全双工的。 |

## Further Reading（延伸阅读）

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — chunked near-streaming Whisper.
  Macháček 等 (2023).《Whisper-Streaming》——分块近流式 Whisper。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — full-duplex 200 ms latency.
  Kyutai (2024).《Moshi》——全双工 200 毫秒延迟。
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — production audio agent orchestration.
  LiveKit Agents 框架 (2024)——生产级音频智能体编排。
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — sub-1 ms VAD, Apache 2.0.
  Silero VAD 仓库——亚 1 毫秒 VAD，Apache 2.0。
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — echo cancellation under open source.
  WebRTC AEC3 论文——开源回声消除。
