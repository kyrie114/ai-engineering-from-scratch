# Build a Voice Assistant Pipeline — The Phase 6 Capstone（构建语音助手流水线——第 6 阶段毕业项目）

> Everything from lessons 01-11, stitched together. Build a voice assistant that listens, reasons, and talks back. In 2026 that is a solved engineering problem, not a research problem — but the integration details decide whether it ships.
> 第 01-11 课的所有内容，缝合在一起。构建一个能听、能推理、能回话的语音助手。2026 年这是一个已解决的工程问题，而非研究问题——但集成细节决定它能否发货。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 05, 06, 07, 11; Phase 11 · 09 (Function Calling); Phase 14 · 01 (Agent Loop)
**Time:** ~120 minutes

## The Problem（问题）

Build an end-to-end assistant:
构建一个端到端助手：

1. Captures mic input (16 kHz mono).
   捕获麦克风输入（16 kHz 单声道）。
2. Detects start/end of user speech.
   检测用户语音的开始/结束。
3. Transcribes streaming.
   流式转录。
4. Passes transcript to an LLM that can call tools (timer, weather, calendar).
   将转录文本传给能调用工具（计时器、天气、日历）的 LLM。
5. Streams LLM text to a TTS.
   将 LLM 文本流式传输到 TTS。
6. Plays audio back to the user.
   向用户播放音频。
7. Stops if the user interrupts mid-response.
   如果用户在响应中途打断，则停止。

Latency target: first TTS audio byte within 800 ms of the user finishing their utterance on a laptop CPU. Quality target: no missed words, no hallucinated subtitles on silence, no voice cloning leakage, no prompt injection success.
延迟目标：在笔记本 CPU 上，用户说完后 800 毫秒内输出第一个 TTS 音频字节。质量目标：不漏词、静默上无幻觉字幕、无声音克隆泄露、无提示词注入成功。

## The Concept（概念）

![Voice assistant pipeline: mic → VAD → STT → LLM+tools → TTS → speaker](../assets/voice-assistant.svg)

### The seven components（七个组件）

1. **Audio capture（音频采集）. ** Mic → 16 kHz mono → 20 ms chunks. Usually `sounddevice` in Python or native AudioUnit/ALSA/WASAPI in production.
   麦克风 → 16 kHz 单声道 → 20 毫秒块。生产环境中通常用 Python 的 `sounddevice` 或原生 AudioUnit/ALSA/WASAPI。
2. **VAD (Lesson 11)（VAD（第 11 课）). ** Silero VAD @ threshold 0.5, min speech 250 ms, silence hang-over 500 ms. Signals "start" and "end."
   Silero VAD，阈值 0.5，最短语音 250 毫秒，静默悬停 500 毫秒。发出"开始"和"结束"信号。
3. **Streaming STT (Lesson 4-5)（流式 STT（第 4-5 课）). ** Whisper-streaming, Parakeet-TDT, or Deepgram Nova-3 (API). Partial + final transcripts.
   Whisper-streaming、Parakeet-TDT 或 Deepgram Nova-3（API）。部分 + 最终转录。
4. **LLM with tool calling（带工具调用的 LLM）. ** GPT-4o / Claude 3.5 / Gemini 2.5 Flash. JSON schema for tools. Stream tokens.
   GPT-4o / Claude 3.5 / Gemini 2.5 Flash。工具的 JSON schema。流式 token。
5. **Streaming TTS (Lesson 7)（流式 TTS（第 7 课）). ** Kokoro-82M (fastest open) or Cartesia Sonic (commercial). Start TTS after 20 LLM tokens.
   Kokoro-82M（最快开源）或 Cartesia Sonic（商业）。在 20 个 LLM token 后开始 TTS。
6. **Playback（播放）. ** Speaker out; opus-encode for low-bandwidth networks.
   扬声器输出；低带宽网络用 opus 编码。
7. **Interruption handler（打断处理器）. ** If VAD fires during TTS playback, stop playback, cancel LLM, restart STT.
   如果在 TTS 播放期间 VAD 触发，停止播放，取消 LLM，重启 STT。

### The three failure modes you will hit（你会遇到的三种失败模式）

1. **First-word clip（首词截断）. ** VAD starts a beat too late. User's "hey" is missing. Start threshold at 0.3, not 0.5.
   VAD 晚启动一拍。用户的"hey" 丢了。起始阈值设为 0.3，不是 0.5。
2. **Mid-response interrupt confusion（响应中途打断混淆）. ** LLM keeps generating after user interrupts; assistant talks over user. Wire VAD → cancel-LLM.
   用户打断后 LLM 继续生成；助手与用户抢话。把 VAD 接到 cancel-LLM。
3. **Silence hallucination（静默幻觉）. ** Whisper outputs "Thanks for watching" on the silent warm-up frames. Always VAD-gate.
   Whisper 在静默预热帧上输出"Thanks for watching"。始终用 VAD 门控。

### 2026 production reference stacks（2026 年生产级参考栈）

| Stack | Latency | License | Notes |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | commercial API | Industry default 2026 |
|                                                         | 行业默认 2026 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | mostly open | DIY-friendly |
|                                                      | 对 DIY 友好 |
| Moshi (full-duplex) | 200-300 ms | CC-BY 4.0 | Single-model; different architecture, lesson 15 |
|                      | 单模型；不同架构，第 15 课 |
| Vapi / Retell (managed) | 300-500 ms | commercial | Fastest to launch; limited customization |
|                           | 最快上线；定制有限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | offline | open | Privacy / edge |
|                                        | 离线 | 开源 | 隐私 / 边缘 |

```figure
v4-voice-latency
```

## Build It（动手实现）

### Step 1: mic capture with chunking (pseudocode)（带分块的麦克风采集（伪代码））

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### Step 2: VAD-gated turn capture（VAD 门控的轮次采集）

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### Step 3: streaming STT → LLM → TTS（流式 STT → LLM → TTS）

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### Step 4: tool calling inside the LLM loop（LLM 循环内的工具调用）

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### Step 5: interruption handling（打断处理）

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## Use It（实际应用）

See `code/main.py` for a runnable simulation that wires all seven components with stub models, so you can see the pipeline shape even without hardware. For a real implementation, swap stubs with:
参见 `code/main.py` 获取一个可运行的模拟，它用桩模型连接全部七个组件，因此即使没有硬件你也可以看到流水线形状。对于真实实现，用以下内容替换桩：

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` or `openai-whisper`
- `openai` (`gpt-4o`) or `anthropic`
- `kokoro` or `cartesia`
- `sounddevice` for I/O

## Pitfalls（陷阱）

- **Logging PII forever（永远记录 PII）. ** Full-turn audio is PII in most jurisdictions. 30-day retention, encrypted at rest.
  完整轮次音频在大多数司法管辖区都是 PII。保留 30 天，静态加密。
- **No barge-in（没有抢话）. ** Users will interrupt. Your assistant must stop talking.
  用户会打断。你的助手必须停止说话。
- **TTS that blocks（阻塞的 TTS）. ** Synchronous TTS blocks the event loop. Use async or a separate thread.
  同步 TTS 会阻塞事件循环。使用异步或单独线程。
- **No tool-call error handling（无工具调用错误处理）. ** Tools fail. LLM must get back the error + retry once, then gracefully degrade.
  工具会失败。LLM 必须取回错误 + 重试一次，然后优雅降级。
- **Overzealous hallucination filters（过度积极的幻觉过滤器）. ** Over-filter and the assistant repeats "I can't help with that." Under-filter and it says anything. Calibrate on a held-out set.
  过度过滤，助手会重复"我无法帮助那个"。过滤不足，它会说任何东西。在保留集上调优。
- **No wake-word option（没有唤醒词选项）. ** Always-listening is a privacy liability. Add a wake-word gate (Porcupine or openWakeWord).
  始终监听是隐私责任。添加唤醒词门控（Porcupine 或 openWakeWord）。

## Ship It（交付成果）

Save as `outputs/skill-voice-assistant-architect.md`. Given budget + scale + language + compliance constraints, produce a full stack spec.
保存为 `outputs/skill-voice-assistant-architect.md`。给定预算 + 规模 + 语言 + 合规约束，输出完整栈规格说明。

## Exercises（练习）

1. **Easy（简单）.** Run `code/main.py`. It simulates one full turn end-to-end with stub modules and prints per-stage latency.
   运行 `code/main.py`。它用桩模块端到端模拟一个完整轮次并打印每阶段延迟。
2. **Medium（中等）.** Replace the STT stub with a real Whisper model on a pre-recorded `.wav`. Measure WER and end-to-end latency.
   在预录制的 `.wav` 上用真实 Whisper 模型替换 STT 桩。测量 WER 和端到端延迟。
3. **Hard（困难）.** Add tool calling: implement `get_weather` (any API) and `set_timer`. Route the LLM through the tools and verify that when the user says "set a 5 minute timer" the right function fires and the spoken reply confirms it.
   添加工具调用：实现 `get_weather`（任意 API）和 `set_timer`。让 LLM 通过这些工具路由，并验证当用户说"set a 5 minute timer" 时正确的函数被触发并且语音回复确认了它。

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Turn | A user + assistant round-trip | One VAD-bounded user speech + one LLM-TTS response. |
|     | 用户 + 助手往返 | 一次 VAD 约束的用户语音 + 一次 LLM-TTS 响应。 |
| Barge-in | Interruption | User speaks while assistant talks; assistant stops. |
|          | 打断 | 用户在助手说话时说话；助手停止。 |
| Wake word | "Hey assistant" | Short keyword detector; Porcupine, Snowboy, openWakeWord. |
|           | "Hey assistant" | 短关键词检测器；Porcupine、Snowboy、openWakeWord。 |
| End-pointing | Turn ending | VAD + min-silence decision that user has finished. |
|              | 轮次结束 | VAD + 最短静默决策，用户已完成。 |
| Pre-roll | Pre-speech buffer | Keep 200-400 ms of audio before VAD fires to avoid first-word clip. |
|          | 语音前缓冲 | 在 VAD 触发前保留 200-400 毫秒音频，避免首词截断。 |
| Tool call | Function invocation | LLM emits JSON; runtime dispatches; result feeds back in-loop. |
|            | 函数调用 | LLM 发出 JSON；运行时分发；结果在循环内反馈。 |

## Further Reading（延伸阅读）

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — production-grade reference.
  LiveKit——语音 agent 快速入门——生产级参考。
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) — DIY-friendly framework.
  Pipecat——语音 agent 示例——对 DIY 友好的框架。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — the managed voice-native path.
  OpenAI Realtime API——托管语音原生路径。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — full-duplex reference (Lesson 15).
  Kyutai Moshi——全双工参考（第 15 课）。
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — wake-word gating.
  Porcupine 唤醒词——唤醒词门控。
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM function calling.
  Anthropic——工具使用指南——LLM 函数调用。
