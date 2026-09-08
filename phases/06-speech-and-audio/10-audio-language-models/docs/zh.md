# Audio-Language Models — Qwen2.5-Omni, Audio Flamingo, GPT-4o Audio（音频语言模型——Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio）

> 2026 audio-language models reason over speech + environmental sound + music. Qwen2.5-Omni-7B matches GPT-4o Audio on MMAU-Pro. Audio Flamingo Next beats Gemini 2.5 Pro on LongAudioBench. The gap between open and closed is essentially closed — except on multi-audio tasks, where everyone is near random.
> 2026 年音频语言模型可以对语音 + 环境音 + 音乐进行推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上与 GPT-4o Audio 相当。Audio Flamingo Next 在 LongAudioBench 上击败 Gemini 2.5 Pro。开源与闭源的差距基本已消除——除了多音频任务，在那里 everyone 都接近随机。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 12 · 03 (Vision-Language Models), Phase 7 · 10 (Audio Transformers)
**Time:** ~45 minutes

## The Problem（问题）

You have 5 seconds of audio: dog barks, someone yells "stop!", then silence. Useful questions span multiple axes:
你有 5 秒音频：狗叫，有人喊"stop!"，然后静默。有用的问题跨越多个轴：

- **Transcription（转录）. ** "What was said?" — ASR territory.
  "说了什么？"——ASR 领域。
- **Semantic reasoning（语义推理）. ** "Is the person in danger?" — requires joint understanding of the bark + yell + silence.
  "这个人处于危险中吗？"——需要对狗叫 + 喊叫 + 静默的联合理解。
- **Music reasoning（音乐推理）. ** "What instruments play the melody?"
  "什么乐器在演奏旋律？"
- **Long-audio retrieval（长音频检索）. ** "Where in this 90-minute lecture did the instructor explain gradient descent?"
  "在这段 90 分钟的讲座中，讲师在哪里解释了梯度下降？"

A single model that answers all of these with one prompt is an **audio-language model** (LALM / ALM). Separate from pure ASR: LALMs produce free-form natural-language answers, not just transcripts.
用一个提示回答所有这些问题的单一模型是**音频语言模型**（LALM / ALM）。与纯 ASR 不同：LALM 产生自由格式的自然语言答案，而不仅仅是转录。

## The Concept（概念）

![Audio-language model: audio encoder + projector + LLM decoder](../assets/alm-architecture.svg)

### The three-component template（三组件模板）

Every 2026 LALM has the same skeleton:
每个 2026 年的 LALM 都有相同的骨架：

1. **Audio encoder（音频编码器）. ** Whisper encoder · BEATs · CLAP · WavLM · or a custom encoder per model.
   Whisper 编码器 · BEATs · CLAP · WavLM · 或每个模型的自定义编码器。
2. **Projector（投影器）. ** Linear or MLP bridging audio-encoder features into the LLM's token embedding space.
   线性或 MLP，将音频编码器特征桥接到 LLM 的 token 嵌入空间。
3. **LLM（大语言模型）. ** Llama / Qwen / Gemma-based decoder. Takes interleaved text + audio tokens; generates text.
   基于 Llama / Qwen / Gemma 的解码器。接收交错文本 + 音频 token；生成文本。

Training（训练）：

- **Stage 1（阶段 1）. ** Freeze encoder + LLM; train projector only on ASR / captioning data.
  冻结编码器 + LLM；仅在 ASR / 字幕数据上训练投影器。
- **Stage 2（阶段 2）. ** Full / LoRA fine-tune on instruction-following audio tasks (QA, reasoning, music understanding).
  在指令遵循音频任务（QA、推理、音乐理解）上进行全参数 / LoRA 微调。
- **Stage 3 (optional)（阶段 3（可选）). ** Voice-in / voice-out adds a speech decoder. Qwen2.5-Omni and AF3-Chat do this.
  语音入 / 语音出添加一个语音解码器。Qwen2.5-Omni 和 AF3-Chat 这样做。

### The 2026 model map（2026 年模型地图）

| Model | Backbone | Audio encoder | Output modality | Access |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | text + speech | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Custom | text + speech | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | text | NVIDIA non-commercial |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | text | NVIDIA non-commercial |
| SALMONN | Vicuna | Whisper + BEATs | text | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | text | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | text | Apache-2.0 |
| Gemini 2.5 Flash/Pro (closed) | Gemini | proprietary | text + speech | API |
| GPT-4o Audio (closed) | GPT-4o | proprietary | text + speech | API |

### Benchmark reality check (2026)（基准现实检验 (2026)）

**MMAU-Pro.** 1800 QA pairs covering speech / sound / music / mixed. Multi-audio subset included.
**MMAU-Pro.** 1800 个 QA 对，覆盖语音 / 声音 / 音乐 / 混合。包含多音频子集。

| Model | Overall | Speech | Sound | Music | Multi-audio |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | SOTA on LongAudioBench | — | — | — | — |

The **multi-audio column is damning for everyone.** Random chance on 4-option multiple choice = 25%; most models score around there. LALMs still struggle to compare two clips.
**多音频列对所有人来说都是 damning 的。** 4 选项多项选择的随机概率 = 25%；大多数模型得分就在那里。LALM 仍然难以比较两个片段。

### Where LALMs are useful in 2026（2026 年 LALM 有用的地方）

- **Compliance audit of call-center recordings（呼叫中心录音的合规审计）. ** "Did the agent mention the required disclosure?"
  "客服人员是否提到了所需的披露？"
- **Accessibility（无障碍）. ** Describe sound events to deaf users (not just transcription).
  向聋人用户描述声音事件（不仅仅是转录）。
- **Content moderation（内容审核）. ** Detect violent language + threatening tone + background context.
  检测暴力语言 + 威胁性语调 + 背景上下文。
- **Podcast / meeting chaptering（播客 / 会议章节划分）. ** Semantic summary, not just speaker turns.
  语义摘要，而不仅仅是说话人轮次。
- **Music catalog analysis（音乐目录分析）. ** "Find all tracks with a B-section key change."
  "找到所有有 B 段调变化的曲目。"

### Where they are NOT (yet) useful（它们（还）没用的地方）

- Fine-grained music theory (below chord-level).
  细粒度音乐理论（和弦级以下）。
- Speaker-attributed reasoning over long conversations (degrades past 10 minutes).
  长对话上的说话人归因推理（超过 10 分钟会退化）。
- Multi-audio comparison (22-26% is barely above random).
  多音频比较（22-26%  barely above random）。
- Real-time streaming reasoning (most are offline batch inference).
  实时流式推理（大多数是离线批处理推理）。

```figure
v4-alm-tokens
```

## Build It（动手实现）

### Step 1: query Qwen2.5-Omni（查询 Qwen2.5-Omni）

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Step 2: the projector pattern（投影器模式）

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

That's it. The projector is usually 1-3 linear layers. Training it on ASR pairs (audio → transcript) is the Stage-1 pretext task.
就是这样。投影器通常是 1-3 个线性层。在 ASR 对（音频 → 转录）上训练它是阶段 1 的前置任务。

### Step 3: benchmarking MMAU / LongAudioBench（基准测试 MMAU / LongAudioBench）

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

Report per-category (speech / sound / music / multi-audio) separately. Aggregate numbers hide where the model fails.
按类别（语音 / 声音 / 音乐 / 多音频）分别报告。聚合数字会隐藏模型在哪里失败。

## Use It（实际应用）

| Task | 2026 pick |
|------|-----------|
| Free-form audio QA (open) | Qwen2.5-Omni-7B |
|                          | 自由格式音频 QA（开源） |
| Best open on long audio | Audio Flamingo Next |
|                         | 长音频最佳开源 |
| Best closed | Gemini 2.5 Pro |
|            | 最佳闭源 |
| Voice-in / voice-out agent | Qwen2.5-Omni or GPT-4o Audio |
|                             | 语音入 / 语音出智能体 |
| Music reasoning | Audio Flamingo 3 or 2 (music-specialized AF-CLAP) |
|                 | 音乐推理 |
| Call-center audit | Gemini 2.5 Pro via API, with RAG over your policy docs |
|                   | 呼叫中心审计 |

## Pitfalls（陷阱）

- **Over-trust on multi-audio（对多音频过度信任）. ** If your task needs "which clip has X," random-chance-level performance is real.
  如果你的任务需要"哪个片段有 X"，随机概率级别的性能是真实的。
- **Long-audio degradation（长音频退化）. ** Past 10 minutes, most models' speaker attribution breaks. Diarize first (Lesson 6), then summarize.
  超过 10 分钟，大多数模型的说话人归分会崩溃。先做说话人分离（第 6 课），然后总结。
- **Hallucinations on silence（静默上的幻觉）. ** Same Whisper-style issue inherited by LALMs that use Whisper encoder. VAD-gate.
  使用 Whisper 编码器的 LALM 继承的同样的 Whisper 风格问题。VAD 门控。
- **Benchmark cherry-picking（基准 cherry-picking）. ** Vendor blog posts highlight best-case categories. Run MMAU-Pro multi-audio subset yourself.
  厂商博客文章突出最佳案例类别。自己运行 MMAU-Pro 多音频子集。

## Ship It（交付成果）

Save as `outputs/skill-alm-picker.md`. Pick LALM + benchmark subset + output-modality (text vs speech) for a given audio-understanding task.
保存为 `outputs/skill-alm-picker.md`。为给定的音频理解任务选择 LALM + 基准子集 + 输出模态（文本 vs 语音）。

## Exercises（练习）

1. **Easy（简单）.** Run `code/main.py` to see a toy projector pattern + fake LALM routing of (audio-embedding, text-tokens) → output tokens.
   运行 `code/main.py` 查看一个玩具投影器模式 + 假的 LALM 路由 (audio-embedding, text-tokens) → output tokens。
2. **Medium（中等）.** Score Qwen2.5-Omni-7B on 100 MMAU-Pro speech items. Compare to the paper's reported number.
   在 100 个 MMAU-Pro 语音项目上给 Qwen2.5-Omni-7B 打分。与论文报告的数字比较。
3. **Hard（困难）.** Build a minimal audio-captioning baseline: BEATs encoder + 2-layer projector + frozen Llama-3.2-1B. Fine-tune only the projector on AudioCaps. Compare to SALMONN on Clotho-AQA.
   构建一个最小音频字幕基线：BEATs 编码器 + 2 层投影器 + 冻结的 Llama-3.2-1B。仅在 AudioCaps 上微调投影器。与 SALMONN 在 Clotho-AQA 上比较。

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| LALM | Audio ChatGPT | Audio encoder + projector + LLM decoder. |
|      | 音频 ChatGPT | 音频编码器 + 投影器 + LLM 解码器。 |
| Projector | Adapter | Small MLP mapping audio features into LLM embedding space. |
|           | 适配器 | 将音频特征映射到 LLM 嵌入空间的小型 MLP。 |
| MMAU | The benchmark | 10k audio-QA pairs across speech, sound, music. |
|      | 基准 | 跨语音、声音、音乐的 1 万个音频 QA 对。 |
| MMAU-Pro | Harder MMAU | 1800 multi-audio / reasoning-heavy questions. |
|           | 更难 MMAU | 1800 个多音频 / 推理重的问题。 |
| LongAudioBench | Long-form eval | Multi-minute clips with semantic queries. |
|                | 长格式评估 | 带语义查询的多分钟片段。 |
| Voice-in / voice-out | Speech-native | Model ingests speech and emits speech without text detour. |
|                        | 语音原生 | 模型摄入语音并发出语音，无需文本绕道。 |

## Further Reading（延伸阅读）

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — reference architecture.
  Chu 等 (2024).《Qwen2-Audio》——参考架构。
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — speech-in-speech-out.
  Alibaba (2025).《Qwen2.5-Omni》——语音入语音出。
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — the open long-audio leader.
  NVIDIA (2025).《Audio Flamingo 3》——开源长音频领导者。
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA.
  NVIDIA (2026).《Audio Flamingo Next》——LongAudioBench SOTA。
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — dual-encoder pioneer.
  Tang 等 (2023).《SALMONN》——双编码器先驱。
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) — live 2026 rankings.
  MMAU-Pro 排行榜——2026 年实时排名。
