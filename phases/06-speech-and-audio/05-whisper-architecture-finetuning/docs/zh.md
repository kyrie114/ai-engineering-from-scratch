# Whisper — Architecture & Fine-Tuning（Whisper——架构与微调）

> Whisper is a 30-second-window transformer encoder-decoder, trained on 680k hours of multilingual weakly-supervised audio-text pairs. One architecture, multiple tasks, robust across 99 languages. The 2026 reference ASR.
> Whisper 是一个 30 秒窗口的 transformer 编码器-解码器，在 68 万小时的多语言弱监督音频-文本对上训练。一个架构，多个任务，在 99 种语言上鲁棒。2026 年参考 ASR。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 5 · 10 (Attention), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## The Problem（问题）

Whisper, released by OpenAI in September 2022, was the first ASR model to ship as a commodity: paste audio, get text, 99 languages, robust to noise, runs on a laptop. By 2024 OpenAI had shipped Large-v3 and Turbo variants; by 2026, Whisper is the default baseline for everything from podcast transcription to voice assistants to YouTube subtitles.
Whisper 由 OpenAI 于 2022 年 9 月发布，是第一个作为商品级模型发布的 ASR 模型：粘贴音频，获取文本，支持 99 种语言，抗噪声，可在笔记本上运行。到 2024 年，OpenAI 已发布 Large-v3 和 Turbo 变体；到 2026 年，Whisper 是从播客转录、语音助手到 YouTube 字幕一切场景的默认基线。

But Whisper is not a pipeline you can treat as a black box forever. Domain shift kills it — technical jargon, speaker accents, proper nouns, short clips, silence. You need to know:
但 Whisper 不是你可以永远当作黑盒处理的流水线。领域偏移会杀死它——技术术语、说话人口音、专有名词、短片段、静默。你需要知道：

1. What it actually is inside.
   它内部究竟是什么。
2. How to give it chunked, streaming, or long-form audio correctly.
   如何正确地给它分块、流式或长音频。
3. When to fine-tune and how.
   何时微调以及如何微调。

## The Concept（概念）

![Whisper encoder-decoder, tasks, chunked inference, fine-tune](../assets/whisper.svg)

**Architecture（架构）.** Standard transformer encoder-decoder.
标准 transformer 编码器-解码器。

- Input: 30-second log-mel spectrogram, 80 mels, 10 ms hop → 3000 frames. Clips shorter are zero-padded, clips longer are chunked.
  输入：30 秒对数梅尔语谱图，80 个梅尔，10 ms 步长 → 3000 帧。更短的片段零填充，更长的片段分块。
- Encoder: conv-downsample (stride 2) + `N` transformer blocks. For Large-v3: 32 layers, 1280-dim, 20 heads.
  编码器：conv-downsample（步幅 2）+ `N` 个 transformer block。对于 Large-v3：32 层，1280 维，20 个头。
- Decoder: `N` transformer blocks with causal self-attn + cross-attn to encoder output. Same size as encoder.
  解码器：`N` 个 transformer block，带因果自注意力 + 对编码器输出的交叉注意力。与编码器大小相同。
- Output: BPE tokens over a 51,865-token vocab.
  输出：覆盖 51,865 token 词表的 BPE token。

Large-v3 has 1.55B params. Turbo uses a 4-layer decoder (from 32), cutting latency 8× with a <1% WER hit.
Large-v3 有 15.5 亿参数。Turbo 使用 4 层解码器（从 32 层减少），延迟降低 8 倍，WER 增加不到 1%。

**The prompt format（提示格式）.** Whisper is a multitask model steered by special tokens in the decoder prompt:
Whisper 是一个多任务模型，由解码器提示中的特殊 token 引导：

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — language tag; forces translation-vs-transcription behavior.
  `<|en|>`——语言标签；强制翻译-转录行为。
- `<|transcribe|>` or `<|translate|>` — translate English output from any-language input, or verbatim.
  `<|transcribe|>` 或 `<|translate|>`——从任意语言输入翻译为英语输出，或逐字转录。
- `<|notimestamps|>` — skip word-level timestamps (faster).
  `<|notimestamps|>`——跳过词级时间戳（更快）。

The prompt is what lets one model do many tasks. Change `<|en|>` to `<|fr|>` and it transcribes French.
正是这个提示让一个模型能做多种任务。把 `<|en|>` 改成 `<|fr|>`，它就会转录法语。

**30-second window（30 秒窗口）.** Everything is pinned to 30 seconds. Longer clips need chunking; shorter clips are padded. Windows are not streamed natively — this is why WhisperX, Whisper-Streaming, and faster-whisper exist.
一切都绑定到 30 秒。更长的片段需要分块；更短的片段做填充。窗口本身不支持流式传输——这就是 WhisperX、Whisper-Streaming 和 faster-whisper 存在的原因。

**Log-mel normalization（对数梅尔归一化）.** `(log_mel - mean) / std` where the stats come from Whisper's own training corpus. You *must* use Whisper's preprocessing (`whisper.audio.log_mel_spectrogram`), not `librosa.feature.melspectrogram`.
`(log_mel - mean) / std`，其中的统计量来自 Whisper 自身的训练语料库。你*必须*使用 Whisper 的预处理（`whisper.audio.log_mel_spectrogram`），而不是 `librosa.feature.melspectrogram`。

### Variants in 2026（2026 年的变体）

| Variant | Params | Latency (A100) | WER (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× realtime | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | streaming | 2.0% |

### Fine-tuning（微调）

Canonical workflow in 2026:
2026 年的标准工作流：

1. Collect 10–100 hours of target-domain audio with aligned transcripts.
   收集 10–100 小时带对齐 transcripts 的目标领域音频。
2. Run `transformers.Seq2SeqTrainer` with `generate_with_loss` callback.
   使用带 `generate_with_loss` 回调的 `transformers.Seq2SeqTrainer` 运行。
3. Parameter-efficient: LoRA on `q_proj`, `k_proj`, `v_proj` of attention layers reduces GPU memory 4× with <0.3 WER cost.
   参数高效：在注意力层的 `q_proj`、`k_proj`、`v_proj` 上应用 LoRA，GPU 内存减少 4 倍，WER 代价 <0.3%。
4. Freeze the encoder if you have <10 hours. Only tune the decoder.
   如果你有不到 10 小时的数据，冻结编码器。只微调解码器。
5. Use Whisper's own tokenizer and prompt format; never swap tokenizers.
   使用 Whisper 自身的分词器和提示格式；绝不要换分词器。

Community results: fine-tuning Medium on 20 hours of medical dictation drops WER from 12% to 4.5% on medical vocabulary. Fine-tuning Turbo on 4 hours of Icelandic drops WER from 18% to 6%.
社区结果：在 20 小时医学听写上微调 Medium，医学词汇上的 WER 从 12% 降到 4.5%。在 4 小时冰岛语上微调 Turbo，WER 从 18% 降到 6%。

```figure
sp-asr-attention
```

## Build It（动手实现）

### Step 1: run Whisper out of the box（直接运行 Whisper）

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

Key defaults you should always override: `temperature=0.0` (sampling defaults to 0.0 → 0.2 → 0.4 … fallback chain), `condition_on_previous_text=False` (prevents the cascading hallucination problem), and `no_speech_threshold=0.6` (silence detection).
你应该始终覆盖的关键默认值：`temperature=0.0`（采样默认为 0.0 → 0.2 → 0.4 … 回退链），`condition_on_previous_text=False`（防止级联幻觉问题），`no_speech_threshold=0.6`（静默检测）。

### Step 2: chunked long-form（分块长音频）

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX adds (1) Silero VAD gating, (2) word-level alignment via wav2vec 2.0, (3) diarization via `pyannote.audio`. The 2026 workhorse for production transcription.
WhisperX 增加了 (1) Silero VAD 门控，(2) 通过 wav2vec 2.0 做词级对齐，(3) 通过 `pyannote.audio` 做说话人分离。2026 年生产级转录的主力工具。

### Step 3: fine-tune with LoRA（使用 LoRA 微调）

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

Then standard Trainer loop. Checkpoint every 1000 steps. Evaluate with WER on held-out.
然后是标准的 Trainer 循环。每 1000 步保存一个 checkpoint。在保留集上用 WER 评估。

### Step 4: inspect what each layer learns（检查每层学到了什么）

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

Visualize with a heatmap — you will see diagonal alignment as decoder steps scan through encoder frames. That diagonal is Whisper's notion of word timestamps.
用热力图可视化——你会看到对角线对齐，因为解码器步扫描编码器帧。那条对角线就是 Whisper 的词时间戳概念。

## Use It（实际应用）

The 2026 stack:
2026 年栈：

| Situation | Pick |
|-----------|------|
| General English, offline | Large-v3-turbo via `whisperx` |
|                            | 通用英语、离线 |
| Mobile / edge | Whisper-Tiny quantized (int8) or Moonshine |
|                | 移动端 / 边缘 |
| Multilingual long-form | Large-v3 via `whisperx` + diarization |
|                          | 多语言长音频 |
| Low-resource language | Fine-tune Medium or Turbo with LoRA |
|                        | 低资源语言 |
| Streaming (2 s latency) | Whisper-Streaming or Parakeet-TDT |
|                           | 流式（2 秒延迟） |
| Word-level timestamps | WhisperX (forced alignment via wav2vec 2.0) |
|                        | 词级时间戳 |

`faster-whisper` (CTranslate2 backend) is the fastest CPU+GPU inference runtime in 2026 — 4× faster than vanilla with identical output.
`faster-whisper`（CTranslate2 后端）是 2026 年最快的 CPU+GPU 推理运行时——比原生快 4 倍，输出相同。

## Pitfalls that still ship in 2026（2026 年仍然上线的坑）

- **Hallucinated text on silence（静默上的幻觉文本）.** Whisper trained on captions includes "Thanks for watching!", "Subscribe!", song lyrics. Always VAD-gate before calling.
  在字幕上训练的 Whisper 包含"Thanks for watching!"、"Subscribe!"、歌词。调用前始终用 VAD 做门控。
- **`condition_on_previous_text` cascade（`condition_on_previous_text` 级联）.** One hallucination pollutes subsequent windows. Set `False` unless you need fluency across chunks.
  一次幻觉会污染后续窗口。除非你需要跨块的流畅性，否则设为 `False`。
- **Short-clip padding（短片段填充）.** A 2-second clip padded to 30 seconds can hallucinate in the trailing silence. Use `pad=False` or VAD-gate.
  一段 2 秒的片段填充到 30 秒，可能在尾部静默中产生幻觉。使用 `pad=False` 或 VAD 门控。
- **Wrong mel stats（错误的梅尔统计量）.** Using librosa's mels instead of Whisper's produces near-random output. Use `whisper.audio.log_mel_spectrogram`.
  使用 librosa 的梅尔而非 Whisper 的梅尔，输出接近随机。使用 `whisper.audio.log_mel_spectrogram`。

## Ship It（交付成果）

Save as `outputs/skill-whisper-tuner.md`. Design a Whisper fine-tune or inference pipeline for a given domain.
保存为 `outputs/skill-whisper-tuner.md`。为给定领域设计 Whisper 微调或推理流水线。

## Exercises（练习）

1. **Easy（简单）.** Run `code/main.py`. It tokenizes a Whisper-style prompt, computes decoded shape budgets, and prints the chunk schedule for a 10-minute clip.
   运行 `code/main.py`。它对一个 Whisper 风格的提示做分词，计算解码形状预算，并打印一段 10 分钟音频的分块计划。
2. **Medium（中等）.** Install `faster-whisper`, transcribe a 10-minute podcast, compare WER against a human transcript. Try `language="auto"` vs forced `language="en"`.
   安装 `faster-whisper`，转录一段 10 分钟的播客，与人工 transcript 比较 WER。尝试 `language="auto"` 与强制 `language="en"`。
3. **Hard（困难）.** Using HF `datasets`, pick a language Whisper struggles with (e.g., Urdu), fine-tune Medium with LoRA for 2 epochs on 2 hours, and report WER delta.
   使用 HF `datasets`，选择一种 Whisper 表现较差的语种（例如乌尔都语），用 LoRA 在 2 小时数据上微调 Medium 2 个 epoch，并报告 WER 变化。

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| 30-sec window | Whisper's limit | Hard input cap; chunk longer audio. |
|               | Whisper 的限制 | 硬输入上限；更长音频需要分块。 |
| SOT | Start-of-transcript | `<|startoftranscript|>` kicks off the decoder prompt. |
|     | 转录开始 | `<|startoftranscript|>` 启动解码器提示。 |
| Timestamps token | Temporal alignment | Every 0.02 s offset is a special token in the 51k vocab. |
|                   | 时间对齐 | 每 0.02 秒偏移都是 51k 词表中的一个特殊 token。 |
| Turbo | The fast variant | 4-decoder layers, 8× faster, <1% WER regression. |
|       | 快速变体 | 4 层解码器，快 8 倍，WER 退化不到 1%。 |
| WhisperX | The long-form wrapper | VAD + Whisper + wav2vec alignment + diarization. |
|          | 长音频包装器 | VAD + Whisper + wav2vec 对齐 + 说话人分离。 |
| LoRA fine-tune | Efficient tuning | Add low-rank adapters to attention; train ~0.3% of params. |
|                | 高效微调 | 给注意力加入低秩适配器；训练约 0.3% 的参数。 |
| Hallucination | The silent failure | Whisper produces fluent English from noise/silence. |
|                | 静默失败 | Whisper 从噪声/静默中生成流畅英语。 |

## Further Reading（延伸阅读）

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — the original architecture and training recipe.
  Radford et al. (2022).《Whisper 论文》——原始架构和训练配方。
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — 4-layer decoder, 8× speedup.
  OpenAI (2024).《Whisper Large-v3-turbo 发布》——4 层解码器，8 倍加速。
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — long-form, word-aligned, diarized.
  Bain et al. (2023).《WhisperX》——长音频、词级对齐、说话人分离。
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-backed, 4× faster.
  Systran——faster-whisper 仓库——基于 CTranslate2，快 4 倍。
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — canonical LoRA / full-FT walkthrough.
  HuggingFace——Whisper 微调教程——规范的 LoRA / 全参数微调演练。
