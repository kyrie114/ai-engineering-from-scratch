# 长上下文评估——NIAH, RULER, LongBench, MRCR（Long-Context Evaluation — NIAH, RULER, LongBench, MRCR）

> Gemini 3 Pro 广告 10M 词元的上下文。在 1M 词元时，8 针 MRCR 下降到 26.3%。广告的 ≠ 可用的。长上下文评估告诉你实际交付的模型容量。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 23 (Chunking Strategies)
**Time:** ~60 minutes

## 问题（The Problem）

你有一份 200 页的合同。模型声称 1M 词元的上下文。你把合同粘贴进去并问："终止条款是什么？"模型回答了——但从封面页回答，因为终止条款在 120k 词元深处，超出了模型实际注意力的范围。

这就是 2026 年的上下文容量差距。规格表说 1M 或 10M。现实说其中 60-70% 是可用的，而 "可用" 取决于任务。

- **检索（海中的一根针）：** 在前沿模型的广告最大值附近近乎完美。
- **多跳/聚合：** 大多数模型在约 128k 之后急剧下降。
- **分散事实上的推理：** 最先失败的任务。

长上下文评估衡量这些轴。本课命名基准、每个基准实际衡量什么，以及如何为你的领域构建自定义针测试。

## 概念（The Concept）

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**海中的针（Needle-in-a-Haystack, NIAH，2023）。** 在长上下文中的受控深度放置一个事实（"magic word is pineapple"）。要求模型检索它。深度 × 长度的扫描。原始的基准测试。前沿模型现在饱和了这个；它是一个必要但不充分的基线。

**RULER（Nvidia，2024）。** 跨 4 个类别的 13 种任务类型：检索（单/多键/多值）、多跳追踪（变量追踪）、聚合（常见词频率）、QA。可配置的上下文长度（4k 到 128k+）。揭示饱和 NIAH 但在多跳上失败的模型。在 2024 版本中，声称 32k+ 上下文的 17 个模型中只有一半在 32k 时保持质量。

**LongBench v2（2024）。** 503 道选择题，8k-2M 词元的上下文，六个任务类别：单文档 QA、多文档 QA、长上下文学习、长对话、代码仓库、长结构化数据。真实世界长上下文行为的生产基准。

**MRCR（多轮指代消解）。** 大规模多轮指代消解。8 针、24 针、100 针变体。暴露模型在注意力退化之前可以处理多少事实。

**NoLiMa。** "非词汇针。"针和查询共享字面上没有重叠；检索需要一个语义推理步骤。比 NIAH 更难。

**HELMET。** 连接许多文档，从任何一个中问问题。测试选择性注意力。

**BABILong。** 将 bAbI 推理链嵌入不相关的海中。测试海中的推理，而不仅仅是检索。

### 实际报告的内容

- **广告上下文窗口。** 规格表数字。
- **有效检索长度。** 在某个阈值（例如 90%）下的 NIAH 通过。
- **有效推理长度。** 在该阈值下的多跳或聚合通过。
- **退化曲线。** 准确性 vs 上下文长度，按任务类型绘制。

你的规格表两个数字：检索有效和推理有效。通常推理有效是广告窗口的 25-50%。

```figure
gx-niah-decay
```

## 构建它（Build It）

### 步骤 1：你的领域的自定义 NIAH

见 `code/main.py`。骨架：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

扫描 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}。绘制热力图。这就是你的目标模型的 NIAH 卡。

### 步骤 2：多针变体

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

像 "三个魔法词是什么？" 这样的问题需要检索所有三个。单针成功不能预测多针成功。

### 步骤 3：多跳变量追踪（RULER 风格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

答案需要链接三个赋值。在 128k 时，前沿模型经常下降到 50-70% 的准确性。

### 步骤 4：在你的栈上使用 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

按类别报告准确性。聚合分数隐藏了大的任务级差异。

## 陷阱

- **仅 NIAH 评估。** 在 1M 词元上通过 NIAH 对多跳来说毫无意义。始终运行 RULER 或自定义多跳测试。
- **均匀深度采样。** 许多实现只在 depth=0.5 测试。测试 0、0.25、0.5、0.75、1.0——"中间的迷失"效应是真实的。
- **与填充的词汇重叠。** 如果针与填充共享关键词，检索就变得微不足道。使用 NoLiMa 风格的非重叠针。
- **忽略延迟。** 1M 词元的提示需要 30-120 秒来预填充。测量时间到第一个词元以及准确性。
- **供应商自我报告的数字。** OpenAI、Google、Anthropic 都发布他们自己的分数。始终在你的用例上独立重新运行。

## 使用它（Use It）

2026 年技术栈：

| 情况 | 基准 |
|-----------|-----------|
| 快速健全性检查 | 3 个深度 × 3 个长度的自定义 NIAH |
| 生产模型选择 | 你目标长度的 RULER（13 个任务） |
| 真实世界 QA 质量 | LongBench v2 单文档 QA 子集 |
| 多跳推理 | BABILong 或自定义变量追踪 |
| 对话/对话 | 你目标长度的 MRCR 8 针 |
| 模型升级回归 | 每次新模型上运行的固定内部 NIAH + RULER 工具 |

生产经验法则：在预期长度上拥有 NIAH + 1 个推理任务之前，不要信任上下文窗口。

## 发布它（Ship It）

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## 练习（Exercises）

1. **简单（Easy）。** 构建 3 个深度（0.25、0.5、0.75）× 3 个长度（1k、4k、16k）的 NIAH。在任何模型上运行。绘制通过率作为 3x3 热力图。
2. **中等（Medium）。** 添加 3 针变体。在每个长度测量所有 3 个的检索。与同一长度的单针通过率比较。
3. **困难（Hard）。** 构建嵌入 64k 填充的变量追踪任务（X1 → X2 → X3，3 跳）。在 3 个前沿模型上测量准确性。报告每个模型的有效推理长度。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| NIAH | 海中的针 | 在填充中植入一个事实，要求模型检索它。 |
| RULER | 加强版 NIAH | 跨检索/多跳/聚合/QA 的 13 种任务类型。 |
| 有效上下文 | 真实容量 | 准确性仍保持高于阈值的长度。 |
| 中间的迷失 | 深度偏见 | 模型对长输入中间的内容关注不足。 |
| 多针 | 一次多个事实 | 多个植入；测试注意力处理，而不仅仅是检索。 |
| MRCR | 多轮指代消解 | 8、24 或 100 针指代消解；暴露注意力饱和。 |
| NoLiMa | 非词汇针 | 针和查询没有字面重叠的令牌；需要推理。 |

## 延伸阅读（Further Reading）

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 原始 NIAH 仓库。
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — 多任务基准。
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 真实世界长上下文评估。
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 更难的针。
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — 海中的推理。
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 深度偏见论文。
