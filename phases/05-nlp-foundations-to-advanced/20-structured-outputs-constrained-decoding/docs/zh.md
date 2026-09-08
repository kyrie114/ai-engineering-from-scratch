# 结构化输出与约束解码（Structured Outputs & Constrained Decoding）

> 要求 LLM 输出 JSON。大部分时候得到 JSON。在生产环境中，"大部分" 是问题。约束解码通过在采样之前编辑 logits 将 "大部分" 变为 "总是"。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 minutes

## 问题（The Problem）

一个分类器提示 LLM："Return one of {positive, negative, neutral}。"模型返回 "The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ..."。你的解析器崩溃。你的分类器的 F1 是 0.0。

自由形式生成不是一个契约。它是一个建议。一个生产系统需要一个契约。

2026 年存在三层。

1. **提示（Prompting）。** 礼貌地询问。"Return only the JSON object。"在前沿模型上约 80% 有效，在小模型上更少。
2. **原生结构化输出 API。** OpenAI `response_format`、Anthropic 工具使用、Gemini JSON 模式。在支持的 schema 上可靠。供应商锁定。
3. **约束解码（Constrained decoding）。** 在每个生成步骤修改 logits，使模型*不能*发出无效词元。100% 有效按构造。适用于任何本地模型。

本课为三者建立直觉，并命名何时使用哪一个。

## 概念（The Concept）

![Constrained decoding masking invalid tokens at each step](../assets/constrained-decoding.svg)

**约束解码如何工作。** 在每个生成步骤，LLM 产生一个约 100k 词元的完整词汇表的 logit 向量。一个 *logit processor* 坐在模型和采样器之间。它计算给定目标语法当前位置哪些词元是有效的——JSON Schema、正则表达式、上下文无关文法——并将所有无效词元的 logits 设置为负无穷。剩余 logits 的 softmax 将概率质量仅放在有效延续上。

2026 年的实现：

- **Outlines。** 将 JSON Schema 或正则表达式编译为有限状态机。每个词元获得 O(1) 有效下一个词元查找。基于 FSM，所以递归 schema 需要展平。
- **XGrammar / llguidance。** 上下文无关文法引擎。处理递归 JSON Schema。接近零解码开销。OpenAI 在 2025 年结构化输出实现中认可了 llguidance。
- **vLLM guided decoding。** 通过 Outlines、XGrammar 或 lm-format-enforcer 后端内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`。
- **Instructor。** 在任何 LLM 之上的 Pydantic 包装器。验证失败时重试。跨提供者，但不修改 logits——它依赖重试 + 结构化输出感知提示。

### 反直觉的结果

约束解码通常比无约束生成*更快*。两个原因。首先，它缩小了下一个词元搜索空间。其次，巧妙的实现对于强制词元完全跳过词元生成（像 `{"name": "` 这样的脚手架——每个字节都是确定的）。

### 让你付出代价的陷阱

字段顺序很重要。把 `answer` 放在 `reasoning` 前面，模型在思考之前就承诺了一个答案。JSON 是有效的。答案是错的。没有任何验证能捕捉到它。

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema 字段顺序是逻辑，不是格式。

```figure
constrained-decoder
```

## 构建它（Build It）

### 步骤 1：正则表达式约束生成（从零开始）

见 `code/main.py` 中的独立 FSM 实现。30 行代码中的核心思想：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 跟踪我们到目前为止满足了语法的哪些部分。`valid_tokens(state, tokenizer)` 计算哪些词汇表词元可以在不离开接受路径的情况下推进 FSM。

### 步骤 2：用于 JSON Schema 的 Outlines

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

零验证错误。永远。FSM 使无效输出无法到达。

### 步骤 3：用于跨提供者的 Pydantic 的 Instructor

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

不同的机制。Instructor 不接触 logits。它将 schema 格式化为提示，解析输出，并在验证失败时重试（默认 3 次）。适用于任何提供者。重试增加延迟和成本。跨提供者可移植性是卖点。

### 步骤 4：原生供应商 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

服务端约束解码。对于支持的 schema，可靠性与 Outlines 相当。没有本地模型管理。锁定到供应商。

## 陷阱

- **递归 schema。** Outlines 将递归展平到固定深度。树形结构化输出（嵌套注释、AST）需要 XGrammar 或 llguidance（基于 CFG）。
- **巨大枚举。** 10,000 选项枚举编译缓慢或超时。切换到检索器：首先预测前 k 个候选，约束到那些。
- **语法太严格。** 强制 `date: "YYYY-MM-DD"` 正则表达式，模型无法为缺失日期输出 `"unknown"`。模型通过发明一个日期来补偿。允许 `null` 或一个哨兵值。
- **过早承诺。** 见上面的字段顺序陷阱。始终将推理放在第一位。
- **没有 schema 的供应商 JSON 模式。** 纯 JSON 模式只保证有效的 JSON，不保证对你的用例有效。始终提供完整的 schema。

## 使用它（Use It）

2026 年技术栈：

| 情况（Situation） | 选择（Pick） |
|-----------|------|
| OpenAI/Anthropic/Google 模型，简单 schema | 原生供应商结构化输出 |
| 任何提供者，Pydantic 工作流，可以容忍重试 | Instructor |
| 本地模型，需要 100% 有效性，平面 schema | Outlines（FSM） |
| 本地模型，递归 schema | XGrammar 或 llguidance |
| 自托管推理服务器 | vLLM guided decoding |
| 批量处理，重试可接受 | Instructor + 最便宜的模型 |

## 发布它（Ship It）

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## 练习（Exercises）

1. **简单（Easy）。** 提示一个小开放权重模型（例如 Llama-3.2-3B），无约束解码，用于 `Review(sentiment, confidence, evidence_span)`。测量 100 条评论中解析为有效 JSON 的比例。
2. **中等（Medium）。** 使用 Outlines JSON 模式相同的语料库。比较合规率、延迟和语义准确性。
3. **困难（Hard）。** 从零开始实现一个电话号码（`\d{3}-\d{3}-\d{4}`）的正则表达式约束解码器。在 1000 个样本上验证 0 个无效输出。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| 约束解码（Constrained decoding） | 强制有效输出 | 在每个生成步骤掩盖无效词元 logits。 |
| Logit processor | 约束的东西 | 函数：`(logits, state) -> masked_logits`。 |
| FSM | 有限状态机 | 编译的语法表示；O(1) 有效下一个词元查找。 |
| CFG | 上下文无关文法 | 处理递归的语法；比 FSM 慢但更具表达力。 |
| Schema field order | 重要吗？ | 是的——第一个字段承诺；始终将推理放在答案之前。 |
| Guided decoding | vLLM 对它的称呼 | 相同的概念，集成到推理服务器中。 |
| JSON mode | OpenAI 的早期版本 | 保证 JSON 语法；不保证 schema 匹配。 |

## 延伸阅读（Further Reading）

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — Outlines 论文。
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — 快速的基于 CFG 的约束解码。
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — 推理服务器集成。
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — API 参考 + 陷阱。
- [Instructor library](https://python.useinstructor.com/) — 跨提供者的 Pydantic + 重试。
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 基准测试 6 个约束解码框架。
