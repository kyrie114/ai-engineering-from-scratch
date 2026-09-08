# 对话状态跟踪（Dialogue State Tracking）

> "I want a cheap restaurant in the north... actually make it moderate... and add Italian。"三轮，三次状态更新。DST 保持槽位值字典同步，以便预订工作。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 20 (Structured Outputs)
**Time:** ~75 minutes

## 问题（The Problem）

在面向任务的对话系统中，用户的目标被编码为一组槽位-值对：`{cuisine: italian, area: north, price: moderate}`。每个用户轮次可以添加、更改或删除一个槽位。系统必须读取整个对话并正确输出当前状态。

得到一个槽位错误，系统就预订了错误的餐厅，安排了错误的航班，或收取了错误的费用。DST 是用户所说与后端执行之间的铰链。

尽管有 LLMs，它在 2026 年仍然重要：

- 合规敏感领域（银行、医疗、航空预订）需要确定性的槽位值，而不是自由形式生成。
- 工具使用智能体仍然需要在调用 API 之前进行槽位解析。
- 多轮修正比看起来难："actually no, make it Thursday。"

现代管道：经典 DST 概念 + LLM 抽取器 + 结构化输出防护栏。

## 概念（The Concept）

![DST: dialog history → slot-value state](../assets/dst.svg)

**任务结构。** 一个 schema 定义领域（restaurant、hotel、taxi）及其槽位（cuisine、area、price、people）。每个槽位可以是空的、用封闭集合中的值填充（price：{cheap, moderate, expensive}），或自由形式值（name："The Copper Kettle"）。

**两种 DST 表述。**

- **分类。** 对于每个（槽位，候选值）对，预测是/否。适用于封闭词汇槽位。标准 2020 年之前。
- **生成。** 给定对话，将槽位生成为自由文本。适用于开放词汇槽位。现代默认。

**度量。** 联合目标准确性（Joint Goal Accuracy, JGA）——*每个*槽位都正确的轮次比例。全有或全无。2026 年 MultiWOZ 2.4 排行榜顶端约 83%。

**架构。**

1. **基于规则（槽位正则 + 关键词）。** 狭窄领域的强基线。可调试。
2. **TripPy / BERT-DST。** 带有 BERT 编码的复制式生成。LLM 之前的标准。
3. **LDST（LLaMA + LoRA）。** 带有领域-槽位提示的指令调优 LLM。在 MultiWOZ 2.4 上达到 ChatGPT 级别的质量。
4. **无本体（2024-26）。** 跳过 schema；直接生成槽位名称和值。处理开放领域。
5. **提示 + 结构化输出（2024-26）。** 带有 Pydantic schema + 约束解码的 LLM。5 行代码，生产就绪。

### 经典失败模式

- **跨轮次指代消解。** "Let's stay with the first option。"需要解析哪个选项。
- **覆盖 vs 追加。** 用户说 "add Italian。"你是替换 cuisine 还是追加？
- **隐式确认。** "OK cool"——那接受提供的预订了吗？
- **修正。** "Actually make it 7 pm。"必须更新时间而不清除其他槽位。
- **对前一个系统话语的指代消解。** "Yes, that one。"哪个 "that"？

```figure
n5-slot-tracker
```

## 构建它（Build It）

### 步骤 1：基于规则的槽位抽取器

见 `code/main.py`。正则 + 同义词字典覆盖狭窄领域中 70% 的规范话语：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

在规范词汇之外脆弱。适用于确定性槽位确认。

### 步骤 2：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三个不变量：

- 永远不要重置用户没有触碰的槽位。
- 显式否定（"never mind the cuisine"）必须清除。
- 用户修正（"actually..."）必须覆盖，而不是追加。

### 步骤 3：使用结构化输出的 LLM 驱动 DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证有效的状态对象。没有正则，没有 schema 不匹配，没有幻觉槽位。

### 步骤 4：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准：系统在所有槽位上获得正确的轮次比例是多少？对于 MultiWOZ 2.4，2026 年的顶级系统：80-83%。你的领域内系统应该在你的狭窄词汇上超过那个，否则 LLM 基线会击败你。

### 步骤 5：处理修正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

在检测到的修正时，覆盖最后更新的槽位而不是追加。没有 LLM 帮助很难做对。现代模式：始终让 LLM 从历史重新生成整个状态，而不是增量更新——这自然地处理修正。

## 陷阱

- **完整历史重新生成成本。** 让 LLM 每轮重新生成状态总共花费 O(n²) 个词元。限制历史或总结旧轮次。
- **Schema 漂移。** 事后添加新槽位破坏旧训练数据。为你的 schema 做版本控制。
- **大小写敏感性。** "Italian" vs "italian" vs "ITALIAN"——到处规范化。
- **隐式继承。** 如果用户之前指定了 "for 4 people，"对不同时间的新请求不应该清除 people。始终传递完整历史。
- **自由形式 vs 封闭集。** 名称、时间和地址需要自由形式槽位；菜系和区域是封闭的。在 schema 中混合两者。

## 使用它（Use It）

2026 年技术栈：

| 情况 | 方法 |
|-----------|----------|
| 狭窄领域（一个或两个意图） | 基于规则 + 正则 |
| 宽领域，有标签数据 | LDST（在 MultiWOZ 风格数据上进行 LLaMA + LoRA 指令调优） |
| 宽领域，无标签，生产就绪 | LLM + Instructor + Pydantic schema |
| 语音/语音 | ASR + 归一化器 + LLM-DST |
| 多领域预订流程 | 每个领域 Pydantic 模型的 Schema-guided LLM |
| 合规敏感 | 基于规则主要，LLM 回退配合确认流程 |

## 发布它（Ship It）

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## 练习（Exercises）

1. **简单（Easy）。** 在 `code/main.py` 中为 3 个槽位（cuisine、area、price）构建基于规则的状态跟踪器。在 10 个手工制作的对话上测试。测量 JGA。
2. **中等（Medium）。** 使用 Instructor + Pydantic + 小 LLM 进行相同的数据集。比较 JGA。检查最难的轮次。
3. **困难（Hard）。** 实现两者并路由：基于规则主要，当基于规则的发出 <2 个槽位时 LLM 回退。测量组合 JGA和每轮推理成本。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| DST | 对话状态跟踪 | 在对话轮次之间维护槽位值字典。 |
| 槽位 | 用户意图的单位 | 后端需要的命名参数（cuisine、date）。 |
| 领域 | 任务领域 | Restaurant、hotel、taxi——槽位集合。 |
| JGA | 联合目标准确性 | 每个槽位都正确的轮次比例。全有或全无。 |
| MultiWOZ | 基准 | 多领域 WOZ 数据集；标准 DST 评估。 |
| 无本体 DST | 没有 schema | 直接生成槽位名称和值，没有固定列表。 |
| 修正 | "Actually..." | 覆盖先前填充槽位的轮次。 |

## 延伸阅读（Further Reading）

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — 规范基准。
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA 用于 DST 的指令调优。
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — 复制式 DST 主力。
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — 基于 EM 的无监督 TOD。
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — 规范 DST 结果。
