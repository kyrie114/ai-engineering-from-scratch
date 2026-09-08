# 聊天机器人——从规则到神经再到 LLM 智能体（Chatbots — Rule-Based to Neural to LLM Agents）

> ELIZA 用模式匹配回复。DialogFlow 映射意图。GPT 从权重回答。Claude 运行工具并验证。每个时代都解决了前一个时代最严重的失败。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 minutes

## 问题（The Problem）

用户说 "I want to change my flight。"系统必须弄清楚他们想要什么，缺少什么信息，如何获取它，以及如何完成操作。然后用户说 "wait, what if I cancel instead?" 系统必须记住上下文，切换任务，并保持状态。

对话对 ML 系统来说很难。输入是开放式的。输出必须在多轮中连贯。系统可能需要对世界采取行动（更改航班，收费）。每个错误步骤都对用户可见。

聊天机器人架构已经循环了四种范式，每一种的引入都是因为前一种失败得太明显。本课按顺序介绍它们。2026 年的生产格局是最后两种的混合。

## 概念（The Concept）

![Chatbot evolution: rule-based → retrieval → neural → agent](../assets/chatbot.svg)

### 脚本化的一半世纪，1950-2001

第一个范式没有持续五年。它持续了五十年。了解它的弧线很重要，因为其中的每个系统都是同一台机器——匹配输入，发出预设回复，更新一点状态——五十年来向这台机器添加规则从未产生一般情况。这就是范式二到四存在的原因。

**1950。** Turing 通过提出一个操作性替代方案回避了 "机器能思考吗？"：如果审讯者无法通过电传打字区分机器和人，哲学问题就无关紧要了。对话成为该领域的基准，在该领域还没有名字之前。

**1956。** 这个名字到来了——达特茅斯的一个夏季研讨会以 "可以原则上如此精确地描述智能的每个特征以至于可以制造一台机器来模拟它" 的猜想 coined 了 "人工智能"。提案预算了两个月来取得实质性进展。

**1966。** ELIZA 发布了你在步骤 1 中构建的反射技巧：分解规则从输入中提取片段，重组规则将它们回显为问题。大约 200 个模式，零状态，零理解——用户仍然对它吐露心声。Weizenbaum 余下的职业生涯都对这么少的机械就能做到这一点感到震惊。

**1972。** PARRY，在斯坦福建造来模拟偏执狂，添加了 ELIZA 缺乏的东西：内部状态。恐惧、愤怒和不信任的数值变量在每一轮更新并控制哪个脚本触发，所以相同的输入根据到目前为止的对话产生不同的响应。在一次盲法转录测试中，精神病学家在机会水平上区分 PARRY 和人类患者。它是人格调节的直接祖先——以三个浮点数实现的系统提示。同年，两个机器人被指向 ARPANET 上的彼此：一个治疗师脚本采访一个偏执狂状态机，网络上的第一个机器人对机器人对话。

**1995。** ALICE 用 AIML（一种用于模式-模板对的 XML 方言）扩展了 ELIZA 配方。大约 40,000 个手写类别，三次 Loebner 奖胜利。它证明了基于规则的系统的扩展定律：更多规则带来覆盖率，永远没有一般性。每个规则都是某人必须维护的责任。

**2001。** SmarterChild 将配方呈现在 3000 万即时通讯用户面前，并添加了后端查找——天气、股票、电影时间——拼接到模板中。眯起眼睛看，它就是一个穿着 2001 服装的工具调用：解析意图，调用服务，将结果渲染到回复中。

五十年，一个机制，不断增加的规则数量。这个范式的结束不是因为任何人驳斥了它，而是因为手写状态机的维护成本随着覆盖范围线性增长，而用户期望随着他们上周看到的东西增长。

```figure
chatbot-lineage
```

**基于规则（Rule-based, ELIZA, AIML, DialogFlow）。** 手写的模式匹配用户输入并产生响应。意图分类器路由到预定义的流程。槽位填充（slot-filling）状态机收集所需信息。在为其设计的狭窄范围内表现出色。在其外部立即失败。仍然在安全关键领域（银行认证、航空预订）交付，其中幻觉不被容忍。

**基于检索（Retrieval-based）。** 一个 FAQ 风格的系统。编码每一对（话语，响应）。在运行时，编码用户的消息并检索最近的存储响应。想想 Zendesk 经典的 "相似文章" 功能。比规则更好地处理释义。没有生成，所以没有幻觉。

**神经（Neural, seq2seq）。** 在对话日志上训练的编码器-解码器。从头开始生成响应。流利但容易出现通用输出（"I don't know"）和事实漂移。从不稳定在主题上。谷歌、Facebook 和微软在 2016-2019 年都有令人失望的聊天机器人的原因。

**LLM 智能体（LLM agents）。** 一个包装在循环中的语言模型，该循环规划、调用工具并验证结果。不是一个有长提示的聊天机器人。一个智能体循环：计划 → 调用工具 → 观察结果 → 决定下一步。检索优先的锚定（RAG）防止它产生幻觉。工具调用让它真正做事。这是 2026 年的架构。

这四个范式不是顺序替换。一个 2026 年的生产聊天机器人路由通过所有四个：基于规则用于认证和破坏性操作，检索用于 FAQ，神经生成用于自然措辞，LLM 智能体用于模糊的开放式查询。

## 构建它（Build It）

### 步骤 1：基于规则的模式匹配

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20 行代码的 ELIZA。反射技巧（"I feel sad" → "Why do you feel sad"）是 Weizenbaum 1966 年的规范心理治疗师演示。仍然有指导意义。

### 步骤 2：基于检索（FAQ）

这个说明性片段需要 `pip install sentence-transformers`（它引入 torch）。本课的 `code/main.py` 使用 stdlib Jaccard 相似度，因此本课无需外部依赖即可运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒绝是关键设计选择。如果最佳匹配不够接近，返回 `None` 并让系统升级。

### 步骤 3：神经生成（基线）

使用一个小的指令调优编码器-解码器（FLAN-T5）或一个微调的对话模型。2026 年单独在生产中无法使用（矛盾、偏离主题漂移、事实废话），但在混合系统内部用于自然措辞交付。DialoGPT 风格的仅解码器模型需要显式的轮次分隔符和 EOS 处理来产生连贯的回复；FLAN-T5 text2text 管道开箱即用于教学示例。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 步骤 4：LLM 智能体循环

2026 年生产形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

三件事要注意。工具（Tools）是 LLM 可以调用的可调用函数。当 LLM 返回最终答案而不是工具调用时，循环终止。步骤预算防止模糊任务上的无限循环。

真实生产添加：检索优先锚定（在每次 LLM 调用之前注入相关文档）、防护栏（无确认拒绝破坏性操作）、可观察性（记录每一步）和评估（自动化检查智能体行为是否符合规范）。

### 步骤 5：混合路由

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式：破坏性操作用确定性规则，FAQ 用检索，其他所有用 LLM 智能体。这就是 2026 年客户支持系统中交付的内容。

## 使用它（Use It）

2026 年技术栈：

| 使用场景 | 架构（Architecture） |
|---------|---------------|
| 预订、支付、认证 | 基于规则的状态机 + 槽位填充 |
| 客户支持 FAQ | 在策划答案上的检索 |
| 开放式帮助聊天 | 带有 RAG + 工具调用的 LLM 智能体 |
| 内部工具 / IDE 助手 | 带有工具调用的 LLM 智能体（搜索、读取、写入） |
| 伴侣 / 角色聊天机器人 | 带有角色系统提示、知识检索的调优 LLM |

始终在生产中使用混合路由。没有单一架构能很好地处理每个请求。路由层本身通常是一个小的意图分类器。

## 仍然交付的失败模式

- **自信的捏造（Confident fabrication）。** LLM 智能体声称它完成了一个它没有完成的操作。缓解：验证结果，记录工具调用，不要让 LLM 在没有成功的工具返回的情况下声称做了什么。
- **提示注入（Prompt injection）。** 用户插入覆盖系统提示的文本。在 OWASP LLM 应用程序 Top 10 2025 中排名 LLM01。两种风格：直接注入（粘贴到聊天中）和间接注入（隐藏在智能体读取的文档、电子邮件或工具输出中）。

  按场景测量的攻击成功率不同。在通用工具使用和编码基准上，前沿模型的测量成功率约为 0.5-8.5%。针对 AI 编码智能体的自适应攻击等特定高风险设置已达到约 84%。生产 CVE 包括 EchoLeak（CVE-2025-32711，CVSS 9.3）——一个由攻击者控制的邮件触发的 Microsoft 365 Copilot 中的零点击数据泄露漏洞。

  缓解措施：在整个循环中将用户输入视为不可信；在工具调用之前进行消毒；将工具输出与主提示隔离；使用计划-验证-执行（Plan-Verify-Execute, PVE）模式，智能体首先计划，然后在执行之前根据该计划验证每个操作（这阻止工具结果注入新的计划外操作）；对破坏性操作要求用户确认；对工具范围应用最小权限。

  没有任何提示工程能完全消除这种风险。需要外部运行时防御层（LLM Guard、允许列表验证、语义异常检测）。

- **范围蔓延（Scope creep）。** 智能体偏离任务，因为工具调用返回了 tangential 相关信息。缓解：狭窄的工具合同；保持系统提示聚焦；添加越轨率评估。
- **无限循环（Infinite loops）。** 智能体不断调用同一个工具。缓解：步骤预算、工具调用去重、"我们是否在进步" 的 LLM 评判。
- **上下文窗口耗尽（Context window exhaustion）。** 长对话将最早的推出上下文。缓解：总结旧轮次，按相似性检索相关过去轮次，或使用长上下文模型。

## 发布它（Ship It）

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 练习（Exercises）

1. **简单（Easy）。** 用 10 个模式实现上述基于规则的响应，用于咖啡店订购机器人。测试边缘情况：双重订单、修改、取消、不明确的意图。
2. **中等（Medium）。** 构建一个混合 FAQ + LLM 回退。50 个 SaaS 产品的策划 FAQ 条目，LLM 回退，通过文档站点检索。在 100 个真实支持问题上测量拒绝率和准确性。
3. **困难（Hard）。** 用三个工具（搜索、读取用户数据、发送邮件）实现上述智能体循环。用 50 个测试场景（包括提示注入尝试）运行评估。报告越轨率、失败任务率和任何注入成功。

## 关键术语（Key Terms）

| 术语（Term） | 人们怎么说（What people say） | 它实际意味着什么（What it actually means） |
|------|-----------------|-----------------------|
| 意图（Intent） | 用户想要什么 | 分类标签（book_flight, reset_password）。路由到处理程序。 |
| 槽位（Slot） | 一块信息 | 机器人需要的参数（日期、目的地）。槽位填充是一系列询问。 |
| RAG | 检索加生成 | 检索相关文档，然后锚定 LLM 的响应。 |
| 工具调用（Tool call） | 函数调用 | LLM 发出带有名称 + 参数的结构化调用。运行时执行，返回结果。 |
| 智能体循环（Agent loop） | 计划、行动、验证 | 控制器，运行 LLM 调用与工具调用交错，直到任务完成。 |
| 提示注入（Prompt injection） | 用户攻击提示 | 试图覆盖系统提示的恶意输入。 |

## 延伸阅读（Further Reading）

- [Turing (1950). Computing Machinery and Intelligence](https://academic.oup.com/mind/article/LIX/236/433/986238) — 使对话成为该领域基准的论文。
- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 原始基于规则的聊天机器人论文。
- [Colby, Weber, Hilf (1971). Artificial Paranoia](https://doi.org/10.1016/0004-3702(71)90002-6) — PARRY 的情感变量架构，第一个有状态的聊天机器人。
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — 谷歌在 LLM 智能体接管之前的晚期神经聊天机器人论文。
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 命名智能体循环模式的论文。
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2024 年生产指导，在 2026 年仍然成立。
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — 提示注入论文。
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 使提示注入成为顶级安全问题的排名。
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — 实用的编排层防御，包括计划-验证-执行和用户确认流程。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 间接提示注入的规范零点击数据泄露 CVE。解释为什么具有写入访问权限的智能体需要运行时防御。
