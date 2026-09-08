# Reward Modeling & RLHF（奖励建模与 RLHF）

> 人类无法为"好的助手响应"编写奖励函数，但他们可以比较两个响应并选出更好的一个。将奖励模型拟合到这些比较上，然后通过 PPO 对语言模型进行强化学习。Christiano 2017。InstructGPT 2022。将 GPT-3 转变为 ChatGPT 的配方。在 2026 年它大部分被 DPO 取代——但心理模型保留。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment), Phase 9 · 08 (PPO)
**Time:** ~45 分钟

## The Problem（问题）

你在下一个 token 预测目标上训练了一个语言模型。它写出语法正确的英语。它也撒谎、啰嗦，并且拒绝拒绝。你不能通过更多预训练来修复它——网络文本是问题，不是解药。

你想要一个标量奖励，它说"响应 A 比响应 B 更适合指令 X"。手动编写这个奖励函数是不可能的。"有帮助"不是 token 上的闭式表达式。但人类可以比较两个输出并标记偏好。这在大规模上收集很便宜。

RLHF (Christiano et al. 2017; Ouyang et al. 2022) 将偏好转换为奖励模型，然后通过 PPO 针对该奖励优化 LM。三步：SFT → RM → PPO。这是 2023-2025 年交付 ChatGPT、Claude、Gemini 和每一个其他对齐 LLM 的配方。

在 2026 年，PPO 步骤大部分被 DPO（Phase 10 · 08）取代，因为它更便宜，在调整对齐上几乎一样好。但*奖励模型*部分仍然是每一个 Best-of-N 采样器、每一个从可验证奖励进行 RL 的管道，以及每一个使用过程奖励模型的推理模型的基础。理解 RLHF，你就理解整个对齐堆栈。

## The Concept（概念）

![Three-stage RLHF: SFT, RM training on pairwise prefs, PPO with KL penalty（三阶段 RLHF：SFT、成对偏好的 RM 训练、带 KL 惩罚的 PPO）](../assets/rlhf.svg)

**Stage 1: 监督微调 (SFT)。** 从预训练的基础模型开始。在目标行为的人类书写演示（指令遵循响应、有帮助的回复等）上微调。结果：一个模型 `π_SFT`，它*偏向于好行为*，但仍有 unbounded 动作空间。

**Stage 2: 奖励模型训练。**

- 收集响应对 `(y_+, y_-)` 到提示 `x`，由人类标记为"y_+ 优于 y_-"。
- 训练奖励模型 `R_φ(x, y)` 给 `y_+` 分配更高的分数。
- 损失：**Bradley-Terry 成对逻辑**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ 是 sigmoid。奖励差异意味着偏好的 log-odds。BT 自 1952 年（Bradley-Terry）以来一直是标准，是现代 RLHF 中的主导选择。

- `R_φ` 通常从带有顶部标量 head 的 SFT 模型初始化。相同的 transformer backbone；一个单独的线性层输出奖励。

**Stage 3: 带 KL 惩罚的针对 RM 的 PPO。**

- 从 `π_SFT` 初始化可训练策略 `π_θ`。保留冻结的*参考* `π_ref = π_SFT`。
- 在响应 `y` 结束时的奖励：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 惩罚防止 `π_θ` 任意漂移 `π_SFT`——它是一个*正则化器*，不是一个硬信任区域。`β` 通常 `0.01`-`0.05`。
- 针对此奖励运行 PPO（Lesson 08）。Advantages 在 token 级轨迹上计算，但 RM 仅对完整响应评分。

**为什么用 KL？** 没有它，PPO 会 happily 找到奖励黑客策略——RM 仅在分布内完成上训练。分布外响应可能比任何人类书写的得分更高。KL 使 `π_θ` 保持在 RM 训练所在的流形附近。它是 RLHF 中最重要的旋钮。

**2026 年状态：**

- **DPO** (Rafailov 2023)：闭式代数将 Stage 2+3 坍缩为偏好数据上的单个监督损失。没有 RM，没有 PPO。在调整对齐基准上质量相同，但计算量是一小部分。Phase 10 · 08 涵盖。
- **GRPO** (DeepSeek 2024-2025)：具有 group-relative baseline 而不是 critic 的 PPO，奖励来自*验证器*（代码运行 / 数学答案匹配）而不是人类训练的 RM。推理模型的主导。Phase 9 · 12 涵盖。
- **过程奖励模型 (PRM)：** 对部分解决方案（每个推理步骤）评分，用于推理管道的 RLHF 和 GRPO 变体。
- **Constitutional AI / RLAIF：** 使用对齐的 LLM 而不是人类生成偏好。扩展偏好预算。

```figure
reward-model
```

## Build It（动手实现）

这节课使用表示为字符串的微小合成"提示"和"响应"。RM 是一个在 bag-of-tokens 表示上的线性评分器。没有真正的 LLM——管道的*形状*重要，而不是规模。见 `code/main.py`。

### Step 1: 合成偏好数据

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

在真实的 RLHF 中，这被人类标注者取代。形状——`(prompt, preferred_response, rejected_response)`——是相同的。

### Step 2: Bradley-Terry 奖励模型

线性评分：`R(x, y) = w · bag(y)`。训练以最小化 BT 成对逻辑损失：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

几百个更新后，`w` 给 good-word tokens 分配正权重，给 bad-word tokens 分配负权重。

### Step 3: RM 上的 PPO 类策略

我们的玩具策略从词汇表中产生单个 token。我们在 RM 下对 token 评分，计算 `log π_θ(token | prompt)`，添加 KL 到参考惩罚，并应用裁剪的 PPO 替代。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### Step 4: 监控 KL

每次更新跟踪平均 `KL(π_θ || π_ref)`。如果它悄悄超过 `~5-10`，策略已经从 `π_SFT` 漂移得很远——降低 `β` 正在上升或奖励黑客开始。这是真实 RLHF 中的顶级诊断。

### Step 5: 使用 TRL 的生产配方

一旦你理解了玩具管道，这里是将相同循环作为真实库用户编写的代码。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现——Stage 2 的 `RewardTrainer` 和 Stage 3 的 `PPOTrainer`（带有内置 KL 到参考）。

```python
# Stage 2: 从成对偏好的奖励模型
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: 针对带 KL 惩罚到 SFT 参考的 RM 的 PPO
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

库为你做的三件事。`adap_kl_ctrl=True` 实现自适应 β 调度：如果观察到的 KL 超过 `target_kl`，β 翻倍；如果低于一半，β 减半。参考模型按约定冻结——你不能意外地与 `policy` 共享参数。Value head 与策略位于相同的 backbone 上（`AutoModelForCausalLMWithValueHead` 附加一个标量 MLP head），这就是为什么 TRL 分别报告 `policy/kl` 和 `value/loss`。

## Pitfalls（陷阱）

- **过度优化 / 奖励黑客。** RM 是不完美的；`π_θ` 找到对抗完成，得分高但不好。症状：奖励无限攀升，而人工评估分数停滞或下降。修复：提前停止，提高 `β`，扩大 RM 训练数据。
- **长度黑客。** 在有用响应上训练的 RM 通常隐式奖励长度。策略学习填充响应。修复：长度归一化奖励，或使用长度感知 RM 的 RLAIF。
- **太小的 RM。** RM 需要至少和策略一样大。小 RM 不能忠实地对策略的输出评分。
- **KL 调优。** 太低的 β → 漂移和奖励黑客。太高的 β → 策略几乎不变。标准技巧是*自适应* β，以固定 KL 每步为目标。
- **偏好数据噪声。** ~30% 的人类标签是嘈杂或模糊的。通过在协议过滤数据上训练 RM 或对 BT 使用温度来校准。
- **离策略问题。** PPO 数据在第一次更新后略微离策略。像 Lesson 08 中那样监控 clip fraction。

## Use It（应用场景）

2026 年的 RLHF 是分层的：

| Layer | Target | Method |
|-------|--------|--------|
| 指令遵循、有帮助、无害 | 对齐 | DPO (Phase 10 · 08) 优于 RLHF-PPO。 |
| 推理正确性（数学、代码） | 能力 | 具有验证器奖励的 GRPO (Phase 9 · 12)。 |
| 长期多步任务 | Agentic | PPO / GRPO 带有步骤上的过程奖励模型。 |
| 安全 / 拒绝行为 | 安全 | 具有单独安全 RM 的 RLHF-PPO，或 Constitutional AI。 |
| 推理时的 Best-of-N | 快速对齐 | 在解码时使用 RM；不需要策略训练。 |
| 奖励蒸馏 | 推理计算 | 在冻结 LM 上训练小"奖励 head"。 |

RLHF 是 2022-2024 年的*方法。在 2026 年，生产对齐管道是 DPO-first，PPO 仅用于 RM 密集型或安全关键步骤。

## Ship It（交付物）

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## Exercises（练习）

1. **Easy. 在 `code/main.py` 中在 500 个合成偏好对训练 Bradley-Terry 奖励模型。在保留的 100 个对上测量成对准确率。应超过 90%。**
2. **Medium. 用 `β ∈ {0.0, 0.1, 1.0}` 运行玩具 PPO-RLHF 循环。对于每个，绘制 RM 分数 vs 更新中的 KL 到参考。哪些运行奖励黑客？**
3. **Hard. 在相同偏好数据上实现 DPO（闭式偏好似然损失）并与 RLHF-PPO 管道比较计算使用和最终 RM 分数。**

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| RLHF | "Alignment RL" | Three-stage SFT + RM + PPO pipeline (Christiano 2017, Ouyang 2022). |
| Reward Model (RM) | "The scoring net" | Learned scalar function fit to pairwise preferences via Bradley-Terry. |
| Bradley-Terry | "Pairwise logistic loss" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; the standard RM objective. |
| KL penalty | "Stay near the reference" | `β · KL(π_θ \|\| π_ref)` in the reward; the anti-reward-hacking regularizer. |
| Reward hacking | "Goodhart's law" | Policy exploits RM flaws; symptoms: reward up, human eval flat. |
| RLAIF | "AI-labeled preferences" | RLHF where labels come from another LM instead of humans. |
| PRM | "Process Reward Model" | Scores partial reasoning steps; used in reasoning pipelines. |
| Constitutional AI | "Anthropic's method" | AI-generated preferences guided by explicit rules. |

## Further Reading（延伸阅读）

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — 开启 RLHF 的论文。
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — ChatGPT 背后的配方。
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — 用于摘要的早期 RLHF。
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO；2026 年后的 RLHF 默认。
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF 和自批评循环。
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — HH 论文。
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — 生产 `RewardTrainer` 和 `PPOTrainer`。阅读 trainer 源代码以了解自适应 KL 和 value head 细节。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla — 带图的三阶段管道的规范演练。
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — 库；`examples/` 有 Llama、Mistral 和 Qwen 的端到端 RLHF 脚本。
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — 奖励假设观点；思考奖励黑客的必要先决条件。
