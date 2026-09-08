# RL for Games — AlphaZero, MuZero, and the LLM-Reasoning Era（游戏强化学习——AlphaZero、MuZero 与 LLM 推理时代）

> 1992 年：TD-Gammon 用纯 TD 在双陆棋上击败人类冠军。2016 年：AlphaGo 击败李世石。2017 年：AlphaZero 从零开始统治国际象棋、将棋和围棋。2024 年：DeepSeek-R1 证明了相同配方，用 GRPO 替换 PPO，在推理上有效。游戏是推动本阶段每一次突破的基准。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 05 (DQN), Phase 9 · 08 (PPO), Phase 9 · 09 (RLHF), Phase 9 · 10 (MARL)
**Time:** ~120 分钟

## The Problem（问题）

游戏拥有强化学习想要的一切。干净的奖励（赢/输）。无限情节（self-play reset）。完美模拟（游戏*就是*模拟器）。离散或小型连续动作空间。强制对抗鲁棒性的多智能体结构。

游戏也是每一次主要强化学习突破的测试场。TD-Gammon（双陆棋，1992）。Atari-DQN（2013）。AlphaGo（2016）。AlphaZero（2017）。OpenAI Five（Dota 2，2019）。AlphaStar（StarCraft II，2019）。MuZero（学习模型，2019）。AlphaTensor（矩阵乘法，2022）。AlphaDev（排序算法，2023）。DeepSeek-R1（数学推理，2025）——最新的演示，证明游戏强化学习技巧在文本上有效。

本课通过单一统一镜头survey 三个标志性架构——AlphaZero、MuZero 和 GRPO：**self-play + 搜索 + 策略改进**。每一个都推广前一个；GRPO 特别是将 AlphaZero 的配方应用于 LLM 推理，token 作为动作，数学验证作为胜利信号。

## The Concept（概念）

![AlphaZero ↔ MuZero ↔ GRPO: same loop, different environments（AlphaZero ↔ MuZero ↔ GRPO：相同循环，不同环境）](../assets/rl-games.svg)

**统一循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero (2017)。** Silver et al. 给定一个有已知规则的游戏（国际象棋、将棋、围棋）：

- 策略-价值网络：一个 tower `f_θ(s) → (p, v)`。`p` 是合法着法上的先验。`v` 是预期的游戏结果。
- 蒙特卡洛树搜索 (MCTS)：在每个着法，扩展可能延续的树。使用 `(p, v)` 作为先验 + 自举。通过 UCB (PUCT) 选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- Self-play：智能体对智能体玩游戏。在着法 `t`，MCTS 访问分布 `π_t` 成为策略训练目标。
- 损失：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是游戏结果（+1 / 0 / -1）。

零人类知识。零手工启发式。一个单一配方，在各自几千万次 self-play 游戏后掌握了国际象棋、将棋和围棋。

**MuZero (2019)。** Schrittwieser et al. 移除了规则必须已知的要求。

- 不固定环境，而是学习一个*潜在动态模型* `(h, g, f)`：
  - `h(s)`：将观察编码为潜在状态。
  - `g(s_latent, a)`：预测下一个潜在状态 + 奖励。
  - `f(s_latent)`：预测策略先验 + 价值。
- MCTS 在*学习的潜在空间*中运行。相同的搜索，相同的训练循环。
- 在 Go、国际象棋、将棋*和* Atari 上有效——一个算法，没有规则知识。

**Stochastic MuZero (2022)。** 添加随机动态和机会节点；扩展到双陆棋类游戏。

**Muesli, Gumbel MuZero (2022-2024)。** 样本效率和确定性搜索的改进。

**GRPO (2024-2025)。** DeepSeek-R1 配方。相同的 AlphaZero 形状循环，应用于语言模型推理：

- "游戏"：回答一个数学 / 代码 / 推理问题。"赢" = 验证器（测试用例通过，数字答案匹配）返回 1。
- 策略：LLM。动作：token。状态：提示 + 响应sofar。
- 没有 critic（PPO 风格 V_φ）。相反，对于每个提示，从策略中采样 `G` 个补全。为每个计算奖励。使用 **group-relative advantage** `A_i = (r_i - mean_r) / std_r` 作为 REINFORCE 风格更新的信号。
- KL 惩罚到参考策略以防止漂移（像 RLHF）。
- 完整损失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

没有奖励模型，没有 critic，没有 MCTS。Group-relative baseline 替代了所有三个。在推理基准上匹配或超过 PPO-RLHF 质量，但计算量是一小部分。

**R1 配方的完整版。** DeepSeek-R1 (DeepSeek 2025) 是一篇论文中的两个模型：

- **R1-Zero。** 从 DeepSeek-V3 基础模型开始。没有 SFT。直接应用 GRPO，有两个奖励组件：*accuracy reward*（基于规则——最终答案是否解析为正确的数字 / 代码是否通过单元测试）和 *format reward*（完成是否将其 chain-of-thought 包裹在 `<think>…</think>` 标签中）。在数千步之后，平均响应长度从 ~100 增长到 ~10,000 token，数学基准分数攀升到接近 o1-preview 水平。模型从零开始学习推理。缺点：它的思维链通常不可读，混合语言，缺乏风格润色。
- **R1。** 用四阶段管道修复 R1-Zero 的可读性问题：
  1. **Cold-start SFT。** 收集几千个长 CoT 演示，具有干净的格式。对它们监督微调基础模型。这给出了可读的起点。
  2. **Reasoning-oriented GRPO。** 应用 GRPO，带有 accuracy+format 奖励加上 *language-consistency* 奖励以防止代码切换。
  3. **Rejection sampling + SFT round 2。** 从 RL checkpoint 采样 ~600K 推理轨迹，只保留那些具有正确最终答案和可读 CoT 的，并与 ~200K 非推理 SFT 示例（写作、QA、自我认知）组合。再次微调基础模型。
  4. **Full-spectrum GRPO。** 另一轮 RL，涵盖推理（基于规则的奖励）和一般对齐（有帮助/无害偏好基于奖励）。

结果在开放权重上匹配 o1 在 AIME 和 MATH-500 上，并且小到可以蒸馏。同一篇论文还发布了六个蒸馏密集模型（Qwen-1.5B 到 Llama-70B），通过在 R1 的推理轨迹上 SFT——学生在没有 RL 的情况下。强 RL 教师的蒸馏 consistently 在学生规模上击败从零开始的 RL。

**为什么用 GRPO 而不是 PPO 进行推理。** DeepSeekMath 论文（2024 年 2 月）中的三个原因：(1) 没有价值网络需要训练，内存减半；(2) group baseline 自然地处理推理任务产生的稀疏轨迹结束奖励；(3) 每提示归一化使不同难度问题的 advantages 可比，PPO 的单个 critic 不能。

**Search-free vs search-based。** 游戏已经分支：

- *具有长视界的完美信息游戏*（Go、国际象棋）：仍然是 search-based。AlphaZero / MuZero 主导。
- *LLM 推理*：生产中没有 MCTS  yet；GRPO 在完整 rollout 上，推理时的 best-of-N。过程奖励模型 (PRM) 暗示步骤级搜索正在被添加回来。

```figure
f3-selfplay-ladder
```

## Build It（动手实现）

`code/main.py` 中的代码实现了 **GRPO 缩影**——一个具有多个样本组的 bandit。该算法与 LLM 上的相同；只有策略和环境更简单。它教授 *损失* 和 *group-relative advantage*，这是 2025 年的创新。

### Step 1: 一个微小的验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

在真实 GRPO 中，验证器运行单元测试或检查数学相等性。

### Step 2: 策略：每个提示上 K 个答案 token 上的 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等同于以提示为条件的 LLM 的最终层输出。

### Step 3: 组采样和 group-relative advantage

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

Group-relative advantage 是 2024 年 DeepSeek 的技巧。不需要 critic。"baseline" 是组均值，归一化使用组标准差。

### Step 4: 与 REINFORCE 基线（无价值）比较

相同的设置，相同的计算，朴素 REINFORCE。GRPO 更快更稳定地收敛。

### Step 5: 观察熵和 KL

与 RLHF 相同的诊断：到参考的平均 KL，策略熵，随时间变化的奖励。一旦这些稳定，训练就完成了。

## Pitfalls（陷阱）

- **通过验证器游戏的奖励黑客。** GRPO 继承 RLHF 的风险：如果验证器是错误的或可利用的，LLM 会找到利用。鲁棒验证器（多个测试用例，形式化证明）重要。
- **组大小太小。** 组 baseline 的方差像 `1/√G`。低于 `G = 4`，advantage 信号嘈杂；标准选择是 `G = 8` 到 `64`。
- **长度偏差。** 不同长度的 LLM 补全有不同的对数概率。按 token 数量归一化，或使用序列级对数概率，或截断到最大长度。
- **纯 self-play 循环。** AlphaZero 风格训练可能陷入一般和游戏中的主导循环。通过多样化对手池（league play、Lesson 10）缓解。
- **搜索-策略不匹配。** AlphaZero 训练策略模仿搜索输出。如果策略网络太小，无法表示搜索的分布，训练就停滞了。
- **计算下限。** MuZero / AlphaZero 需要大规模计算。一个单一消融通常是数百 GPU 小时。缩影演示存在（例如，Connect Four 上的 AlphaZero）用于学习。
- **验证器覆盖。** 通过错误解决方案的单元测试会强化错误。设计能捕捉边缘情况的验证器。

## Use It（应用场景）

2026 年的游戏强化学习格局，按领域：

| 领域 | 主导方法 |
|--------|----------------|
| 双人零和棋盘游戏（Go、国际象棋、将棋） | AlphaZero / MuZero / KataGo |
| 不完美信息纸牌游戏（扑克） | CFR + 深度学习 (DeepStack, Libratus, Pluribus) |
| Atari / 像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大型多人策略（Dota、StarCraft） | PPO + self-play + league (OpenAI Five, AlphaStar) |
| LLM 数学 / 代码推理 | GRPO (DeepSeek-R1, Qwen-RL, open replications) |
| LLM 对齐 | DPO / RLHF-PPO（不是 GRPO；验证器是偏好，不是可验证的） |
| 机器人 | PPO + DR（不是游戏强化学习，但使用相同的策略梯度工具） |
| 组合问题 | AlphaZero 变体 (AlphaTensor, AlphaDev) |

*配方*——self-play、搜索增强改进、策略蒸馏——跨越文本、像素和物理控制。GRPO 是最年轻的实例；更多正在到来。

## Ship It（交付物）

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## Exercises（练习）

1. **Easy. 在 `code/main.py` 中实现 GRPO bandit。在 2 个提示 × 每个 4 个答案 token 上训练。用 `G=8` 在 < 1,000 次更新中收敛。**
2. **Medium. 插入 PPO（裁剪）和朴素 REINFORCE。在同一 bandit 上与 GRPO 比较样本效率和奖励方差。**
3. **Hard. 扩展到长度 2 的"推理链"：智能体发出两个 token，验证器奖励这对。测量 GRPO 如何处理两步序列的信用分配。（提示：每个*完整序列*计算 group advantage，传播到两个 token 位置。）**

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| MCTS | "Tree search with learned net" | Monte Carlo Tree Search; UCB1/PUCT selection with learned `(p, v)` priors. |
| AlphaZero | "Self-play + MCTS" | Policy-value net trained to match MCTS visits and game outcome. |
| MuZero | "Learned-model AlphaZero" | Same loop but in latent space via learned dynamics. |
| GRPO | "Critic-free PPO" | Group Relative Policy Optimization; REINFORCE with group-mean baseline + KL. |
| PUCT | "AlphaZero's UCB" | `Q + c · p · √N / (1 + N_a)` — balances value estimate with prior. |
| Self-play | "Agent vs past self" | Standard for zero-sum; symmetric training signal. |
| League play | "Population-based self-play" | Past + current + exploiters sampled as opponents. |
| Verifier reward | "Verifiable RL" | Reward comes from a deterministic checker (tests pass, answer matches). |
| Process reward | "PRM" | Scores each reasoning step, not just the final answer. |

## Further Reading（延伸阅读）

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)。
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)。
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4)。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z)。
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — 引入 GRPO 和 group-relative baseline 的论文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 完整的四阶段 R1 配方加上 R1-Zero 消融。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — 大规模 CFR + 深度学习。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — 开创一切的论文。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — 应用具有自定义奖励函数的 GRPO 的生产参考。
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — 多个规模下的 R1 配方的开放复制。
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书中对 self-play、搜索和 R1 在 LLM 规模上实例化的"设计的奖励"的框架。
