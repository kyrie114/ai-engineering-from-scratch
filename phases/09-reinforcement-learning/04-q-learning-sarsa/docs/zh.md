# Temporal Difference — Q-Learning & SARSA（时序差分——Q-Learning 与 SARSA）

> 蒙特卡洛方法要等到情节结束才更新。TD 通过自举下一状态的价值估计，在每一步之后都进行更新。Q-learning 是离策略且乐观的；SARSA 是 on-policy 且谨慎的。两者都是一行代码。两者都支撑本阶段中的每一个深度强化学习方法。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming), Phase 9 · 03 (Monte Carlo)
**Time:** ~75 分钟

## The Problem（问题）

蒙特卡洛方法有效，但它有两个昂贵的要求。它需要终止的情节，并且只有在最终回报到达后才更新。如果你的情节是 1000 步，MC 要等 1000 步才更新任何东西。它高方差、低偏差，实践中速度慢。

动态规划有相反的 profile——零方差自举备份——但需要已知模型。

时序差分（TD）学习取两者之长。从单个转移 `(s, a, r, s')`，形成一步目标 `r + γ V(s')` 并将 `V(s)` 向它 nudges。不需要模型。不需要完整情节。由于在 RHS 使用近似 `V` 带来的偏差，但方差比 MC  dramatically 低，并且从第一步开始在线更新。

这是现代强化学习——DQN、A2C、PPO、SAC——旋转的枢纽。Phase 9 的其余部分都是在这节课中你将编写的单步 TD 更新之上叠加的函数逼近和技巧的层次。

## The Concept（概念）

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')（Q-learning vs SARSA：离策略 max vs on-policy Q(s', a')）](../assets/td.svg)

**V 的 TD(0) 更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

括号中的量是 TD 误差 `δ = r + γ V(s') - V(s)`。它是 MC 中 `G_t - V(s_t)` 的在线类比。收敛要求 `α` 满足 Robbins-Monro（`Σ α = ∞`，`Σ α² < ∞`）并且所有状态被无限次访问。

**Q-learning。** 一种用于控制的离策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` 假设从 `s'`  onward 将遵循*贪婪*策略，无论智能体实际采取什么动作。这种解耦使 Q-learning 在智能体通过 ε-贪婪探索时学习 `Q*`。Mnih et al. (2015) 将其转换为 Atari 上的深度 Q-learning（Lesson 05）。

**SARSA。** 一种 on-policy TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

这个名字是元组 `(s, a, r, s', a')`。SARSA 使用智能体*实际*下一步采取的动作 `a'`，而不是贪婪的 `argmax`。收敛到当前运行的 ε-贪婪 `π` 的 `Q^π`，在极限 `ε → 0` 时变为 `Q*`。

**悬崖行走的差异。** 在经典的悬崖行走任务（掉下悬崖 = 奖励 -100）上，Q-learning 学习沿着悬崖边缘的最优路径，但在探索期间偶尔会受到惩罚。SARSA 学习离悬崖一步远的更安全路径，因为它将探索噪声纳入其 Q 值。随着训练，两者在 `ε → 0` 时都达到最优。实践中这很重要：当探索在部署时实际发生时，SARSA 的行为更保守。

**Expected SARSA。** 将 `Q(s', a')` 替换为它在 `π` 下的期望值：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

比 SARSA 方差更低（没有 `a'` 的样本），相同的 on-policy 目标。通常是现代教科书中的默认选择。

**n-step TD 和 TD(λ)。** 通过在自举前等待 `n` 步，在 TD(0) 和 MC 之间插值。`n=1` 是 TD，`n=∞` 是 MC。TD(λ) 用几何权重 `(1-λ)λ^{n-1}` 平均所有 `n`。大多数深度强化学习使用 3 到 20 之间的 `n`。

```figure
qlearning-gridworld
```

## Build It（动手实现）

### Step 1: ε-贪婪策略上的 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行代码。与 Q-learning 的*唯一*区别是目标行。

### Step 2: Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 将目标与行为解耦。那一个符号就是 on-policy 和 off-policy 之间的区别。

### Step 3: 学习曲线

跟踪每 100 个情节的平均回报。Q-learning 在简单的确定性 GridWorld 上收敛更快；SARSA 在悬崖行走上更保守。在 `code/main.py` 中的 4×4 GridWorld 上，两者在约 2000 个情节后都接近最优，其中 `α=0.1, ε=0.1`。

### Step 4: 与 DP 真相比较

运行值迭代（Lesson 02）以获得 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。一个健康的表格 TD 智能体在 10,000 个情节后在 4×4 GridWorld 上落在 `~0.5` 范围内。

## Pitfalls（陷阱）

- **初始 Q 值很重要。** 乐观初始化（`Q = 0` 对于负奖励任务）鼓励探索。悲观初始化可以永远困住贪婪策略。
- **α 调度。** 常数 `α` 对于非平稳问题没问题。衰减 `α_n = 1/n` 在理论上给出收敛但在实践中太慢——将 `α` 固定在 `[0.05, 0.3]` 中并监控学习曲线。
- **ε 调度。** 从高（`ε=1.0`）开始，衰减到 `ε=0.05`。"GLIE"（极限贪婪与无限探索）是收敛条件。
- **Q-learning 中的 max 偏差。** 当 Q 嘈杂时，`max` 操作符向上偏差。导致高估——Hasselt 的 Double Q-learning（Lesson 05 中 DDQN 使用）用两个 Q 表修复这个问题。
- **非终止情节。** TD 可以在没有终点的情况下学习，但你需要要么限制步数，要么在限制处正确处理自举。标准做法：将限制视为非终点，继续自举。
- **状态哈希。** 如果状态是元组/张量，使用可哈希的键（tuple，不是 list；四舍五入的 float 元组，不是原始值）。

## Use It（应用场景）

2026 年的 TD 格局：

| 任务 | 方法 | 原因 |
|------|--------|--------|
| 小型表格环境 | Q-learning | 直接学习最优策略。 |
| On-policy 安全关键 | SARSA / Expected SARSA | 探索期间保守。 |
| 高维状态 | DQN（Phase 9 · 05） | 具有 replay 和目标网络的神经网络 Q 函数。 |
| 连续动作 | SAC / TD3（Phase 9 · 07） | Q 网络上的 TD 更新；策略网络发出动作。 |
| LLM RL（基于奖励模型） | PPO / GRPO（Phase 9 · 08, 12） | 通过 GAE 具有 TD 风格优势的 actor-critic。 |
| 离线强化学习 | CQL / IQL（Phase 9 · 08） | 具有保守正则化的 Q-learning。 |

2026 年论文中你读到的 90% 的"强化学习"都是 Q-learning 或 SARSA 的某种变体。在阅读更深之前，先在指尖理解表格更新。

## Ship It（交付物）

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## Exercises（练习）

1. **Easy. 在 4×4 GridWorld 上实现 Q-learning 和 SARSA。绘制 2000 个情节的学习曲线（每 100 个情节的平均回报）。谁收敛更快？**
2. **Medium. 构建一个悬崖行走环境（4×12，最后一行是奖励 -100 并重置到起点的悬崖）。比较 Q-learning 和 SARSA 的最终策略。截取每条路径的屏幕截图。哪条更接近悬崖？**
3. **Hard. 实现 Double Q-learning。在嘈杂奖励 GridWorld 上（每步奖励添加高斯噪声 σ=5），展示 Q-learning 高估 `V*(0,0)` 一个显著量，而 Double Q-learning 没有。**

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| TD error | "The update signal" | `δ = r + γ V(s') - V(s)`, the bootstrapped residual. |
| TD(0) | "One-step TD" | Update after every transition using only the next state's estimate. |
| Q-learning | "Off-policy RL 101" | TD update with `max` over next-state actions; learns `Q*` regardless of behavior policy. |
| SARSA | "On-policy Q-learning" | TD update using the actual next action; learns `Q^π` for current ε-greedy π. |
| Expected SARSA | "The low-variance SARSA" | Replace sampled `a'` with its expectation under π. |
| GLIE | "Correct exploration schedule" | Greedy in the Limit with Infinite Exploration; needed for Q-learning convergence. |
| Bootstrapping | "Using current estimate in the target" | What distinguishes TD from MC. Source of bias but massive variance reduction. |
| Maximization bias | "Q-learning overestimates" | `max` over noisy estimates is upward-biased; fixed by Double Q-learning. |

## Further Reading（延伸阅读）

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始论文和收敛证明。
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q-learning、Expected SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 对最大化偏差的修复。
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — expected SARSA 动机。
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — 提出 SARSA 的论文（当时称为"modified connectionist Q-learning"）。
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — 将 TD(0) 推广到 TD(n)，通往 PPO 中 GAE 的路径。
