# Multi-Agent RL（多智能体强化学习）

> 单智能体强化学习假设环境是平稳的。将两个学习智能体放在同一个世界中，这个假设就失效了：每个智能体都是另一个环境的一部分，并且两者都在变化。多智能体强化学习是当马尔可夫假设不再成立时使学习收敛的一组技巧。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (Q-learning), Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~45 分钟

## The Problem（问题）

一个机器人学习在房间中导航是单智能体强化学习问题。一个足球队不是。AlphaStar vs StarCraft 对手不是。一个投标智能体市场不是。两辆车协商一个四向停车标志不是。许多对多的现实世界问题不是。

在每一个多智能体设置中，从任何一个智能体的角度来看，其他智能体*就是*环境的一部分。当它们学习和改变行为时，环境变得非平稳。马尔可夫性质——"下一个状态只依赖于当前状态和我的动作"——被违反，因为下一个状态也依赖于*其他*智能体的选择，并且它们的策略是移动的目标。

这打破了表格收敛证明（Q-learning 的保证假设平稳环境）。它也打破了朴素的深度强化学习：智能体在循环中追逐彼此，从不会收敛到稳定策略。你需要多智能体特定技术：集中式训练 / 分散式执行、反事实 baseline、league play、self-play。

2026 年应用：机器人集群、交通路由、自动驾驶车队、市场模拟器、多智能体 LLM 系统（Phase 16），以及任何有多于一个智能玩家的游戏。

## The Concept（概念）

![Four MARL regimes: indep, centralized critic, self-play, league（四种 MARL 制度：独立、集中式 critic、self-play、league）](../assets/marl.svg)

**形式化：马尔可夫博弈。** MDP 的推广：状态 `S`，联合动作 `a = (a_1, …, a_n)`，转移 `P(s' | s, a)`，每个智能体的奖励 `R_i(s, a, s')`。每个智能体 `i` 在其自己的策略 `π_i` 下最大化其自己的回报。如果奖励是相同的，它是**完全合作的**。如果是零和的，它是**对抗的**。如果是混合的，它是**一般和**。

**核心挑战：**

- **非平稳性。** 从智能体 `i` 的视角看，`P(s' | s, a_i)` 依赖于 `π_{-i}`，它在变化。
- **信用分配。** 共享奖励；哪个智能体导致了它？
- **探索协调。** 智能体必须探索互补策略，而不是冗余地探索相同状态。
- **可扩展性。** 联合动作空间随 `n` 呈指数增长。
- **部分可观察性。** 每个智能体只看到它自己的观察；全局状态是隐藏的。

**四种主导制度：**

**1. 独立 Q-learning / 独立 PPO (IQL, IPPO)。** 每个智能体学习它自己的 Q 或策略，将其他智能体视为环境的一部分。简单，有时有效（特别是具有作为 smoothing agent-modeling 技巧的 experience replay）。理论收敛：无。实践中：对松散耦合的任务没问题，对紧密耦合的任务不好。

**2. 集中式训练，分散式执行 (CTDE)。** 最常见的现代范式。每个智能体有自己的*策略* `π_i`，它依赖于本地观察 `o_i`——部署时的标准分散式执行。在*训练*期间，集中式 critic `Q(s, a_1, …, a_n)` 依赖于完整全局状态和联合动作。例子：
- **MADDPG** (Lowe et al. 2017)：每个智能体具有集中式 critic 的 DDPG。
- **COMA** (Foerster et al. 2017)：反事实 baseline——问"如果我采取动作 `a'` 代替，我的奖励会是什么？"——隔离我的贡献。
- **MAPPO** / **IPPO** 具有共享 critic (Yu et al. 2022)：具有集中式价值函数的 PPO。2026 年合作 MARL 的主导。
- **QMIX** (Rashid et al. 2018)：价值分解——`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))` 具有单调混合。

**3. Self-play。** 同一个智能体的两个副本互相玩。对手的策略*就是*我过去快照的策略。AlphaGo / AlphaZero / MuZero。OpenAI Five。对零和游戏最有效；训练信号是对称的。

**4. League play。** Self-play 到一般和 / 对抗环境的推广：保留过去和当前策略的种群，从 league 中采样对手，针对它们训练。添加 exploiters（专门击败当前最好的）和 main exploiters（专门击败 exploiters）。AlphaStar（StarCraft II）。当游戏承认"石头-剪刀-布"策略循环时需要。

**通信。** 允许智能体互相发送学习消息 `m_i`。在合作设置中有效。Foerster et al. (2016) 表明可微分智能体间通信可以端到端训练。今天的基于 LLM 的多智能体系统（Phase 16）本质上用自然语言通信。

```figure
f3-marl-orbit
```

## Build It（动手实现）

这节课使用带有两个合作智能体的 6×6 GridWorld。它们从对角开始，必须到达共享目标。共享奖励：当任一智能体仍在移动时每步 `-1`，当两者都到达时 `+10`。见 `code/main.py`。

### Step 1: 多智能体环境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*联合*动作空间是 `|A|² = 16`。全局状态是两个位置。

### Step 2: 独立 Q-learning

每个智能体运行它自己的 Q 表，键是联合状态。在每一步：两者都选择 ε-贪婪动作，收集联合转移，每个用共享奖励更新它自己的 Q。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

在这个任务上有效，因为奖励是密集和对齐的。在紧密耦合的任务上失败（例如，一个智能体必须*等待*另一个）。

### Step 3: 具有分解价值更新的集中式 Q

使用联合动作 `Q(s, a_1, a_2)` 的一个 Q。从共享奖励更新。在通过边缘化分散执行：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。以指数联合动作空间换取*正确*全局视图。

### Step 4: 简单 self-play（对抗 2 智能体）

相同的智能体，两个角色。训练智能体 A 对抗智能体 B；`K` 个情节后，将 A 的权重复制到 B。对称训练，稳定进展。AlphaZero 配方的缩影。

## Pitfalls（陷阱）

- **非平稳 replay。** 独立智能体的 experience replay 比单智能体更差，因为旧转移是由现在过时的对手生成的。修复：重新标记或按新近度加权。
- **信用分配歧义。** 长情节后的共享奖励；没有明确的方法说哪个智能体有贡献。修复：反事实 baseline (COMA)，或每个智能体的奖励塑造。
- **策略漂移 / 追逐。** 每个智能体的最佳响应随着其他智能体的更新而变化。修复：集中式 critic，慢学习率，或一次冻结一个。
- **通过协调的奖励黑客。** 智能体找到设计者没有预料到的协调利用。投标智能体收敛到出价零。修复：小心的奖励设计，行为约束。
- **探索冗余。** 两个智能体探索相同的状态-动作对。修复：每个智能体的熵奖励，或角色条件。
- **League 循环。** 纯 self-play 可能陷入主导循环。修复：具有多样化对手的 league play。
- **样本爆炸。** `n` 个智能体 × 状态空间 × 联合动作。用函数逼近近似；分解动作空间（每个智能体一个策略输出 head）。

## Use It（应用场景）

2026 年的 MARL 应用地图：

| 领域 | 方法 | 备注 |
|--------|--------|-------|
| 合作导航 / 操作 | MAPPO / QMIX | CTDE；共享 critic + 分散式 actor。 |
| 双人游戏（国际象棋、围棋、扑克） | 带 MCTS 的 Self-play (AlphaZero) | 零和；对称训练。 |
| 复杂多人（Dota、StarCraft） | League play + 模仿预训练 | OpenAI Five, AlphaStar。 |
| 自动驾驶车队 | CTDE MAPPO / PPO 带有注意力 | 部分 obs；可变团队规模。 |
| 拍卖市场 | 博弈论均衡 + RL | 当 `n` → ∞ 时的平均场 RL。 |
| LLM 多智能体系统 (Phase 16) | 自然语言通信 + 角色条件 | 在智能体计划层的 RL 循环。 |

在 2026 年，MARL 最大的增长领域是基于 LLM 的：语言模型智能体集群谈判、辩论、构建软件。RL 显示为*轨迹级*输出上的偏好优化，不是 token 级（Phase 16 · 03）。

## Ship It（交付物）

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## Exercises（练习）

1. **Easy. 在 2 智能体合作 GridWorld 上训练独立 Q-learning。多少个情节后平均回报 > 0？绘制联合学习曲线。**
2. **Medium. 添加一个"协调"任务：只有当两个智能体在同一回合踏上目标时才能达到。独立 Q 仍然收敛吗？什么坏了？**
3. **Hard. 实现 MAPPO 风格训练的集中式 critic 并在协调任务上与独立 PPO 比较收敛速度。**

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Markov game | "Multi-agent MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`; each agent has its own reward. |
| CTDE | "Centralized training, decentralized execution" | Joint critic at training time; each agent's policy uses only local obs. |
| IPPO | "Independent PPO" | Each agent runs PPO separately. Simple baseline; often underrated. |
| MAPPO | "Multi-agent PPO" | PPO with a centralized value function conditioned on global state. |
| QMIX | "Monotonic value decomposition" | `Q_tot = f_monotone(Q_1, …, Q_n)` allows decentralized argmax. |
| COMA | "Counterfactual multi-agent" | Advantage = my Q minus expected Q marginalizing over my action. |
| Self-play | "Agent vs past self" | Single agent, two roles; standard for zero-sum games. |
| League play | "Population training" | Cache past policies, sample opponents from the pool; handles strategy cycles. |

## Further Reading（延伸阅读）

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 具有集中式 critic 的 CTDE。
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — 用于信用分配的反事实 baseline。
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — 具有单调性的价值分解。
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO 在 MARL 中出奇地强大。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 大规模 league play。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 零和游戏中的纯 self-play。
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — 包括教科书中对多智能体设置和非平稳性问题的简短处理，CTDE 旨在解决这个问题。
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — 涵盖合作、竞争和混合 MARL 以及收敛结果的调查。
