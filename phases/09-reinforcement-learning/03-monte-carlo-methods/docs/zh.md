# Monte Carlo Methods — Learning from Complete Episodes（蒙特卡洛方法——从完整情节中学习）

> 动态规划需要一个模型。蒙特卡洛方法只需要情节。执行策略，观察回报，取平均值。强化学习中最简单的想法——也是解锁下游一切的关键。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming)
**Time:** ~75 分钟

## The Problem（问题）

动态规划很优雅，但它假设你可以查询每个状态和动作的 `P(s' | s, a)`。现实世界中几乎没有东西是这样工作的。机器人无法分析计算机器人在关节扭矩后相机像素的分布。定价算法无法对所有可能的客户反应进行积分。LLM 无法枚举一个 token 之后所有可能的延续。

你需要一种只需要从环境*采样*能力的方法。执行策略。获取轨迹 `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`。用它来估计价值。这就是蒙特卡洛。

从 DP 到 MC 的转变在哲学上很重要：我们从*已知模型 + 精确备份*转向*采样 rollout + 平均回报*。方差增加了，但适用性爆炸了。这节课之后的每一个强化学习算法——TD、Q-learning、REINFORCE、PPO、GRPO——在本质上都是一个蒙特卡洛估计器，有时在上面叠加了自举。

## The Concept（概念）

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit（蒙特卡洛：rollout、计算回报、平均；首次访问 vs 每次访问）](../assets/monte-carlo.svg)

**核心思想，一句话：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)` 其中 `G^{(i)}(s)` 是在策略 `π` 下访问 `s` 后观察到的回报。

**首次访问 vs 每次访问 MC。** 给定一个多次访问状态 `s` 的情节，首次访问 MC 只计算第一次访问的回报；每次访问 MC 计算所有访问的回报。两者在极限下都是无偏的。首次访问更易于分析（iid 样本）。每次访问在每个情节中使用更多数据，实践中通常收敛更快。

**增量均值。** 不用存储所有回报，而是更新运行平均值：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

重新组织：`V_new = V_old + α · (target - V_old)` 其中 `α = 1/n`。将 `1/n` 替换为常数步长 `α ∈ (0, 1)`，你就得到了一个跟踪 `π` 变化的非平稳 MC 估计器。这一步就是从 MC 到 TD 再到每一个现代强化学习算法的整个跳跃。

**探索现在变成了一个问题。** DP 通过枚举接触每一个状态。MC 只看到策略访问的状态。如果 `π` 是确定性的，状态空间的整个区域永远不会被采样，它们的价值估计将永远停留在零。三个修复方法，按历史顺序：

1. **探索起点。** 每个情节从随机的 (s, a) 对开始。保证覆盖；在实践中不现实（你不能将机器人"重置"到任意状态）。
2. **ε-贪婪。** 对当前 Q 贪婪行动，但以概率 `ε` 选择一个随机动作。所有状态-动作对都会被渐进地采样。
3. **离策略 MC。** 在行为策略 `μ` 下收集数据，通过重要性采样学习关于目标策略 `π` 的知识。高方差，但它是通往 replay buffer 方法（如 DQN）的桥梁。

**蒙特卡洛控制。** 评估 → 改进 → 评估，就像策略迭代一样，但评估是基于采样的：

1. 运行 `π`，得到一个情节。
2. 从观察到的回报更新 `Q(s, a)`。
3. 使 `π` 关于 `Q` 是 ε-贪婪的。
4. 重复。

在温和条件下以概率 1 收敛到 `Q*` 和 `π*`（每一对都被无限次访问，`α` 满足 Robbins-Monro）。

```figure
epsilon-greedy
```

## Build It（动手实现）

### Step 1: rollout → (s, a, r) 列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

没有模型，只有 `env.reset()` 和 `env.step(s, a)`。与 gym 环境相同的接口但被简化了。

### Step 2: 计算回报（反向 sweep）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

一次通过，`O(T)`。反向递推 `G_t = r_{t+1} + γ G_{t+1}` 避免了重新求和。

### Step 3: 首次访问 MC 评估

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

三行代码完成工作：在首次访问时标记状态，增加计数，更新运行均值。

### Step 4: ε-贪婪 MC 控制（on-policy）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### Step 5: 与 DP 金标准比较

你对 `V^π` 的 MC 估计应该与 Lesson 02 中的 DP 结果在情节 → ∞ 时一致。实践中：在 4×4 GridWorld 上运行 50,000 个情节，你会在 DP 答案的 `~0.1` 范围内。

## Pitfalls（陷阱）

- **无限情节。** MC 要求情节*终止*。如果你的策略可以永远循环，限制 `max_steps` 并将限制视为隐式失败。随机策略的 GridWorld 经常超时——这是正常的，只要确保你正确计数。
- **方差。** MC 使用完整回报。在长情节中，方差很大——最后的一个不幸奖励会以相同幅度偏移 `V(s_0)`。TD 方法（Lesson 04）通过自举来减少这一点。
- **状态覆盖。** 在 fresh Q 上贪婪 MC 且存在平局时将只尝试一个动作。你*必须*探索（ε-贪婪、探索起点、UCB）。
- **非平稳策略。** 如果 `π` 改变（如在 MC 控制中），旧回报来自不同的策略。常数-α MC 处理这个；样本平均 MC 不处理。
- **离策略重要性采样。** 权重 `π(a|s)/μ(a|s)` 在整个轨迹上相乘。方差随着视界爆炸。用每决策加权 IS 限制或切换到 TD。

## Use It（应用场景）

蒙特卡洛方法在 2026 年的角色：

| 使用场景 | 为什么用 MC |
|----------|--------|
| 短视界游戏（21 点、扑克） | 情节自然终止；回报干净。 |
| 记录策略的离线评估 | 在存储的轨迹上平均折扣回报。 |
| 蒙特卡洛树搜索（AlphaZero） | 从树叶子开始的 MC rollout 引导选择。 |
| LLM 强化学习评估 | 对于给定策略，在采样的补全上计算平均奖励。 |
| PPO 中的基线估计 | 优势目标 `A_t = G_t - V(s_t)` 使用 MC `G_t`。 |
| 强化学习教学 | 实际工作的最简单算法——剥离自举以查看核心。 |

现代深度强化学习算法（PPO、SAC）通过 `n` 步回报或 GAE 在纯 MC（完整回报）和纯 TD（单步自举）之间插值。两个端点都是同一估计器的实例。

## Ship It（交付物）

保存为 `outputs/skill-mc-evaluator.md`：

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## Exercises（练习）

1. **Easy. 在 4×4 GridWorld 上实现均匀随机策略的首次访问 MC 评估。运行 10,000 个情节。绘制 `V(0,0)` 作为情节数的函数，与 DP 答案对比。**
2. **Medium. 实现 ε-贪婪 MC 控制，其中 `ε ∈ {0.01, 0.1, 0.3}`。在 20,000 个情节后比较平均回报。曲线是什么样子的？偏差-方差权衡在哪里？**
3. **Hard. 实现*离策略* MC 重要性采样：在均匀随机策略 `μ` 下收集数据，为确定性最优策略 `π` 估计 `V^π`。比较普通 IS vs 每决策 IS vs 加权 IS。哪个方差最低？**

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Monte Carlo | "Random sampling" | Estimate expectations by averaging over iid samples from the distribution. |
| Return `G_t` | "Future reward" | Sum of discounted rewards from step `t` to episode end: `Σ_{k≥0} γ^k r_{t+k+1}`. |
| First-visit MC | "Count each state once" | Only the first visit in an episode contributes to the value estimate. |
| Every-visit MC | "Use all visits" | Every visit contributes; slightly biased but more sample-efficient. |
| ε-greedy | "Exploration noise" | Pick greedy action with prob `1-ε`; random action with prob `ε`. |
| Importance sampling | "Correcting for sampling from the wrong distribution" | Reweight returns by `π(a\|s)/μ(a\|s)` products to estimate `V^π` from `μ` data. |
| On-policy | "Learn from my own data" | Target policy = behavior policy. Vanilla MC, PPO, SARSA. |
| Off-policy | "Learn from someone else's data" | Target policy ≠ behavior policy. Importance-sampled MC, Q-learning, DQN. |

## Further Reading（延伸阅读）

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书。第 5 章涵盖蒙特卡洛方法。
- [Silver, Veness (2010). Monte-Carlo Planning in Large POMDPs](https://papers.nips.cc/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html) — POMCP 中的 MC 规划。
- [Mnih et al. (2015). Human-level control through deep reinforcement learning (DQN)](https://arxiv.org/abs/1312.5602) — 将 MC 思想与函数逼近结合。
