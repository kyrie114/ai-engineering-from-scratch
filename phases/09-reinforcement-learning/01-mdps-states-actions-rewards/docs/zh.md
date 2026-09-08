# MDPs, States, Actions & Rewards（MDP、状态、动作与奖励）

> 马尔可夫决策过程由五部分组成：状态、动作、转移、奖励、折扣。强化学习中的一切——Q-learning、PPO、DPO、GRPO——都优化这个结构。学一次，就能读懂强化学习的其余部分。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Probability & Distributions), Phase 2 · 01 (ML Taxonomy)
**Time:** ~45 分钟

## The Problem（问题）

你正在写一个国际象棋机器人。或者一个库存规划器。或者一个交易代理。或者一个训练推理模型的 PPO 循环。四个不同的领域，一个令人惊讶的事实：这四个领域都坍缩为同一个数学对象。

监督学习给你 `(x, y)` 对，让你拟合一个函数。强化学习不给你标签——只有状态流、你采取的动作，以及一个标量奖励。这步棋赢了吗？补货决策省钱了吗？交易盈利了吗？LLM 刚产生的 token 是否带来了更高的奖励？

在形式化之前，你无法从这个流中学习。"我看到了什么""我做了什么""接下来发生了什么""这有多好"——每一个都必须变成一个可以推理的对象。这种形式化就是马尔可夫决策过程。本阶段中的每一个强化学习算法，包括最后的 RLHF 和 GRPO 循环，都是在这个结构上进行优化的。

## The Concept（概念）

![Markov decision process: states, actions, transitions, rewards, discount（马尔可夫决策过程：状态、动作、转移、奖励、折扣）](../assets/mdp.svg)

**五个对象。**

- **状态** `S`。智能体做出决策所需的一切。在 GridWorld 中是单元格；在国际象棋中是棋盘；在 LLM 中是上下文窗口加上任何记忆。
- **动作** `A`。可选的动作。上下左右；走一步；输出一个 token。
- **转移** `P(s' | s, a)`。给定状态 `s` 和动作 `a` 后下一状态的概率分布。在国际象棋中是确定性的，在库存问题中是随机的，在 LLM 解码中几乎是确定性的。
- **奖励** `R(s, a, s')`。标量信号。赢 = +1，输 = -1。收入减去成本。GRPO 中的对数似然比项。
- **折扣** `γ ∈ [0, 1)`。未来奖励相对于当前奖励的重要性。`γ = 0.99` 的视界约为 100 步；`γ = 0.9` 的视界约为 10 步。

**马尔可夫性质** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`。未来只依赖于当前状态。如果不是这样，那是状态表示不完整——不是方法的失败，而是状态的失败。

**策略与回报。** 策略 `π(a | s)` 将状态映射到动作分布。回报 `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` 是未来奖励的折扣和。状态价值 `V^π(s) = E[G_t | s_t = s]` 是在策略 `π` 下从状态 `s` 出发的期望回报。动作价值 `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` 是从状态 `s` 出发并首先执行动作 `a` 的期望回报。本阶段中的每一个强化学习算法都在估计这两个值之一，然后据此改进 `π`。

**贝尔曼方程。** 本阶段中所有方法都使用的定点方程：

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

这些方程将期望回报分解为"这一步的奖励"加上"你到达状态的折扣价值"。递归的。本阶段中的每一个算法要么迭代这个方程直到收敛（动态规划），从中采样（蒙特卡洛），要么单步自举（时序差分）。

```figure
discount-horizon
```

## Build It（动手实现）

### Step 1: 一个微小的确定性 MDP

一个 4×4 的 GridWorld。智能体从左上角开始，终点在右下角，每步奖励 -1，动作集合 `{up, down, left, right}`。见 `code/main.py`。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

五行代码。这就是整个环境。确定性转移，恒定步 penalty，吸收态终点。

### Step 2:  rollout 一个策略

策略是从状态到动作分布的函数。最简单的：均匀随机。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

运行随机策略 1000 次。这个 4×4 棋盘的平均回报约为 -60 到 -80。最优回报是 -6（直线右下角路径）。缩小这个差距是本阶段的一切。

### Step 3: 通过贝尔曼方程精确计算 `V^π`

对于小型 MDP，贝尔曼方程是一个线性系统。枚举状态，应用期望，迭代直到值停止变化。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

这是迭代策略评估。它是 Sutton & Barto 中的第一个算法，也是后续每一个强化学习算法的理论基础。

### Step 4: `γ` 是一个具有物理意义的超参数

有效视界大约是 `1 / (1 - γ)`。`γ = 0.9` → 10 步。`γ = 0.99` → 100 步。`γ = 0.999` → 1000 步。

太低会让智能体短视。太高会让信用分配变得嘈杂，因为许多早期步骤都要为遥远的未来奖励负责。LLM 的 RLHF 通常使用 `γ = 1`，因为情节短且有界。控制任务使用 `0.95–0.99`。长期策略游戏使用 `0.999`。

## Pitfalls（陷阱）

- **非马尔可夫状态。** 如果你需要最后三个观察来决定，"状态"不仅仅是当前观察。修复：堆叠帧（DQN on Atari 堆叠 4 个）或使用循环状态（LSTM/GRU over observations）。
- **稀疏奖励。** 只有赢的奖励会让大型状态空间中的学习几乎不可能。塑造奖励（中间信号）或用模仿学习引导（Phase 9 · 09）。
- **奖励黑客。** 优化代理奖励经常产生病态行为。OpenAI 的赛艇代理为了无限收集能量而永远转圈，而不是完成比赛。总是从目标结果定义奖励，而不是从代理定义。
- **折扣误设。** 在无限视界任务上使用 `γ = 1` 会让每个值都变成无穷大。总是用有限视界或 `γ < 1` 来限制。
- **奖励尺度。** `{+100, -100}` 和 `{+1, -1}` 的奖励给出相同的最优策略，但梯度幅度截然不同。在插入 PPO/DQN 之前，标准化到 `[-1, 1]` 左右。

## Use It（应用场景）

2026 年的强化学习堆栈在接触代码之前将每个强化学习管道都简化为 MDP：

| 场景 | 状态 | 动作 | 奖励 | γ |
|-----------|-------|--------|--------|---|
| 控制（运动、操作） | 关节角度 + 速度 | 连续扭矩 | 任务特定的塑造奖励 | 0.99 |
| 游戏（国际象棋、围棋、扑克） | 棋盘 + 历史 | 合法着法 | 赢=+1 / 输=-1 | 1.0（有限） |
| 库存 / 定价 | 库存 + 需求 | 订货量 | 收入 - 成本 | 0.95 |
| LLM 的 RLHF | 上下文 token | 下一个 token | 结束时的奖励模型分数 | 1.0（情节 ~200 token） |
| GRPO 推理 | 提示 + 部分响应 | 下一个 token | 结束时的验证器 0/1 | 1.0 |

在写任何训练循环之前先写出五个元组。大多数"强化学习不工作"的错误报告都可以追溯到纸上 broken 的 MDP 公式。

## Ship It（交付物）

保存为 `outputs/skill-mdp-modeler.md`：

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## Exercises（练习）

1. **Easy. 在 `code/main.py` 中实现 4×4 GridWorld 和随机策略 rollout。运行 10,000 个情节。报告回报的均值和标准差。与最优回报 (-6) 比较。**
2. **Medium. 对均匀随机策略运行 `policy_evaluation`，其中 `γ ∈ {0.5, 0.9, 0.99}`。为每个值打印 4×4 网格形式的 `V`。解释为什么靠近终点的状态值随着更大的 `γ` 增长更快。**
3. **Hard. 将 GridWorld 随机化：每个动作以概率 `p = 0.1` 滑动到相邻方向。重新评估均匀策略。`V[start]` 会变好还是变差？为什么？**

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| MDP | "Reinforcement learning setup" | Tuple `(S, A, P, R, γ)` satisfying the Markov property. |
| State | "What the agent sees" | Sufficient statistic for future dynamics under the chosen policy class. |
| Policy | "Agent's behavior" | Conditional distribution `π(a \| s)` or deterministic map `s → a`. |
| Return | "Total reward" | Discounted sum `Σ γ^t r_t` from the current step. |
| Value | "How good a state is" | Expected return under `π` starting from `s`. |
| Q-value | "How good an action is" | Expected return under `π` starting from `s` with first action `a`. |
| Bellman equation | "Dynamic programming recursion" | Fixed-point decomposition of value / Q into one-step reward plus discounted successor value. |
| Discount `γ` | "Future vs present" | Geometric weight on far-future reward; effective horizon `~1/(1-γ)`. |

## Further Reading（延伸阅读）

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — the textbook. Ch. 3 covers MDPs and Bellman equations; Ch. 1 motivates the reward hypothesis that underlies every subsequent lesson.
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — the origin of the Bellman equation.
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — concise MDP primer from a deep-RL angle.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — the operations-research reference on MDPs and exact solution methods.
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — the cleanest derivation of MDPs as a dynamic-programming specialization.
