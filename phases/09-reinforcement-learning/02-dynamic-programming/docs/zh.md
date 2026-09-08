# Dynamic Programming — Policy Iteration & Value Iteration（动态规划——策略迭代与值迭代）

> 动态规划是"开卷考试"的强化学习。你已经知道了转移和奖励函数；你只需要迭代贝尔曼方程，直到 `V` 或 `π` 停止移动。它是每一个基于采样的方法都试图接近的基准。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs)
**Time:** ~75 分钟

## The Problem（问题）

你有一个已知模型的 MDP：你可以查询任何状态-动作对的 `P(s' | s, a)` 和 `R(s, a, s')`。库存经理知道需求分布。棋盘游戏有确定性转移。GridWorld 是四行 Python。你有一个*模型*。

无模型强化学习（Q-learning、PPO、REINFORCE）是为没有模型的情况发明的——你只能从环境中采样。但当你确实有一个模型时，有更快、更好的方法：动态规划。贝尔曼在 1957 年设计了它们。它们仍然定义正确性：当人们说"这个 MDP 的最优策略"时，他们的意思是 DP 会返回的策略。

你在 2026 年需要它们有三个原因。首先，强化学习研究中的每一个表格环境（GridWorld、FrozenLake、CliffWalking）都是用 DP 求解以产生金标准策略的。其次，精确值可以让你*调试*采样方法：如果 Q-learning 对 `V*(s_0)` 的估计与 DP 答案相差 30%，你的 Q-learning 有 bug。第三，现代离线强化学习和规划方法（MCTS、AlphaZero 的搜索、Phase 9 · 10 中的基于模型的强化学习）都在学习或给定的模型上迭代贝尔曼备份。

## The Concept（概念）

![Policy iteration and value iteration, side by side（策略迭代与值迭代，并排对比）](../assets/dp.svg)

**两种算法，都是对贝尔曼方程的定点迭代。**

**策略迭代。** 交替两个步骤直到策略停止变化。

1. *评估：* 给定策略 `π`，通过反复应用 `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` 直到收敛来计算 `V^π`。
2. *改进：* 给定 `V^π`，使其关于 `V^π` 贪婪：`π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

收敛是有保证的，因为 (a) 每次改进步骤要么保持 `π` 不变，要么严格增加某些状态的 `V^π`，(b) 确定性策略的空间是有限的。即使对于大型状态空间，通常也在约 5-20 次外迭代中收敛。

**值迭代。** 将评估和改进合并为一个 sweep。应用贝尔曼*最优性*方程：

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

重复直到 `max_s |V_{new}(s) - V(s)| < ε`。最后通过取贪婪动作提取策略。每次迭代严格更快——没有内部评估循环——但通常需要更多迭代才能收敛。

**广义策略迭代（GPI）。** 统一的框架。价值函数和策略被锁定在一个双向改进循环中；任何将两者推向一致性的方法（异步值迭代、修改后的策略迭代、Q-learning、actor-critic、PPO）都是 GPI 的一个实例。

**为什么 `γ < 1` 很重要。** 贝尔曼算子是 sup-norm 中的一个 `γ` 压缩：`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。压缩意味着唯一不动点和几何收敛。去掉 `γ < 1` 就会失去这个保证——你需要有限视界或吸收终止状态。

```figure
value-iteration-gamma
```

## Build It（动手实现）

### Step 1: 构建 GridWorld MDP 模型

使用与 Lesson 01 相同的 4×4 GridWorld。我们添加一个随机变体：以概率 `0.1`，智能体滑动到一个随机垂直方向。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` 返回一个 `(s', r, p)` 列表。这就是整个模型。

### Step 2: 策略评估

给定策略 `π(s) = {action: prob}`，迭代贝尔曼方程直到 `V` 停止移动：

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### Step 3: 策略改进

用关于 `V` 的贪婪策略替换 `π`。如果 `π` 没有改变，返回——我们已经在最优解了。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### Step 4: 将它们缝合在一起

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

在 4×4 上的典型收敛：4-6 次外迭代。输出 `V*(0,0) ≈ -6` 和一个严格减少步数的策略。

### Step 5: 值迭代（单循环版本）

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

相同的不动点，更少的代码行。

## Pitfalls（陷阱）

- **忘记处理终点。** 如果你对吸收态应用贝尔曼方程，它仍然会选择一个什么都不改变的动作。用 `if s == terminal: V[s] = 0` 来保护。
- **Sup-norm 与 L2 收敛。** 使用 `max |V_new - V|`，而不是平均值。理论保证是在 sup-norm 上的。
- **原地与同步更新。** 原地更新 `V[s]`（Gauss-Seidel）比单独的 `V_new` 字典（Jacobi）收敛得更快。生产代码使用原地更新。
- **策略平局。** 如果两个动作有相等的 Q 值，`argmax` 可能在每次迭代中以不同方式打破平局，导致"策略稳定"检查振荡。使用稳定的平局打破（固定顺序中的第一个动作）。
- **状态空间爆炸。** DP 每次 sweep 是 `O(|S| · |A|)`。可以工作到约 10⁷ 个状态。超过这个范围，你需要函数逼近（从 Phase 9 · 05 开始）。

## Use It（应用场景）

在 2026 年，DP 是正确性基准和规划器的内部循环：

| 使用场景 | 方法 |
|----------|--------|
| 精确求解小型表格 MDP | 值迭代（更简单）或策略迭代（更少的外步骤） |
| 验证 Q-learning / PPO 实现 | 在玩具环境上与 DP 最优 `V*` 比较 |
| 基于模型的强化学习（Phase 9 · 10） | 在学习到的转移模型上的贝尔曼备份 |
| AlphaZero / MuZero 中的规划 | 蒙特卡洛树搜索 = 异步贝尔曼备份 |
| 离线强化学习（CQL, IQL） | 保守 Q 迭代——对 OOD 动作有惩罚的 DP |

每当有人说"最优价值函数"时，他们的意思是"DP 不动点"。当你在论文中看到 `V*` 或 `Q*` 时，想象这个循环。

## Ship It（交付物）

保存为 `outputs/skill-dp-solver.md`：

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## Exercises（练习）

1. **Easy. 在 4×4 GridWorld 上运行值迭代，其中 `γ ∈ {0.9, 0.99}`。多少次 sweep 后 `max |ΔV| < 1e-6`？以 4×4 网格形式打印 `V*`。**
2. **Medium. 在*随机* GridWorld（滑动概率 `0.1`）上比较策略迭代与值迭代。计数：sweep 次数、挂钟时间、最终 `V*(0,0)`。哪个在迭代中收敛更快？在挂钟中？**
3. **Hard. 构建修改后的策略迭代：在评估步骤中，只运行 `k` 次 sweep 而不是直到收敛。绘制 `V*(0,0)` 误差 vs `k`，其中 `k ∈ {1, 2, 5, 10, 50}`。这条曲线告诉你关于评估/改进权衡的什么信息？**

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Policy iteration | "DP algorithm" | Alternating evaluation (`V^π`) and improvement (greedy `π` w.r.t. `V^π`) until the policy stops changing. |
| Value iteration | "Faster DP" | Bellman optimality backup applied in one sweep; converges to `V*` geometrically. |
| Bellman operator | "The recursion" | `(T V)(s) = max_a Σ P (r + γ V(s'))`; a `γ`-contraction in sup-norm. |
| Contraction | "Why DP converges" | Any operator `T` with `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|` has a unique fixed point. |
| GPI | "Everything is DP" | Generalized Policy Iteration: any method driving `V` and `π` to mutual consistency. |
| Synchronous update | "Jacobi-style" | Use old `V` throughout a sweep; cleanly analyzable but slower. |
| In-place update | "Gauss-Seidel-style" | Use `V` as it's being updated; converges faster in practice. |

## Further Reading（延伸阅读）

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书。第 4 章涵盖动态规划。
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — 贝尔曼方程的起源。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 关于 MDP 和精确求解方法的运筹学参考。
