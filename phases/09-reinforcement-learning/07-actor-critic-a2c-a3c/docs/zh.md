# Actor-Critic — A2C and A3C（Actor-Critic——A2C 与 A3C）

> REINFORCE 是嘈杂的。添加一个学习 `V̂(s)` 的 critic，从回报中减去它，你就得到了一个具有相同期望但方差低得多的 advantage。这就是 actor-critic。A2C 同步运行它；A3C 跨线程运行它。两者都是每一个现代深度强化学习方法的心理模型。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (TD Learning), Phase 9 · 06 (REINFORCE)
**Time:** ~75 分钟

## The Problem（问题）

朴素 REINFORCE 有效，但它的方差很糟糕。蒙特卡洛回报 `G_t` 在情节之间可以摆动一个数量级。将这种噪声乘以 `∇ log π` 并平均产生一个梯度估计器，需要数千个情节才能将策略移动与用更少的 DQN 更新能移动的距离相同。

方差来自使用原始回报。如果你减去一个 baseline `b(s_t)`——任何状态的函数，包括学习的——期望不变，方差下降。最佳可处理 baseline 是 `V̂(s_t)`。现在乘以 `∇ log π` 的量是*advantage*：

`A(s, a) = G - V̂(s)`

一个动作是好的，如果它产生了高于平均水平的回报；如果低于平均水平则是坏的。具有学习 critic 的 REINFORCE 是 *actor-critic*。Critic 给 actor 一个低方差教师。这是 2015 年之后的每一个深度策略方法（A2C、A3C、PPO、SAC、IMPALA）。

## The Concept（概念）

![Actor-critic: policy net plus value net, TD residual as advantage（Actor-Critic：策略网络加价值网络，TD 残差作为 advantage）](../assets/actor-critic.svg)

**两个网络，一个共享损失：**

- **Actor** `π_θ(a | s)`：策略。采样以行动。用策略梯度训练。
- **Critic** `V_φ(s)`：估计从状态的期望回报。训练以最小化 `(V_φ(s) - target)²`。

**Advantage。** 两个标准形式：

- *MC advantage:* `A_t = G_t - V_φ(s_t)`。无偏，高方差。
- *TD advantage:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏（使用 `V_φ`），方差低得多。也称为 *TD residual* `δ_t`。

**n-step advantage。** 在两者之间插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 是纯 TD。`n = ∞` 是 MC。大多数实现使用 Atari 的 `n = 5`，MuJoCo 上 PPO 的 `n = 2048`。

**Generalized Advantage Estimation (GAE)。** Schulman et al. (2016) 提出了对所有 n-step advantages 的指数加权平均：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。`λ = 0` 是 TD（低方差，高偏差）。`λ = 1` 是 MC（高方差，无偏）。`λ = 0.95` 是 2026 年默认值——调整直到偏差/方差拨盘在你想要的位置。

**A2C: 同步 advantage actor-critic。** 在 `N` 个并行环境上收集 `T` 步。为每个步骤计算 advantages。在组合 batch 上更新 actor 和 critic。重复。A3C 的更简单、更可扩展的兄弟。

**A3C: 异步 advantage actor-critic。** Mnih et al. (2016)。生成 `N` 个工作线程，每个运行一个 env。每个工作线程在其自己的 rollout 上进行本地梯度计算，然后异步地将它们应用到共享参数服务器。不需要 replay buffer——工作线程通过运行不同轨迹来去相关。A3C 证明了你可以在 CPU 上进行大规模训练。在 2026 年，基于 GPU 的 A2C（批量并行 env）占主导地位，因为 GPU 想要大批量。

**组合损失。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三项：策略梯度损失、价值回归、熵奖励。`c_v ~ 0.5`，`c_e ~ 0.01` 是规范的起点。

```figure
actor-critic
```

## Build It（动手实现）

### Step 1: 一个 critic

线性 critic `V_φ(s) = w · features(s)` 用 MSE 更新：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在表格环境上，critic 在几百个情节内收敛。在 Atari 上，用共享 CNN trunk + value head 替换线性 critic。

### Step 2: n-step advantage

给定长度为 `T` 的 rollout 和自举的最终 `V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是 critic 目标。`advantages` 是乘以 `∇ log π` 的量。

### Step 3: 组合更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

On-policy，每个更新一次 rollout，actor 和 critic 分开的学习率。

### Step 4: 并行化 (A3C vs A2C)

- **A3C:** 生成 `N` 个线程。每个运行自己的 env 和自己的前向传播。定期将梯度更新推送到共享 master。master 上没有锁——竞争是可以的，它们只是增加噪声。
- **A2C:** 在单个进程中运行 `N` 个 env 实例，将观察堆叠成 `[N, obs_dim]` batch，批量前向传播，批量反向传播。更高的 GPU 利用率，确定性，更容易推理。2026 年的默认选择。

我们的玩具代码是单线程的以保持清晰；重写为批量 A2C 是三行 numpy。

## Pitfalls（陷阱）

- **Actor 梯度之前的 critic 偏差。** 如果 critic 是随机的，它的 baseline 是无信息的，你是在纯噪声上训练。在打开策略梯度之前预热 critic 几百步，或使用慢 actor 学习率。
- **Advantage 归一化。** 将 advantages 归一化为每 batch 的零均值/单位标准差。以近零成本大规模稳定训练。
- **Shared trunk。** 在图像输入上使用共享特征提取器给 actor 和 critic。单独的 heads。共享特征免费乘坐两个损失。
- **On-policy 契约。** A2C 将数据重用恰好一次更新。更多，你的梯度是有偏的（重要性采样修正是 PPO 添加的）。
- **熵坍缩。** 没有 `c_e > 0`，策略在几百个更新内变成近确定性并停止探索。
- **奖励尺度。** Advantage 幅度取决于奖励尺度。归一化奖励（例如，running-std 除法）以跨任务的一致梯度幅度。

## Use It（应用场景）

A2C/A3C 在 2026 年很少是最终选择，但它们是一切后续方法精炼的架构：

| 方法 | 与 A2C 的关系 |
|--------|----------------|
| PPO | A2C + 裁剪重要性比率用于多轮更新 |
| IMPALA | A3C + V-trace 离策略修正 |
| SAC（Phase 9 · 07） | 具有软价值 critic 的离策略 A2C（下一课） |
| GRPO（Phase 9 · 12） | 没有 critic 的 A2C——group-relative advantage |
| DPO | 将 A2C 坍缩为偏好排序损失，没有采样 |
| AlphaStar / OpenAI Five | 具有 league 训练 + 模仿预训练的 A2C |

当你在 2026 年的论文中看到"advantage"时，想想 actor-critic。

## Ship It（交付物）

保存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## Exercises（练习）

1. **Easy. 在 4×4 GridWorld 上用 MC advantage（`G_t - V(s_t)`）训练 actor-critic。与 Lesson 06 中的 REINFORCE-with-running-mean-baseline 比较样本效率。**
2. **Medium. 切换到 TD-residual advantage（`r + γ V(s') - V(s)`）。测量 advantage batch 的方差。它降低了多少？**
3. **Hard. 实现 GAE(λ)。扫描 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`。绘制最终回报 vs 样本效率。这个任务上的偏差/方差最佳点在哪里？**

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Actor | "The policy net" | `π_θ(a\|s)`, updated by policy gradient. |
| Critic | "The value net" | `V_φ(s)`, updated by MSE regression to returns / TD targets. |
| Advantage | "How much better than average" | `A(s, a) = Q(s, a) - V(s)` or its estimators. Multiplier for `∇ log π`. |
| TD residual | "δ" | `δ_t = r + γ V(s') - V(s)`; one-step advantage estimate. |
| GAE | "The interpolation knob" | Exponentially weighted sum of n-step advantages, parameterized by `λ`. |
| A2C | "Synchronous actor-critic" | Batched across envs; one gradient step per rollout. |
| A3C | "Async actor-critic" | Worker threads push gradients to a shared param server. Original paper; less common in 2026. |
| Bootstrap | "Use V at the horizon" | Truncate the rollout, add `γ^n V(s_{t+n})` to close the sum. |

## Further Reading（延伸阅读）

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C，原始的异步 actor-critic 论文。
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE。
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 基础；当 critic 是神经网络时，与 Ch. 9 函数逼近配对。
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — 可扩展的分布式 actor-critic 与 V-trace 离策略修正。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 值得阅读的生产 A2C/PPO 实现。
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — 两时间尺度 actor-critic 分解的基础收敛结果。
