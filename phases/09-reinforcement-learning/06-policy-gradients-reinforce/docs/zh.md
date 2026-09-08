# Policy Gradient — REINFORCE from Scratch（策略梯度——从零实现 REINFORCE）

> 停止估计价值。直接参数化策略，计算期望回报的梯度，向山上走。Williams (1992) 用一个定理写下了它。这就是为什么 PPO、GRPO 和每一个 LLM 强化学习循环存在的原因。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 03 (Monte Carlo), Phase 9 · 04 (TD Learning)
**Time:** ~75 分钟

## The Problem（问题）

Q-learning 和 DQN 参数化*价值*函数。你通过 `argmax Q` 选择动作。这对于离散动作和离散状态是可行的。当动作是连续的（对 10 维扭矩进行 `argmax`？）或当你想要随机策略（`argmax` 在构造上是确定性的）时，它就会失效。

策略梯度改为参数化*策略*。`π_θ(a | s)` 是一个输出动作分布的神经网络。从中采样以行动。计算期望回报关于 `θ` 的梯度。向山上走。没有 `argmax`。没有贝尔曼递归。只有 `J(θ) = E_{π_θ}[G]` 上的梯度上升。

REINFORCE 定理（Williams 1992）告诉你这个梯度是可计算的：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。运行一个情节。计算回报。在每一步乘以 `∇ log π_θ(a | s)`。平均。梯度上升。完成。

2026 年的每一个 LLM 强化学习算法——PPO、DPO、GRPO——都是 REINFORCE 的改进。在指尖理解它是本阶段其余部分以及 Phase 10 · 07（RLHF 实现）和 Phase 10 · 08（DPO）的先决条件。

## The Concept（概念）

![Policy gradient: softmax policy, log-π gradient, return-weighted update（策略梯度：softmax 策略、log-π 梯度、回报加权更新）](../assets/policy-gradient.svg)

**策略梯度定理。** 对于任何由 `θ` 参数化的策略 `π_θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 是从步骤 `t` 开始的折扣回报。期望是在从 `π_θ` 采样的完整轨迹 `τ` 上。

**证明很短。** 在期望下对 `J(θ) = Σ_τ P(τ; θ) G(τ)` 求导。使用 `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`（对数导数技巧）。分解 `log P(τ; θ) = Σ log π_θ(a_t | s_t) + 不依赖于 θ 的环境项`。环境项消失。两行代数给你定理。

**方差减少技巧。** 朴素 REINFORCE 有巨大的方差——回报是嘈杂的，`∇ log π` 是嘈杂的，它们的乘积非常嘈杂。两个标准修复：

1. **Baseline 减法。** 将 `G_t` 替换为 `G_t - b(s_t)`，对于任何不依赖于 `a_t` 的 baseline `b(s_t)`。无偏，因为 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`。典型选择：`b(s_t) = V̂(s_t)` 由 critic 学习 → actor-critic（Lesson 07）。
2. **Reward-to-go。** 将 `Σ_t G_t · ∇ log π_θ(a_t | s_t)` 替换为 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`。只有未来回报对给定动作重要——过去奖励贡献零均值噪声。

结合起来，你得到：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

这是带 baseline 的 REINFORCE——A2C（Lesson 07）和 PPO（Lesson 08）的直接祖先。

**离散动作的 Softmax 策略参数化。** 标准选择：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中 `f_θ` 是任何输出每个动作分数的神经网络。梯度有一个干净的形式：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

即，所取动作的分数减去它在策略下的期望值。

**连续动作的高斯策略。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 有闭式解。这就是 Phase 9 · 07 的 SAC 所需要的全部。

```figure
policy-gradient-landscape
```

## Build It（动手实现）

### Step 1: softmax 策略网络

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

对于表格环境，使用线性策略（每个动作一个权重向量）。对于 Atari，换入 CNN 并保持 softmax head。

### Step 2: 采样和对数概率

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### Step 3: 捕获对数概率的 rollout

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### Step 4: REINFORCE 更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

梯度 `∇ log π(a|s) = e_a - π(·|s)`（`a` 的 onehot 减去概率）是 softmax 策略梯度的核心。把它刻进肌肉记忆。

### Step 5: baselines

最近情节的 `G` 的 running mean 足以让 4×4 GridWorld 运行方差减少；它需要约 500 个情节来收敛。将 baseline 升级为学习的 `V̂(s)`，你就得到了 actor-critic。

## Pitfalls（陷阱）

- **爆炸梯度。** 回报可能很大。总是在乘以 `∇ log π` 之前将 `G` 归一化到 `~N(0, 1)` 跨 batch。
- **熵坍缩。** 策略过早收敛到近确定性动作，停止探索，陷入困境。修复：向目标添加熵奖励 `β · H(π(·|s))`。
- **高方差。** 朴素 REINFORCE 需要数千个情节。Critic baseline（Lesson 07）或 TRPO/PPO 的 trust region（Lesson 08）是标准修复。
- **样本低效。** On-policy 意味着你在一次更新后丢弃每个转移。通过重要性采样进行离策略修正带回数据，但以方差为代价（PPO 的比率是一个裁剪的 IS 权重）。
- **非平稳梯度。** 来自 100 个情节前的相同梯度使用旧的 `π`。出于这个原因，on-policy 方法每几个 rollout 更新一次。
- **信用分配。** 没有 reward-to-go，过去奖励贡献噪声。总是使用 reward-to-go。

## Use It（应用场景）

在 2026 年，REINFORCE 很少直接运行，但其梯度公式无处不在：

| 使用场景 | 派生方法 |
|----------|---------------|
| 连续控制 | 具有高斯策略的 PPO / SAC |
| LLM RLHF | PPO 带有 KL 惩罚，在 token 级策略上运行 |
| LLM 推理 (DeepSeek) | GRPO — 具有 group-relative baseline 的 REINFORCE，没有 critic |
| 多智能体 | 集中式 critic REINFORCE (MADDPG, COMA) |
| 离散动作机器人 | A2C, A3C, PPO |
| 仅偏好设置 | DPO — 将 REINFORCE 重写为偏好似然损失，没有采样 |

当你在 2026 年训练脚本中读到 `loss = -advantage * log_prob` 时，那就是带 baseline 的 REINFORCE。整篇论文（DPO、GRPO、RLOO）都是这一行的方差减少技巧。

## Ship It（交付物）

保存为 `outputs/skill-policy-gradient-trainer.md`：

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## Exercises（练习）

1. **Easy. 在 4×4 GridWorld 上使用线性 softmax 策略实现 REINFORCE。训练 1,000 个情节，不带 baseline。绘制学习曲线；测量方差（回报的标准差）。**
2. **Medium. 添加 running-mean baseline。再次训练。与朴素运行比较样本效率和方差。Baseline 将收敛步数减少了多少？**
3. **Hard. 添加熵奖励 `β · H(π)`。扫描 `β ∈ {0, 0.01, 0.1, 1.0}`。绘制最终回报和策略熵。这个任务上的最佳点在哪里？**

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Policy gradient | "Train the policy directly" | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`; derived from the log-derivative trick. |
| REINFORCE | "The original PG algorithm" | Williams (1992); Monte Carlo returns multiplied by log-policy gradient. |
| Log-derivative trick | "Score function estimator" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; makes gradients of expectations tractable. |
| Baseline | "Variance reduction" | Any `b(s)` subtracted from `G`; unbiased because `E[b · ∇ log π] = 0`. |
| Reward-to-go | "Only future returns count" | `G_t^{from t}` instead of the full `G_0`; correct and lower-variance. |
| Entropy bonus | "Encourage exploration" | `+β · H(π(·\|s))` term keeps the policy from collapsing. |
| On-policy | "Train on what you just saw" | Gradient expectation is w.r.t. the current policy — cannot reuse old data directly. |
| Advantage | "How much better than average" | `A(s, a) = G(s, a) - V(s)`; the signed quantity REINFORCE-with-baseline multiplies. |

## Further Reading（延伸阅读）

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — 原始 REINFORCE 论文。
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — 具有函数逼近的现代策略梯度定理。
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书阐述。
