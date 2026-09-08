# Proximal Policy Optimization (PPO)（近端策略优化 (PPO)）

> A2C 在每次更新后丢弃 rollout。PPO 将策略梯度包裹在裁剪的重要性比率中，以便你可以在同一数据上进行 10 多次轮次，而策略不会爆炸。Schulman et al. (2017)。在 2026 年仍然是默认的策略梯度算法。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~75 分钟

## The Problem（问题）

A2C（Lesson 07）是 on-policy 的：梯度 `E_{π_θ}[A · ∇ log π_θ]` 需要从*当前* `π_θ` 采样的数据。进行一次更新，`π_θ` 就改变了；你使用的数据现在是离策略的。重新使用它，你的梯度就有偏差。

Rollout 是昂贵的。在 Atari 上，跨 8 个 env × 128 步 = 1024 个转移和十几秒的环境时间的一个 rollout。在一次梯度步后丢弃它是浪费的。

Trust Region Policy Optimization (TRPO, Schulman 2015) 是第一个修复方法：约束每个更新，使旧策略和新策略之间的 KL 散度保持在 `δ` 以下。理论上干净，但每次更新需要共轭梯度求解。2026 年没有人运行 TRPO。

PPO (Schulman et al. 2017) 用简单的裁剪目标替换了硬信任区域约束。多一行代码。每个 rollout 十轮。没有共轭梯度。足够好的理论保证。九年之后，它仍然是 MuJoCo 到 RLHF 所有内容的默认策略梯度算法。

## The Concept（概念）

![PPO clipped surrogate objective: ratio clipping at 1 ± ε（PPO 裁剪替代目标：在 1 ± ε 处的比率裁剪）](../assets/ppo.svg)

**重要性比率。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略与收集数据的策略的似然比。`r_t = 1` 意味着没有变化。`r_t = 2` 意味着新策略是旧策略的两倍可能采取 `a_t`。

**裁剪替代。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两项：

- 如果 advantage `A_t > 0` 且比率试图超过 `1 + ε`，裁剪使梯度变平——不要将好动作推得比旧概率高 `+ε` 以上。
- 如果 advantage `A_t < 0` 且比率试图低于 `1 - ε`（意味着我们会使坏动作比其裁剪减少更可能），裁剪限制梯度——不要将坏动作推得比 `-ε` 低。

`min` 处理另一个方向：如果比率已经在*有益*方向上移动，你仍然得到梯度（在会伤害你的一侧没有裁剪）。

典型的 `ε = 0.2`。将目标绘制为 `r_t` 的函数：一个分段线性函数，在"好一侧"有一个平坦的屋顶，在"坏一侧"有一个平坦的地板。

**完整的 PPO 损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

与 A2C 相同的 actor-critic 结构。三个系数，通常 `c_v = 0.5`，`c_e = 0.01`，`ε = 0.2`。

**训练循环。**

1. 跨 `N` 个并行 env 收集 `N × T` 个转移，每个 `T` 步。
2. 计算 advantages（GAE），将它们冻结为常数。
3. 将 `π_{θ_old}` 快照为当前 `π_θ` 的快照。
4. 对于 `K` 轮，对于每个 `(s, a, A, V_target, log π_old(a|s))` 的小批量：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + 价值损失 + 熵。
   - 梯度步。
5. 丢弃 rollout。返回步骤 1。

`K = 10` 和 64 的小批量是一个标准的超参数集。PPO 是鲁棒的：在 ±50% 内确切数字很少重要。

**KL-惩罚变体。** 原始论文提出了一个使用自适应 KL 惩罚的替代方案：`L = L^{PG} - β · KL(π_θ || π_old)`，其中 `β` 根据观察到的 KL 调整。裁剪版本成为主导；KL 变体在 RLHF 中存活（那里 KL 到参考策略是一个你无论如何都想要的单独约束）。

```figure
ppo-clip
```

## Build It（动手实现）

### Step 1: 在 rollout 时捕获 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照在 rollout 时只取一次。它在更新轮次期间不会改变。

### Step 2: 计算 GAE advantages（Lesson 07）

与 A2C 相同。跨 batch 归一化。

### Step 3: 裁剪替代更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

"裁剪 → 零梯度"模式是 PPO 的核心。如果新策略已经在有益方向上漂移得太远，更新就停止。

### Step 4: 价值和熵

添加标准 MSE 到 critic 目标和 actor 上的熵奖励，与 A2C 相同。

### Step 5: 诊断

每次更新要观察三件事：

- **Mean KL** `E[log π_old - log π_θ]`。应该保持在 `[0, 0.02]`。如果它超过 `0.1`，减少 `K_EPOCHS` 或 `LR`。
- **Clip fraction**——比率落在 `[1-ε, 1+ε]` 之外的样本比例。应该是 `~0.1-0.3`。如果 `~0`，裁剪从未触发 → 提高 `LR` 或 `K_EPOCHS`。如果 `~0.5+`，你正在过拟合 rollout → 降低它们。
- **Explained variance** `1 - Var(V_target - V_pred) / Var(V_target)`。Critic 质量指标。随着 critic 学习应该接近 1。

## Pitfalls（陷阱）

- **Clip 系数调优不当。** `ε = 0.2` 是事实上的标准。到 `0.1` 使更新太胆小；`0.3+` 邀请不稳定性。
- **太多轮次。** `K > 20`  routinely  destabilizes，因为策略从 `π_old` 漂移得太远。限制轮次，特别是对于大型网络。
- **没有奖励归一化。** 大奖励尺度会侵蚀裁剪范围。在计算 advantages 之前归一化奖励（running std）。
- **忘记 advantage 归一化。** 每 batch 零均值/单位标准差归一化是标准的。跳过它在大多数基准上破坏 PPO。
- **学习率没有衰减。** PPO 受益于线性 LR 衰减到零。常数 LR 通常更差。
- **重要性比率数学错误。** 总是 `exp(log_new - log_old)` 以获得数值稳定性，不是 `new / old`。
- **错误的梯度符号。** 最大化替代 = *最小化* `-L^{CLIP}`。翻转符号是最常见的 PPO bug。

## Use It（应用场景）

PPO 是 2026 年跨令人惊讶多的领域的默认强化学习算法：

| 使用场景 | PPO 变体 |
|----------|-------------|
| MuJoCo / 机器人控制 | 具有高斯策略的 PPO，GAE(0.95) |
| Atari / 离散游戏 | 具有分类策略的 PPO，滚动 128 步 rollout |
| LLM 的 RLHF | PPO 带有 KL 惩罚到参考模型，在响应结束时从 RM 奖励 |
| 大规模游戏智能体 | IMPALA + PPO (AlphaStar, OpenAI Five) |
| 推理 LLM | GRPO（Lesson 12）——没有 critic 的 PPO 变体 |
| 仅偏好数据 | DPO——将 PPO+KL 坍缩为闭式，没有在线采样 |

PPO *损失形状*——裁剪替代 + 价值 + 熵——是 DPO、GRPO 和几乎所有 RLHF 管道的脚手架。

## Ship It（交付物）

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## Exercises（练习）

1. **Easy. 在 4×4 GridWorld 上用 `ε=0.2, K=4` 运行 PPO。在匹配的环境步数下与 A2C（每个 rollout 一个轮次）比较样本效率。**
2. **Medium. 扫描 `K ∈ {1, 4, 10, 30}`。绘制回报 vs 环境步数并跟踪每次更新的平均 KL。在这个任务上 `K` 何时导致 KL 爆炸？**
3. **Hard. 用自适应 KL 惩罚（如果 `KL > 2·target` 则 `β` 翻倍，如果 `KL < target/2` 则 `β` 减半）替换裁剪替代。比较最终回报、稳定性和无裁剪性。**

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Importance ratio | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`; deviation from the policy that collected the data. |
| Clipped surrogate | "PPO's main trick" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`; flat gradient past the clip on beneficial side. |
| Trust region | "TRPO / PPO intent" | Limit each update's KL to guarantee monotone improvement. |
| KL penalty | "Soft trust region" | Alternative PPO: `L - β · KL(π_θ \|\| π_old)`. Adaptive `β`. |
| Clip fraction | "How often clipping triggers" | Diagnostic — should be 0.1-0.3; outside means mistuned. |
| Multi-epoch training | "Data reuse" | K epochs on each rollout; variance cost traded for sample efficiency. |
| On-policy-ish | "Mostly on-policy" | PPO is nominally on-policy but K>1 epochs uses slightly-off-policy data safely. |
| PPO-KL | "The other PPO" | KL-penalty variant; used in RLHF where KL-to-reference is already a constraint. |

## Further Reading（延伸阅读）

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 论文。
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO，PPO 的前身。
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — 每一个 PPO 超参数的消融。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT；RLHF 中的 PPO 配方。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — 使用 PyTorch 的清晰现代阐述。
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — 许多论文使用的参考单文件 PPO。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — 语言模型 PPO 的生产配方；与 Lesson 09（RLHF）一起阅读。
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — "37 个代码级优化"论文；哪些 PPO 技巧是承重的，哪些是民间传说。
