# Deep Q-Networks (DQN)（深度 Q 网络 (DQN)）

> 2013 年：Mnih 在原始像素上训练了一个 Q-learning 网络，在 7 个 Atari 游戏上击败了所有经典强化学习智能体。2015 年：扩展到 49 个游戏，发表在 Nature 上，开启了深度强化学习时代。DQN 是 Q-learning 加上三个使函数逼近稳定的技巧。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 04 (Q-learning, SARSA)
**Time:** ~75 分钟

## The Problem（问题）

表格 Q-learning 需要为每个 (状态, 动作) 对存储一个单独的 Q 值。一个国际象棋棋盘有约 10⁴³ 个状态。一个 Atari 帧是 210×160×3 = 100,800 个特征。表格强化学习在数千个状态时就会失效，更不用说数十亿了。

事后看来，修复方法很明显：用神经网络 `Q(s, a; θ)` 替换 Q 表。但这个"事后显而易见"的想法花了数十年才实现。使用 Q-learning 的朴素函数逼近在"致命 triad"下会发散——函数逼近 + 自举 + 离策略学习。Mnih et al. (2013, 2015) 识别出了三个稳定学习的工程技巧：

1. **Experience replay** 去相关转移。
2. **Target network** 冻结自举目标。
3. **Reward clipping** 归一化梯度幅度。

Atari 上的 DQN 是第一次一个单一架构、单一超参数集从原始像素解决数十个控制问题。此后构建的所有"深度强化学习"——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——都堆叠在这个三技巧基础之上。

## The Concept（概念）

![DQN training loop: env, replay buffer, online net, target net, Bellman TD loss（DQN 训练循环：环境、replay buffer、在线网络、目标网络、贝尔曼 TD 损失）](../assets/dqn.svg)

**目标。** DQN 最小化神经 Q 函数上的单步 TD 损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 在线网络，每步通过梯度下降更新。`θ^-` = 目标网络，定期从 `θ` 复制（每 ~10,000 步）。`D` = 过去转移的 replay buffer。

**三个技巧，按重要性顺序：**

**Experience replay。** 一个容量为 `~10⁶` 的环形 buffer。每个训练步骤随机采样一个小批量。这打破了时间相关性（连续帧几乎相同），让网络多次从稀有奖励转移中学习，并使连续梯度更新去相关。没有它，神经网络上的 on-policy TD 在 Atari 上会发散。

**Target network。** 在贝尔曼方程的两边使用相同的网络 `Q(·; θ)` 会使目标每次更新都移动——"追逐自己的尾巴"。修复方法：保留第二个网络 `Q(·; θ^- )` 权重冻结。每 `C` 步，复制 `θ → θ^-`。这使回归目标稳定数千个梯度步。软更新 `θ^- ← τ θ + (1-τ) θ^-`（在 DDPG、SAC 中使用）是一个更平滑的变体。

**Reward clipping。** Atari 奖励幅度从 1 到 1000+ 不等。裁剪到 `{-1, 0, +1}` 阻止任何单个游戏主导梯度。当奖励幅度重要时是错误的；对于 Atari 来说没问题，那里只有符号重要。

**Double DQN。** Hasselt (2016) 修复了最大化偏差：使用在线网络*选择*动作，目标网络*评估*它。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

直接替换，一致更好。默认使用它。

**其他改进（Rainbow, 2017）：** 优先 replay（采样高 TD-error 转移更多）、dueling 架构（单独的 `V(s)` 和 advantage heads）、noisy networks（学习的探索）、n-step 回报、distributional Q（C51/QR-DQN）、多步自举。每个增加几个百分点；收益大致是可加的。

```figure
f3-dqn-stability
```

## Build It（动手实现）

这里的代码是纯 stdlib，没有 numpy——我们使用手写的单隐藏层 MLP 在一个微小的连续 GridWorld 上，所以每个训练步都以微秒运行。该算法在规模上与 Atari DQN 相同。

### Step 1: replay buffer

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

对于 Atari 约 50,000 容量；对于我们的玩具环境 5,000 就足够了。

### Step 2: 一个微小的 Q 网络（手动 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传播：线性 → ReLU → 线性。这就是整个网络。

### Step 3: DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

形状是 Lesson 04 中的 Q-learning，有两个区别：(a) 我们通过可微的 `Q(·; θ)` 反向传播，而不是索引一个表格，(b) 目标使用 `Q(·; θ^-)`。

### Step 4: 外循环

对于每个情节，在 `Q(·; θ)` 上 ε-贪婪行动，将转移推入 buffer，采样一个小批量，取一个梯度步，定期同步 `θ^- ← θ`。模式：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我们带有 16 维 one-hot 状态的微小 GridWorld 上，智能体在约 500 个情节中学习接近最优的策略。在 Atari 上，将其扩展到 200M 帧并添加 CNN 特征提取器。

## Pitfalls（陷阱）

- **致命 triad。** 函数逼近 + 离策略 + 自举可能发散。DQN 用目标网络 + replay 缓解；不要删除任何一个。
- **探索。** ε 必须衰减，通常从 1.0 到前 ~10% 训练中的 0.01。没有足够的早期探索，Q 网络会收敛到局部 basin。
- **高估。** 嘈杂 Q 上的 `max` 向上偏差。生产环境中总是使用 Double DQN。
- **奖励尺度。** 裁剪或归一化奖励；梯度幅度与奖励幅度成正比。
- **Replay buffer 冷启动。** 在 buffer 有几千个转移之前不要训练。早期梯度在 ~20 个样本上过拟合。
- **目标同步频率。** 太频繁 ≈ 没有目标网络；太不频繁 ≈ 过时目标。Atari DQN 使用 10,000 环境步。经验法则：每 ~1/100 的训练视界同步一次。
- **观察预处理。** Atari DQN 堆叠 4 帧以使状态马尔可夫。任何具有速度信息的 env 需要帧堆叠或循环状态。

## Use It（应用场景）

在 2026 年，DQN 很少是最先进的，但仍然作为参考离策略算法：

| 任务 | 首选方法 | 为什么不用 DQN？ |
|------|----------|--------------|
| 离散动作 Atari 类 | Rainbow DQN 或 Muesli | 相同框架，更多技巧。 |
| 连续控制 | SAC / TD3（Phase 9 · 07） | DQN 没有策略网络。 |
| On-policy / 高吞吐量 | PPO（Phase 9 · 08） | 没有 replay buffer；更容易扩展。 |
| 离线强化学习 | CQL / IQL / Decision Transformer | 保守 Q 目标，没有自举爆炸。 |
| 大型离散动作空间（推荐器） | 带动作嵌入的 DQN，或 IMPALA | 可以；但装饰很重要。 |
| LLM RL | PPO / GRPO | 序列级别，不是步级别；不同的损失。 |

教训仍然适用。Replay 和目标网络出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 的 self-play buffer，以及每一个离线强化学习方法中。Reward clipping 作为 PPO 中的优势归一化继续存在。该架构是蓝图。

## Ship It（交付物）

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## Exercises（练习）

1. **Easy. 运行 `code/main.py`。绘制每情节回报曲线。多少个情节后运行均值超过 -10？**
2. **Medium. 禁用目标网络（对贝尔曼目标的两边都使用在线网络）。测量训练不稳定性——回报是否振荡或发散？**
3. **Hard. 添加 Double DQN：使用在线网络选择 `argmax a'`，目标网络评估。在嘈杂奖励 GridWorld 上比较 `Q(s_0, best_a)` 与真实 `V*(s_0)` 的偏差，有 vs 没有 Double DQN 经过 1,000 个情节后。**

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| DQN | "Deep Q-learning" | Q-learning with a neural Q-function, replay buffer, and target network. |
| Experience replay | "Shuffled transitions" | Ring buffer sampled uniformly each gradient step; decorrelates data. |
| Target network | "Frozen bootstrap" | Periodic copy of Q used in the Bellman target; stabilizes training. |
| Deadly triad | "Why RL diverges" | Function approximation + bootstrapping + off-policy = no convergence guarantee. |
| Double DQN | "Fix for maximization bias" | Online net selects action, target net evaluates it. |
| Dueling DQN | "V and A heads" | Decompose Q = V + A - mean(A); same output, better gradient flow. |
| Rainbow | "All the tricks" | DDQN + PER + dueling + n-step + noisy + distributional in one. |
| PER | "Prioritized Replay" | Sample transitions proportional to TD-error magnitude. |

## Further Reading（延伸阅读）

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 开启深度强化学习的 2013 年 NeurIPS 研讨会论文。
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Nature 论文，49 游戏 DQN。
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN。
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — dueling DQN。
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — 堆叠技巧论文。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 清晰的现代阐述。
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书中关于 DQN 的目标网络和 replay buffer 旨在驯服的"致命 triad"（函数逼近 + 自举 + 离策略）的处理。
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — 用于消融研究的参考单文件 DQN；最好与本课的 from-scratch 版本一起阅读。
