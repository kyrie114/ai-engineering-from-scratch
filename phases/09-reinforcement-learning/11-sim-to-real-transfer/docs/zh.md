# Sim-to-Real Transfer（Sim-to-Real 迁移）

> 在模拟器中训练但在硬件上失败的策略是一个记住了模拟器的策略。领域随机化、领域自适应和系统识别是使学习到的控制器跨越现实差距的三个工具。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 9 · 08 (PPO), Phase 2 · 10 (Bias/Variance)
**Time:** ~45 分钟

## The Problem（问题）

训练真实机器人是缓慢、危险和昂贵的。一个双足机器人需要数百万次训练情节来学习走路；真实双足机器人即使摔倒一次也会损坏硬件。模拟器给你无限的 reset、确定性可复现性、并行环境，以及没有物理损坏。

但模拟器是错误的。轴承的摩擦比 MuJoCo 模型大。相机有模拟器不包括的镜头畸变。电机有延迟、回差和饱和度，99% 的 sim 模型跳过这些。风、灰尘和可变照明破坏了在无菌渲染上训练的策略。**现实差距**——sim 分布和真实分布之间的系统性差异——是部署的机器人强化学习的核心问题。

你需要一个对 sim-to-real 分布偏移*鲁棒*的策略。三种历史方法：随机化模拟器（领域随机化），用少量真实数据调整策略（领域自适应 / 微调），或识别真实系统的参数并匹配它们（系统识别）。在 2026 年，主导配方结合了所有三种与大规模并行模拟（Isaac Sim、Isaac Lab、GPU 上的 Mujoco MJX）。

## The Concept（概念）

![Three sim-to-real regimes: domain randomization, adaptation, system identification（三种 sim-to-real 制度：领域随机化、自适应、系统识别）](../assets/sim-to-real.svg)

**领域随机化 (DR)。** Tobin et al. 2017，Peng et al. 2018。在训练期间，随机化真实机器人可能不同的每个 sim 参数：质量、摩擦系数、电机 PD 增益、传感器噪声、相机位置、光照、纹理、接触模型。策略学习"今天它在哪个 sim 中"的条件分布，并在整个跨度上泛化。如果真实机器人落在训练包络内，策略就有效。

- **优点：** 不需要真实数据。一个配方，许多机器人。
- **缺点：** 过度随机化训练产生"通用"但过于谨慎的策略。太多噪声 ≈ 太多正则化。

**系统识别 (SI)。** 在训练前将模拟器的参数拟合到真实世界数据。如果你能测量真实机器人上的臂关节摩擦，将其插入 sim。然后训练期望这些值的策略。需要访问真实系统，但直接减少现实差距。

- **优点：** 精确，低噪声训练目标。
- **缺点：** 残差模型错误对策略是不可见的；小的未识别效果（例如，电机死区）仍然破坏部署。

**领域自适应。** 在 sim 中训练，用少量真实数据微调。两种风格：

- **Real2Sim2Real：** 使用真实 rollout 学习残差模拟器 `f(s, a, z) - f_sim(s, a)`，在修正的 sim 中训练。关闭差距而不需要太多真实数据。
- **观察自适应：** 训练一个通过学习的特征提取器（例如，GAN 像素到像素）将真实 obs 映射为 sim-like obs 的策略。控制器保持在 sim 中。

**特权学习 / 师生。** Miki et al. 2022（ANYmal 四足机器人）。在模拟器中训练一个*教师*，它有权访问特权信息（真实摩擦、地形高度、IMU 漂移）。蒸馏一个*学生*，它只看到真实传感器观察。学生学会从历史中推断特权特征，跨物理参数鲁棒。

**大规模并行模拟。** 2024-2026。Isaac Lab、Mujoco MJX、Brax 都在单个 GPU 上运行数千个并行机器人。具有 4,096 个并行人形的 PPO 在数小时内收集了数年的经验。随着训练分布变宽，"现实差距"缩小；当那 4,096 个 env 中的每一个都有不同的随机化参数时，DR 几乎变得免费。

**真实世界 2026 配方（四足行走例子）：**

1. 具有领域随机化重力、摩擦、电机增益、载荷的大规模并行 sim。
2. 用特权信息（地形图、真实身体速度）训练的教师策略。
3. 只使用本体感觉（腿关节编码器）从教师蒸馏的学生策略。
4. 通过真实 IMU 上的自动编码器的可选观察自适应。
5. 部署。在 10+ 环境上零样本。如果失败，用安全约束 PPO 进行几分钟的真实世界微调。

```figure
f3-reality-gap
```

## Build It（动手实现）

这节课的代码是在具有*嘈杂*转移的 GridWorld 上进行领域随机化的小演示。我们训练一个在"sim"中经历随机化滑动概率的策略，并在"真实"上评估，滑动水平它在训练中从未见过。形状直接映射到 MuJoCo 到硬件的迁移。

### Step 1: 参数化 sim

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` 是模拟器暴露的一个参数。在真实机器人学中，它可能是摩擦、质量、电机增益——任何在 sim 和真实之间偏移的东西。

### Step 2: 用 DR 训练

在每个情节的开始，采样 `slip ~ Uniform[0.0, 0.4]`。训练 PPO / Q-learning / 任何东西。这样做许多情节。

### Step 3: 在"真实"滑动上评估零样本

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上评估。前四个在训练支持内；`0.5` 和 `0.7` 在外面。DR 训练的策略应在支持内保持接近最优，并在外优雅降级。固定滑动训练的策略在其训练滑动之外将是脆弱的。

### Step 4: 与窄训练比较

用 `slip = 0.0` 仅训练第二个策略。评估相同的 `slip` sweep。你应该看到，一旦真实滑动 > 0，就会有灾难性下降。

## Pitfalls（陷阱）

- **随机化太多。** 在 `slip ∈ [0, 0.9]` 上训练，你的策略如此规避风险，以至于从不尝试最优路径。匹配*预期的*真实世界分布，不是"任何事情都可能发生。"
- **随机化太少。** 在薄切片上训练，策略根本无法泛化。使用自适应课程（自动领域随机化），随着策略改进加宽分布。
- **参数空间识别错误。** 随机化错误的东西（相机色调，当真实差距是电机延迟时）DR 没有帮助。首先分析真实机器人。
- **特权信息泄漏。** 一个使用全局状态进行动作的教师，不仅仅是观察，可以产生一个学生无法赶上。确保教师的策略是学生给定观察历史可以实现的形式。
- **Sim-to-sim 迁移失败。** 如果你的策略对更难的 sim 变体不鲁棒，它对真实世界也不会鲁棒。在部署之前总是测试在保留的 sim 变体上。
- **没有真实安全包络。** 一个在 sim 中工作并在真实中"工作"的策略，没有低级安全护盾，仍然可以损坏硬件。在非学习控制器中添加速率限制、扭矩限制、关节限制。

## Use It（应用场景）

2026 年的 sim-to-real 堆栈：

| 领域 | 堆栈 |
|--------|-------|
| 腿部运动（ANYmal、Spot、人形） | Isaac Lab + DR + 特权教师 / 学生 |
| 操作（灵巧手、拣放） | Isaac Lab + DR + 视觉 DR-GAN |
| 自动驾驶 | CARLA / NVIDIA DRIVE Sim + DR + 真实微调 |
| 无人机竞速 | RotorS / Flightmare + DR + 在线自适应 |
| 手指 / 手中操作 | OpenAI Dactyl（前所未有的规模的 DR） |
| 工业臂 | MuJoCo-Warp + SI + 小真实微调 |

对于所有规模的控制，工作流程是一致的：尽可能拟合 sim，随机化你无法拟合的东西，训练巨大的策略，蒸馏，用安全护盾部署。

## Ship It（交付物）

保存为 `outputs/skill-sim2real-planner.md`：

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## Exercises（练习）

1. **Easy. 在固定滑动 GridWorld（slip=0.0）上训练 Q-learning 智能体。在 `slip ∈ {0.0, 0.1, 0.3, 0.5}` 上评估。绘制回报 vs 滑动。**
2. **Medium. 用 `slip ~ Uniform[0, 0.3]` 训练 DR Q-learning 智能体。评估相同的 sweep。DR 在 slip=0.5（分布外）买了多少？**
3. **Hard. 实现课程：从 slip=0.0 开始，每当策略达到最优的 90% 时加宽 DR 范围。测量达到 slip=0.3 零样本的总环境步数 vs 固定 DR 基线。**

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Reality gap | "Sim-to-real difference" | Distribution shift between training and deployment physics/sensing. |
| Domain randomization (DR) | "Train across random sims" | Randomize sim parameters during training so policy generalizes. |
| System identification (SI) | "Measure real and fit sim" | Estimate real physical parameters; set sim to match. |
| Domain adaptation | "Fine-tune on real data" | Small real-world fine-tune after sim training; may adapt obs or dynamics. |
| Privileged info | "Ground truth for teacher" | Information only the sim has; student must infer it from obs history. |
| Teacher/student | "Distill privileged -> observable" | Teacher trained with shortcuts; student learns to mimic without them. |
| ADR | "Automatic Domain Randomization" | Curriculum that widens DR ranges as the policy improves. |
| Real2Sim | "Close the gap with real data" | Learn a residual to make the sim mimic real rollouts. |

## Further Reading（延伸阅读）

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — 原始 DR 论文（机器人视觉）。
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — 动态 DR，四足运动。
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl，大规模 ADR。
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — ANYmal 的师生。
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — 驱动 2025-2026 部署的大规模并行 sim。
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — ADR 课程方法。
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — Dyna 框架（使用模型进行规划 + rollout），它支撑现代 sim-to-real 管道。
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — 具有基准结果的 sim-to-real 方法分类。
