# DualPipe 并行（DualPipe Parallelism）

> DeepSeek-V3 在 2,048 张 H800 GPU 上训练，MoE 专家散落在各节点。跨节点专家 all-to-all 通信每 1 GPU 小时计算就要 1 GPU 小时通信。GPU 有一半时间在闲着。DualPipe（DeepSeek，2024 年 12 月）是一种双向流水线，把前向与反向计算与它们触发的 all-to-all 通信重叠起来。气泡下降，吞吐上升；保存两份模型参数（名字里的「dual」）一旦专家并行（Expert Parallelism）已经把专家铺开到各 rank，代价就便宜。本课是 Learn 型走读：DualPipe 实际做什么，以及为什么 Sea AI Lab 的 DualPipeV 改进在略微更紧的气泡代价下丢掉 2 倍参数成本。

**Type:** Learn
**Languages:** Python (stdlib, schedule simulator)
**Prerequisites:** Phase 10 · 05 (distributed training, FSDP, DeepSpeed), Phase 10 · 14 (open-model architectures and MoE)
**Time:** ~60 minutes

## 学习目标（Learning Objectives）

- 说出 DualPipe 前向-反向块的四个组成部分，以及为什么每一部分都有自己的重叠窗口
- 解释规模下的流水线气泡问题，以及实践中的「无气泡」与营销话术有何不同
- 手工追踪 8 个 PP rank、16 个微批次的 DualPipe 调度，确认正向与反向流填满彼此的空闲槽
- 陈述 DualPipeV（Sea AI Lab，2025）的权衡：丢掉 2 倍参数复制，代价是专家并行未启用时气泡略大

## 问题（The Problem）

在 2k 张 H800 GPU 上训练 671B MoE 模型会撞上三个叠加的瓶颈：

1. **内存压力。** 每张 GPU 只拿模型的一片。序列 8k、61 层、128 头时的激活内存巨大。
2. **流水线气泡（pipeline bubbles）。** 传统流水线并行（GPipe、1F1B）让 GPU 在等待本阶段输入或梯度时闲着。8 个阶段时，即使 1F1B 调度，大约 12% 的 GPU 时间仍可能是气泡。
3. **跨节点 all-to-all。** 带专家并行的 MoE 把专家散到各节点。每次前向都会触发一次 all-to-all 把 token 派到专家，再一次把结果合回来。在 2k GPU 上这很容易变成 1:1 的计算-通信比。

每一项都有单独解法：梯度检查点管内存，Zero Bubble（Sea AI Lab，2023）管流水线气泡，专家并行通信内核管 all-to-all。DualPipe 做的是让它们一起工作。调度在单个前向-反向块内重叠计算与通信，从流水线两端同时注入微批次，并用得到的调度把 all-to-all 藏进计算窗口。

报告结果：流水线气泡近乎消除，DeepSeek-V3 的 14.8T token 训练运行中 GPU 利用率超过 95%。

## 概念（The Concept）

### 流水线并行复习（Pipeline parallelism refresher）

把 N 层模型切到 P 台设备上。设备 `i` 持有层 `i * N/P .. (i+1) * N/P - 1`。一个微批次前向流过设备 0 到 P-1，再从 P-1 反向流到 0。每台设备只有在上一设备送来输出后才能开始前向阶段，只有在下游设备送来上游梯度后才能开始反向。

GPipe（Huang 等人，2019）一次调度一个微批次，浪费大部分 GPU 时间。1F1B（Narayanan 等人，2021）为多个微批次交错前向与反向。Zero Bubble（Qi 等人，2023）把反向拆成两部分——对输入的反向（B）和对权重的反向（W）——并调度它们去填气泡。Zero Bubble 之后，流水线几乎已经绷紧。

DualPipe 是下一步。它在上面加了两个想法：

### 想法 1：块分解（Idea 1: chunk decomposition）

每个前向块拆成四个组成部分：

- **注意力（Attention）。** Q/K/V 投影、注意力、输出投影。
- **All-to-all 分发（dispatch）。** 把 token 送到其专家的跨节点通信。
- **MLP。** MoE 专家计算。
- **All-to-all 合并（combine）。** 把专家输出带回来的跨节点通信。

反向块为这些各自加上梯度版本。DualPipe 调度它们，使 all-to-all 分发与下一块的注意力计算并行，all-to-all 合并与后续块的 MLP 计算并行。

### 想法 2：双向调度（Idea 2: bidirectional scheduling）

大多数流水线调度从阶段 0 注入微批次，流向阶段 P-1。DualPipe 从两端同时注入。阶段 0 看到从那里出发的前向微批次；阶段 P-1 也看到从那里出发的前向微批次。两股流在中间相遇。

为此，设备 `i` 必须同时持有早期流水线层 `i` 和晚期流水线层 `P - 1 - i`。这就是 DualPipe 的「dual」：每台设备保存两份它需要服务的模型层（每个方向一份）。在 DeepSeek-V3 的规模上，这是 2 倍参数复制成本。它负担得起，因为专家并行已经把 MoE 专家铺得很薄，把非专家层复制两次只是小菜。

关键在于：一个方向的前向流与另一方向的反向流，恰好在单方向调度会出现气泡的地方重叠。气泡消失。

### 手工追踪一份调度（A hand-traced schedule）

考虑 P = 4 个 rank、8 个微批次，分成 4 个前向 / 4 个反向。时间从左到右；行是设备 rank。

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

读「F4/F5R」这种记法：rank 1 在同一时间槽里跑微批次 4 的前向（沿流水线从左到右）以及微批次 5 的前向（从右到左）。这就是「双向」在操作上的含义。

在 rank 2，交叉流更早重叠；在 rank 0 和 P-1，最晚重叠。在调度的稳定中间阶段，每个 rank 都在跑 X 方向的前向，并与 Y 方向的反向重叠。计算是忙的。前向的 all-to-all 分发藏在反向计算里。All-to-all 合并藏在前向计算里。气泡被挤掉。

### 气泡核算（Bubble accounting）

标准 1F1B 流水线气泡（每 rank 浪费的时间）：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 的改进把它压下去，但不到零。DualPipe 在稳定阶段，若微批次数量能被 2 倍流水线深度整除，则气泡为零。稳定阶段之外（预热与冷却）仍有一些气泡，但它不随微批次数量增长——论文强调的关键性质。

营销说法：「无气泡」。技术说法：气泡不随微批次数量增长。Sea AI Lab 的后续分析（DualPipeV / Cut-in-half）表明，只有当专家并行不是瓶颈时才有完全零气泡；在 EP 驱动的 all-to-all 下，调度上总有某种妥协。

### DualPipeV —— 改进（The refinement）

Sea AI Lab（2025）观察到，当 EP 通信重叠不是重点时，2 倍参数复制是浪费。他们的 DualPipeV 调度把双向注入折成「V 形」调度，跑在单份参数拷贝上。气泡比 DualPipe 略大，但内存节省可观。DeepSeek 在开源 DualPipe 实现里把 DualPipeV 作为 EP-off 模式采用。

权衡：

| 特性 | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| 每设备参数拷贝 | 2 | 1 | 1 | 1 |
| 气泡相对微批次 | 常数 | 小幅增长 | 增长 | 增长 |
| 计算-通信重叠 | 完整 | 部分 | 最小 | 部分 |
| 何时用 | EP 很重的 MoE | 稠密或 EP 很轻 | 基线 | 任意流水线 |

### 对 14.8T token 运行意味着什么（What it means for a 14.8T-token run）

DeepSeek-V3 的预训练在 2,048 张 H800 上消耗了 14.8T token，大约 2.8M GPU 小时。若用朴素 1F1B，他们会把其中 12–15% 丢给流水线气泡——340–420K GPU 小时，够训完一个完整 70B 模型。DualPipe 回收了其中大部分。没有内部日志很难直接量化贡献，但论文声称训练平均 GPU 利用率超过 95%。

对较小运行（低于 1k GPU），DualPipe 过头了——流水线气泡相对总成本更小，稠密模型训练也很少撞上 all-to-all 瓶颈。对多千 GPU 规模的前沿 MoE 训练，它实际上是必需的。

### 它在栈中的位置（Where it sits in the stack）

- 与 **FSDP**（Phase 10 · 05）互补。FSDP 跨 rank 切分模型参数；DualPipe 跨 rank 调度计算。它们可以组合。
- 与 **ZeRO-3** 梯度分片兼容。两份复制的簿记需要与 ZeRO 的分片梯度协作。
- 需要针对具体集群拓扑调过的 **自定义 all-to-all 内核**。DeepSeek 的开源内核是参考实现。

```figure
expert-capacity
```

## 使用它（Use It）

`code/main.py` 是流水线调度模拟器。它接受 `(P, n_micro_batches, schedule)`，并打印 1F1B、Zero Bubble、DualPipe 和 DualPipeV 各自的稳定阶段利用率。它是教学工具——数字匹配论文中的定性主张，不是生产实测加速的声明。

模拟器的价值：用不同的 P 和微批次数量运行，观察气泡占比如何对 1F1B 增长、对 DualPipe 不增长。

真实训练运行的集成考虑：

- 选一个能被微批次数量干净整除的流水线并行深度。
- 确保专家并行网格支持双向 all-to-all。DeepSeek 的内核是参考。
- 第一次预期要烧掉一周调试调度本身。簿记很琐碎。
- 监控每个 rank 的 GPU 利用率，而不只是聚合。DualPipe 的收益来自收紧拖后腿的那些。

## 交付（Ship It）

本课产出 `outputs/skill-dualpipe-planner.md`。给定训练集群规格（GPU 数量、拓扑、互联、模型形状），它推荐流水线并行策略、要用的调度算法，以及目标规模下的预期气泡占比。

## 练习（Exercises）

1. 在 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)` 上运行 `code/main.py`。计算 GPU 利用率差，并把它表示成每百万训练 token 回收的 GPU 小时。

2. 手工画出 `(P=4, micro_batches=8, schedule=dualpipe)` 的调度表。给每个时间槽标上微批次 ID 和方向。找出气泡首次消失的时间槽。

3. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）图 5。找出 DualPipe 前向块内 all-to-all 分发的重叠窗口。解释计算调度如何把它藏起来。

4. 计算 DualPipe 对稠密 70B 模型（P=8 个流水线阶段）和 671B MoE 模型（P=16 个流水线阶段）的 2 倍参数开销。说明为什么 MoE 情形的开销比例更小（大多数参数是专家，分片在很大的 EP 组上）。

5. 把 DualPipe 与 Chimera（2021 年的竞争性双向调度器）比较。用论文第 3.4 节作参考，指出 DualPipe 新增而 Chimera 没有的两个具体性质。

## 关键术语（Key Terms）

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| 流水线气泡 | 「每 rank 的空闲时间」 | 流水线阶段在等输入或梯度时浪费的 GPU 周期 |
| 1F1B | 「默认流水线调度」 | 一次前向 / 一次反向交错调度；DualPipe 要击败的基线 |
| Zero Bubble | 「Sea AI Lab 2023」 | 把反向拆成 B（输入梯度）和 W（权重梯度）；几乎完全绷紧流水线 |
| DualPipe | 「DeepSeek-V3 调度」 | 双向流水线 + 计算-通信重叠；气泡不随微批次数量增长 |
| DualPipeV | 「一切两半」 | V 形改进，丢掉 2 倍参数复制，代价是气泡略大 |
| 块（Chunk） | 「流水线工作单元」 | 一个微批次穿过一个流水线阶段的前向或反向 |
| All-to-all 分发 | 「把 token 送到专家」 | 把 token 路由到指定 MoE 专家的跨节点通信 |
| All-to-all 合并 | 「把专家输出带回来」 | MLP 之后收集专家输出的跨节点通信 |
| 专家并行（EP） | 「专家跨 GPU」 | 把 MoE 专家分片到各 rank，不同 GPU 持有不同专家 |
| 流水线并行（PP） | 「层跨 GPU」 | 把模型层分片到各 rank；DualPipe 调度的那个维度 |
| 气泡占比 | 「浪费的 GPU 时间」 | (bubble_time / total_time)；DualPipe 把它推向零 |

## 延伸阅读（Further Reading）

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) — DualPipe 的主要参考
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) — 开源参考实现，含 DualPipeV（Cut-in-half）模式
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — Zero Bubble 前身
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) — 影响 DeepSeek EP-off 模式的 DualPipeV 分析
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — DualPipe 对照的 1F1B 调度
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — 最初的流水线并行论文和气泡问题
