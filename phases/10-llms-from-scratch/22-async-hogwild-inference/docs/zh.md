# 异步与 Hogwild! 推理（Async and Hogwild! Inference）

> 投机解码（Phase 10 · 15）在一条序列内部并行化 token。多智能体框架跨整条序列并行，但强迫显式协调（投票、子任务拆分）。Hogwild! 推理（Rodionov 等人，arXiv:2504.06261）做的是另一件事：让同一 LLM 的 N 个实例并行跑在一份共享键值缓存（key-value cache）上。每个工作者立刻看到其他工作者生成的 token。现代推理模型——QwQ、DeepSeek-R1——可以不经微调，通过那份共享缓存自我协调。方法仍是实验性的，但它打开了一条与投机解码正交的全新推理并行轴。本课用标准库 Python 实现双工作者 Hogwild! 模拟器，并解释为什么共享缓存协作从现有模型的推理能力里涌现出来。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 12 (inference optimization), Phase 10 · 15 (speculative decoding)
**Time:** ~60 minutes

## 学习目标（Learning Objectives）

- 描述三种常见的并行 LLM 拓扑（投票、子任务、Hogwild!），并指出各自瞄准哪些问题
- 陈述 Hogwild! 的核心设置：多个工作者、一份共享 KV 缓存、经自我提示涌现的协调
- 把 Hogwild! 的墙钟加速表示为工作者数 `N`、任务级并行度 `p` 和协调开销 `c` 的函数
- 在玩具问题上实现双工作者 Hogwild! 模拟器，并观察涌现的任务划分

## 问题（The Problem）

现代 LLM 靠产出长推理链来解难题——五千 token 的逐步逻辑很常见，深层数学题上会出现数万 token。70B 模型以 35 tokens/sec 解码，50k token 就是 24 分钟。交互式它不是。

投机解码（Phase 10 · 15）通过在一条序列内部并行化给你 3–5 倍加速。再往上，自回归解码的顺序依赖就是硬天花板。每个新 token 依赖此前每一个 token。

显而易见的问题：我们能跨序列并行吗？在同一问题上跑同一模型的多份拷贝，让它们合作，让它们划分工作？

先前工作：投票集成（跑 N 个模型，选多数答案）、思维树（tree-of-thought，分支推理路径再重组），以及多智能体框架（给每个智能体分配子任务，用协调者）。这些在特定任务域都有帮助。它们也都引入显式协调机械——投票规则、分支剪枝逻辑、智能体间消息协议。

Hogwild! 推理走另一条路。N 个工作者共享一份 KV 缓存。每个工作者立刻看到其他工作者生成的 token，仿佛那就是自己的上下文。这些工作者——没有任何训练或微调——自己想出如何划分工作。现代推理模型（QwQ、DeepSeek-R1、Claude 家族推理模式）能读共享缓存，并说出类似「我看到工作者 2 已经处理了基本情况，所以我做归纳步骤」。

加速依赖工作负载，截至 2026 年 4 月仍是实验性的。但这个想法值得知道，因为它打开了一条新的推理并行轴。

## 概念（The Concept）

### 设置（The setup）

初始化 N 个工作者进程，都跑同一个 LLM。不是每工作者一份 KV 缓存，而是维护一份共享缓存。当工作者 `i` 生成 token `t_j`，该 token 被写进共享缓存的下一位置。当工作者 `k` 走下一步，它读缓存的当前状态（其中包含所有 N 个工作者到目前生成的一切）。

在步进时刻，工作者竞相写 token。没有每工作者位置索引——缓存是一条单一增长的序列。顺序由写入到达时间决定。

### 为什么协调会涌现（Why coordination emerges）

工作者共享一份提示。通常类似「你是 N 个实例之一，一起解这个问题。每个实例读共享内存，能看到其他实例写了什么。避免重复工作。」提示加上共享缓存就够了。推理模型读缓存，注意到问题的哪些部分已经被尝试，并（常常但并非总是）转向尚未探索的部分。

Hogwild! 论文（Rodionov 等人，2025）报告的观察包括：

- 工作者制定计划，并通过缓存把计划传达给其他工作者。
- 工作者注意到其他工作者推理中的错误并指出来。
- 计划失败时工作者会适应并提出替代方案。
- 被提示检查冗余时，工作者会检测并转向。

这些都不需要微调。涌现行为来自模型已经具备的推理能力。

### 命名（The naming）

论文名字戏仿 Hogwild! SGD（Recht 等人，2011），一种异步更新优化器。类比：SGD 的异步工作者都写向一份共享参数向量；Hogwild! 推理的工作者都写向一份共享 KV 缓存。两者都依赖经验收敛，而不是同步保证。

### RoPE 让这可行（RoPE makes this tractable）

旋转位置嵌入（Rotary Position Embeddings，RoPE，Su 等人 2021）通过 Q 和 K 向量中的旋转编码位置信息。因为位置是旋转而不是烤进去的偏移，token 的位置可以移动而不必重算 KV 缓存条目。当工作者 `i` 在位置 `p` 写入共享缓存，其他工作者读该位置可以直接用缓存条目——无需再旋转。

在学习位置或绝对位置模型里，Hogwild! 每次并发写入都需要缓存失效。RoPE 让缓存保持稳定。

### 墙钟数学（Wall-time math）

令 `T_serial` 为一个工作者单独解题的时间。令 `p` 为任务级可并行比例。令 `c` 为每步协调开销（读扩展后的缓存、决定写什么）。

单工作者时间：`T_serial`。
N 工作者 Hogwild! 时间，若协调免费：`T_serial * ((1 - p) + p / N)`。经典 Amdahl。
带协调开销：`T_serial * ((1 - p) + p / N) + c * steps_per_worker`。

工作者要有产出，`c` 必须相对每步解码时间很小。在产出 5k+ token 的推理模型上，工作者负担得起几百 token 的协调开销仍然领先。在短聊天任务上，协调占主导，Hogwild! 比串行更差。

### 具体例子（Concrete example）

推理问题：1 万 token 的思维链。假设问题有 `p = 0.7` 可并行内容（不同证明策略、不同案例分析），每工作者 `c = 200` token 协调开销。取 `N = 4` 个工作者：

- 串行时间：10000 次解码步。
- Hogwild! 时间：10000 * (0.3 + 0.7 / 4) + 200 * 4 = 10000 * 0.475 + 800 = 5550 次解码步。
- 加速：10000 / 5550 = 1.8x。

这很温和。但在更长的推理问题（50k token）上，协调开销被摊销，加速推向 2.5–3x。Hogwild! 相当于一种语言里线程级并行的推理版本，让你自然写出多线程代码。

### 何时伸手去拿 Hogwild!（When to reach for Hogwild!）

- 长推理问题（数千 token），任务能跨独立子目标并行。
- 被训练成逐步思考的推理模型。非推理模型自我协调不好。
- 单节点部署，VRAM 够放下共享缓存加上 N 个工作者进程。缓存是共享的，但每个工作者有自己的激活内存。

### 何时不要（When not to）

- 短交互聊天。协调开销占主导。
- 不能并行的任务（单一线性证明、单一编译）。N=1 就是上限。
- 非推理模型。没有协调涌现。
- 多节点部署。共享缓存需要非常快的跨工作者同步。节点内没问题；跨节点是延迟灾难。

### 实验状态（The experimental status）

截至 2026 年 4 月，Hogwild! 是带开源 PyTorch 实现的研究方法。生产采用尚未发生。三个阻碍：

1. 跨并发进程管理共享 KV 缓存是不平凡的工程。
2. 涌现协调依赖任务；基准仍在建设。
3. 加速相对投机解码已经交付的成果偏温和，两者可以组合，但组合工程又是一层。

值得知道。值得实验。还不值得拿产品去赌。

```figure
continuous-batching
```

## 动手构建（Build It）

`code/main.py` 实现一个玩具 Hogwild! 模拟器：

- 两个工作者进程，各自是确定性「LLM」，以已知概率产出若干 token 类别之一（工作 token、观察 token、协调 token）。
- 一份共享缓存（只是一个 token 列表），两个工作者都读都写。
- 简单协调逻辑：当一个工作者看到另一个已经在某类别产出足够工作 token，它就选另一个类别。

模拟器跑固定步数预算并报告：

- 产出的工作 token 总数。
- 总墙钟时间（工作者步数）。
- 相对单工作者的有效加速。
- 哪个工作者写了哪个 token 的轨迹。

### 第 1 步：共享缓存（Step 1: the shared cache）

一个两边都往上追加的列表。真实实现里用简单锁（Python `threading.Lock`）；我们用计数器模拟。

### 第 2 步：工作者循环（Step 2: the worker loop）

每个工作者每一步：

- 读当前共享缓存。
- 根据已经在那里的内容决定写哪一类 token。
- 写一个 token。

### 第 3 步：协调启发式（Step 3: the coordination heuristic）

如果类别 X 在缓存里已经有 K 个 token，而工作者打算写的类别是 X，工作者就切到类别 Y。这是对推理模型「注意到这块已经被覆盖，改做别的」行为的玩具替身。

### 第 4 步：测得的加速（Step 4: measured speedup）

用 N=1 工作者和 N=2 工作者跑模拟器，总步数预算相同。统计产出的工作 token。N=2 应大约多产出 1.5–1.8 倍工作 token，因为协调驱动的任务划分。

### 第 5 步：给协调加压（Step 5: stress the coordination）

降低协调启发式的敏感度。再跑。观察到没有好的协调时，N=2 冗余产出相同 token，加速掉到 1 以下。这匹配论文观察：技巧只有在工作者具备自我协调的推理能力时才成立。

## 使用它（Use It）

截至 2026 年 4 月，Hogwild! 在生产中的集成仍是研究级。Yandex/HSE/IST 的参考实现基于 PyTorch，面向 DeepSeek-R1 和 QwQ 模型上的单节点多进程设置。

务实采用路径：

1. 给推理任务工作负载画像。测量探索性 token（多种策略、案例分析、搜索）相对线性 token 的比例。
2. 若探索占主导，跑一次双工作者 Hogwild! 实验。测量墙钟改进。
3. 若改进低于 1.3x，你处在协调主导区。回到单工作者。
4. 若改进超过 1.5x，推到 N=4 再测。收益递减通常在 N=4–8 附近出现。

与投机解码组合：每个 Hogwild! 工作者可以独立使用投机解码。两次加速大致相乘，把 3x 投机解码和 1.8x Hogwild! 变成相对朴素单工作者解码约 5.4x。

## 交付（Ship It）

本课产出 `outputs/skill-parallel-inference-router.md`。给定推理工作负载画像（token 预算、任务并行画像、模型家族、部署目标），它在投票、思维树、多智能体、Hogwild! 和投机解码策略之间路由。

## 练习（Exercises）

1. 用默认设置运行 `code/main.py`。确认 N=2 Hogwild! 配置在相同墙钟时间内比 N=1 基线产出更多工作 token。

2. 降低协调启发式强度（设 `coordination_weight=0.1`）。再跑。展示加速崩溃。解释原因：工作者无法协调时会重复劳动。

3. 计算 50k token 推理任务在 `p=0.8, c=500`、N=4 工作者下的预期 Hogwild! 加速。对 1k token 聊天任务、`p=0.3, c=200`、N=4 做同样计算。为什么一个赢、另一个输？

4. 阅读 Hogwild! 论文第 4 节（初步评测）。指出作者报告的两种失败模式。描述更好的协调提示如何可能缓解每一种。

5. 在玩具里把 Hogwild! 与投机解码组合：每个工作者内部用 2 token 投机解码。报告相乘加速。当两个工作者都想延长同一共享缓存前缀时，会出现什么簿记问题？

## 关键术语（Key Terms）

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Hogwild! | 「并行工作者，共享缓存」 | 同一 LLM 的 N 个实例并发跑在一份共享 KV 缓存上；经自我提示涌现协调 |
| 共享 KV 缓存 | 「协调媒介」 | 所有工作者都读都写的单一增长 KV 缓冲；实现跨工作者即时 token 可见 |
| 涌现协调 | 「不需要训练」 | 有推理能力的 LLM 能读共享缓存并划分工作，无需微调或显式协议 |
| 协调开销（c） | 「花在定向的 token」 | 每工作者读扩展缓存并决定做什么的成本；必须相对总解码时间保持很小 |
| 可并行比例（p） | 「什么能并行跑」 | 任务级并行：总工作中并非内在顺序的那部分比例 |
| RoPE 使能 Hogwild! | 「旋转位置是平移不变的」 | 因为位置是旋转，写入共享缓存不需要重算先前 token |
| 投票集成 | 「跑 N 个，选多数」 | 最简单的并行推理拓扑；对分类有用，对长文推理较差 |
| 思维树 | 「分支并剪枝」 | 探索多条分支再剪枝的推理策略；显式协调逻辑 |
| 多智能体框架 | 「分配子任务」 | 每个智能体有角色；协调者编排；协议开销重 |

## 延伸阅读（Further Reading）

- [Rodionov et al. — Hogwild! Inference: Parallel LLM Generation via Concurrent Attention (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261) — Hogwild! 论文，在 QwQ 和 DeepSeek-R1 上的初步评测
- [Recht, Re, Wright, Niu — Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730) — 最初的 Hogwild!，命名来源
- [Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) — RoPE，使共享缓存推理可行的性质
- [Yao et al. — Tree of Thoughts: Deliberate Problem Solving with Large Language Models (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — Hogwild! 与之正交的思维树推理策略
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — 投机解码，Hogwild! 与之组合的序列内并行
- [Hogwild! reference PyTorch implementation](https://github.com/eqimp/hogwild_llm) — 论文实验的单一事实来源
