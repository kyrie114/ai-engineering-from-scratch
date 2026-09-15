# 投机解码与 EAGLE（Speculative Decoding and EAGLE）

> 前沿 LLM 生成一个 token 需要在数十亿参数上做一次完整前向。那次前向被严重超额配置：大多数时候，一个小得多的模型就能正确猜出接下来 3–5 个 token，大模型只需要*验证*这个猜测。猜对时，你用一次的价钱拿到 5 个 token。投机解码（speculative decoding，Leviathan 等人 2023）把这件事做精确了，EAGLE-3（2025）把接受率推到每次验证约 4.5 个 token——在匹配输出分布下 4–5 倍加速。

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10 Lesson 12 (Inference Optimization), Phase 10 Lesson 04 (Pre-training Mini-GPT)
**Time:** ~75 minutes

## 问题（The Problem）

70B 级模型在 H100 上的解码吞吐通常是 40–80 tokens/second。每个 token 都需要一次完整前向，从 HBM 读出全部模型权重。你不能把模型变小而不改变输出。你不能把批量加到超出内存。你卡住了——除非能让模型每次前向输出不止一个 token。

自回归生成看起来天生串行：`x_{t+1} = sample(p(· | x_{1:t}))`。但存在并发机会。如果你有一个廉价预测器说「接下来 4 个 token 大概是 [a, b, c, d]」，你可以在**大模型的单次前向**里验证全部 5 个位置，并接受最长匹配前缀。

Leviathan、Kalai、Matias（2023，「Fast Inference from Transformers via Speculative Decoding」）通过巧妙的接受/拒绝规则把这件事做精确，保住目标模型的采样分布。同样的输出分布，快 2–4 倍。

## 概念（The Concept）

### 双模型设置（The Two-Model Setup）

- **目标模型（target model）** `M_p`：你真正想从中采样的那个大、慢、高质量模型。分布：`p(x)`。
- **草稿模型（draft model）** `M_q`：小、快、质量较低的模型。分布：`q(x)`。小 5–30 倍。

每一步：

1. 草稿模型自回归提出 `K` 个 token：`x_1, x_2, ..., x_K ~ q`。
2. 目标模型对全部 `K+1` 个位置并行跑一次前向，为每个被提出的 token 产出 `p(x_k)`。
3. 用下面的修正拒绝采样规则从左到右接受/拒绝每个 token。接受最长匹配前缀。
4. 若有任何 token 被拒绝，从校正分布采样替换并停止。否则从 `p(· | x_1...x_K)` 再采样一个奖励 token（bonus token）。

若草稿与目标完美匹配，每次目标前向得到 K+1 个 token。若草稿在位置 1 就错，你只得到 1 个 token。

### 精确性规则（The Exactness Rule）

投机解码在分布上**可证明等价于从 p 采样**。拒绝规则：

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

其中 `(p - q)+` 表示逐点差的正部。当草稿与目标一致（`p ≈ q`）时接受率接近 1。当它们不一致时，残差分布被构造成使总体样本仍精确为 `p`。

**贪心情形。** 对 temperature=0 采样只需检查 `argmax(p) == x_t`。是则接受；否则输出 `argmax(p)` 并停止。

### 期望加速（Expected Speedup）

若草稿模型的 token 级接受率是 `α`，每次目标前向产出的期望 token 数是：

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

在 `α = 0.8, K = 4`：`(1 - 0.8^5)/(1 - 0.8) = 3.36` 个 token 每次前向。单次目标前向的代价大约是 `cost_q * K + cost_p`（K 次草稿步加上一次目标验证）。若 `cost_p >> cost_q * K`，吞吐加速比是 `3.36× / 1 = 3.36×`。

唯一真正的参数是 `α`，它完全取决于草稿-目标对齐。好草稿就是一切。

### 训练草稿：蒸馏（Training the Draft: Distillation）

随机小模型是差草稿。标准配方是从目标蒸馏：

1. 选一个小架构（70B 目标大约 1B，7B 目标大约 500M）。
2. 在大规模文本语料上跑目标模型；存储它的下一 token 分布。
3. 用相对目标分布的 KL 散度训练草稿（不是相对真值 token）。

结果：编码上 `α` 通常 0.6–0.8，自然语言聊天上 0.7–0.85。生产中加速 2–3 倍。

### EAGLE：树草稿 + 特征复用（EAGLE: Tree Drafting + Feature Reuse）

Li、Wei、Zhang、Zhang（2024，「EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty」）观察到标准投机解码的两处低效：

1. 草稿做 K 次串行步，每步全栈。但草稿可以复用最近一次验证里目标的特征（隐状态）——目标已经算出丰富表示，草稿却从头再推一遍。
2. 草稿输出一条线性链。如果草稿能输出候选的*树*（每个节点多个猜测），目标的单次前向就能通过树注意力掩码并行验证多条候选路径，并挑最长被接受的分支。

EAGLE-1 的改动：
- 草稿输入 = 位置 t 处目标的最终隐状态，而不是原始 token。
- 草稿架构 = 1 个 transformer 解码器层（不是单独的小模型）。
- 输出 = 每层深度 K = 4–8 个候选的树，深度 4–6。

EAGLE-2（2024）加入动态树拓扑：草稿不确定的地方树变宽，有把握的地方保持窄。在不增加验证成本的情况下提高 `α_effective`。

EAGLE-3（Li 等人 2025，「EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test」）去掉固定顶层特征依赖，用新的「测试时模拟」损失训练草稿——草稿在匹配目标测试时分布的输出上训练，而不是教师强制的训练分布。接受率从 0.75（EAGLE-2）升到 0.82（EAGLE-3），每次验证的平均 token 从 3.0 升到 4.5。

### 树注意力验证（Tree Attention Verification）

草稿输出一棵树时，目标模型用**树注意力掩码（tree attention mask）**在单次前向里验证它——编码树拓扑而不是纯直线的因果掩码。每个 token 只关注它在树中的祖先。验证遍仍是一次前向、一次矩阵乘；拓扑掩码只多几个 KV 条目。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

若 `a, b` 是竞争的第一 token 候选，`c, d, e, f` 是第二 token 候选，六个位置都在一次前向里验证。输出是任意被接受路径上的最长前缀。

### 何时赢，何时不赢（When It Wins, When It Doesn't）

**赢：**
- 文本可预测的聊天 / 补全（代码、常见英语、结构化输出）。`α` 高。
- 解码期间有未用 GPU 算力的设置（内存受限阶段）。树草稿用上可用 FLOPs。

**输 / 不赢：**
- 高度随机的输出（高温创造性写作）。`α` 掉向 `1/|vocab|`。
- 并发非常高的批量服务——批处理已经填满 FLOPs，树验证空间很小。
- 目标模型很小、草稿也大不了多少。

生产团队通常报告聊天上 2–3 倍墙钟加速，代码生成上 3–5 倍，创造性写作上接近零。

```figure
speculative-decoding
```

## 动手构建（Build It）

`code/main.py`：

- 一份参考 `speculative_decode(target, draft, prompt, K, temperature)`，实现精确拒绝规则，并验证它保住目标分布（相对朴素目标采样的经验 KL < 0.01）。
- 一个 EAGLE 风格树草稿器，构建带 top-p 分支的深度 K 树。
- 一个树注意力掩码构建器，为验证器产出正确的因果模式。
- 一个接受率测试架，在微型 LM 上跑两者（从 GPT-2-medium 目标蒸馏一个 GPT-2-small）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 使用它（Use It）

- **vLLM** 和 **SGLang** 提供一等公民投机解码。标志：`--speculative_model`、`--num_speculative_tokens`。EAGLE-2/3 支持经由 `--spec_decoding_algorithm eagle` 标志。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE 树。
- **参考草稿模型**：`Qwen/Qwen3-0.6B-spec`（为 Qwen3-32B 草稿），`meta-llama/Llama-3.2-1B-Instruct-spec`（为 70B 草稿）。
- **Medusa 头**（Cai 等人 2024，「Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads」）：不是草稿模型，而是给目标本身加 K 个并行预测头。部署更简单，接受率略低于 EAGLE。

## 交付（Ship It）

本课产出 `outputs/skill-speculative-tuning.md`——一份技能，给目标模型的工作负载画像并选择：草稿模型、K（草稿长度）、树宽、温度，以及何时回退到朴素解码。

## 练习（Exercises）

1. 实现精确拒绝规则并经验验证。经 `speculative_decode` 和朴素目标采样各跑 10K 样本；计算两个输出分布之间的 TV 距离。应小于 0.01。

2. 计算加速公式。给定固定 `α` 和 `K`，画出每次目标前向的期望 token。找出 α ∈ {0.5, 0.7, 0.9} 的最优 K。

3. 训练一个微型草稿。拿 124M GPT-2 目标，用 KL 损失在 100M token 上蒸馏 30M GPT-2 草稿。在留出文本上测 `α`。预期：0.6–0.7。

4. 实现 EAGLE 风格树草稿。不是链，让草稿在每一深度输出 top-3 分支。构建树注意力掩码。验证目标接受最长正确分支。

5. 测量失败模式。在 temperature=1.5（高随机性）跑投机解码。展示 α 崩溃，算法因草稿开销比朴素解码更慢。

## 关键术语（Key Terms）

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 目标模型 | 「那个大模型」 | 你想从中采样的慢、高质量模型（p 分布） |
| 草稿模型 | 「投机器」 | 小、快的预测器（q 分布）；小 5–30 倍 |
| K / 草稿长度 | 「前瞻」 | 每次验证提出的投机 token 数 |
| α / 接受率 | 「命中率」 | 草稿提案被接受的每 token 概率 |
| 精确拒绝规则 | 「接受测试」 | 保住目标分布的 r < p/q 比较 |
| 残差分布 | 「校正后的 p-q」 | (p - q)+ / ||(p - q)+||_1，拒绝时从中采样的分布 |
| 树草稿 | 「分支投机」 | 草稿输出候选树，用树结构注意力掩码在一次验证 |
| 树注意力掩码 | 「拓扑掩码」 | 编码树拓扑的因果掩码，使每个节点只关注祖先 |
| Medusa 头 | 「并行头」 | 目标本身上的 K 个额外预测头；没有单独草稿模型 |
| EAGLE 特征复用 | 「隐状态草稿」 | 草稿输入是目标的最后隐状态，而不是原始 token，从而缩小草稿 |
| 测试时模拟损失 | 「EAGLE-3 训练」 | 在匹配目标测试时分布的输出上训练草稿，而不是教师强制 |

## 延伸阅读（Further Reading）

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 精确拒绝规则与理论加速分析
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind 同期投机采样论文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — 相对草稿模型的并行头替代
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — 特征复用与树草稿
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — 动态树拓扑
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — 训练时与测试时匹配
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi/前瞻解码，一种无投机器替代
