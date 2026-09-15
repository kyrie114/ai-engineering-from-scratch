# 投机解码与 EAGLE-3（Speculative Decoding and EAGLE-3）

> 第 7 阶段 · 第 16 课证明了数学：Leviathan 拒绝规则精确地保留验证器（verifier）的分布。本课是 2026 年生产级投机解码（speculative decoding）的训练栈视角。EAGLE-3 把草稿模型（draft model）从廉价近似变成在验证器自身隐状态上训练的专用微型网络，再加入训练时测试（training-time test）循环，让训练分布与推理分布对齐。结果：端到端加速 3× 到 6.5×，对话上每 token 接受率超过 0.9，没有分布层面的折中。2026 年每一套生产推理栈都默认带上它。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 16 (speculative decoding math), Phase 10 · 12 (inference optimization)
**Time:** ~75 minutes

## 学习目标（Learning Objectives）

- 用一句话陈述 Leviathan 定理，并证明投机循环产生的样本与直接从验证器采样同分布
- 走一遍从朴素投机解码（Leviathan 2023）经 EAGLE、EAGLE-2 到 EAGLE-3 的两年演进，并指出每一步消除的那条精确限制
- 由接受率 `α` 与草稿-验证器成本比 `c` 计算期望加速，并为每种工况选择最优草稿长度 `N`
- 从零实现完整投机循环：草稿、验证、从残差拒绝采样、拒绝时回滚 KV 缓存、全部接受时发出奖励 token（bonus token）

## 问题（The Problem）

70B 模型的自回归解码在 H100 上大约每秒 35 个 token。GPU 远未饱和。内存带宽才是天花板：每个 token 从 HBM 加载 70B 权重，做一步算术，产出一个浮点数。计算单元大部分时间闲着。

投机解码把这件事变成一个你真正能解决的吞吐问题。廉价草稿用 `N` 次小型前向提出 `N` 个 token。验证器对前缀加上全部 `N` 个草稿只跑一次。若验证器在位置 `i` 的分布与草稿一致（我们将给出精确的统计含义），就接受；否则拒绝，并从残差分布中采样一次校正。一次大模型前向最多产出 `N+1` 个被接受的 token，而不是一个。

真正关键的定理是 Leviathan、Kalman、Matias（ICML 2023）：输出分布与直接从验证器采样完全相同。不是近似。是恒等。这就是投机解码能进生产的全部理由——它是纯延迟优化，没有质量折中。

第 7 阶段 · 第 16 课给你的是数学。本课给你的是训练栈。好草稿比廉价草稿多值 2× 加速。EAGLE、EAGLE-2 和 EAGLE-3（Li et al., 2024–2025）把“草稿 = 同一家族的更小模型”变成一门精确工程纪律。2026 年的生产推理服务器默认用 EAGLE-3。

## 概念（The Concept）

### 不变量：Leviathan 拒绝采样（The invariant: Leviathan rejection sampling）

令 `p(t)` 为给定前缀下次 token 的草稿分布，`q(t)` 为验证器分布。采样草稿 token `d ~ p`。以概率 `min(1, q(d) / p(d))` 接受。拒绝时，从残差分布 `(q - p)_+ / ||(q - p)_+||_1` 采样。得到的样本按 `q` 分布。这与 `p` 有多差无关——越差拒绝越频繁，但输出仍然精确。

把 `N` 次这样的调用首尾相接，用一次验证器前向跑在 `prefix + d_1 + ... + d_N` 上。验证器同时返回 `q_1, q_2, ..., q_{N+1}`。从左往右走。第一次在位置 `j` 拒绝时，从 `residual(q_j, p_j)` 采样并停止。全部接受时，从 `q_{N+1}` 再采一个奖励 token。

### 什么决定加速（What determines speedup）

令 `α` 为每个草稿 token 的期望接受率。令 `c = cost(draft) / cost(verifier)` 为成本比。每次验证器前向期望接受的 token 数为：

```
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

每个被接受 token 的期望墙钟时间为 `(N * c + 1) / E[accepted]`。对 `N` 最小化它，就得到甜点。对 `α = 0.8, c = 0.05`：最优 `N` 大约 5–7，加速约 3.2×。对 `α = 0.95, c = 0.02`：最优 `N` 大约 8–10，加速推向 5×。

最大的杠杆是 `α`。在固定 `N = 5` 时，从 `α = 0.6`（朴素草稿）到 `α = 0.9`（EAGLE-3），每次验证器前向的期望接受 token 从 2.2 升到 4.1。同一验证器几乎多出 2× 吞吐。

### 两年演进（The two-year progression）

**朴素投机（Vanilla speculative，Leviathan, 2023）。** 草稿模型是同一家族独立训练的更小 LLM。容易接线，`α ≈ 0.6`，加速最好大约 2×。

**EAGLE-1（Li et al., 2024）。** 草稿是一个微型 Transformer——通常一两层——以验证器最后一层隐状态为输入，直接预测下一 token。因为草稿看到了验证器的特征表示，其分布更接近验证器。`α` 升到 0.7–0.8。

**EAGLE-2（Li et al., 2024）。** 加入动态草稿树：不再提出一条长度为 `N` 的单一序列，而是提出一棵小候选树，用一次验证器前向（树注意力，tree attention）给每个节点打分，再走最高概率路径。草稿长度逐步自适应。被接受路径上每 token 的 `α` 升到 0.85 以上。

**EAGLE-3（Li et al., 2025, NeurIPS）。** 再改两处。第一，完全丢掉特征预测损失——EAGLE-1/2 训练草稿去匹配验证器隐状态，这会封顶数据能帮多少。EAGLE-3 直接在 token 预测上训练。第二，训练时测试（TTT）：草稿训练期间，把草稿自己的先前预测作为多步输入回喂，方式和推理时一样。这让训练分布与测试分布对齐，并阻止误差累积。测得加速：对话上最高 6.5×；SGLang 在 H100 上 batch 64 时吞吐提升 38%。

### KV 缓存回滚（KV cache rollback）

验证一步把验证器的 KV 缓存扩展 `N` 条。若在位置 `j` 拒绝，`j-1` 之后的缓存内容就错了。两种常见实现：写到暂存缓冲、接受时提交（vLLM、TensorRT-LLM）；或者保留物理 KV 缓存加逻辑长度，拒绝时截断。无论哪种，回滚代价是每层每头若干字节，相对前向代价可以忽略。

对 EAGLE-2 树搜索，验证器带着尊重树拓扑的非因果掩码跑注意力。工程琐碎，但计算就是一次带自定义掩码的标准 flash-attention 调用。

### 2026 年的草稿架构（Draft architectures in 2026）

| 策略 | 草稿类型 | `α` | 加速 | 训练成本 |
|----------|-----------|-----|---------|---------------|
| Vanilla | 独立小 LLM | 0.55-0.70 | 1.8-2.3× | 无（复用已有小模型） |
| Medusa | 验证器上的额外 LM 头 | 0.65-0.75 | 2-3× | ~1B SFT token |
| EAGLE-1 | 作用在隐状态上的 1 层 Transformer | 0.70-0.80 | 2.5-3× | ~60B token |
| EAGLE-2 | EAGLE-1 + 动态草稿树 | 0.80-0.88 | 3-4× | ~60B token |
| EAGLE-3 | 多层特征融合 + TTT | 0.88-0.92 | 3.5-6.5× | ~60-200B token |
| Lookahead | 无草稿（Jacobi 迭代） | N/A | 1.3-1.6× | 无 |

2026 年生产中：vLLM 和 SGLang 在可用时默认 EAGLE-3，否则 EAGLE-2。TensorRT-LLM 对 Meta 与 NVIDIA 公开模型有最快的 Medusa 路径。llama.cpp 为 CPU 部署提供朴素草稿。

```figure
l5-spec-decode-eagle
```

## 动手实现（Build It）

见 `code/main.py`。这是带齐全部零件的完整 Leviathan 投机循环：长度为 N 的草稿、验证器并行前向、按位置拒绝、残差采样、奖励 token、KV 回滚，以及输出分布与直接从 `q` 采样匹配的经验验证。

### 步骤 1：拒绝规则（Step 1: the rejection rule）

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### 步骤 2：残差分布（Step 2: residual distribution）

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### 步骤 3：一次完整投机步（Step 3: a full speculative step）

`spec_step` 函数从 `p` 草稿 `N` 个 token，再在一次并行 `q` 评估中验证全部。对每个草稿 token 应用拒绝规则，第一次拒绝时从残差采样校正。若全部接受，从 `q_{N+1}` 发出一个奖励 token。

### 步骤 4：KV 回滚记账（Step 4: KV rollback bookkeeping）

模拟器为每个 worker 跟踪逻辑 `kv_length`。接受 `k` 个草稿时，`kv_length += k`。在位置 `j` 拒绝时，缓存已经写过 `j`，但逻辑长度设为 `prefix_length + j + 1`——校正 token 之后一位。后续读取截断到逻辑长度。

### 步骤 5：Leviathan 检验（Step 5: the Leviathan check）

跑 50,000 次投机步。统计被接受 token 的经验分布。与从 `q` 直接采样 50,000 次比较。卡方统计量应远低于临界值。定理在实践中成立。

### 步骤 6：加速 vs. α（Step 6: speedup vs. α）

通过不同幅度把 `p` 从 `q` 扰动开，扫描草稿质量。测量 `α`，再画出每次验证器调用的期望 token 数随 `α` 和 `N` 的变化。代码打印一张表，展示 EAGLE-3 级草稿质量（`α ≈ 0.9`）如何解锁每次验证器调用 4–5 个 token。

## 实际应用（Use It）

带 EAGLE-3 的生产级 `vllm serve`：

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

SGLang 在 H100 上以 batch 64 跑 EAGLE-3：相对 batch-64 朴素解码大约多 1.38× 吞吐，数据来自 EAGLE-3 论文。

何时该用投机解码：

- 任何交互式对话负载，p50 延迟比峰值吞吐更重要。
- 代码生成与结构化输出（JSON、SQL）。目标分布高度可预测，`α` 超过 0.9。
- 长文生成（数千 token）。摊销后的加速持续兑现。

何时不该用：

- 非常小的模型（< 3B）。草稿并不比验证器便宜多少。
- 极小的 batch-1 CPU 部署。草稿模型的内存开销可能不值。
- 很高温度的创意采样，`α` 会塌掉。

## 交付产物（Ship It）

本课产出 `outputs/skill-eagle3-tuner.md`。给定推理负载（模型、批次大小、目标延迟、任务画像），它推荐一套投机解码策略和调参（草稿家族、`N`、树深度、温度感知切换）。

## 练习（Exercises）

1. 运行 `code/main.py`。确认 Leviathan 分布检验的卡方统计量在 50,000 个样本上低于 95% 临界值。

2. 把 `N` 从 1 扫到 10，`α` 固定 0.9、`c` 固定 0.04。画出每次验证器调用的期望 token 数，以及每 token 实际墙钟时间。找出使墙钟时间最小的 `N`。解释曲线形状。

3. 改代码模拟 EAGLE-2 树搜索：每一步草稿提出形状 `[2, 2, 2]` 的树（八条候选路径）。验证器跑一次，最高概率的被接受路径胜出。计算每叶 `α` 以及每次验证器调用的总 token 数。与等价计算量的线性链投机解码比较。

4. 实现两条并发序列的批处理 KV 回滚模拟器。序列 A 全部草稿被接受；序列 B 在位置 2 拒绝。证明正确的 `kv_length` 按序列更新，且没有浪费工作。

5. 阅读 EAGLE-3 论文第 4 节（Training-Time Test）。用两句话解释：为什么没有 TTT 的朴素草稿训练会受暴露偏差（exposure bias）之苦，以及为什么训练时把草稿自己的预测喂回去能修好它。把它与 seq2seq 里的计划采样（scheduled sampling）文献连起来。

## 关键术语（Key Terms）

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Leviathan rule | “min(1, q 除以 p)” | 以概率 `min(1, q(d)/p(d))` 做伯努利接受/拒绝；拒绝时从残差采样，精确保留验证器分布 |
| Residual distribution | “(q 减 p) 取正，再归一化” | `(q - p)_+` 在零处截断并重新归一化——拒绝时正确的采样分布 |
| Acceptance rate α | “草稿多对” | 拒绝规则下每 token 伯努利成功的期望概率；支配全部加速数学 |
| EAGLE-1 | “隐状态草稿” | 以验证器最后一层隐状态为条件的微型 Transformer 草稿（Li et al., 2024） |
| EAGLE-2 | “动态草稿树” | EAGLE-1 加上一棵候选续写树，用一次验证器前向的树注意力打分 |
| EAGLE-3 | “训练时测试” | 丢掉特征预测损失，直接做 token 预测，训练时把草稿自己的输出喂回去 |
| Training-time test (TTT) | “暴露偏差修复” | 训练时自回归地跑草稿，使训练与测试输入分布匹配——计划采样的直接类比 |
| KV rollback | “撤销被拒绝的草稿” | 拒绝后把验证器 KV 缓存重置到被接受前缀长度的记账 |
| Bonus token | “白送的那个” | 当全部 `N` 个草稿都被接受时，从 `q_{N+1}` 再采一个，不额外花验证器代价 |
| Tree attention | “一次验证许多候选” | 带尊重草稿树拓扑的非因果掩码的注意力；一次前向为树中每个节点计算 `q_i` |

## 延伸阅读（Further Reading）

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — 奠基论文与等价定理
- [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) — 同期独立引入，证明干净
- [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — EAGLE-1，以隐状态为条件的草稿
- [Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — 动态树搜索
- [Li et al. — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) — 2026 年生产默认
- [Cai et al. — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) — 另一种无独立草稿的路径
- [vLLM Speculative Decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 把全部策略接好的典范生产参考
