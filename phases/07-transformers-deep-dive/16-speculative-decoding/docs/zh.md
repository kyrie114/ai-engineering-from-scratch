# Speculative Decoding — Draft, Verify, Repeat（推测解码 —— 草稿、验证、重复）

> 自回归解码是串行的。每个 token 等待前一个。推测解码打破链：一个廉价模型草拟 N 个 token，昂贵的模型在一次前向传播中验证所有 N 个。当草稿正确时，你为 N 次生成支付了一次大模型前向传播。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 07 (GPT Causal LM)（GPT 因果语言模型）, Phase 7 · 12 (KV Cache & Flash Attention)（KV Cache 与 Flash Attention）
**Time:** ~60 minutes（约 60 分钟）

## The Problem（问题）

一个 70B LLM 采样一个 token 在 H100 上需要 ~30 ms。一个 3B 草稿模型需要 ~3 ms。如果我们让 3B 草拟 5 个 token ahead，然后运行 70B *一次* 来验证所有 5 个，总计是 `5×3 + 30 = 45 ms` 用于最多 5 个 accepted token —— versus `5×30 = 150 ms` 用于 straight-line generation。这就是完整的推测解码 pitch：交换少量额外 GPU 内存（草稿模型）来获得 2–4× 更低解码延迟。

这个技巧必须保留分布。由 Leviathan et al. (2023) 引入并由 Chen et al. 同时引入的推测采样，保证输出序列与如果大模型自己采样会是 **identically distributed** 的。没有质量权衡。只是更快。

四个 draft-verifier pair 家族主导 2026 推理：

1. **Vanilla speculative (Leviathan 2023)。** 单独的草稿模型（例如，Llama 3 1B）+ 验证器（例如，Llama 3 70B）。
2. **Medusa (Cai 2024)。** 验证器上的多个解码头并行预测位置 `t+1..t+k`。没有单独的草稿模型。
3. **EAGLE family (Li 2024, 2025)。** 轻量级草稿，重用验证器的隐藏状态；比 vanilla 更高的接受率；典型 3–4×。
4. **Lookahead decoding (Fu 2024)。** Jacobi iteration；根本不需要草稿模型。Self-speculation。利基但 dependency-free。

2026 年每个生产推理栈默认 shipped 推测解码。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 都至少支持 vanilla + EAGLE-2。

## The Concept（概念）

### The core algorithm（核心算法）

给定一个验证器 `M_q` 和一个更便宜的草稿 `M_p`：

1. 让 `x_1..x_k` 是已经解码的前缀。
2. **草稿**：使用 `M_p` 自回归地提出 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，带有草稿概率 `p_1..p_N`。
3. **并行验证**：在 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 上运行 `M_q` 一次，得到位置 `k+1..k+N+1` 的验证器概率 `q_1..q_{N+1}`。
4. **从左到右接受/拒绝每个草稿 token**：对于每个 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受。
5. 在位置 `j` 第一次拒绝时：从“residual”分布 `(q_j - p_j)_+` 归一化采样 `t_j`。`j` 之后的所有草稿都被丢弃。
6. 在接受所有 `N` 时：从 `q_{N+1}`（免费 bonus token）采样一个额外 token `t_{N+1}`。

Residual distribution 技巧是保持输出与如果 `M_q` 从头开始采样将产生的 marginal distribution 完全相同的数学洞察。

### What determines speedup（什么决定加速）

让 `α` = 每个草稿 token 的预期接受率。让 `c` = 草稿到验证器成本比率。每步：

- Naive generation 为每个 token 做 1 次大模型调用。
- Speculative 为 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个 token 做 1 次大模型调用，当 `α` 高时。

在 `α = 0.75` 和 `N = 5` 时的典型经验法则：3× 更少的大模型调用。草稿成本是 5× cheap。总 wall-clock 下降 ~2.5×。

**`α` 取决于：**

- 草稿在多大程度上近似验证器。相同家族 / 相同训练数据显著提升 `α`。
- 解码策略。Greedy 草稿对 greedy 验证器：高 `α`。Temperature sampling：更难匹配；接受率下降。
- 任务类型。代码和结构化输出接受更多（可预测）；free-form creative writing 接受更少。

### Medusa —— 没有草稿模型的草稿

Medusa 用验证器上的额外输出头替换草稿模型。在位置 `t`：

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

每个头输出它自己的 logits。在推理时，你从每个头采样得到一个候选序列，然后用一个 tree-attention scheme 一次性用一次前向传播验证所有候选延续。

优点：没有第二个模型。缺点：添加可训练参数；需要一个监督微调阶段（~1B token）；接受率比 vanilla speculative 用好的草稿略低。

### EAGLE —— 通过重用隐藏状态改进草稿

EAGLE-1/2/3（Li et al., 2024–2025）把草稿模型变成一个 tiny transformer（通常 1 层），摄入验证器的最后层隐藏状态。因为草稿看到验证器的特征表示，它的预测与验证器的输出分布强相关。接受率从 ~0.6（vanilla）上升到 0.85+。

EAGLE-3 (2025) 添加了候选延续上的 tree search。vLLM 和 SGLang 为 Llama 3/4 和 Qwen 3  shipped EAGLE-2/3 作为默认 spec pathway。

### The KV cache dance（KV cache 舞蹈）

验证在一次性前向传播中把 N 个草稿 token 送入验证器。这通过 N 个条目扩展验证器的 KV cache。如果一些草稿被拒绝，你必须把 cache rollback 到接受的前缀长度。

生产实现（vLLM 的 `--speculative-model`、TensorRT-LLM 的 LookaheadDecoder）用 scratch KV buffer 处理这个。先写，在 acceptance 时提交。概念上不难，但它很 fiddly。

```figure
draft-verify-tokens
```

## Build It（动手实现）

见 `code/main.py`。我们实现核心推测采样算法（拒绝步骤 + residual distribution），包含：

- 一个“big model”，它是一个 hand-coded distribution 上的 deterministic-softmax（这样我们可以分析性地验证 acceptance math）。
- 一个“draft model”，它是大模型的扰动。
- 一个产生与直接采样验证器相同的 marginal distribution 的 acceptance / rejection loop。

### Step 1: the rejection step（拒绝步骤）

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是一个 uniform random number。`q_prob` 是验证器对草稿 token 的概率。`p_prob` 是草稿模型的概率。Leviathan theorem 是这个 Bernoulli decision， followed by sampling from the residual on rejection，精确保留验证器的分布。

### Step 2: residual distribution（残差分布）

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

从 `q` 中逐元素减去 `p`，把负值 clamp 到零，在拒绝时从这个 renormalize 采样。

### Step 3: one speculative step（一个推测步骤）

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

五个 accepted → 一个 bonus → 在一个验证器 pass 中产生六个 token。

### Step 4: measure acceptance rate（测量接受率）

在不同草稿质量级别运行 10,000 个推测步骤。绘制接受率 vs 草稿和验证器分布之间的 KL divergence。你应该看到一个干净的单调关系。

### Step 5: verify distribution equivalence（验证分布等价）

经验上：推测循环产生的 token 直方图应该匹配直接从验证器采样产生的直方图。这是实践中的 Leviathan theorem。卡方检验在采样误差内确认。

## Use It（实际应用）

生产：

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

TensorRT-LLM 在 2026 年中期拥有最快的 Medusa path。`faster-whisper` 用一个小草稿包装推测解码用于 Whisper-large。

**挑选草稿：**

| Strategy（策略） | When to pick（何时选择） | Speedup（加速） |
|----------|--------------|---------|
| Vanilla draft (1B/3B Llama family)（vanilla 草稿） | Fast prototype, no training（快速原型，无训练） | 1.8–2.3× |
| Medusa heads（Medusa 头） | You can fine-tune the verifier（你可以微调验证器） | 2–3× |
| EAGLE-2 / 3 | Production, max speed（生产，最大速度） | 3–4× |
| Lookahead（前视） | No draft, no training, no extra params（没有草稿，没有训练，没有额外参数） | 1.3–1.6× |

**何时不推测解码：**

- 1–5 token 的单一序列生成。开销占主导。
-  wildly creative / high-temperature sampling（`α` 下降）。
- 内存受限部署（草稿模型增加 VRAM）。

## Ship It（交付）

见 `outputs/skill-spec-decode-picker.md`。这个 skill 为一个新的推理工作负载选择推测解码策略（vanilla / Medusa / EAGLE / lookahead）和 tuning 参数（N、draft temperature）。

## Exercises（练习）

1. **Easy（简单）。** 运行 `code/main.py`。确认在 50,000 token 上，推测 token 分布与验证器的直接采样分布在卡方 p > 0.05 内匹配。
2. **Medium（中等）。** 绘制 speedup（每 big-model forward 的 token）作为 `N` 的函数，对于 `α = 0.5, 0.7, 0.85`。为每个 `α` 确定最佳 `N`。（提示：每个 verify call 的预期 token = `(1 - α^{N+1}) / (1 - α)`。）
3. **Hard（困难）。** 实现一个 tiny Medusa：拿 Lesson 14 的毕业设计 GPT，添加 3 个额外的 LM 头，预测位置 t+2、t+3、t+4。在 tinyshakespeare 上用联合多头损失训练。与通过截断相同模型制作的 vanilla 草稿比较接受率。
4. **Hard（困难）。** 实现 rollback：从一个 10 token 前缀 KV cache 开始，feed 5 个草稿 token，模拟在位置 3 的拒绝。验证你的 cache 读取在下一个迭代中与“prefix + first 2 accepted drafts”匹配。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Draft model（草稿模型） | "The cheap one"（便宜的那个） | 一个更小的模型，提出候选 token；通常比验证器便宜 10–50 倍。 |
| Verifier（验证器） | "The big one"（大的那个） | 我们保留其分布的目标模型；每个推测步骤运行一次。 |
| Acceptance rate (α)（接受率） | "How often the draft is right"（草稿多频繁正确） | 验证器接受草稿的每个 token 概率。典型 0.7–0.9。 |
| Residual distribution（残差分布） | "The rejection fallback"（拒绝回退） | `(q - p)_+` 归一化；在拒绝时从中采样保留验证器的分布。 |
| Bonus token（bonus token） | "The free one"（免费的那个） | 当所有 N 个草稿被接受时，从验证器的下一步分布采样一个额外 token。 |
| Medusa | "Draft-less speculative"（无草稿推测） | 验证器上的多个 LM 头并行预测位置 t+1..t+k。 |
| EAGLE | "Hidden-state draft"（隐藏状态草稿） | 条件于验证器最后层隐藏状态的 tiny transformer 草稿。 |
| Lookahead decoding（前视解码） | "Jacobi iteration"（Jacobi 迭代） | 使用不动点迭代的 self-speculation；没有草稿模型。 |
| Tree attention（树注意力） | "Verify many candidates at once"（一次性验证许多候选） | 一次考虑几个草稿延续的 branching 验证。 |
| KV rollback（KV 回滚） | "Undo rejected drafts"（撤销拒绝的草稿） | Scratch KV buffer；accept 时提交，reject 时丢弃。 |

## Further Reading（延伸阅读）

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) —— 核心算法和等价定理。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) —— 同时引入；clean Bernoulli-rejection proof。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) —— Medusa 论文；tree-attention verification。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) —— EAGLE-1；hidden-state-conditioned 草稿。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) —— EAGLE-2；dynamic tree depth。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) —— EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) —— lookahead，无草稿方法。
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) —— 所有四种策略连接的规范生产参考。
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) —— EAGLE-1/2/3 的参考代码。
