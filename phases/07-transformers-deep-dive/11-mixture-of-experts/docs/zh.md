# Mixture of Experts (MoE)（混合专家模型）

> 一个 dense 70B transformer 为每个 token 激活每个参数。一个 671B MoE 每个 token 只激活 37B，并在每个基准上击败它。稀疏性是这十年最重要的扩展思想。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer)（完整 Transformer）, Phase 7 · 07 (GPT)（GPT）
**Time:** ~45 minutes（约 45 分钟）

## The Problem（问题）

一个 dense transformer 的推理 FLOPs 等于它的参数数量（前向传播乘以 2）。扩展一个 dense 模型，每个 token 支付全额费用。到 2024 年，前沿 hitting 了一个 compute wall：要真正更智能，你需要指数级更多的每 token FLOPs。

Mixture of Experts 打破了这个联系。把每个 FFN 替换为 `E` 个独立的专家 + 一个为每个 token 挑选 `k` 个专家的路由器。总参数 = `E × FFN_size`。每个 token 的活跃参数 = `k × FFN_size`。典型 2026 配置：`E=256`，`k=8`。存储随 `E` 扩展，计算随 `k` 扩展。

2026 年的前沿几乎全是 MoE：DeepSeek-V3（671B total / 37B active）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立排行榜上，前 10 个开源模型全是 MoE。

## The Concept（概念）

![MoE layer: router selects k of E experts per token（MoE 层：路由器为每个 token 从 E 个专家中选择 k 个）](../assets/moe.svg)

### FFN swap（FFN 替换）

Dense transformer block：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE block：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token（为每个 token 从 E 中选 k 个）
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个专家是一个独立的 FFN（通常是 SwiGLU）。路由器是一个单一的 linear layer。每个 token 挑选它自己的 `k` 个专家，并获得它们输出的 gated mixture。

### 负载均衡问题

如果路由器把 90% 的 token 放在 expert 3 上，其他专家就会饿死。已经尝试了三种修复：

1. **Auxiliary load-balancing loss**（辅助负载均衡损失）（Switch Transformer、Mixtral）。添加一个与专家使用方差成比例的惩罚。有效，但添加一个超参数和第二个梯度信号。
2. **Expert capacity + token dropping**（专家容量 + token 丢弃）（早期 Switch）。每个专家最多处理 `C × N/E` 个 token；溢出 token 跳过该层。伤害质量。
3. **Auxiliary-loss-free balancing**（无辅助损失均衡）（DeepSeek-V3）。添加一个学到的 per-expert bias，偏移路由器的 top-k 选择。Bias 在训练损失之外更新。主目标上没有惩罚。2024 年的大解锁。

DeepSeek-V3 的方法：在每个训练步骤之后，对于每个专家，检查其使用率是否高于或低于目标。通过 `±γ` 微调 bias。选择使用 `scores + bias`。用于门控的专家概率是原始 `scores` 不变。解耦路由和表达。

### Shared experts（共享专家）

DeepSeek-V2/V3 也把专家分成 *shared* 和 *routed*。每个 token 通过所有共享专家。Routed 专家通过 top-k 挑选。共享专家捕获通用知识；routed 专家专门化。V3 运行 1 个共享专家加上 256 个 routed 中的 top-8。

### Fine-grained experts（细粒度专家）

经典 MoE（GShard、Switch）：每个专家和完整 FFN 一样宽。`E` 小（8–64），`k` 小（1–2）。

现代细粒度 MoE（DeepSeek-V3、Qwen-MoE）：每个专家更窄（1/8 FFN 大小）。`E` 大（256+），`k` 更大（8+）。相同总参数，但组合扩展快得多。`C(256, 8) = 400 trillion` 每个 token 可能的“专家”。质量上升，延迟保持平坦。

### 成本概况

每个 token，每层：

| Config（配置） | Active params / token（活跃参数/ token） | Total params（总参数） |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (dense)（dense） | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3 在几乎每个基准上都击败 Llama 3 70B（dense），同时做**更少的每 token 活跃 FLOPs**。更多参数 = 更多知识。更多活跃 FLOPs = 每个 token 更多计算。MoE 解耦它们。

### 陷阱：内存

所有专家都活在 GPU 上，无论哪个触发。一个 671B 模型需要 ~1.3 TB 的 VRAM 用于 fp16 权重。前沿 MoE 部署需要 expert parallelism —— 在 GPU 之间分片专家，跨网络路由 token。延迟由 all-to-all 通信主导，而不是矩阵乘法。

```figure
expert-routing
```

## Build It（动手实现）

见 `code/main.py`。一个紧凑的纯 stdlib MoE 层，包含：

- `n_experts=8` SwiGLU-ish 专家（每个一个 linear，用于说明）
- top-k=2 路由
- softmax-normalized 门控权重
- 通过 per-expert bias 实现 auxiliary-loss-free balancing

### Step 1: 路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts（对所选专家的原始分数做 softmax）
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

Bias 影响选择，不影响门控权重。这就是 DeepSeek-V3 技巧 —— bias 修正负载不平衡而不 steering 模型的预测。

### Step 2: 运行 100 token 通过路由器

跟踪哪些专家触发多频繁。没有 bias 时，使用率倾斜。有了 bias update loop（过度使用的专家 `-γ`，低于使用的 `+γ`），使用率在几次迭代中收敛到均匀分布。

### Step 3: 参数数量比较

打印 MoE 配置的“dense equivalent”。DeepSeek-V3-shaped：256 routed + 1 shared，8 active，d_model=7168。总参数数量令人眼花缭乱。活跃数量是 dense Llama 3 70B 的七分之一。

## Use It（实际应用）

HuggingFace 加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 生产推理：vLLM 原生支持 MoE 路由。SGLang 有最快的 expert-parallel path。两者自动处理 top-k 选择和 expert parallelism。

**何时选择 MoE：**

- 你希望在较低推理成本 per token 下获得前沿质量。
- 你有 VRAM / expert-parallel 基础设施。
- 你的工作负载是 token-heavy（chat、code）而不是 context-heavy（长文档）。

**何时不选 MoE：**

- 边缘部署 —— 你为任何活跃 FLOP 支付全额存储。
- 延迟关键的单一用户服务 —— expert routing 增加开销。
- 小模型（<7B）—— MoE 的质量优势只出现在计算阈值以上（~6B active params）。

## Ship It（交付）

见 `outputs/skill-moe-configurator.md`。这个 skill 在给定参数预算、训练 token 和部署目标的情况下，为一个新 MoE 选择 E、k 和 shared-expert layout。

## Exercises（练习）

1. **Easy（简单）。** 运行 `code/main.py`。观察 auxiliary-loss-free bias update 如何在 50 次迭代内使专家使用均匀化。
2. **Medium（中等）。** 用基于 hash 的路由器替换学到的路由器（确定性，无学习）。比较质量和平衡。为什么学到的路由器更好？
3. **Hard（困难）。** 实现 GRPO-style "rollout-matched routing"（DeepSeek-V3.2 技巧）：记录推理时哪些专家触发，在梯度计算期间强制相同的路由。在 toy policy-gradient 设置上测量效果。

## Key Terms（关键术语）

| Term（术语） | What people say（常听到的说法） | What it actually means（实际含义） |
|------|-----------------|-------|
| Expert（专家） | "One FFN among many"（许多 FFN 中的一个） | 一个独立的前馈网络；参数专用于稀疏的 FFN 计算切片。 |
| Router（路由器） | "The gate"（门） | 一个 tiny linear layer，把每个 token 对每个 expert 评分；top-k 选择。 |
| Top-k routing（top-k 路由） | "k active experts per token"（每个 token k 个活跃专家） | 每个 token 的 FFN 计算恰好通过 k 个专家，按门控加权。 |
| Auxiliary loss（辅助损失） | "Load-balance penalty"（负载均衡惩罚） | 惩罚倾斜的专家使用的额外损失项。 |
| Auxiliary-loss-free（无辅助损失） | "DeepSeek-V3's trick"（DeepSeek-V3 的技巧） | 通过路由器选择上的 per-expert bias 平衡；没有额外梯度。 |
| Shared expert（共享专家） | "Always on"（总是开启） | 每个 token 都通过的额外专家；捕获通用知识。 |
| Expert parallelism（专家并行） | "Shard by expert"（按专家分片） | 把不同的专家分配到不同的 GPU；跨网络路由 token。 |
| Sparsity（稀疏性） | "Active params < total params"（活跃参数 < 总参数） | 比率 `k × expert_size / (E × expert_size)`；DeepSeek-V3 是 37/671 ≈ 5.5%。 |

## Further Reading（延伸阅读）

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — 这个想法。
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch，经典 MoE。
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B。
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + auxiliary-loss-free MoE + MTP。
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — 基于 bias 的均衡论文。
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — 本课路由器使用的细粒度 + shared-expert 拆分。
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 原始 shared-expert 论文。
